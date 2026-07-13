"""
corr_signal.py
==============
Channel B input: realized daily bond-equity correlation, sampled causally at
month-ends for the correlation-conditioned core split.

    rho63_t = corr( r_eq_d , r_bond_d ) over the trailing 63 trading days,
              evaluated at the last trading day <= month-end t.

Design invariants (mirror vol_signal.py)
----------------------------------------
1. SIZING ONLY. rho63 feeds corr_adjusted_core_split (Stage 2). It never
   touches Stage-1 selection, candidate definitions, or the HMM.
2. CAUSAL. The value stamped on month-end t uses daily data through the
   last trading day <= t only. Verified by a truncation test in _test().
3. PROXY-AWARE. The bond leg may be an ETF proxy (IEF, corr 0.998 with
   LT09TRUU monthly, beta 1.01, TE 0.44%/yr on 2004-2025). The proxy is
   signal-side only; LT09TRUU remains the modeled/traded asset.
4. Months with fewer than min_days trailing observations get NaN, and the
   engine falls back to the mixture rho (or the fixed split) rather than
   acting on an under-sampled estimate.

Run unit tests:  python -m src.corr_signal
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # package layout (src/) or side-by-side
    from src.vol_signal import load_daily_close, daily_log_returns
except ImportError:
    from vol_signal import load_daily_close, daily_log_returns


def aligned_daily_returns(
    equity_csv: str | Path,
    bond_csv: str | Path,
    price_field: str = "Close",
) -> pd.DataFrame:
    """Inner-join daily log returns of the two legs. Columns: ['eq', 'bond']."""
    eq = daily_log_returns(load_daily_close(equity_csv, price_field=price_field))
    bd = daily_log_returns(load_daily_close(bond_csv, price_field=price_field))
    out = pd.DataFrame({"eq": eq, "bond": bd}).dropna()
    if out.empty:
        raise ValueError("No overlapping daily observations between the two legs.")
    return out


def rolling_realized_corr(
    daily: pd.DataFrame,
    window: int = 63,
    min_periods: int = 40,
) -> pd.Series:
    """Trailing rolling correlation on daily log returns (daily index)."""
    return daily["eq"].rolling(window, min_periods=min_periods).corr(daily["bond"])


def month_end_series(daily_series: pd.Series) -> pd.Series:
    """
    Sample a daily series at calendar month-ends: the LAST available daily
    value <= each month-end (strictly causal). Index: month-end timestamps,
    matching the engine's monthly convention.
    """
    s = daily_series.dropna()
    per = s.index.to_period("M")
    out = s.groupby(per).last()
    out.index = out.index.to_timestamp("M")
    return out


def build_monthly_realized_corr(
    equity_csv: str | Path,
    bond_csv: str | Path,
    window: int = 63,
    min_periods: int = 40,
    min_days_in_month: int = 10,
    price_field: str = "Close",
) -> pd.Series:
    """
    One-call convenience: two daily CSVs -> month-end rho63 series.

    Months whose calendar month contributed fewer than min_days_in_month
    joint trading days are masked NaN (partial first/last months).
    The engine consumes this directly:  rho_realized = build_monthly_realized_corr(...)
    """
    daily = aligned_daily_returns(equity_csv, bond_csv, price_field=price_field)
    rho_d = rolling_realized_corr(daily, window=window, min_periods=min_periods)
    rho_m = month_end_series(rho_d)

    n_days = daily["eq"].groupby(daily.index.to_period("M")).size()
    n_days.index = n_days.index.to_timestamp("M")
    rho_m[n_days.reindex(rho_m.index).fillna(0) < min_days_in_month] = np.nan
    rho_m.name = f"rho{window}d"
    return rho_m


def build_monthly_horizon_corr(
    equity_csv: str | Path,
    bond_csv: str | Path,
    horizon_days: int = 21,
    window_days: int = 252,
    min_periods: int = 150,
    min_days_in_month: int = 10,
    price_field: str = "Close",
) -> pd.Series:
    """
    HORIZON-MATCHED realized correlation: correlation of overlapping
    horizon_days-day log returns over a trailing window_days window,
    sampled causally at month-ends.

    Rationale (validated Jul 2026): stock-bond correlation is
    frequency-dependent. In 2022 the 63d daily-return correlation stayed
    NEGATIVE through H1 (flight-to-quality micro-dynamics) while the
    monthly-horizon comovement had turned positive in mid-2021. A monthly
    rebalanced engine must condition on the monthly-horizon correlation;
    overlapping 21d returns deliver that horizon with daily updating and
    far less noise than a 12-month rolling monthly correlation.
    """
    daily = aligned_daily_returns(equity_csv, bond_csv, price_field=price_field)
    eq_h = daily["eq"].rolling(horizon_days).sum()
    bd_h = daily["bond"].rolling(horizon_days).sum()
    rho_d = eq_h.rolling(window_days, min_periods=min_periods).corr(bd_h)
    rho_m = month_end_series(rho_d)

    n_days = daily["eq"].groupby(daily.index.to_period("M")).size()
    n_days.index = n_days.index.to_timestamp("M")
    rho_m[n_days.reindex(rho_m.index).fillna(0) < min_days_in_month] = np.nan
    rho_m.name = f"rho_h{horizon_days}x{window_days}"
    return rho_m


def blend_rho(
    rho_realized: float | None,
    rho_mixture: float | None,
    w_realized: float,
) -> float:
    """
    rho_eff = w * rho_realized + (1 - w) * rho_mixture, NaN-safe:
    if one input is missing, fall back to the other; if both, NaN.
    """
    rr = np.nan if rho_realized is None else float(rho_realized)
    rm = np.nan if rho_mixture is None else float(rho_mixture)
    if np.isfinite(rr) and np.isfinite(rm):
        return float(w_realized * rr + (1.0 - w_realized) * rm)
    if np.isfinite(rr):
        return rr
    if np.isfinite(rm):
        return rm
    return float("nan")


# ──────────────────────────────────────────────────────────────────────
# Unit tests (synthetic). Run: python -m src.corr_signal
# ──────────────────────────────────────────────────────────────────────

def _test():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2005-01-03", periods=1260)  # ~5 years
    n1 = 630

    def draw(n, corr, sd_e=0.010, sd_b=0.004):
        cov = np.array([[sd_e**2, corr*sd_e*sd_b], [corr*sd_e*sd_b, sd_b**2]])
        return rng.multivariate_normal([0.0003, 0.0001], cov, size=n)

    X = np.vstack([draw(n1, -0.6), draw(len(days)-n1, +0.5)])
    daily = pd.DataFrame({"eq": X[:, 0], "bond": X[:, 1]}, index=days)

    rho_d = rolling_realized_corr(daily, window=63, min_periods=40)
    rho_m = month_end_series(rho_d)

    # regime-1 plateau near -0.6; regime-2 plateau near +0.5
    r1 = rho_m.loc["2006-06-30":"2007-03-31"].mean()
    r2 = rho_m.loc["2009-01-31":"2009-10-31"].mean()
    assert -0.75 < r1 < -0.45, r1
    assert 0.35 < r2 < 0.65, r2

    # the 63d window crosses zero within ~4 months of the true switch
    switch = days[n1]
    after = rho_m.loc[switch:]
    first_pos = after[after > 0].index[0]
    lag_months = (first_pos.to_period("M") - switch.to_period("M")).n
    assert lag_months <= 4, lag_months

    # causality: truncating the future never changes the past
    cut = pd.Timestamp("2008-06-30")
    daily_t = daily.loc[:cut]
    rho_m_t = month_end_series(rolling_realized_corr(daily_t, 63, 40))
    a, b = rho_m.loc[:cut], rho_m_t.loc[:cut]
    assert len(a) == len(b) and np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)

    # month-end sampling uses only days <= label
    lab = rho_m.index[20]
    manual = rho_d.loc[:lab].dropna().iloc[-1]
    assert abs(rho_m.loc[lab] - manual) < 1e-15

    # blend semantics
    assert abs(blend_rho(-0.4, +0.2, 0.5) - (-0.1)) < 1e-12
    assert blend_rho(float("nan"), +0.2, 0.5) == 0.2
    assert blend_rho(-0.4, None, 0.5) == -0.4
    assert np.isnan(blend_rho(None, None, 0.5))

    print(f"regime plateaus: {r1:+.2f} / {r2:+.2f} | sign-flip lag after switch: {lag_months} month(s)")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _test()
