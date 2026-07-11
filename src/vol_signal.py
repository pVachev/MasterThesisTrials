"""
vol_signal.py
=============
Channel A: vol-sharpened effective bear probability (sizing-only overlay).

Builds a month-end downside-realized-volatility signal from DAILY equity
closes and maps it, together with the HMM's one-step-ahead bear probability,
into an effective bear probability:

    RS-_t   = sum over trading days d in month t of r_d^2 * 1(r_d < 0)
    vol-_t  = sqrt(RS-_t)                       (monthly downside vol)
    z_t     = vol-_t / trailing median(vol-)    (ratio, 1.0 = typical month)
    p_eff   = min(cap, p_bear + eta * max(0, z_t - z_star))

Design invariants
-----------------
1. SIZING ONLY. p_eff is consumed exclusively by
   compute_regime_conviction_weights via the p_bear_override argument.
   Stage-1 selection and the predictive probability row are untouched
   (the same pred_row feeds Stage-1 moment aggregation — mutating it
   would contaminate selection and break the two-stage identity).
2. ONE-DIRECTIONAL. The boost can only RAISE bearishness (max(0, .)):
   calm daily vol never overrides the HMM toward risk-on, so the overlay
   can shrink cyclical tilts / raise defensive conviction but never adds
   equity beta the HMM did not already sanction.
3. CAUSAL. z_t uses only daily data through the last trading day of
   month t, and the standardization baseline is the trailing median of
   vol- over months t-window .. t-1 (current month EXCLUDED). The signal
   at the rebalance date t is therefore known at decision time; weights
   apply to t+1. Verified by a truncation test in _test().
4. SELF-CALIBRATING. Ratio-to-trailing-median absorbs secular vol-level
   shifts (no absolute thresholds); the median is robust to the very
   spikes the signal is trying to flag.

Units: raw daily LOG returns from total-return closes. The daily risk-free
contribution to squared variation is negligible, so no excess adjustment
is applied on the signal side (the modeled/traded return space is
unchanged elsewhere in the pipeline).

Run unit tests:  python -m src.vol_signal
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_daily_close(csv_path: str | Path, price_field: str = "Close") -> pd.Series:
    """
    Load a daily close series from a Yahoo-Finance-style CSV.

    Handles the yfinance multi-header layout used in data/raw
    (row 0: Price,Close,High,Low,Open,Volume; row 1: Ticker,...;
    row 2: Date,,,,,) as well as a generic single-header
    Date,Price file. Returns a float Series indexed by trading day.
    """
    head = pd.read_csv(csv_path, nrows=2)
    is_yf_multiheader = (
        str(head.columns[0]).strip() == "Price"
        and str(head.iloc[0, 0]).strip() == "Ticker"
    )

    if is_yf_multiheader:
        df = pd.read_csv(csv_path, skiprows=[1, 2])
        df = df.rename(columns={"Price": "Date"})
        if price_field not in df.columns:
            raise KeyError(
                f"Column '{price_field}' not found in {csv_path}; "
                f"available: {list(df.columns)}"
            )
    else:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={df.columns[0]: "Date"})
        if price_field not in df.columns:
            price_field = df.columns[1]

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    s = (
        df.set_index("Date")[price_field]
        .astype(float)
        .sort_index()
        .dropna()
    )
    s = s[~s.index.duplicated(keep="last")]
    s.name = price_field
    return s


def daily_log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns from a close series."""
    return np.log(close.astype(float)).diff().dropna()


# ──────────────────────────────────────────────────────────────────────
# Monthly realized measures
# ──────────────────────────────────────────────────────────────────────

def monthly_realized_measures(daily_log_ret: pd.Series) -> pd.DataFrame:
    """
    Aggregate daily log returns into month-end realized measures.

    Index: calendar month-end timestamps (to_period('M').to_timestamp('M')),
    matching the engine's monthly convention in diff_data().

    Columns
    -------
    n_days      trading days observed in the month
    rv          realized variance          sum r_d^2
    rs_minus    downside semivariance      sum r_d^2 * 1(r_d < 0)
    rs_plus     upside semivariance        rv - rs_minus
    vol_total   sqrt(rv)      (monthly total vol, decimal)
    vol_down    sqrt(rs_minus) (monthly downside vol, decimal)
    down_share  rs_minus / rv  in [0, 1]; NaN if rv == 0
    """
    r = daily_log_ret.dropna().astype(float)
    if r.empty:
        raise ValueError("daily_log_ret is empty after dropna().")

    per = r.index.to_period("M")
    tmp = pd.DataFrame({"r2": r**2})
    tmp["r2_neg"] = tmp["r2"].where(r < 0, 0.0)

    g = tmp.groupby(per)
    out = pd.DataFrame(
        {
            "n_days": g["r2"].size(),
            "rv": g["r2"].sum(),
            "rs_minus": g["r2_neg"].sum(),
        }
    )
    out["rs_plus"] = out["rv"] - out["rs_minus"]
    out["vol_total"] = np.sqrt(out["rv"])
    out["vol_down"] = np.sqrt(out["rs_minus"])
    out["down_share"] = np.where(out["rv"] > 0, out["rs_minus"] / out["rv"], np.nan)
    out.index = out.index.to_timestamp("M")
    out.index.name = "month_end"
    return out


def trailing_median_ratio(
    x: pd.Series,
    window: int = 60,
    min_periods: int = 36,
) -> pd.Series:
    """
    z_t = x_t / median(x_{t-window} .. x_{t-1}).

    The current month is EXCLUDED from its own baseline (shift(1)), so the
    ratio reads "this month's level relative to trailing history" and is
    strictly causal. NaN until min_periods trailing observations exist.
    """
    base = x.shift(1).rolling(window=window, min_periods=min_periods).median()
    base = base.where(base > 0)
    return x / base


def build_monthly_vol_signal(
    csv_path: str | Path,
    price_field: str = "Close",
    window: int = 60,
    min_periods: int = 36,
    min_days_per_month: int = 15,
) -> pd.DataFrame:
    """
    One-call convenience: daily CSV -> monthly signal frame.

    Adds column 'z' = trailing-median ratio of vol_down. Months with fewer
    than min_days_per_month trading days (partial first/last months) get
    z = NaN so the engine falls back to the raw p_bear rather than acting
    on an under-sampled month.

    The engine consumes only the 'z' column:  vol_z = sig["z"].
    """
    close = load_daily_close(csv_path, price_field=price_field)
    rets = daily_log_returns(close)
    sig = monthly_realized_measures(rets)
    sig["z"] = trailing_median_ratio(sig["vol_down"], window=window, min_periods=min_periods)
    sig.loc[sig["n_days"] < min_days_per_month, "z"] = np.nan
    return sig


# ──────────────────────────────────────────────────────────────────────
# Effective bear probability (Channel A transform)
# ──────────────────────────────────────────────────────────────────────

def effective_bear_probability(
    p_bear: float,
    z: float | None,
    eta: float,
    z_star: float,
    cap: float = 1.0,
) -> float:
    """
    p_eff = min(cap, p_bear + eta * max(0, z - z_star)).

    NaN / None / non-finite z passes p_bear through unchanged (no boost).
    """
    p = float(p_bear)
    if z is None:
        return p
    z = float(z)
    if not np.isfinite(z):
        return p
    return float(min(cap, p + eta * max(0.0, z - z_star)))


def effective_bear_probability_series(
    p_bear: pd.Series,
    z: pd.Series,
    eta: float,
    z_star: float,
    cap: float = 1.0,
) -> pd.Series:
    """Vectorized version for studies; indices are aligned by pandas."""
    zz = z.reindex(p_bear.index)
    boost = eta * (zz - z_star).clip(lower=0.0)
    out = (p_bear + boost.fillna(0.0)).clip(upper=cap)
    return out


# ──────────────────────────────────────────────────────────────────────
# Unit tests (synthetic, no data files). Run: python -m src.vol_signal
# ──────────────────────────────────────────────────────────────────────

def _test():
    rng = np.random.default_rng(0)

    # ── build 10 years of synthetic daily returns with one crash month ──
    days = pd.bdate_range("2000-01-03", periods=2520)  # ~120 months
    r = pd.Series(rng.normal(0.0003, 0.007, size=len(days)), index=days)

    per = days.to_period("M")
    months = per.unique()
    crash_month = months[90]
    crash_mask = per == crash_month
    r[crash_mask] = rng.normal(-0.01, 0.03, size=crash_mask.sum())

    sig = monthly_realized_measures(r)
    sig["z"] = trailing_median_ratio(sig["vol_down"], window=60, min_periods=36)

    crash_me = crash_month.to_timestamp("M")
    assert sig.loc[crash_me, "z"] > 1.5, sig.loc[crash_me, "z"]
    assert sig.loc[crash_me, "down_share"] > 0.5

    # calm months hover near 1 (median-ratio), e.g. month 80
    calm_me = months[80].to_timestamp("M")
    assert 0.5 < sig.loc[calm_me, "z"] < 1.6, sig.loc[calm_me, "z"]

    # ── semivariance share limits ────────────────────────────────────
    d_neg = pd.Series([-0.01, -0.02, -0.005], index=pd.bdate_range("2010-03-01", periods=3))
    d_pos = pd.Series([0.01, 0.02, 0.005], index=pd.bdate_range("2010-04-01", periods=3))
    mm = monthly_realized_measures(pd.concat([d_neg, d_pos]))
    assert abs(mm.iloc[0]["down_share"] - 1.0) < 1e-12
    assert abs(mm.iloc[1]["down_share"] - 0.0) < 1e-12
    assert abs(mm.iloc[0]["vol_down"] - np.sqrt((d_neg**2).sum())) < 1e-15

    # ── causality: truncating the future never changes the past ──────
    cut = months[100].to_timestamp("M")
    r_trunc = r[r.index.to_period("M").to_timestamp("M") <= cut]
    sig_t = monthly_realized_measures(r_trunc)
    sig_t["z"] = trailing_median_ratio(sig_t["vol_down"], window=60, min_periods=36)
    a = sig.loc[:cut, "z"]
    b = sig_t.loc[:cut, "z"]
    assert len(a) == len(b)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)

    # ── p_eff transform ───────────────────────────────────────────────
    assert effective_bear_probability(0.2, 1.0, eta=0.4, z_star=1.25) == 0.2       # below z*
    assert abs(effective_bear_probability(0.2, 2.0, 0.4, 1.25) - 0.5) < 1e-12      # 0.2+0.4*0.75
    assert effective_bear_probability(0.9, 4.0, 0.4, 1.25) == 1.0                  # cap
    assert effective_bear_probability(0.33, float("nan"), 0.4, 1.25) == 0.33       # NaN passthrough
    assert effective_bear_probability(0.33, None, 0.4, 1.25) == 0.33

    s_p = pd.Series([0.2, 0.9, 0.33], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    s_z = pd.Series([2.0, 4.0, np.nan], index=s_p.index)
    s_e = effective_bear_probability_series(s_p, s_z, eta=0.4, z_star=1.25)
    assert np.allclose(s_e.to_numpy(), [0.5, 1.0, 0.33])

    print(f"crash-month z = {sig.loc[crash_me, 'z']:.2f}, "
          f"calm-month z = {sig.loc[calm_me, 'z']:.2f}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _test()
