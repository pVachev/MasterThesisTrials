"""
corr_signal_study.py
====================
Channel B pre-registration study. Pure post-processing — no engine code.

Establishes, before any backtest integration:
  1. Proxy_Check      : IEF (daily, Yahoo) vs LT09TRUU (monthly) tracking.
  2. Estimator_Race   : core-only rule bond_w = 0.40 - 0.2*rho_x, monthly,
                        outcomes on LT09TRUU — fixed vs corr12/24 (monthly)
                        vs rho63 (daily-horizon) vs h21x126/252 (horizon-
                        matched overlapping 21d returns).
  3. Flip_2022        : month-end rho by source through 2021-2023. KEY
                        finding (Jul 2026): stock-bond correlation is
                        FREQUENCY-DEPENDENT — rho63 stayed negative through
                        H1-2022 while monthly-horizon comovement flipped
                        positive in mid-2021. Condition monthly decisions on
                        the monthly-horizon correlation (h21), not the
                        highest-frequency one.
  4. Interaction      : fired (z >= z*) x sign(rho) -> next-month bond
                        return and bond-minus-equity spread, for rho63 and
                        h21x252. Validates the armed-state amplifier
                        (corr_lambda) two-sidedly.

Usage:
    python corr_signal_study.py \
        --equity-csv "data/raw/^SP500TR.csv" --bond-csv "data/raw/IEF.csv" \
        --workbook <any exported EW workbook> -o corr_signal_study.xlsx
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

try:
    from src.corr_signal import (
        aligned_daily_returns, month_end_series,
        build_monthly_realized_corr, build_monthly_horizon_corr,
    )
    from src.vol_signal import load_daily_close
except ImportError:
    from corr_signal import (
        aligned_daily_returns, month_end_series,
        build_monthly_realized_corr, build_monthly_horizon_corr,
    )
    from vol_signal import load_daily_close

Z_STAR = 2.0
KAPPA = 0.2
BOUNDS = (0.20, 0.60)


def _norm_me(idx):
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp("M")


def load_monthly_assets(workbook: str) -> tuple[pd.Series, pd.Series]:
    ar = pd.read_excel(workbook, sheet_name="Asset_Returns", index_col=0)
    ar.index = _norm_me(ar.index)
    to_simple = lambda s: np.exp(s) - 1 if s.abs().max() < 0.2 else s
    return to_simple(ar["^SP500TR"].astype(float)), to_simple(ar["LT09TRUU"].astype(float))


def proxy_check(bond_csv: str, lt_monthly: pd.Series) -> pd.DataFrame:
    ief = load_daily_close(bond_csv)
    ief_m = ief.groupby(ief.index.to_period("M")).last()
    ief_m.index = ief_m.index.to_timestamp("M")
    d = pd.DataFrame({"IEF": ief_m.pct_change(), "LT": lt_monthly}).dropna()
    beta = np.polyfit(d["LT"], d["IEF"], 1)[0]
    return pd.DataFrame([{
        "n_months": len(d),
        "corr": d["IEF"].corr(d["LT"]),
        "beta": beta,
        "tracking_error_ann": (d["IEF"] - d["LT"]).std() * np.sqrt(12),
        "mean_diff_ann": (d["IEF"] - d["LT"]).mean() * 12,
    }])


def core_rule(rho_sig: pd.Series, sp: pd.Series, bd: pd.Series,
              start="2004-01-01", end="2025-12-31") -> dict:
    d = pd.DataFrame({"rho": rho_sig, "sp_n": sp.shift(-1), "bd_n": bd.shift(-1)}).dropna().loc[start:end]
    bw = (0.40 - KAPPA * d["rho"]).clip(*BOUNDS)
    r = (1 - bw) * d["sp_n"] + bw * d["bd_n"]
    w = (1 + r).cumprod()
    return {
        "n": len(d),
        "CAGR": (1 + r).prod() ** (12 / len(r)) - 1,
        "Sharpe": r.mean() / r.std() * np.sqrt(12),
        "MaxDD": (w / w.cummax() - 1).min(),
        "cum_2022_JanOct": (1 + r.loc["2022-01-01":"2022-10-31"]).prod() - 1,
    }


def interaction_table(rho: pd.Series, z: pd.Series, sp: pd.Series, bd: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"rho": rho, "z": z, "bd_next": bd.shift(-1),
                       "spread_next": (bd - sp).shift(-1)}).dropna()
    rows = []
    for f in [True, False]:
        for ng in [True, False]:
            m = ((df["z"] >= Z_STAR) == f) & ((df["rho"] < 0) == ng)
            rows.append({
                "cell": ("FIRED" if f else "silent") + " & rho" + ("<0" if ng else ">=0"),
                "n": int(m.sum()),
                "bd_next_mean": df.loc[m, "bd_next"].mean(),
                "spread_next_mean": df.loc[m, "spread_next"].mean(),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-csv", default="data/raw/^SP500TR.csv")
    ap.add_argument("--bond-csv", default="data/raw/IEF.csv")
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--study-xlsx", default="vol_signal_study.xlsx",
                    help="for the z series (Signal_Series sheet)")
    ap.add_argument("-o", "--out", default="corr_signal_study.xlsx")
    a = ap.parse_args()
    wb = a.workbook or sorted(glob.glob("allocation_backtest_EW_*.xlsx"))[0]
    print(f"equity: {a.equity_csv}\nbond  : {a.bond_csv}\nwbook : {wb}")

    sp, bd = load_monthly_assets(wb)
    z = pd.read_excel(a.study_xlsx, sheet_name="Signal_Series", index_col=0)["z"]
    z.index = _norm_me(z.index)

    px = proxy_check(a.bond_csv, bd)
    print("\n── Proxy_Check ──"); print(px.round(4).to_string(index=False))

    rho63 = build_monthly_realized_corr(a.equity_csv, a.bond_csv)
    h126 = build_monthly_horizon_corr(a.equity_csv, a.bond_csv, window_days=126, min_periods=80)
    h252 = build_monthly_horizon_corr(a.equity_csv, a.bond_csv, window_days=252, min_periods=150)
    corr12 = sp.rolling(12).corr(bd)
    corr24 = sp.rolling(24).corr(bd)

    sources = {"fixed_60_40": pd.Series(0.0, index=corr12.index),
               "corr12_monthly": corr12, "corr24_monthly": corr24,
               "rho63_daily": rho63, "h21x126": h126, "h21x252": h252}
    race = pd.DataFrame([{"source": k, **core_rule(v, sp, bd)} for k, v in sources.items()])
    print("\n── Estimator_Race (core-only rule) ──"); print(race.round(4).to_string(index=False))

    flip = pd.DataFrame({"rho63": rho63, "h21x252": h252, "corr12": corr12}
                        ).loc["2021-01-31":"2023-06-30"]
    print("\n── Flip_2022 (excerpt) ──"); print(flip.loc["2021-10-31":"2022-09-30"].round(2).to_string())

    it63 = interaction_table(rho63, z, sp, bd)
    ith = interaction_table(h252, z, sp, bd)
    print("\n── Interaction, h21x252 ──"); print(ith.round(4).to_string(index=False))

    rho_all = pd.DataFrame({"rho63": rho63, "h21x126": h126, "h21x252": h252,
                            "corr12": corr12, "corr24": corr24})
    with pd.ExcelWriter(a.out, engine="openpyxl") as xl:
        px.to_excel(xl, sheet_name="Proxy_Check", index=False)
        race.to_excel(xl, sheet_name="Estimator_Race", index=False)
        rho_all.to_excel(xl, sheet_name="Rho_Series")
        flip.to_excel(xl, sheet_name="Flip_2022")
        it63.to_excel(xl, sheet_name="Interaction_rho63", index=False)
        ith.to_excel(xl, sheet_name="Interaction_h21x252", index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
