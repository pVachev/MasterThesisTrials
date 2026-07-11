"""
vol_signal_study.py
===================
Standalone Channel-A signal validation. Pure post-processing: reads the
DAILY equity CSV and ONE exported backtest workbook (for p_bear and the
60/40 benchmark returns); runs NO model code.

Answers, before any backtest is touched:
  1. Does month-t downside realized vol (z_t) predict bad month-t+1
     benchmark / equity returns?                          -> Event_Study
  2. Does z_t add anything BEYOND the HMM's p_bear?       -> Double_Sort,
     Logit_LR (likelihood-ratio test)
  3. How often / how hard would each (eta, z*) pair boost p_bear, and do
     the boosted months earn their keep? Where does the overlay disagree
     with the HMM (p_bear low, z high), and were those calls good?
                                                          -> Fire_Table
  4. Where should z* sit given the empirical z distribution?
                                                          -> Z_Distribution

p_bear is read from the workbook's Predictive_Probabilities sheet
(column 0 = bear: RegimePostProcessor sorts regimes by ascending mean, and
the sheet preserves that order). The p_bear path is identical across
investor types / sleeve / floor configs, so any exported workbook works.

Usage:
    python vol_signal_study.py \
        --daily-csv "data/raw/^SP500TR.csv" \
        --workbook  allocation_backtest_EW_45pct_floor001_MVS_core_repl.xlsx \
        -o vol_signal_study.xlsx
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

try:  # vol_signal.py lives in src/ when run from the repo root
    from src.vol_signal import (
        build_monthly_vol_signal,
        load_daily_close,
        effective_bear_probability_series,
    )
except ImportError:  # or side-by-side with this script
    from vol_signal import (
        build_monthly_vol_signal,
        load_daily_close,
        effective_bear_probability_series,
    )

BAD_MONTH = -0.02          # "bad next month" threshold on the 60/40 benchmark
ETA_GRID = [0.2, 0.3, 0.4, 0.5]
ZSTAR_GRID = [1.25, 1.5, 1.75, 2.0, 2.5]


def _norm_me(idx) -> pd.DatetimeIndex:
    """Normalize any monthly index to calendar month-end timestamps."""
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp("M")


# ──────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────

def build_master_frame(daily_csv: str, workbook: str) -> tuple[pd.DataFrame, str]:
    sig = build_monthly_vol_signal(daily_csv)

    pp = pd.read_excel(workbook, sheet_name="Predictive_Probabilities", index_col=0)
    pp.index = _norm_me(pp.index)
    bear_name = str(pp.columns[0])
    p_bear = pp.iloc[:, 0].astype(float)

    wd = pd.read_excel(workbook, sheet_name="Wealth_Drawdown", index_col=0)
    wd.index = _norm_me(wd.index)
    bench = wd["benchmark_return"].astype(float)

    close = load_daily_close(daily_csv)
    close_m = close.groupby(close.index.to_period("M")).last()
    close_m.index = close_m.index.to_timestamp("M")
    sp_ret = close_m.pct_change()

    mf = pd.DataFrame(index=sig.index)
    mf["z"] = sig["z"]
    mf["vol_down"] = sig["vol_down"]
    mf["down_share"] = sig["down_share"]
    mf["p_bear"] = p_bear.reindex(mf.index)
    mf["bench_ret"] = bench.reindex(mf.index)
    mf["bench_next"] = mf["bench_ret"].shift(-1)
    mf["sp_next"] = sp_ret.reindex(mf.index).shift(-1)
    return mf, bear_name


def sanity_check_bear_column(an: pd.DataFrame, bear_name: str) -> pd.DataFrame:
    """Confirm column 0 of the prob sheet behaves like a bear probability."""
    worst = an.nsmallest(max(5, len(an) // 10), "bench_ret")
    rows = [
        {"check": "bear column name", "value": bear_name},
        {"check": "mean p_bear, all months", "value": round(float(an["p_bear"].mean()), 3)},
        {"check": "mean p_bear, worst-decile bench months",
         "value": round(float(worst["p_bear"].mean()), 3)},
        {"check": "n analysis months", "value": int(len(an))},
        {"check": "window", "value": f"{an.index.min().date()} -> {an.index.max().date()}"},
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Analyses
# ──────────────────────────────────────────────────────────────────────

def z_distribution(mf: pd.DataFrame, an: pd.DataFrame) -> pd.DataFrame:
    qs = [0.05, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for label, z in [("full history", mf["z"].dropna()), ("analysis window", an["z"])]:
        row = {"sample": label, "n": len(z)}
        for q in qs:
            row[f"q{int(q*100)}"] = z.quantile(q)
        for t in ZSTAR_GRID:
            row[f"pct_gt_{t}"] = float((z > t).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_stats(sub: pd.DataFrame, target: str) -> dict:
    x = sub[target]
    return {
        "n": len(sub),
        "mean": x.mean(),
        "median": x.median(),
        "pct_neg": float((x < 0).mean()),
        f"pct_lt_{abs(BAD_MONTH):.0%}": float((x < BAD_MONTH).mean()),
        "p5": x.quantile(0.05),
        "worst": x.min(),
    }


def event_study(an: pd.DataFrame, sort_col: str, target: str, n_bins: int = 5) -> pd.DataFrame:
    ranked = an.dropna(subset=[sort_col, target]).copy()
    ranked["bin"] = pd.qcut(ranked[sort_col].rank(method="first"), n_bins,
                            labels=[f"Q{i+1}" for i in range(n_bins)])
    rows = []
    for b, sub in ranked.groupby("bin", observed=True):
        row = {"bin": str(b), f"{sort_col}_lo": sub[sort_col].min(),
               f"{sort_col}_hi": sub[sort_col].max()}
        row.update(_bucket_stats(sub, target))
        rows.append(row)
    return pd.DataFrame(rows)


def double_sort(an: pd.DataFrame, z_star: float = 1.5, p_thresh: float = 0.5) -> pd.DataFrame:
    rows = []
    for p_lab, p_mask in [("p_bear<%.2f" % p_thresh, an["p_bear"] < p_thresh),
                          ("p_bear>=%.2f" % p_thresh, an["p_bear"] >= p_thresh)]:
        for z_lab, z_mask in [("z<%.2f" % z_star, an["z"] < z_star),
                              ("z>=%.2f" % z_star, an["z"] >= z_star)]:
            sub = an[p_mask & z_mask]
            row = {"p_bear_cell": p_lab, "z_cell": z_lab}
            if len(sub):
                row.update(_bucket_stats(sub, "bench_next"))
                row["mean_sp_next"] = sub["sp_next"].mean()
            else:
                row["n"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_logit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    Xd = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))

    def nll(b):
        eta = Xd @ b
        return float(np.logaddexp(0.0, eta).sum() - y @ eta)

    res = minimize(nll, np.zeros(Xd.shape[1]), method="BFGS")
    return res.x, -res.fun


def logit_lr(an: pd.DataFrame) -> pd.DataFrame:
    d = an.dropna(subset=["z", "p_bear", "bench_next"]).copy()
    y = (d["bench_next"] < BAD_MONTH).astype(float).to_numpy()

    b0, ll0 = _fit_logit(np.empty((len(y), 0)), y)                       # const only
    b1, ll1 = _fit_logit(d[["p_bear"]].to_numpy(), y)                    # + p_bear
    b2, ll2 = _fit_logit(d[["p_bear", "z"]].to_numpy(), y)               # + z
    b3, ll3 = _fit_logit(d[["z"]].to_numpy(), y)                         # z only

    lr_z_given_p = 2.0 * (ll2 - ll1)
    lr_p_given_z = 2.0 * (ll2 - ll3)
    rows = [
        {"model": "const", "logL": ll0, "coefs": np.round(b0, 3).tolist()},
        {"model": "const + p_bear", "logL": ll1, "coefs": np.round(b1, 3).tolist()},
        {"model": "const + z", "logL": ll3, "coefs": np.round(b3, 3).tolist()},
        {"model": "const + p_bear + z", "logL": ll2, "coefs": np.round(b2, 3).tolist()},
        {"model": f"LR: z | p_bear (target: bench_next < {BAD_MONTH:.0%})",
         "logL": lr_z_given_p, "coefs": f"p = {chi2.sf(lr_z_given_p, 1):.4f}"},
        {"model": "LR: p_bear | z",
         "logL": lr_p_given_z, "coefs": f"p = {chi2.sf(lr_p_given_z, 1):.4f}"},
    ]
    return pd.DataFrame(rows)


def fire_table(an: pd.DataFrame, pure_thresh: float = 0.3) -> pd.DataFrame:
    rows = []
    base = an.dropna(subset=["z", "p_bear", "bench_next"])
    for eta in ETA_GRID:
        for zs in ZSTAR_GRID:
            p_eff = effective_bear_probability_series(base["p_bear"], base["z"], eta, zs)
            boost = p_eff - base["p_bear"]
            fired = boost > 1e-12
            pure = fired & (base["p_bear"] < pure_thresh)
            f, nf = base[fired], base[~fired]
            rows.append({
                "eta": eta, "z_star": zs,
                "n_boost": int(fired.sum()),
                "pct_months": float(fired.mean()),
                "mean_boost": float(boost[fired].mean()) if fired.any() else np.nan,
                "max_boost": float(boost.max()),
                "n_pure_vol_calls": int(pure.sum()),
                "mean_next_boosted": f["bench_next"].mean() if len(f) else np.nan,
                "mean_next_unboosted": nf["bench_next"].mean() if len(nf) else np.nan,
                "pct_bad_boosted": float((f["bench_next"] < BAD_MONTH).mean()) if len(f) else np.nan,
                "pct_bad_unboosted": float((nf["bench_next"] < BAD_MONTH).mean()) if len(nf) else np.nan,
                "mean_next_pure_calls": base.loc[pure, "bench_next"].mean() if pure.any() else np.nan,
                "pct_bad_pure_calls": float((base.loc[pure, "bench_next"] < BAD_MONTH).mean()) if pure.any() else np.nan,
            })
    return pd.DataFrame(rows)


def top_spikes(an: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["z", "down_share", "p_bear", "bench_ret", "bench_next", "sp_next"]
    return an.nlargest(n, "z")[cols].reset_index()


def make_plot(mf: pd.DataFrame, an: pd.DataFrame, png_path: str,
              eta: float = 0.3, z_star: float = 1.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(mf.index, mf["z"], lw=0.9, color="tab:blue", label="z (downside RV / trailing median)")
    for t, c in [(1.5, "orange"), (2.0, "red")]:
        ax1.axhline(t, color=c, ls="--", lw=0.8, label=f"z* = {t}")
    bad = mf["bench_ret"] < -0.03
    for dt in mf.index[bad.fillna(False)]:
        ax1.axvspan(dt - pd.offsets.MonthEnd(1), dt, color="grey", alpha=0.25, lw=0)
    ax1.set_title("Downside realized-vol signal z_t (grey: benchmark months < -3%)")
    ax1.legend(loc="upper right", fontsize=8)

    p_eff = effective_bear_probability_series(an["p_bear"], an["z"], eta, z_star)
    ax2.plot(an.index, an["p_bear"], lw=0.9, color="tab:grey", label="p_bear (HMM)")
    ax2.plot(an.index, p_eff, lw=0.9, color="tab:red", alpha=0.8,
             label=f"p_eff (eta={eta}, z*={z_star})")
    ax2.set_title("HMM bear probability vs vol-sharpened effective bear probability")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-csv", default="data/raw/^SP500TR.csv")
    ap.add_argument("--workbook", default=None,
                    help="Any exported allocation_backtest_EW_*.xlsx (p_bear path is config-invariant)")
    ap.add_argument("-o", "--out", default="vol_signal_study.xlsx")
    ap.add_argument("--png", default="vol_signal_study.png")
    a = ap.parse_args()

    wb = a.workbook or sorted(glob.glob("allocation_backtest_EW_*.xlsx"))[0]
    print(f"daily csv: {a.daily_csv}\nworkbook : {wb}")

    mf, bear_name = build_master_frame(a.daily_csv, wb)
    an = mf.dropna(subset=["z", "p_bear", "bench_next"]).copy()

    notes = sanity_check_bear_column(an, bear_name)
    print(notes.to_string(index=False))

    zdist = z_distribution(mf, an)
    ev_z = event_study(an, "z", "bench_next")
    ev_z_sp = event_study(an, "z", "sp_next")
    ev_p = event_study(an, "p_bear", "bench_next")
    ds = double_sort(an, z_star=1.5, p_thresh=0.5)
    lr = logit_lr(an)
    ft = fire_table(an)
    ts = top_spikes(an)

    with pd.ExcelWriter(a.out, engine="openpyxl") as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False)
        mf.to_excel(xl, sheet_name="Signal_Series")
        zdist.to_excel(xl, sheet_name="Z_Distribution", index=False)
        ev_z.to_excel(xl, sheet_name="Event_Study_z_bench", index=False)
        ev_z_sp.to_excel(xl, sheet_name="Event_Study_z_sp500", index=False)
        ev_p.to_excel(xl, sheet_name="Event_Study_pbear", index=False)
        ds.to_excel(xl, sheet_name="Double_Sort", index=False)
        lr.to_excel(xl, sheet_name="Logit_LR", index=False)
        ft.to_excel(xl, sheet_name="Fire_Table", index=False)
        ts.to_excel(xl, sheet_name="Top_Spikes", index=False)
    make_plot(mf, an, a.png)

    print(f"\nwrote {a.out} and {a.png}\n")
    print("── Event study: next-month 60/40 benchmark by z quintile ──")
    print(ev_z.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n── Double sort (z* = 1.5, p_bear split 0.5) ──")
    print(ds.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n── Logit LR ──")
    print(lr.to_string(index=False))


if __name__ == "__main__":
    main()
