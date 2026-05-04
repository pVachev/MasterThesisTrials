"""
generate_thesis_figures.py
Master Thesis — Regime-Switching Tactical Sector Allocation
Preslav Vachev, emlyon business school, 2026

Reads directly from allocation_export.py output schema:
  Sheets per backtest Excel: Config | Performance | Decision_Log |
                              Weights | Wealth_Drawdown | Asset_Returns |
                              Predictive_Probabilities | Candidate_Scores

Usage:
    python generate_thesis_figures.py

    Edit the FILE PATHS section below to match your actual output filenames.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

SAVE_DPI = 300

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS — edit to match your actual output filenames
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = "ThesisDoc/figures"

# One allocation backtest Excel per investor type.
# Keys become legend labels in Fig 2 / Fig 4.
BACKTEST_FILES = {
    "MV":       "allocation_backtest_EW_45pct_floor001_MV.xlsx",
    "MVS_cons": "allocation_backtest_EW_45pct_floor001_MVS_cons.xlsx",
    "MVS":      "allocation_backtest_EW_45pct_floor001_MVS.xlsx",
    "MVK":      "allocation_backtest_EW_45pct_floor001_MVK.xlsx",
}

# Fig 1 and Fig 3 pull from this single file (primary config = MVS_cons)
PRIMARY_BACKTEST = "allocation_backtest_EW_45pct_floor001_MVS_cons.xlsx"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    "Regime 0": "#c0392b",   # Bear
    "Regime 1": "#e67e22",   # Transitional
    "Regime 2": "#27ae60",   # Bull
}

STRATEGY_COLORS = {
    "MV":        "#2980b9",
    "MVS_cons":  "#27ae60",
    "MVS":       "#8e44ad",
    "MVK":       "#e67e22",
    "Benchmark": "#7f8c8d",
}

SECTORS   = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XAU"]
FEATURED  = {"XAU", "XLK", "XLE"}

SECTOR_NAMES = {
    "XLB": "Materials",   "XLE": "Energy",      "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology",  "XLP": "Cons. Staples",
    "XLU": "Utilities",   "XLV": "Health Care", "XLY": "Cons. Disc.",
    "XAU": "Gold",
}

SECTOR_COLORS = {
    "XLB": "#95a5a6", "XLE": "#e67e22", "XLF": "#2980b9",
    "XLI": "#7f8c8d", "XLK": "#8e44ad", "XLP": "#27ae60",
    "XLU": "#1abc9c", "XLV": "#f39c12", "XLY": "#e74c3c",
    "XAU": "#f1c40f",
}

CRISIS_BANDS = [
    ("2000-03", "2002-09", "Dotcom"),
    ("2007-10", "2009-06", "GFC"),
    ("2020-02", "2020-06", "COVID-19"),
    ("2022-01", "2022-12", "Rate shock"),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _read_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _shade_crises(ax):
    for start, end, label in CRISIS_BANDS:
        try:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            ax.axvspan(s, e, alpha=0.08, color="#c0392b", lw=0)
            ax.text(s + (e - s) / 2, 0.97, label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5,
                    color="#c0392b", style="italic")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Predictive Regime Probability Stacked Area
# Sheet: Predictive_Probabilities
# Columns: "Regime 0 (Bear)", "Regime 1 (Transitional)", "Regime 2 (Bull)"
#          (set by res.pp.regime_names in allocation_regime.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1(path=PRIMARY_BACKTEST):
    print("  fig1_regime_probabilities.png ...")
    df = _read_sheet(path, "Predictive_Probabilities")

    regime_cols = [c for c in df.columns if str(c).startswith("Regime")]
    if not regime_cols:
        regime_cols = df.select_dtypes(include="number").columns[:3].tolist()
    df = df[regime_cols].dropna(how="all").clip(lower=0)
    df = df.div(df.sum(axis=1), axis=0)

    labels = {
        "Regime 0": "Regime 0  —  Bear",
        "Regime 1": "Regime 1  —  Transitional",
        "Regime 2": "Regime 2  —  Bull",
    }
    colors = [REGIME_COLORS.get(c, ["#c0392b", "#e67e22", "#27ae60"][i])
              for i, c in enumerate(df.columns)]

    n = len(df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(15, 6.5), sharex=True)

    for i, (col, ax) in enumerate(zip(df.columns, axes)):
        color = colors[i]
        ax.fill_between(df.index, 0, df[col], color=color, alpha=0.55, lw=0)
        ax.plot(df.index, df[col], color=color, lw=1.1)
        ax.axhline(0.5, color="#888888", lw=0.6, ls="--", alpha=0.6)

        for start, end, label in CRISIS_BANDS[1:]:
            try:
                s, e = pd.Timestamp(start), pd.Timestamp(end)
                ax.axvspan(s, e, alpha=0.10, color="#c0392b", lw=0)
                if i == 0:   # annotate only the top panel
                    ax.text(s + (e - s) / 2, 0.94, label,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=6.5,
                            color="#c0392b", style="italic")
            except Exception:
                pass

        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        ax.set_ylabel(labels.get(col, col), fontsize=8,
                      color=color, labelpad=6)

    axes[-1].set_xlabel("")
    fig.suptitle(
        "Figure 1 — One-Step-Ahead Predictive Regime Probabilities"
        " (3-State HMM, Rolling 60-Month Window, 2004–2026)",
        fontsize=10, y=1.01)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_regime_probabilities.png")
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cumulative Wealth
# Sheet: Wealth_Drawdown
# Columns used: strategy_wealth, benchmark_wealth
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig2():
    print("  fig2_cumulative_wealth.png ...")
    fig, ax = plt.subplots(figsize=(15, 6))
    bm_done = False

    for label, path in BACKTEST_FILES.items():
        if not os.path.exists(path):
            print(f"    [skip] {path}")
            continue
        df    = _read_sheet(path, "Wealth_Drawdown")
        color = STRATEGY_COLORS.get(label, "#333333")

        if "strategy_wealth" in df.columns:
            ax.plot(df.index, df["strategy_wealth"],
                    color=color, lw=1.8, label=label, zorder=4)

        if not bm_done and "benchmark_wealth" in df.columns:
            ax.plot(df.index, df["benchmark_wealth"],
                    color=STRATEGY_COLORS["Benchmark"], lw=1.2,
                    ls="--", label="Benchmark (60/40)", zorder=3)
            bm_done = True

    _shade_crises(ax)
    ax.set_ylabel("Portfolio Value ($1 invested)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax.set_title(
        "Figure 2 — Cumulative Wealth: All Investor Types vs 60/40 Benchmark\n(45% sleeve, floor = 0.001, 2004–2026)",
        fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_cumulative_wealth.png")
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Satellite Weight Mini-Chart Grid
# Sheet: Weights
# Columns: all portfolio tickers; we isolate the 9 sector ETFs + XAU
# Layout: 2 rows × 5 cols; XAU, XLK, XLE get 1.6× wider panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig3(path=PRIMARY_BACKTEST):
    print("  fig3_satellite_weights_grid.png ...")
    df = _read_sheet(path, "Weights")

    # Strip core tickers; keep only satellite columns
    core = {"^SP500TR", "LT09TRUU", "SP500TR", "BONDS"}
    sat_cols = [c for c in SECTORS if c in df.columns]
    if not sat_cols:
        sat_cols = [c for c in df.columns if c not in core]
    weights = df[sat_cols].fillna(0.0)
    dates   = weights.index

    row_order = [
        ["XLB", "XLE", "XLF", "XLI", "XLK"],
        ["XLP", "XLU", "XLV", "XLY", "XAU"],
    ]

    fig = plt.figure(figsize=(16, 8))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)
    axes  = {}

    for row_idx, row_secs in enumerate(row_order):
        w_ratios = [1.6 if s in FEATURED else 1.0 for s in row_secs]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(row_secs), subplot_spec=outer[row_idx],
            wspace=0.35, width_ratios=w_ratios)
        for col_idx, sec in enumerate(row_secs):
            axes[sec] = fig.add_subplot(inner[col_idx])

    flat_order = [s for row in row_order for s in row]
    for sec in flat_order:
        ax       = axes[sec]
        color    = SECTOR_COLORS.get(sec, "#555555")
        featured = sec in FEATURED

        w = (weights[sec].values * 100) if sec in weights.columns else np.zeros(len(dates))

        ax.fill_between(dates, 0, w, color=color, alpha=0.70, lw=0)
        ax.plot(dates, w, color=color, lw=1.4 if featured else 0.9, zorder=3)

        for start, end, _ in CRISIS_BANDS:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                           alpha=0.09, color="#c0392b", lw=0)
            except Exception:
                pass

        ax.set_ylim(0, max(w.max() * 1.15, 10) if w.max() > 0 else 25)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        tick_years = pd.date_range(
            dates[0].replace(month=1, day=1),
            dates[-1].replace(month=1, day=1),
            freq="4YS")
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(t.year) for t in tick_years],
                           fontsize=6, rotation=45, ha="right")
        ax.set_xlim(dates[0], dates[-1])

        ax.set_title(
            f"{sec}\n{SECTOR_NAMES.get(sec, sec)}",
            fontsize=9 if featured else 8,
            fontweight="bold" if featured else "normal",
            color=color if featured else "#444444", pad=3)

        if featured:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.6)
                spine.set_visible(True)

        col_in_row = flat_order.index(sec) % 5
        ax.set_ylabel("Weight (%)" if col_in_row == 0 else "", fontsize=7)

    fig.suptitle(
        "Figure 3 — Monthly Satellite Allocation Weights by Sector (MVS_cons, 45% sleeve, floor=0.001, 2004–2026)\n"
        "Bold frames: XLE (energy cycle 2004–09)  ·  XLK (tech cycle 2016–26)  ·  XAU (gold, crisis periods)",
        fontsize=10, y=1.02)

    out = os.path.join(OUT_DIR, "fig3_satellite_weights_grid.png")
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Drawdown
# Sheet: Wealth_Drawdown
# Columns used: strategy_drawdown, benchmark_drawdown
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig4():
    print("  fig4_drawdown.png ...")
    fig, ax = plt.subplots(figsize=(15, 5))
    bm_done = False

    for label, path in BACKTEST_FILES.items():
        if not os.path.exists(path):
            print(f"    [skip] {path}")
            continue
        df    = _read_sheet(path, "Wealth_Drawdown")
        color = STRATEGY_COLORS.get(label, "#333333")

        if "strategy_drawdown" in df.columns:
            ax.plot(df.index, df["strategy_drawdown"] * 100,
                    color=color, lw=1.6, label=label, alpha=0.9)

        if not bm_done and "benchmark_drawdown" in df.columns:
            ax.plot(df.index, df["benchmark_drawdown"] * 100,
                    color=STRATEGY_COLORS["Benchmark"], lw=1.2,
                    ls="--", label="Benchmark (60/40)")
            bm_done = True

    _shade_crises(ax)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
    ax.set_title(
        "Figure 4 — Strategy Drawdown vs 60/40 Benchmark (2004–2026)", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_drawdown.png")
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    saved {out}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nThesis Figure Generator  |  DPI={SAVE_DPI}  |  out={OUT_DIR}\n")
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    print("\nDone. LaTeX:")
    print(f"  \\graphicspath{{{{ThesisDoc/figures/}}}}")
    for f in ["fig1_regime_probabilities", "fig2_cumulative_wealth",
              "fig3_satellite_weights_grid", "fig4_drawdown"]:
        print(f"  \\includegraphics[width=\\textwidth]{{{f}}}")
