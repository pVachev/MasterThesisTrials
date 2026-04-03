import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import matplotlib.ticker as mtick
from scipy import stats
from src.transform import yld_to_lnr, simple_to_log_m

def diff_data(
    df: pd.DataFrame,
    cols: list[str],
    rf_col: str = "^IRX",
    freq: str = "ME",                      # "D", "ME", or "W-FRI"
    monthly_cols: list[str] | None = None,
    rf_mode: str = "yield_annualized",
) -> pd.DataFrame:
    if rf_col not in df.columns:
        raise KeyError(f"Missing risk-free column in df: {rf_col}")

    cols = [c for c in cols if c != rf_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in df: {missing}")

    df = df.sort_index().copy()
    monthly_set = set(monthly_cols or [])

    # ----------------------------
    # MONTHLY
    # ----------------------------
    if freq.upper() in {"ME", "M"}:
        needed = [rf_col] + cols
        out = pd.DataFrame(index=df.index.to_period("M").to_timestamp("M").unique()).sort_index()

        for c in needed:
            s = df[c].dropna().sort_index()

            if c in monthly_set:
                s.index = s.index.to_period("M").to_timestamp("M")
                s = s[~s.index.duplicated(keep="last")]
                out[c] = s
            else:
                out[c] = s.resample("ME").last()

        if rf_mode == "yield_annualized":
            rf = yld_to_lnr(out[rf_col], periods_per_year=12)
        elif rf_mode == "simple_return_monthly_decimal":
            rf = simple_to_log_m(out[rf_col])
        else:
            raise ValueError("Unsupported rf_mode for monthly frequency.")

        ex_cols = []
        for c in cols:
            out[f"Log{c}"] = np.log(out[c]).diff()
            out[f"ExcessLog{c}"] = out[f"Log{c}"] - rf
            ex_cols.append(f"ExcessLog{c}")

        return out.dropna(subset=ex_cols)

    # ----------------------------
    # WEEKLY
    # ----------------------------
    elif freq.upper().startswith("W"):
        # reject monthly-only assets in weekly models
        weekly_ineligible = [c for c in cols if c in monthly_set]
        if weekly_ineligible:
            raise ValueError(
                f"Weekly models cannot include monthly-only assets: {weekly_ineligible}"
            )

        if rf_col in monthly_set:
            raise ValueError(
                f"Weekly models require a daily RF source, but rf_col='{rf_col}' is monthly."
            )

        needed = [rf_col] + cols
        out = pd.DataFrame()

        for c in needed:
            s = df[c].dropna().sort_index()
            out[c] = s.resample("W-FRI").last()

        if rf_mode != "yield_annualized":
            raise ValueError(
                "Weekly models currently support only yield-based RF logic "
                "(e.g. ^IRX with rf_mode='yield_annualized')."
            )

        rf = yld_to_lnr(out[rf_col], periods_per_year=52)

        ex_cols = []
        for c in cols:
            out[f"Log{c}"] = np.log(out[c]).diff()
            out[f"ExcessLog{c}"] = out[f"Log{c}"] - rf
            ex_cols.append(f"ExcessLog{c}")

        return out.dropna(subset=ex_cols)

    # ----------------------------
    # DAILY
    # ----------------------------
    else:
        if rf_mode != "yield_annualized":
            raise ValueError(
                "Daily models currently support only yield-based RF logic "
                "(e.g. ^IRX with rf_mode='yield_annualized')."
            )

        rf = yld_to_lnr(df[rf_col], periods_per_year=360).reindex(df.index).ffill()

        ex_cols = []
        for c in cols:
            df[f"Log{c}"] = np.log(df[c]).diff()
            df[f"ExcessLog{c}"] = df[f"Log{c}"] - rf
            ex_cols.append(f"ExcessLog{c}")

        return df.dropna(subset=ex_cols)
    
    
def prepare_data(
    df: pd.DataFrame,
    cols: list[str],
    rf_col: str = "^IRX",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Keeps only the ExcessLog{col} columns for col in `cols`, ignoring rf_col if present.

    Optional comparability control
    ------------------------------
    `start_date` and `end_date` let you force all models to live inside the same
    sample window from main.py.

    Behavior
    --------
    - if start_date is earlier than the first available observation, nothing breaks
    - if end_date is later than the last available observation, nothing breaks
    - if both are provided, the output is sliced to [start_date, end_date]
    """
    cols = [c for c in cols if c != rf_col]
    ex_cols = [f"ExcessLog{c}" for c in cols]

    missing = [c for c in ex_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in df: {missing}")

    out = df[ex_cols].dropna().copy()

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        out = out.loc[out.index >= start_ts]

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        out = out.loc[out.index <= end_ts]

    return out


def plot_data(bnd, gspc): 

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8), 
                           sharex=True)
    ax1.plot(bnd, label = "BND", color =  "red")
    ax2.plot(gspc, label = "GSPC", color = "blue")
    ax1.legend()
    ax2.legend()


   

def plot_returns(bnd, gspc):

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8), 
                           sharex=True)
    ax1.plot(bnd, label = "BND Log Returns", color =  "red")
    ax2.plot(gspc, label = "GSPC Log Returns", color = "blue")
    ax1.legend()
    ax2.legend()


def dist_plot(data):

    x = data.to_numpy()
    mu = x.mean()
    sigma = x.std(ddof=1)

    fig, ax = plt.subplots(figsize=(16,10))

    weights = np.ones_like(x) / len(x)
    
    heights, edges, _ = ax.hist(x, bins=300,
            weights=weights, 
            density=False, 
            color='skyblue', 
            edgecolor='black')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    
    grid = np.linspace(edges[0], edges[-1], 400)
    bin_width = np.mean(np.diff(edges))
   
    ax.plot(grid, stats.norm.pdf(grid, mu, sigma) * bin_width)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Ten Year Bonds Log Returns")
    ax.set_ylabel("Density")

    print("Sum of bar heights =", heights.sum())


def qq_normal(data):
    x = data.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title("Normal Q–Q plot")
    return fig, ax