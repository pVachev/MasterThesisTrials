import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import matplotlib.ticker as mtick
from scipy import stats
from src.transform import yld_to_lnr

def diff_data(
    df: pd.DataFrame,
    cols: list[str],
    rf_col: str = "^IRX",
    freq: str = "ME",                    # "D" or "ME"
    monthly_cols: list[str] | None = None,  # only matters when freq="ME"
) -> pd.DataFrame:
    """
    Computes Log{col} and ExcessLog{col} at the requested frequency.

    freq="D":
      - use df as-is (daily levels)
      - Log{col} = log(level).diff()
      - rf = yld_to_lnr(IRX, periods_per_year=360)

    freq="ME":
      - build a month-end LEVEL panel for rf_col + cols:
          * daily series -> resample("ME").last() (labels at calendar month-end even if weekend)
          * monthly Bloomberg series -> force index to month-end (Period -> Timestamp), de-dup by month
      - Log{col} = log(level).diff() on monthly index
      - rf = yld_to_lnr(IRX, periods_per_year=12)
    """
    if rf_col not in df.columns:
        raise KeyError(f"Missing risk-free column in df: {rf_col}")

    cols = [c for c in cols if c != rf_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in df: {missing}")

    df = df.sort_index().copy()

    if freq.upper() in {"ME", "M"}:
        monthly_set = set(monthly_cols or [])
        needed = [rf_col] + cols
        out = pd.DataFrame(index=df.index.to_period("M").to_timestamp("M").unique()).sort_index()

        for c in needed:
            s = df[c].dropna().sort_index()

            if c in monthly_set:
                # Bloomberg monthly: its date is already month-end, but make it deterministic
                s.index = s.index.to_period("M").to_timestamp("M")
                s = s[~s.index.duplicated(keep="last")]
                out[c] = s
            else:
                # Daily series -> last available obs in month, indexed at calendar month-end
                out[c] = s.resample("ME").last()

        rf = yld_to_lnr(out[rf_col], periods_per_year=12)

        ex_cols = []
        for c in cols:
            out[f"Log{c}"] = np.log(out[c]).diff()
            out[f"ExcessLog{c}"] = out[f"Log{c}"] - rf
            ex_cols.append(f"ExcessLog{c}")

        return out.dropna(subset=ex_cols)

    else:
        # Daily
        rf = yld_to_lnr(df[rf_col], periods_per_year=360).reindex(df.index).ffill()

        ex_cols = []
        for c in cols:
            df[f"Log{c}"] = np.log(df[c]).diff()
            df[f"ExcessLog{c}"] = df[f"Log{c}"] - rf
            ex_cols.append(f"ExcessLog{c}")

        return df.dropna(subset=ex_cols)

def prepare_data(df: pd.DataFrame, cols: list[str], rf_col: str = "^IRX") -> pd.DataFrame:
    """
    Keeps only the ExcessLog{col} columns for col in `cols`, ignoring rf_col if present.
    """
    cols = [c for c in cols if c != rf_col]
    ex_cols = [f"ExcessLog{c}" for c in cols]

    missing = [c for c in ex_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in df: {missing}")

    return df[ex_cols].dropna()





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