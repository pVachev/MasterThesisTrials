import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import matplotlib.ticker as mtick
from scipy import stats
from src.transform import yld_to_lnr



# def diff_data(df: pd.DataFrame) -> pd.DataFrame:
#     rf = yld_to_lnr(df["^IRX"]).reindex(df.index).ffill()

#     df = df.sort_index().copy()
#     df["LogBND"] = np.log(df["BND"]).diff()
#     df["LogSPY"] = np.log(df["SPY"]).diff()

#     df["ExcessLogBND"] = df["LogBND"] - rf
#     df["ExcessLogSPY"] = df["LogSPY"] - rf

#     df = df.dropna(subset=["ExcessLogBND", "ExcessLogSPY"])
#     return df

import numpy as np
import pandas as pd

def diff_data(df: pd.DataFrame, cols: list[str], rf_col: str = "^IRX") -> pd.DataFrame:
    """
    Adds Log{col} and ExcessLog{col} for each col in `cols`.
    ExcessLog{col} = log-diff(col) - rf, where rf comes from rf_col via yld_to_lnr.
    """

        # ensure rf exists
    if rf_col not in df.columns:
        raise KeyError(f"Missing risk-free column in df: {rf_col}")

    # ignore rf_col if user passed it in cols
    cols = [c for c in cols if c != rf_col]



    missing = [c for c in ([rf_col] + cols) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in df: {missing}")

    rf = yld_to_lnr(df[rf_col]).reindex(df.index).ffill()

    df = df.sort_index().copy()

    ex_cols: list[str] = []
    for col in cols:
        log_name = f"Log{col}"          # if you truly meant first character: f"Log{col[0]}"
        ex_name  = f"ExcessLog{col}"    # if you truly meant first character: f"ExcessLog{col[0]}"

        df[log_name] = np.log(df[col]).diff()
        df[ex_name] = df[log_name] - rf
        ex_cols.append(ex_name)

    df = df.dropna(subset=ex_cols)
    return df


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