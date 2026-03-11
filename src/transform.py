import pandas as pd 
import numpy as np
import os 
from data.getData import FILEPATH, fetch_data

def clean_data(
    tickers: list[str],
    monthly_tickers: list[str] | None = None,  # e.g. ["LBUSTRUU"]
) -> pd.DataFrame:

    fetch_data(tickers)

    monthly_set = set(monthly_tickers or [])
    dfs: list[pd.DataFrame] = []

    for file in os.listdir(FILEPATH):
        if file.endswith(".csv"):
            ticker = file.removesuffix(".csv")

            if ticker not in tickers:
              continue

            df = pd.read_csv(
                FILEPATH + file,
                skiprows=lambda x: x in [0,1],
                index_col=0,
                usecols=[0,1],
            )
            raw_idx = df.index.astype(str)

            try:
                if ticker == "RF":
                    # Fama-French style monthly RF file: dates come as YYYYMM
                    # like 192607, 192608, ... and must be parsed explicitly.
                    # We do this BEFORE the generic monthly parsing because
                    # dayfirst-based parsing is not reliable for this compact format.
                    df.index = pd.to_datetime(raw_idx.str.strip(), format="%Y%m", errors="raise")
                elif ticker in monthly_set:
                    # Monthly series that already come with normal calendar dates
                    # (for example Bloomberg monthly exports) keep the existing logic.
                    df.index = pd.to_datetime(raw_idx, dayfirst=True, errors="raise")
                else:
                    # Daily series keep the existing daily parsing path.
                    df.index = pd.to_datetime(raw_idx, dayfirst=False, errors="raise")
            except Exception:
                # Last resort fallback.
                # Keep RF explicit here as well so we do not accidentally send a
                # YYYYMM monthly code through generic dayfirst parsing.
                if ticker == "RF":
                    df.index = pd.to_datetime(raw_idx.str.strip(), format="%Y%m", errors="raise")
                else:
                    df.index = pd.to_datetime(raw_idx, dayfirst=(ticker not in monthly_set), errors="raise")

            # rename the single value column to the ticker
            df.rename(columns={df.columns[0]: ticker}, inplace=True)

            # --- NEW: if this ticker is monthly, keep one obs per month and label at month-end
            if ticker in monthly_set:
                df = df.sort_index()
                # label each row by month-end (no aggregation if already monthly)
                df.index = df.index.to_period("M").to_timestamp("M")
                # if duplicates arise after relabeling, keep last
                df = df[~df.index.duplicated(keep="last")]

            dfs.append(df)

    final_data = pd.concat(dfs, axis=1).sort_index()



    return final_data


def yld_to_lnr(y: pd.Series, periods_per_year: int) -> pd.Series:
    """
    Convert an annualized yield in % to per-period log return:
      r_t = log(1 + (y_{t-1}/100)/periods_per_year)
    """
    y = y.astype(float) / 100.0
    r = np.log1p(y.shift(1) / periods_per_year)
    return r

def simple_to_log_m(y: pd.Series) -> pd.Series:
    """
    Convert a SIMPLE MONTHLY return already expressed in DECIMAL form
    into a MONTHLY LOG return.
    """
    y = y.astype(float)
    return np.log1p(y)





