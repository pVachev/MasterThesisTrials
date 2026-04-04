import pandas as pd 
import numpy as np
import os 
from data.getData import FILEPATH, fetch_data

import pandas as pd
import numpy as np
import os
from data.getData import FILEPATH, fetch_data


def clean_data(
    tickers: list[str],
    monthly_tickers: list[str] | None = None,
    weekly_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load raw files and standardize date indices.

    Supported raw frequencies
    -------------------------
    - daily raw series
    - weekly raw series
    - monthly raw series
    - special Fama-French-style RF files:
        * RF  = monthly compact dates YYYYMM
        * RFW = weekly  compact dates YYYYMMDD

    Weekly convention
    -----------------
    Weekly raw series are relabeled to week-ending Friday using:
        to_period("W-FRI").to_timestamp("W-FRI")
    """
    fetch_data(tickers)

    monthly_set = set(monthly_tickers or [])
    weekly_set = set(weekly_tickers or [])
    dfs: list[pd.DataFrame] = []

    for file in os.listdir(FILEPATH):
        if not file.endswith(".csv"):
            continue

        ticker = file.removesuffix(".csv")
        if ticker not in tickers:
            continue

        df = pd.read_csv(
            FILEPATH / file,
            skiprows=lambda x: x in [0, 1],
            index_col=0,
            usecols=[0, 1],
        )

        raw_idx = df.index.astype(str).str.strip()

        try:
            if ticker == "RF":
                df.index = pd.to_datetime(raw_idx, format="%Y%m", errors="raise")

            elif ticker == "RFW":
                df.index = pd.to_datetime(raw_idx, format="%Y%m%d", errors="raise")

            else:
                parsed = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"):
                    try:
                        parsed = pd.to_datetime(raw_idx, format=fmt, errors="raise")
                        break
                    except ValueError:
                        pass

                if parsed is None:
                    raise ValueError(f"Unsupported date format in ticker {ticker}")

                df.index = parsed

        except Exception as e:
            raise ValueError(f"Could not parse dates for ticker {ticker}") from e

        df.rename(columns={df.columns[0]: ticker}, inplace=True)

        # monthly raw series -> month-end labels
        if ticker in monthly_set or ticker == "RF":
            df = df.sort_index()
            df.index = df.index.to_period("M").to_timestamp("M")
            df = df[~df.index.duplicated(keep="last")]

        # weekly raw series -> week-ending Friday labels
        elif ticker in weekly_set or ticker == "RFW":
            df = df.sort_index()
            df.index = df.index.to_period("W-FRI").to_timestamp("W-FRI")
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

def simple_to_log_w(y: pd.Series, in_percent: bool = True) -> pd.Series:
    """
    Convert a SIMPLE WEEKLY return into a WEEKLY LOG return.

    This is meant for a Fama-French-style weekly RF series where each observation
    is already the realized one-week return for that SAME week.

    Important
    ---------
    - NO SHIFT is applied
    - the weekly RF value already belongs to the same week
    """
    y = y.astype(float)
    if in_percent:
        y = y / 100.0
    r = np.log1p(y)
    return r





