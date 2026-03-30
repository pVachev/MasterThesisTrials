import yfinance as yf
import pandas as pd 
import time
import os
from pathlib import Path


FILEPATH = Path(__file__).resolve().parent / "raw"
FILEPATH.mkdir(parents=True, exist_ok=True)


def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start="1988-01-05",
        threads=False,
        progress=False,
        auto_adjust=True
    )
    time.sleep(3)
    return df


def fetch_data(
    tickers: list[str],
    verbose: bool = True
) -> None:

    for ticker in tickers:
        path = FILEPATH / f"{ticker}.csv"

        if path.exists():
            try:
                pd.read_csv(path)
                if verbose:
                    print(f"{ticker} is available to be loaded")
            except Exception as e:
                if verbose:
                    print(f"{ticker}: failed ({type(e).__name__}: {e})")

        else:
            df = get_data(ticker)

            if df is None or df.empty:
                if verbose:
                    print(f"{ticker} failed to download or returned no data")
                continue

            df.to_csv(path)

            if verbose:
                print(f"{ticker} downloaded and saved in {path}")



