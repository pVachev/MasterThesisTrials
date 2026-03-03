import yfinance as yf
import pandas as pd 
import time
import os

FILEPATH = os.path.dirname(os.path.realpath(__file__)) + "\\raw\\"


def get_data(ticker: list) -> pd.DataFrame: 

    df = yf.download(tickers=ticker,
                        start="1990-01-18",
                        threads=False,
                        progress=False,
                        auto_adjust=True)
    time.sleep(3)
    
    return df

def data_test(ticker:str):
    raw_data = {"Time_Series": ["18-01-2025", "19-01-2025"],
                ticker: [6743.09, 6873.25]}
    df = pd.DataFrame(raw_data)
    return df 

    

# ^GSPC
# BND
def fetch_data(
        tickers:list[str],
        verbose: bool = True
    ) -> None:

    dfs: list[pd.DataFrame] = []

    for ticker in tickers:

        path = FILEPATH + ticker + ".csv"

        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if verbose:
                    print(f"{ticker} is available to be loaded")
            except Exception as e:
                if verbose:
                    print(f"{ticker}: failed ({type(e).__name__}: {e})")

        else: 
            df = get_data(ticker)
            df.to_csv(path)
            if verbose:
                print(f"{ticker} downloaded and saved in data\\raw \
                      {ticker}.csv \n Run again to check availability")



