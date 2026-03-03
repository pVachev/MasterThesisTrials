import pandas as pd 
import numpy as np
import os 
from data.getData import FILEPATH, fetch_data


def clean_data(tickers:list[str]) -> pd.DataFrame:

  fetch_data(tickers)

  dfs: list[pd.DataFrame] = []
  
  for file in os.listdir(FILEPATH): 
    if file.endswith(".csv"):
      df = pd.read_csv(FILEPATH + file,
                       skiprows=lambda x: x in [0,1],
                       index_col="Date",
                       usecols=[0,1])
      df.index = pd.to_datetime(df.index)
      df.rename({"Unnamed: 1":file.removesuffix(".csv")}, 
                axis=1, 
                inplace=True)      

      dfs.append(df)
  final_data = pd.concat(dfs, axis=1) 
  final_data.dropna(inplace=True)

  
  # print(final_data)

  return final_data

def yld_to_lnr(df:pd.DataFrame):
  
  """
  Function changes yields to a return (Useful for t-bills)
  """

  df = df / 100

  df = df.shift(1) / 360
  
  df = np.log1p(df)

  return df.dropna()



