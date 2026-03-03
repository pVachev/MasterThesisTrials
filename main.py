from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
import tabulate
from src.transform import clean_data
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from hmmlearn import hmm
from src.hmm import hmm_converge


def main():

    tickers = ["SPY", "^IRX", "WFBIX"]

    df = clean_data(tickers)

    n_states = 3

    df = diff_data(df, tickers)

    x = prepare_data(df, tickers)

    out, df_m, model = hmm_converge(x, 
                                    n_states, 
                                    tickers, 
                                    cov_type="full",
                                    return_details=True

    )
    
    print(df_m)
    print(df)
    # # plot_returns(bnd,gspc)

    
    


    # dist_plot(bnd)

    # qq_normal(bnd)

    # print(bnd)


    # plt.show()
    




#    print(new_path)
    
    # if os.path.exists(FILEPATH):

        # print((__file__))
        # print((FILEPATH))
    
if __name__ == "__main__":

    main()