from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
from src.transform import clean_data
plt.style.use('ggplot')
from src.hmm import hmm_converge


def main():
    tickers = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU"]
    m_tickers = ["LBUSTRUU", "LT09TRUU"]
    m_tickersA = ["LBUSTRUU"]
    m_tickersB = ["LT09TRUU"]
   

    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    df = clean_data(tickers, m_tickers)

    # print(df)
    # df.to_excel("df_debug.xlsx")
 

    tickers_A = ["SPY", "WFBIX", "^IRX"]
    tickers_B = ["SPY", "LBUSTRUU", "^IRX"]
    tickers_C = ["SPY", "LT09TRUU", "^IRX"]

    n_states = 4


    dfA = diff_data(df, tickers_A)
    dfB = diff_data(df, tickers_B, monthly_cols=m_tickersA)
    dfC = diff_data(df, tickers_C, monthly_cols=m_tickersB)

    # print(dfA)
    # print(dfB)

    xA = prepare_data(dfA, tickers_A)
    xB = prepare_data(dfB, tickers_B)
    xC = prepare_data(dfC, tickers_C)

    # print(xA)
    # print(xB)
    # print(xC)


    print("--------------------------THIS IS MODEL A (BOND ETF)--------------------------")
    out_A, df_mA, model_A = hmm_converge(xA, 
                                    n_states, 
                                    tickers_A, 
                                    cov_type="diag",
                                    return_details=True)
    

    print(model_A)
    print(df_mA)
    print(out_A)

    print("--------------------------THIS IS MODEL B (LBUSTRUU)--------------------------")
    out_B, df_mB, model_B = hmm_converge(xB, 
                                    n_states, 
                                    cols=tickers_B, 
                                    cov_type="diag",
                                    return_details=True)
    
    

    print(model_B)
    print(df_mB)
    print(out_B)



    print("--------------------------THIS IS MODEL C (LT09TRUU)--------------------------")
    out_C, df_mC, model_C = hmm_converge(xC, 
                                    n_states, 
                                    cols=tickers_C, 
                                    cov_type="diag",
                                    return_details=True)
    
    

    print(model_C)
    print(df_mC)
    print(out_C)


    # df_mB.to_excel("df_mB.xlsx")
    # df_mA.to_excel("df_mA.xlsx")
    
    
 
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





#    lbustruu = pd.read_csv("LBUSTRUU.csv",
#                        skiprows=lambda x: x in [0,1],
#                        index_col=0)
    
#     print(lbustruu)
#     print(type(lbustruu))