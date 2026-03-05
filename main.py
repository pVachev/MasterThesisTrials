from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
from src.transform import clean_data
plt.style.use('ggplot')
from src.hmm import hmm_converge


def main():
    tickers = ["SPY", "WFBIX","^IRX", "LBUSTRUU"]
    m_tickers = ["LBUSTRUU"]
   

    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    df = clean_data(tickers, m_tickers)

    # print(df)

 

    tickers_A = ["SPY", "WFBIX", "^IRX"]
    tickers_B = ["SPY", "LBUSTRUU", "^IRX"]
    n_states = 3


    dfA = diff_data(df, tickers_A)
    dfB = diff_data(df, tickers_B, monthly_cols=m_tickers)

    # print(dfA)
    # print(dfB)

    xA = prepare_data(dfA, tickers_A)
    xB = prepare_data(dfB, tickers_B)

    # print(xA)
    # print(xB)


    print("--------------------------THIS IS MODEL A (BOND ETF)--------------------------")
    out_A, df_mA, model_A = hmm_converge(xA, 
                                    n_states, 
                                    tickers_A, 
                                    cov_type="full",
                                    return_details=True)
    

    print(model_A)
    print(df_mA)
    print(out_A)

    print("--------------------------THIS IS MODEL B (LBUSTRUU)--------------------------")
    out_B, df_mB, model_B = hmm_converge(xB, 
                                    n_states=3, 
                                    cols=tickers_B, 
                                    cov_type="full",
                                    return_details=True)
    
    

    print(model_B)
    print(df_mB)
    print(out_B)




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