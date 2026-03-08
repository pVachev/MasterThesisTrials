from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
from src.transform import clean_data
plt.style.use('ggplot')
from src.hmm import hmm_converge, hmm_sweep_seeds
from src.postprocess import RegimePostProcessor, diagnose_hmm
from src.plot import plot_regime_dashboard_stack


def main():
    tickers = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU", "GC=F", "^GSPC"]
    m_tickers = ["LBUSTRUU", "LT09TRUU"]
   

    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """


    df = clean_data(tickers, m_tickers)
    




    tickers_A = ["SPY", "WFBIX", "^IRX"]
    tickers_B = ["SPY", "LBUSTRUU", "^IRX"]
    tickers_C = ["SPY", "LT09TRUU", "^IRX"]

    n_states = 3

    dfA = diff_data(df, tickers_A)
    dfB = diff_data(df, tickers_B, monthly_cols=m_tickers)
    dfC = diff_data(df, tickers_C, monthly_cols=m_tickers)



    xA = prepare_data(dfA, tickers_A)
    xB = prepare_data(dfB, tickers_B)
    xC = prepare_data(dfC, tickers_C)


    # xA_std = (xA - xA.mean()) / xA.std()
    # xB_std = (xB - xB.mean()) / xB.std()
    # xC_std = (xC - xC.mean()) / xC.std()



    # df_test = xB_std[["ExcessLogLBUSTRUU"]].join(xC_std[["ExcessLogLT09TRUU"]], how="inner")
    # df_test = df_test.join(xA_std[["ExcessLogWFBIX"]], how="inner")
    # print(df_test.corr())
    # print(df_test.describe())

    # print(xA)
    # print(xB)
    # print(xC)


    # print("--------------------------THIS IS MODEL A (BOND ETF)--------------------------")
    # out_A, df_mA, model_A = hmm_converge(xA, 
    #                                 n_states, 
    #                                 tickers_A, 
    #                                 cov_type="full",
    #                                 return_details=True)
    

    # print(model_A)
    # print(df_mA)
    # print(out_A)

    # print("--------------------------THIS IS MODEL B (LBUSTRUU)--------------------------")
    # out_B, df_mB, model_B = hmm_converge(xB, 
    #                                 n_states, 
    #                                 cols=tickers_B, 
    #                                 cov_type="full",
    #                                 return_details=True)
    
    

    # print(model_B)
    # print(df_mB)
    # print(out_B)



    # print("--------------------------THIS IS MODEL C (LT09TRUU)--------------------------")
    # out_C, df_mC, model_C = hmm_converge(xC, 
    #                                 n_states, 
    #                                 cols=tickers_C, 
    #                                 cov_type="full",
    #                                 return_details=True)
    
    

    # print(model_C)
    # print(df_mC)
    # print(out_C)


    seeds = range(1, 41)
    sumA, best_seedA, outA, dfmA, modelA = hmm_sweep_seeds(xA, n_states, tickers_A, "full", seeds=seeds)
    print("\n--- SWEEP MODEL A (WFBIX) ---")
    print(outA)
    print(modelA)

    ppA = RegimePostProcessor("Model A (WFBIX)", n_states).fit(dfmA, outA)
    regA = ppA.regime_summary("ExcessLogWFBIX")
    print(regA)
    diagnose_hmm("Model a (WFBIX)", modelA, dfmA)
    
    sumB, best_seedB, outB, dfmB, modelB = hmm_sweep_seeds(xB, n_states, tickers_B, "full", seeds=seeds)
    print("\n--- SWEEP MODEL B (LBUSTRUU) ---")
    print(outB)
    print(modelB)

    ppB = RegimePostProcessor("Model B (LBUSTRUU)", n_states).fit(dfmB, outB)
    regB = ppB.regime_summary("ExcessLogLBUSTRUU")
    print(regB)
    diagnose_hmm("Model B (LBUSTRUU)", modelB, dfmB)

    
    sumC, best_seedC, outC, dfmC, modelC = hmm_sweep_seeds(xC, n_states, tickers_C, "full", seeds=seeds)
    print("\n--- SWEEP MODEL C (LT09TRUU) ---")
    print(outC)
    print(modelC)


    # outC, dfmC, modelC = hmm_converge(
    # xC, 4, tickers_C, "full", seed=6, return_details=True,
    # sticky=True, stay_prob=0.95)

    ppC = RegimePostProcessor("Model C (LT09TRUU)", n_states).fit(dfmC, outC)
    regC = ppC.regime_summary("ExcessLogLT09TRUU")
    print(regC)
    diagnose_hmm("Model C (LT09TRUU)", modelC, dfmC)


    # comparison = pd.concat([regA, regB, regC], ignore_index=True)
    # print(comparison)

    

    # bear_dates = dfmC[dfmC["state"] == 0].index  # adjust which state is bear in your output
    # print(bear_dates[:30])
    # print(bear_dates[-30:])

    # bull_dates = dfmC[dfmC["state"] == 3].index  # adjust which state is bear in your output
    # print(bull_dates[:30])



    output_file = "hmm_regime_results.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # main regime summary tables
        regA.to_excel(writer, sheet_name="Regime_A", index=False)
        regB.to_excel(writer, sheet_name="Regime_B", index=False)
        regC.to_excel(writer, sheet_name="Regime_C", index=False)
        # comparison.to_excel(writer, sheet_name="Comparison", index=False)

        # sweep summary tables
        sumA.to_excel(writer, sheet_name="Sweep_A")
        sumB.to_excel(writer, sheet_name="Sweep_B")
        sumC.to_excel(writer, sheet_name="Sweep_C")

        # detailed monthly/state data if useful
        dfmA.to_excel(writer, sheet_name="Detail_A")
        dfmB.to_excel(writer, sheet_name="Detail_B")
        dfmC.to_excel(writer, sheet_name="Detail_C")

        # best seed info
        best_seeds = pd.DataFrame({
            "model": ["A", "B", "C"],
            "best_seed": [best_seedA, best_seedB, best_seedC]
        })
        best_seeds.to_excel(writer, sheet_name="Best_Seeds", index=False)

    print(f"Saved results to {output_file}")


    plot_regime_dashboard_stack([
        ("Model A (WFBIX)", ppA.df_m),
        ("Model B (LBUSTRUU)", ppB.df_m),
        ("Model C (LT09TRUU)", ppC.df_m),
    ], figsize=(26, 18))


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