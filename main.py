from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
from src.transform import clean_data
plt.style.use('ggplot')
from src.hmm import hmm_converge, hmm_sweep_seeds
from src.postprocess import RegimePostProcessor, diagnose_hmm
from src.plot import plot_regime_dashboard_stack, plot_regime_distribution_grid


def main():
    tickers = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR","G1BM","RF", "XAU"]
    m_tickers = ["LBUSTRUU", "LT09TRUU", "XAU"]
   

    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    df = clean_data(tickers, m_tickers)

   
    # tickers_A = ["SPY", "WFBIX", "^IRX"]
    # tickers_B = ["SPY", "LBUSTRUU", "^IRX"]
    # tickers_C = ["SPY", "LT09TRUU", "^IRX"]

    
    # tickers_A = ["SPY", "WFBIX", "RF"]
    # tickers_B = ["SPY", "LBUSTRUU", "RF"]
    # tickers_C = ["SPY", "LT09TRUU", "RF"]

    tickers_A = ["^SP500TR", "WFBIX", "RF", "XAU"]
    tickers_B = ["^SP500TR", "LBUSTRUU", "RF", "XAU"]
    tickers_C = ["^SP500TR", "LT09TRUU", "RF", "XAU"]



    n_states = 3
    dfA = diff_data(df, tickers_A, rf_col="RF", monthly_cols=m_tickers, rf_mode="simple_return_monthly_decimal")
    dfB = diff_data(df, tickers_B, rf_col="RF", monthly_cols=m_tickers, rf_mode="simple_return_monthly_decimal")
    dfC = diff_data(df, tickers_C, rf_col="RF", monthly_cols=m_tickers, rf_mode="simple_return_monthly_decimal")


    xA = prepare_data(dfA, tickers_A, rf_col="RF")
    xB = prepare_data(dfB, tickers_B, rf_col="RF")
    xC = prepare_data(dfC, tickers_C, rf_col="RF")



    # xA.to_excel("newrftest.xlsx")

    # print("xA shape:", xA.shape)
    # print(xA.head())
    # print(xA.tail())
    # print(xA.isna().sum())
    # print(np.isfinite(xA.to_numpy(dtype=float)).all())



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
    transA, durA, chatA = diagnose_hmm(
        "Model A (WFBIX)",
        modelA,
        ppA.df_m,
        return_tables=True,
        order_old=ppA.order_old,
        regime_names=ppA.regime_names,
    )
    
    sumB, best_seedB, outB, dfmB, modelB = hmm_sweep_seeds(xB, n_states, tickers_B, "full", seeds=seeds)
    print("\n--- SWEEP MODEL B (LBUSTRUU) ---")
    print(outB)
    print(modelB)

    ppB = RegimePostProcessor("Model B (LBUSTRUU)", n_states).fit(dfmB, outB)
    regB = ppB.regime_summary("ExcessLogLBUSTRUU")
    print(regB)
    transB, durB, chatB = diagnose_hmm(
        "Model B (LBUSTRUU)",
        modelB,
        ppB.df_m,
        return_tables=True,
        order_old=ppB.order_old,
        regime_names=ppB.regime_names,
    )

    
    sumC, best_seedC, outC, dfmC, modelC = hmm_sweep_seeds(xC, n_states, tickers_C, "full", seeds=seeds)
    print("\n--- SWEEP MODEL C (LT09TRUU) ---")
    print(outC)
    print(modelC)




    ppC = RegimePostProcessor("Model C (LT09TRUU)", n_states).fit(dfmC, outC)
    regC = ppC.regime_summary("ExcessLogLT09TRUU")
    print(regC)
    transC, durC, chatC = diagnose_hmm(
        "Model C (LT09TRUU)",
        modelC,
        ppC.df_m,
        return_tables=True,
        order_old=ppC.order_old,
        regime_names=ppC.regime_names,
    )


    comparison = pd.concat([regA, regB, regC], ignore_index=True)
    print(comparison)




    output_file = "hmm_regime_results.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # combined summary sheets per model
        startrow = 0
        regA.to_excel(writer, sheet_name="Summary_A", index=False, startrow=startrow)
        startrow += len(regA) + 3
        transA.to_excel(writer, sheet_name="Summary_A", startrow=startrow)
        startrow += len(transA) + 3
        durA.to_excel(writer, sheet_name="Summary_A", startrow=startrow)
        startrow += len(durA) + 3
        chatA.to_excel(writer, sheet_name="Summary_A", index=False, startrow=startrow)

        startrow = 0
        regB.to_excel(writer, sheet_name="Summary_B", index=False, startrow=startrow)
        startrow += len(regB) + 3
        transB.to_excel(writer, sheet_name="Summary_B", startrow=startrow)
        startrow += len(transB) + 3
        durB.to_excel(writer, sheet_name="Summary_B", startrow=startrow)
        startrow += len(durB) + 3
        chatB.to_excel(writer, sheet_name="Summary_B", index=False, startrow=startrow)

        startrow = 0
        regC.to_excel(writer, sheet_name="Summary_C", index=False, startrow=startrow)
        startrow += len(regC) + 3
        transC.to_excel(writer, sheet_name="Summary_C", startrow=startrow)
        startrow += len(transC) + 3
        durC.to_excel(writer, sheet_name="Summary_C", startrow=startrow)
        startrow += len(durC) + 3
        chatC.to_excel(writer, sheet_name="Summary_C", index=False, startrow=startrow)

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


    plot_regime_distribution_grid([
        ("Model A (WFBIX)", ppA.df_m),
        ("Model B (LBUSTRUU)", ppB.df_m),
        ("Model C (LT09TRUU)", ppC.df_m),
    ], value_col="ExcessLog^SP500TR", bins=70, figsize=(20, 14), add_kde=True)

    plot_regime_distribution_grid([
        ("Model A (WFBIX)", ppA.df_m.rename(columns={"ExcessLogWFBIX": "_bond_col"})),
        ("Model B (LBUSTRUU)", ppB.df_m.rename(columns={"ExcessLogLBUSTRUU": "_bond_col"})),
        ("Model C (LT09TRUU)", ppC.df_m.rename(columns={"ExcessLogLT09TRUU": "_bond_col"})),
    ], value_col="_bond_col", bins=70, figsize=(20, 14), add_kde=True)



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