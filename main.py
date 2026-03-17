from src.transform import clean_data
from src.runner import (
    GlobalRunConfig,
    build_model_specs,
    build_model_input,
    run_one_model,
)
from src.export import export_model_results_to_excel
from src.plot import plot_results_dashboard, plot_requested_distributions


def main():
    tickers_all = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR","G1BM", "RF" ,
                   "XAU", "USGG3M","LT01TRUU","LT12TRUU","LT13TRUU", "DEMUSD", "Oil COMP",
                   "IYW", "XLE", "EEM"]
    m_tickers = ["LBUSTRUU", "LT09TRUU","LT01TRUU","LT12TRUU", "XAU", "USGG3M", "RF", "LT13TRUU", "DEMUSD","Oil COMP"]
   
    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    cfg = GlobalRunConfig(
        n_states=3,
        rf_col="RF", # change in hmm.py too 
        rf_mode="simple_return_monthly_decimal",   # "simple_return_monthly_decimal" "yield_annualized" 
        start_date="1985-12-31",
        freq="ME",
        cov_type="full",
        output_file="hmm_regime_results.xlsx",
    )


    model_asset_sets = [
        # ["^SP500TR", "LT01TRUU"],
        # ["^SP500TR", "LT13TRUU", "XAU"],
        # ["^SP500TR", "LT09TRUU","XAU"],
        # ["^SP500TR","DEMUSD"], 
        # ["^SP500TR","DEMUSD", "LT13TRUU"],
        # ["^SP500TR","DEMUSD","XAU"],
        # ["^SP500TR","DEMUSD","LT13TRUU", "XAU"],
        # ["^SP500TR", "Oil COMP"],
        # ["^SP500TR","Oil COMP", "XAU", "LT09TRUU"],
        # ["^SP500TR", "Oil COMP", "DEMUSD"],
        # ["^SP500TR", "Oil COMP", "DEMUSD", "XAU","LT09TRUU"],
        ["^SP500TR", "EEM", "IYW", "XLE"], 
        ["^SP500TR", "EEM", "IYW"],
        ["^SP500TR", "EEM", "XLE"]

    ]


    # Build inferred specs automatically
    model_specs = build_model_specs(model_asset_sets, rf_col=cfg.rf_col)

    # Load / clean once
    df = clean_data(tickers_all, m_tickers)

    # Build features up to prepare_data()
    prepared_inputs = {}
    for spec in model_specs:
        df_model, x_model = build_model_input(
            raw_df=df,
            spec=spec,
            monthly_tickers=m_tickers,
            rf_mode=cfg.rf_mode,
            start_date=cfg.start_date,
            freq=cfg.freq
        )
        prepared_inputs[spec.code] = (df_model, x_model)

    # Run all models
    results = []
    for spec in model_specs:
        _, x_model = prepared_inputs[spec.code]
        res = run_one_model(spec, x_model, cfg)
        results.append(res)

        print(f"\n--- {spec.label} ---")
        if res.regime_summary is not None:
            print(res.regime_summary)
        print(res.moment_table)
        if res.corr_table is not None:
            print(res.corr_table)

    # Export
    if cfg.export_excel:
        export_model_results_to_excel(results, cfg.output_file)

    # Plots
    if cfg.make_dashboard:
        plot_results_dashboard(results)

    if cfg.make_distribution_plots:
        plot_requested_distributions(results)


  
if __name__ == "__main__":

    main()
