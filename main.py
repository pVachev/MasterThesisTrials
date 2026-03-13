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
    tickers_all = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR","G1BM", "RF" ,"XAU", "USGG3M"]
    m_tickers = ["LBUSTRUU", "LT09TRUU", "XAU", "USGG3M", "RF"]
   

    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    cfg = GlobalRunConfig(
        n_states=3,
        rf_col="RF", # change in hmm.py too 
        rf_mode="simple_return_monthly_decimal",   # "simple_return_monthly_decimal" "yield_annualized" 
        start_date="1993-08-31",
        cov_type="full",
        output_file="hmm_regime_results.xlsx",
    )


    model_asset_sets = [
        ["^SP500TR", "WFBIX"],
        ["^SP500TR", "LBUSTRUU"],
        ["^SP500TR", "LT09TRUU", "XAU"],
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
