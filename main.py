from src.transform import clean_data
from src.runner import (
    GlobalRunConfig,
    build_model_specs,
    build_model_input,
    run_one_model,
)
from src.export import export_model_results_to_excel
from src.plot import plot_results_dashboard, plot_requested_distributions
from src.allocation_config import (
    InvestorPreferenceConfig,
    SatelliteSpec,
    AllocationConfig,
)

from src.allocation_regime import (
    extract_filtered_probabilities,
    extract_reordered_transition_matrix,
    build_predictive_probability_panel,
)
from src.load import diff_data


from src.allocation_regime import build_predictive_probability_panel
from src.allocation_moments import evaluate_candidate_tilt_moments
from src.allocation_scoring import select_best_tilt_at_date



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
        ["^SP500TR", "LT13TRUU", "XAU"],
        ["^SP500TR", "LT09TRUU","Oil COMP"],
        # ["^SP500TR","DEMUSD"], 
        # ["^SP500TR","DEMUSD", "LT13TRUU"],
        # ["^SP500TR","DEMUSD","XAU"],
        # ["^SP500TR","DEMUSD","LT09TRUU", "XAU"],
        # ["^SP500TR", "Oil COMP"],
        ["^SP500TR", "LT09TRUU","XAU"],
        # ["^SP500TR", "Oil COMP", "DEMUSD"],
        # ["^SP500TR", "Oil COMP", "DEMUSD", "XAU","LT09TRUU"],
        # ["^SP500TR", "EEM", "IYW", "XLE"], 
        # ["^SP500TR", "EEM", "IYW"],
        # ["^SP500TR", "EEM", "XLE"]

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


    # -------------------------------------------------------------
    # SMOKE TEST STARTS HERE
    #--------------------------------------------------------------

    # pick one existing HMM model result as the future core regime engine
    res_core = next(r for r in results if r.spec.code == "C")   # example: LT09TRUU core model

    # investor types
    mv_pref = InvestorPreferenceConfig(
        name="MV Investor",
        investor_type="MV",
        lambda_=3.0,
    )

    mvs_pref = InvestorPreferenceConfig(
        name="MVS Investor",
        investor_type="MVS",
        lambda_=3.0,
        gamma=0.5,
    )

    mvk_pref = InvestorPreferenceConfig(
        name="MVK Investor",
        investor_type="MVK",
        lambda_=3.0,
        delta=0.2,
    )

    # satellite candidates
    gold_sat = SatelliteSpec(
        ticker="XAU",
        label="Gold",
        allowed_weights=[0.00, 0.05, 0.10, 0.15, 0.20],
        group="commodity",
    )

    # later you can add more:
    # oil_sat = SatelliteSpec(...)
    # fx_sat = SatelliteSpec(...)

    # allocation-layer config
    alloc_cfg = AllocationConfig(
        rebalance_frequency="ME",
        top_n_satellites=1,
        max_satellite_weight=0.20,
        fixed_core_weights={
            "^SP500TR": 0.60,
            "LT09TRUU": 0.40,
        },
        long_only=True,
        no_leverage=True,
        transaction_cost_bps=5.0,
        turnover_limit=None,
        min_regime_obs=24,
        shrinkage_intensity=0.0,
        score_improvement_floor=0.0,
        export_file="allocation_results.xlsx",
    )

    alloc_cfg.validate()

    print("\n=== STAGE 2 SMOKE TEST ===")
    print("Core regime engine:", res_core.spec.label)
    print(mv_pref)
    print(mvs_pref)
    print(mvk_pref)
    print(gold_sat)
    print(alloc_cfg)


    # -------------------------------------------------------------
    # ENDS HERE
    #--------------------------------------------------------------

    print("\n=== STAGE 3 SMOKE TEST ===")

    filtered_probs = extract_filtered_probabilities(res_core)
    print("\nFiltered probabilities:")
    print(filtered_probs.head())

    P = extract_reordered_transition_matrix(res_core)
    print("\nReordered transition matrix:")
    print(P)

    pred_probs = build_predictive_probability_panel(res_core, steps_ahead=1)
    print("\nOne-step predictive probabilities:")
    print(pred_probs.head())

    # optional checks
    print("\nFiltered row sums:")
    print(filtered_probs.sum(axis=1).head())

    print("\nPredictive row sums:")
    print(pred_probs.sum(axis=1).head())

    print(filtered_probs.idxmax(axis=1).head(10))
    print(pred_probs.idxmax(axis=1).head(10))


    allocation_df = diff_data(
        df,
        cols=["^SP500TR", "LT09TRUU", "XAU", cfg.rf_col],
        rf_col=cfg.rf_col,
        monthly_cols=m_tickers,
        rf_mode=cfg.rf_mode,
)
    

    test_date = pred_probs.index[24]

    stage4_baseline = evaluate_candidate_tilt_moments(
        res=res_core,
        allocation_df=allocation_df,
        alloc_cfg=alloc_cfg,
        predictive_probability_panel=pred_probs,
        rebalance_date=test_date,
        satellite_weights={},
    )

    stage4_gold_10 = evaluate_candidate_tilt_moments(
        res=res_core,
        allocation_df=allocation_df,
        alloc_cfg=alloc_cfg,
        predictive_probability_panel=pred_probs,
        rebalance_date=test_date,
        satellite_weights={"XAU": 0.10},
    )

    print("\n=== STAGE 4 SMOKE TEST ===")
    print("Rebalance date:", test_date)

    print("\nBaseline weights:")
    print(stage4_baseline["total_weights"])
    print("Baseline aggregated moments:")
    print(stage4_baseline["aggregated_moments"])

    print("\nGold 10% weights:")
    print(stage4_gold_10["total_weights"])
    print("Gold 10% aggregated moments:")
    print(stage4_gold_10["aggregated_moments"])

    print("\nState-conditional moments for Gold 10% candidate:")
    print(stage4_gold_10["state_moment_table"])

    print("\n=== STAGE 5 SMOKE TEST ===")

    # choose one core regime model result
    # IMPORTANT:
    # if gold is core, do not also include gold as a satellite candidate
    res_core = next(r for r in results if r.spec.code == "C")   # example only

    # build predictive probability panel
    pred_probs = build_predictive_probability_panel(res_core, steps_ahead=1)

    # choose one rebalance date
    test_date = pred_probs.index[24]

    # choose satellite candidates
    # example if gold is NOT satellite in this specification:
    # satellite_specs = [oil_sat, fx_sat]
    # for a first smoke test, if you still want to tesßt gold in a non-gold-core spec:
    satellite_specs = [gold_sat]

    decision, candidate_table = select_best_tilt_at_date(
        res=res_core,
        allocation_df=allocation_df,
        alloc_cfg=alloc_cfg,
        investor_cfg=mv_pref,
        satellite_specs=satellite_specs,
        predictive_probability_panel=pred_probs,
        rebalance_date=test_date,
    )

    print("Rebalance date:", test_date)
    print("\nChosen decision:")
    print(decision)

    print("\nTop candidate rows:")
    print(candidate_table.head(10))


  
if __name__ == "__main__":

    main()
