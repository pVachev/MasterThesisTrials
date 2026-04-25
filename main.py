import pandas as pd 
import numpy as np

from src.transform import clean_data
from src.load import diff_data
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
    TrainTestConfig,
    CashSleeveConfig,
    calibrate_investor_params
)
from src.allocation_backtest import (
    run_fixed_parameter_train_test_backtest,
    run_expanding_window_backtest,
    ExpandingWindowConfig,
)


from src.allocation_export import export_allocation_backtest_to_excel
from src.allocation_plot import plot_allocation_dashboard, plot_distribution_comparison




def main():
    tickers_all = ["SPY", "WFBIX","^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR","G1BM", "RF",
                   "XAU", "USGG3M","LT01TRUU","LT12TRUU","LT13TRUU", "DEMUSD", "Oil COMP",
                   "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "LT09TRUUW", "RFW", "EEM"]
    m_tickers = ["LBUSTRUU", "LT09TRUU","LT01TRUU","LT12TRUU", "XAU", "USGG3M", "RF", "LT13TRUU", "DEMUSD","Oil COMP"]
    w_tickers = ["LT09TRUUW", "RFW"]
    """
    ModelA -> Bond ETF
    ModelB -> 10Y bonds 
    """

    cfg = GlobalRunConfig(
        n_states=3,
        cov_type="full",
        seeds=range(1, 26),
        rf_col="RF",
        rf_mode="simple_return_monthly_decimal",
        freq="ME",
        start_date="1999-01-31",
        end_date="2026-03-31",
        output_file="hmm_regime_results_monthly.xlsx",

    )

    model_asset_sets = [
        ["^SP500TR", "LT09TRUU"],
        ["^SP500TR", "LT09TRUU","XAU"],
        # ["^SP500TR", "LT09TRUU", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XAU"],
        # ["^SP500TR", "EEM"],
        # ["^SP500TR", "WFBIX"],
        # ["^SP500TR", "LT09TRUU","Oil COMP"],
        # ["^SP500TR","DEMUSD"], 
        # ["^SP500TR","DEMUSD", "LT13TRUU"],
        # ["^SP500TR","DEMUSD","XAU"],
        # ["^SP500TR","LT09TRUU", "XAU"],
        # ["^SP500TR", "Oil COMP"],
        # ["^SP500TR", "LT09TRUU", "XAU", "XLK", "XLP"],
        # ["^SP500TR", "WFBIX", "XAU", "XLK", "XLP"],
        # ["^SP500TR", "Oil COMP", "DEMUSD"],
        # ["^SP500TR", "Oil COMP", "DEMUSD", "XAU","LT09TRUU"],
        # ["^SP500TR", "EEM", "IYW", "XLE"], 
        # ["^SP500TR", "EEM", "XLE"]
    ]


    # Build inferred specs automatically
    model_specs = build_model_specs(model_asset_sets, rf_col=cfg.rf_col)

    # Load / clean once
    df = clean_data(tickers_all, m_tickers, w_tickers)

    # Build features up to prepare_data()
    prepared_inputs = {}
    for spec in model_specs:
        df_model, x_model = build_model_input(
            raw_df=df,
            spec=spec,
            monthly_tickers=m_tickers,
            weekly_tickers=w_tickers,
            rf_mode=cfg.rf_mode,
            freq=cfg.freq,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
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


    # ============================================================
    # ALLOCATION PARAMETERS
    # ============================================================

    

    # ── Preference parameter calibration ──────────────────────────────────
    # Parameters γ and δ are calibrated to the unconditional moments of the
    # 60/40 benchmark portfolio over the full backtest period (2004–2026):
    #
    #   σ²_bm  = 0.000672,  |skew_bm| = 0.7014,  |ekurt_bm| = 1.9482
    #
    # Formula (no factorial scaling — direct score contribution targeting):
    #
    #   γ = target_pct × λ × σ²_bm / |skew_bm|
    #   δ = target_pct × λ × σ²_bm / |ekurt_bm|
    #
    # Resulting score contribution hierarchy at benchmark moments:
    #   mean return    100%  →  primary driver
    #   variance term   31%  →  meaningful risk penalty   (λ·σ²  = 0.002016)
    #   skewness term    6%  →  secondary tiebreaker      (γ·|skew| = 0.000403 at moderate)
    #   kurtosis term    6%  →  secondary tiebreaker      (δ·|ekurt| = 0.000403 at moderate)

    investor_configs = {
        "MV": InvestorPreferenceConfig(
            name="MV Investor",
            investor_type="MV",
            lambda_=3.0,
        ),
        "MVS_cons": InvestorPreferenceConfig(
            name="MVS Investor (conservative)",
            investor_type="MVS",
            lambda_=3.0,
            gamma=0.000431,  # 15% × 3.0 × 0.000672 / 0.7014
        ),
        "MVS": InvestorPreferenceConfig(
            name="MVS Investor",
            investor_type="MVS",
            lambda_=3.0,
            gamma=0.000574,  # 20% × 3.0 × 0.000672 / 0.7014
        ),
        "MVK": InvestorPreferenceConfig(
            name="MVK Investor",
            investor_type="MVK",
            lambda_=3.0,
            gamma=0.000574,  # 20% × 3.0 × 0.000672 / 0.7014
            delta=0.000207,  # 20% × 3.0 × 0.000672 / 1.9482
        ),
    }


    sector_specs_weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


    sector_specs = [
        # ── Cyclical sectors: scale with (1 - p_bear) ──────────────────
        # These benefit from bull markets and should shrink in bear regimes.
        SatelliteSpec(ticker="XLB", label="Materials",             allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        SatelliteSpec(ticker="XLE", label="Energy",                allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        SatelliteSpec(ticker="XLF", label="Financials",            allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        SatelliteSpec(ticker="XLI", label="Industrials",           allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        SatelliteSpec(ticker="XLK", label="Technology",            allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        SatelliteSpec(ticker="XLY", label="Consumer Discretionary",allowed_weights=sector_specs_weights, group="sector", style="cyclical"),
        # ── Defensive sectors: scale with p_bear ───────────────────────
        # These preserve capital in stress and should grow in bear regimes.
        SatelliteSpec(ticker="XLP", label="Consumer Staples",      allowed_weights=sector_specs_weights, group="sector", style="defensive"),
        SatelliteSpec(ticker="XLU", label="Utilities",             allowed_weights=sector_specs_weights, group="sector", style="defensive"),
        SatelliteSpec(ticker="XLV", label="Health Care",           allowed_weights=sector_specs_weights, group="sector", style="defensive"),
        SatelliteSpec(ticker="XAU", label="Gold",                  allowed_weights=sector_specs_weights, group="commodity", style="defensive"),
    ]

    alloc_cfg = AllocationConfig(
        rebalance_frequency="ME",
        top_n_satellites=2,
        max_satellite_weight=0.35,
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
        score_improvement_floor=0.002,
        export_file="allocation_results.xlsx",
        # Satellites displace SP500 only; LT09TRUU stays fixed at 40%.
        # A 20% max sleeve reduces SP500 from 60% → min 40% at full tilt.
        equity_only_displacement=True,
        equity_ticker="^SP500TR",
    )
    alloc_cfg.validate()

    tt_cfg = TrainTestConfig(
        train_start="1999-01-31",
        train_end="2016-12-31",
        test_start="2017-01-31",
        test_end=None,
        min_train_observations=60,
    )

    benchmark_weights = {
        "^SP500TR": 0.60,
        "LT09TRUU": 0.40,
    }

    cash_sleeve = CashSleeveConfig(
        enabled=False,
        activation_threshold=0.55,   # only activate when p_bear > 55%
        max_cash_weight=0.4,        # up to 25% cash at p_bear = 1.0
        rf_ticker="RF",
    )

    # ============================================================
    # A1 HONEST TRAIN/TEST ALLOCATION BACKTEST
    # ============================================================

 
    RUN_ALLOCATION = False
    EXPORT_ALLOCATION = False
    PLOT_ALLOCATION = False
    STORE_CANDIDATE_SCORES = False
 
    CORE_MODEL_CODE = "A"
 
    if RUN_ALLOCATION:
        res_core = next(r for r in results if r.spec.code == CORE_MODEL_CODE)
 
        satellite_tickers = [s.ticker for s in sector_specs]
        allocation_cols = ["^SP500TR", "LT09TRUU"] + satellite_tickers + [cfg.rf_col]
 
        allocation_df = diff_data(
            df,
            cols=allocation_cols,
            rf_col=cfg.rf_col,
            monthly_cols=m_tickers,
            rf_mode=cfg.rf_mode,
            freq=cfg.freq,
        )
 
        allocation_results = {}
 
        for inv_key, investor_cfg in investor_configs.items():
            bt, frozen_state = run_fixed_parameter_train_test_backtest(
                res_core=res_core,
                allocation_df=allocation_df,
                hmm_cfg=cfg,
                tt_cfg=tt_cfg,
                alloc_cfg=alloc_cfg,
                investor_cfg=investor_cfg,
                satellite_specs=sector_specs,
                benchmark_weights=benchmark_weights,
                signal_return_prefix="ExcessLog",
                realized_return_prefix="Log",
                periods_per_year=12,
                store_candidate_scores=STORE_CANDIDATE_SCORES,
                cash_sleeve_cfg=cash_sleeve,
            )
 
            allocation_results[inv_key] = bt
 
            print(f"\n=== A1 HONEST BACKTEST | {investor_cfg.name} ===")
            print(bt.performance_summary)
 
            if EXPORT_ALLOCATION:
                export_allocation_backtest_to_excel(
                    backtest_res=bt,
                    alloc_cfg=alloc_cfg,
                    investor_cfg=investor_cfg,
                    satellite_specs=sector_specs,
                    res_core=res_core,
                    output_file=f"allocation_backtest_A1_{inv_key}.xlsx",
                )
 
            if PLOT_ALLOCATION:
                plot_allocation_dashboard(bt)
                plot_distribution_comparison(bt)
 
    # ============================================================
    # EXPANDING WINDOW IN-SAMPLE BACKTEST
    # ============================================================
    # Complements A1 by asking: if the model could re-learn
    # from all data up to each rebalance date, would regime-aware
    # allocation have added value over the full history?
    #
    # Key design differences vs A1:
    #   - HMM is re-fit from scratch at every rebalance date
    #   - Candidate moment library is re-estimated on the same window
    #   - Both the regime signal AND the satellite moments update
    #     as history accumulates — fixing the stale-library problem
    #
    # Expected thesis result:
    #   - Crisis period (2000–2009): satellite rotation earns its keep,
    #     regime transitions are frequent and informative
    #   - Bull period (2010–2026): modest gains, regime rarely switches
    #   - This asymmetry IS the finding — frame it as such
    #
    # Runtime note:
    #   With seeds=range(1,41) and ~300 rebalance dates this takes
    #   several minutes. Set seeds=range(1,11) for faster development.
    # ============================================================
 
    RUN_EXPANDING_WINDOW = True
    EXPORT_EXPANDING_WINDOW = True
    ew_cfg = ExpandingWindowConfig(
        burn_in_periods=60,
        refit_every_n_periods=1,
        verbose=True,
        verbose_every=12,
        rolling_window=60,    # 5-year rolling window — prevents stale moments
                              # from dominating. Set to None for expanding window.
    )
 
    if RUN_EXPANDING_WINDOW:
        res_core = next(r for r in results if r.spec.code == CORE_MODEL_CODE)
 
        satellite_tickers = [s.ticker for s in sector_specs]
        allocation_cols   = ["^SP500TR", "LT09TRUU"] + satellite_tickers + [cfg.rf_col]
 
        allocation_df = diff_data(
            df,
            cols=allocation_cols,
            rf_col=cfg.rf_col,
            monthly_cols=m_tickers,
            rf_mode=cfg.rf_mode,
            freq=cfg.freq,
        )
 
        ew_results = {}
 
        for inv_key, investor_cfg in investor_configs.items():
            print(f"\n=== EXPANDING WINDOW BACKTEST | {investor_cfg.name} ===")
 
            bt_ew = run_expanding_window_backtest(
                res_core=res_core,
                allocation_df=allocation_df,
                hmm_cfg=cfg,
                alloc_cfg=alloc_cfg,
                investor_cfg=investor_cfg,
                satellite_specs=sector_specs,
                benchmark_weights=benchmark_weights,
                ew_cfg=ew_cfg,
                signal_return_prefix="ExcessLog",
                realized_return_prefix="Log",
                periods_per_year=12,
                store_candidate_scores=False,
                cash_sleeve_cfg=cash_sleeve,
            )
 
            ew_results[inv_key] = bt_ew
            print(bt_ew.performance_summary)
 
            if EXPORT_EXPANDING_WINDOW:
                export_allocation_backtest_to_excel(
                    backtest_res=bt_ew,
                    alloc_cfg=alloc_cfg,
                    investor_cfg=investor_cfg,
                    satellite_specs=sector_specs,
                    res_core=res_core,
                    output_file=f"allocation_backtest_EW_{inv_key}.xlsx",
                )

if __name__ == "__main__":

    main()