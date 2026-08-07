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
from src.plot import plot_results_dashboard, plot_requested_distributions, plot_asset_price_levels
from src.allocation_config import (
    InvestorPreferenceConfig,
    SatelliteSpec,
    AllocationConfig,
    TrainTestConfig,
    CashSleeveConfig,
)
from src.allocation_moments import calibrate_investor_params
from src.allocation_backtest import (
    run_fixed_parameter_train_test_backtest,
    run_expanding_window_backtest,
    ExpandingWindowConfig,
)
from src.allocation_export import export_allocation_backtest_to_excel, export_model_results_to_excel
from src.vol_signal import build_monthly_vol_signal
from src.corr_signal import build_monthly_horizon_corr
from src.allocation_plot import plot_allocation_dashboard, plot_distribution_comparison


# ── Shared configuration ───────────────────────────────────────────────────

TICKERS_ALL = [
    "SPY", "WFBIX", "^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR", "G1BM", "RF",
    "XAU", "USGG3M", "LT01TRUU", "LT12TRUU", "LT13TRUU", "DEMUSD", "Oil COMP",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "LT09TRUUW", "RFW", "EEM", "IEF"
]
M_TICKERS = [
    "LBUSTRUU", "LT09TRUU", "LT01TRUU", "LT12TRUU", "XAU",
    "USGG3M", "RF", "LT13TRUU", "DEMUSD", "Oil COMP",
]
W_TICKERS = ["LT09TRUUW", "RFW"]

MODEL_ASSET_SETS = [
    ["^SP500TR", "LT09TRUU"],         # Model A — primary thesis model
    ["^SP500TR", "LT09TRUU", "XAU"],  # Model B — gold inclusion diagnostic
]

HMM_CFG = GlobalRunConfig(
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

# ── Preference parameters ──────────────────────────────────────────────────
# γ and δ calibrated to unconditional 60/40 benchmark moments (2004–2026):
#   σ²_bm = 0.000672, |skew_bm| = 0.7014, |ekurt_bm| = 1.9482
# Formula: γ = target_pct × λ × σ²_bm / |skew_bm|
#          δ = target_pct × λ × σ²_bm / |ekurt_bm|
# Score contribution at benchmark moments:
#   mean return  100%  — primary driver
#   variance      31%  — λ·σ² = 0.002016
#   skewness       6%  — γ·|skew| = 0.000403 (moderate)
#   kurtosis       6%  — δ·|ekurt| = 0.000403 (moderate)

INVESTOR_CONFIGS = {
    "MV": InvestorPreferenceConfig(
        name="MV Investor", investor_type="MV", lambda_=3.0,
    ),
    "MVS_cons": InvestorPreferenceConfig(
        name="MVS Investor (conservative)", investor_type="MVS",
        lambda_=3.0, gamma=0.000431,  # 15% × 3.0 × 0.000672 / 0.7014
    ),
    "MVS": InvestorPreferenceConfig(
        name="MVS Investor", investor_type="MVS",
        lambda_=3.0, gamma=0.000574,  # 20% × 3.0 × 0.000672 / 0.7014
    ),
    "MVK": InvestorPreferenceConfig(
        name="MVK Investor", investor_type="MVK",
        lambda_=3.0,
        gamma=0.000574,   # 20% × 3.0 × 0.000672 / 0.7014
        delta=0.000207,   # 20% × 3.0 × 0.000672 / 1.9482
    ),
}

SECTOR_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

SECTOR_SPECS = [
    # Cyclical: conviction scaled by (1 - p_bear)
    SatelliteSpec(ticker="XLB", label="Materials",              allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    SatelliteSpec(ticker="XLE", label="Energy",                 allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    SatelliteSpec(ticker="XLF", label="Financials",             allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    SatelliteSpec(ticker="XLI", label="Industrials",            allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    SatelliteSpec(ticker="XLK", label="Technology",             allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    SatelliteSpec(ticker="XLY", label="Consumer Discretionary", allowed_weights=SECTOR_WEIGHTS, group="sector",    style="cyclical"),
    # Defensive: conviction scaled by p_bear
    SatelliteSpec(ticker="XLP", label="Consumer Staples",       allowed_weights=SECTOR_WEIGHTS, group="sector",    style="defensive"),
    SatelliteSpec(ticker="XLU", label="Utilities",              allowed_weights=SECTOR_WEIGHTS, group="sector",    style="defensive"),
    SatelliteSpec(ticker="XLV", label="Health Care",            allowed_weights=SECTOR_WEIGHTS, group="sector",    style="defensive"),
    SatelliteSpec(ticker="XAU", label="Gold",                   allowed_weights=SECTOR_WEIGHTS, group="commodity", style="defensive"),
]

BENCHMARK_WEIGHTS = {"^SP500TR": 0.60, "LT09TRUU": 0.40}

CASH_SLEEVE = CashSleeveConfig(
    enabled=False,
    activation_threshold=0.55,
    max_cash_weight=0.40,
    rf_ticker="RF",
)

# ── Channel A: vol-sharpened bear probability (sizing-only overlay) ────────
# Flip VOL_SIGNAL_ENABLED only after vol_signal_study.py validates locally.
# Empirical z quantiles 2004+: 1.5 ~ 74th pct, 2.0 ~ 85th pct (Z_Distribution).
VOL_SIGNAL_ENABLED = True
VOL_ETA            = 0.3
VOL_Z_STAR         = 2.0
DAILY_EQUITY_CSV   = "data/raw/^SP500TR.csv"

# ── Correlation-conditioned core split ─────────────────────────────────────
# Ablation (Jul 2026, 20pct MV core-repl, vol on): kappa 0->0.2 is worth
# +0.033 Sharpe, -7.7pp DownCapture, +2.7pp MaxDD for -12bp CAGR — the
# largest single Stage-2 marginal. Mechanism = persistent bond ballast
# (mixture rho < 0 most months) + sign-timing at the 2022 rate shock.
# Estimation caveats (mixture-rho sign flips, thin regimes, no shrinkage)
# remain; Channel B will replace/blend with realized daily bond-equity corr.
CORE_SPLIT_KAPPA = 0.2

# ── Hybrid displacement (core-repl refinement) ─────────────────────────────
# Cyclicals fund from equity only; defensives fund pro-rata. Ship dark;
# flip per run and compare against hybrid-off at matched configs (ablation).
HYBRID_DISPLACEMENT = True
HYBRID_VARIANT     = "B"   # "A" or "B"; see AllocationConfig.hybrid_variant

# ── Channel B: realized horizon-matched bond-equity correlation ────────────
# rho_h21x252 from daily SP500TR + IEF proxy (corr 0.998 with LT09TRUU).
# CORR_BLEND_W blends mixture and realized rho into ONE rho_eff; CORR_LAMBDA
# raises the tilt slope to (kappa + lambda) on rho_eff when armed (z >= z*).
DAILY_BOND_CSV = "data/raw/IEF.csv"
CORR_BLEND_W   = 0.5
CORR_LAMBDA    = 0.2

SENSITIVITY_GRID = [
    # {"sleeve": 0.20, "floor": 0.001},
    # {"sleeve": 0.20, "floor": 0.002},
    {"sleeve": 0.25, "floor": 0.001},
    {"sleeve": 0.25, "floor": 0.002},
    # {"sleeve": 0.30, "floor": 0.001},
    # {"sleeve": 0.30, "floor": 0.002},
    # {"sleeve": 0.35, "floor": 0.001},
    # {"sleeve": 0.35, "floor": 0.002},
    # {"sleeve": 0.40, "floor": 0.001},
    # {"sleeve": 0.40, "floor": 0.002},
    # {"sleeve": 0.45, "floor": 0.001},
    # {"sleeve": 0.45, "floor": 0.002},
    # {"sleeve": 0.50, "floor": 0.001},
    # {"sleeve": 0.50, "floor": 0.002},
]


# ── Run flags ──────────────────────────────────────────────────────────────
RUN_ALLOCATION          = False   # A1 honest train/test backtest
EXPORT_ALLOCATION       = False
PLOT_ALLOCATION         = False
STORE_CANDIDATE_SCORES  = False

RUN_EXPANDING_WINDOW    = True   # Full sensitivity grid
EXPORT_EXPANDING_WINDOW = True

ONLY_SLEEVE    = None  # e.g. 0.45
ONLY_FLOOR     = 0.001   # e.g. 0.001
ONLY_INVESTORS = None  # e.g. ["MV", "MVS"]

CORE_MODEL_CODE = "A"

# ── Pipeline functions ─────────────────────────────────────────────────────

def run_hmm_models(df: pd.DataFrame) -> list:
    """Fit HMM on all model asset sets. Returns list of ModelResult."""
    model_specs = build_model_specs(MODEL_ASSET_SETS, rf_col=HMM_CFG.rf_col)
    results = []
    for spec in model_specs:
        df_model, x_model = build_model_input(
            raw_df=df, spec=spec,
            monthly_tickers=M_TICKERS, weekly_tickers=W_TICKERS,
            rf_mode=HMM_CFG.rf_mode, freq=HMM_CFG.freq,
            start_date=HMM_CFG.start_date, end_date=HMM_CFG.end_date,
        )
        res = run_one_model(spec, x_model, HMM_CFG)
        results.append(res)
        print(f"\n--- {spec.label} ---")
        if res.regime_summary is not None:
            print(res.regime_summary)
        print(res.moment_table)
        if res.corr_table is not None:
            print(res.corr_table)

    if HMM_CFG.export_excel:
        export_model_results_to_excel(results, HMM_CFG.output_file)
    if HMM_CFG.make_dashboard:
        plot_results_dashboard(results)
    if HMM_CFG.make_distribution_plots:
        plot_requested_distributions(results)

    # Regenerate thesis Figure 1 (price level plot)
    # plot_asset_price_levels(
    #     backtest_excel_path="allocation_backtest_EW_45pct_floor001_MVS_cons.xlsx",
    #     out_path="ThesisDoc/figures/fig1_asset_price_levels.png",
    # )

    return results


def run_a1_backtest(results: list, df: pd.DataFrame) -> dict:
    """
    A1 honest train/test backtest.
    HMM and satellite moments frozen on 1999-2016 training window,
    evaluated out-of-sample from 2017 onwards.
    Primary thesis configuration: 45% sleeve, floor = 0.001.
    """
    if not RUN_ALLOCATION:
        return {}

    res_core = next(r for r in results if r.spec.code == CORE_MODEL_CODE)
    satellite_tickers = [s.ticker for s in SECTOR_SPECS]
    allocation_cols = ["^SP500TR", "LT09TRUU"] + satellite_tickers + [HMM_CFG.rf_col]
    allocation_df = diff_data(
        df, cols=allocation_cols, rf_col=HMM_CFG.rf_col,
        monthly_cols=M_TICKERS, rf_mode=HMM_CFG.rf_mode, freq=HMM_CFG.freq,
    )

    tt_cfg = TrainTestConfig(
        train_start="1999-01-31", train_end="2016-12-31",
        test_start="2017-01-31", test_end=None,
        min_train_observations=60,
    )

    # ── Primary thesis config: 45% sleeve / floor 0.001 ───────────────────
    alloc_cfg_a1 = AllocationConfig(
        rebalance_frequency="ME",
        top_n_satellites=2,
        max_satellite_weight=0.45,           # primary thesis config
        fixed_core_weights=BENCHMARK_WEIGHTS,
        long_only=True,
        no_leverage=True,
        transaction_cost_bps=5.0,
        turnover_limit=None,
        min_regime_obs=24,
        shrinkage_intensity=0.0,
        score_improvement_floor=0.001,       # primary thesis config
        export_file="allocation_results.xlsx",
        equity_only_displacement=True,
        equity_ticker="^SP500TR",
    )
    alloc_cfg_a1.validate()

    allocation_results = {}
    for inv_key, investor_cfg in INVESTOR_CONFIGS.items():
        bt, _ = run_fixed_parameter_train_test_backtest(
            res_core=res_core,
            allocation_df=allocation_df,
            hmm_cfg=HMM_CFG,
            tt_cfg=tt_cfg,
            alloc_cfg=alloc_cfg_a1,
            investor_cfg=investor_cfg,
            satellite_specs=SECTOR_SPECS,
            benchmark_weights=BENCHMARK_WEIGHTS,
            signal_return_prefix="ExcessLog",
            realized_return_prefix="Log",
            periods_per_year=12,
            store_candidate_scores=STORE_CANDIDATE_SCORES,
            cash_sleeve_cfg=CASH_SLEEVE,
        )
        allocation_results[inv_key] = bt
        print(f"\n=== A1 HONEST BACKTEST | {investor_cfg.name} ===")
        print(bt.performance_summary)

        if EXPORT_ALLOCATION:
            export_allocation_backtest_to_excel(
                backtest_res=bt,
                alloc_cfg=alloc_cfg_a1,
                investor_cfg=investor_cfg,
                satellite_specs=SECTOR_SPECS,
                res_core=res_core,
                output_file=f"allocation_backtest_A1_{inv_key}.xlsx",
            )
        if PLOT_ALLOCATION:
            plot_allocation_dashboard(bt)
            plot_distribution_comparison(bt)

    return allocation_results


def run_sensitivity_grid(results: list, df: pd.DataFrame) -> None:
    """
    Expanding-window rolling HMM backtest across all sleeve/floor combinations.
    Toggle RUN_EXPANDING_WINDOW = True to execute.
    Use ONLY_SLEEVE, ONLY_FLOOR, ONLY_INVESTORS to run a subset.

    Design:
      - HMM re-fit at every rebalance date on a rolling 60-month window
      - Both regime signal and satellite moments update recursively
      - Fixes the stale-moment problem of the expanding window
    """
    if not RUN_EXPANDING_WINDOW:
        return

    res_core = next(r for r in results if r.spec.code == CORE_MODEL_CODE)
    satellite_tickers = [s.ticker for s in SECTOR_SPECS]
    allocation_cols = ["^SP500TR", "LT09TRUU"] + satellite_tickers + [HMM_CFG.rf_col]
    allocation_df = diff_data(
        df, cols=allocation_cols, rf_col=HMM_CFG.rf_col,
        monthly_cols=M_TICKERS, rf_mode=HMM_CFG.rf_mode, freq=HMM_CFG.freq,
    )

    ew_cfg = ExpandingWindowConfig(
        burn_in_periods=60,
        refit_every_n_periods=1,
        verbose=True,
        verbose_every=12,
        rolling_window=60,
    )

    vol_z = None
    if VOL_SIGNAL_ENABLED:
        vol_z = build_monthly_vol_signal(DAILY_EQUITY_CSV)["z"]
        print(f"vol signal: {vol_z.notna().sum()} usable months, "
              f"eta={VOL_ETA}, z*={VOL_Z_STAR}")

    rho_realized = None
    if CORR_BLEND_W > 0 or CORR_LAMBDA != 0:
        rho_realized = build_monthly_horizon_corr(DAILY_EQUITY_CSV, DAILY_BOND_CSV)
        print(f"corr signal: {rho_realized.notna().sum()} usable months, "
              f"blend_w={CORR_BLEND_W}, lambda={CORR_LAMBDA}")

    grid = [
        g for g in SENSITIVITY_GRID
        if (ONLY_SLEEVE is None or g["sleeve"] == ONLY_SLEEVE)
        and (ONLY_FLOOR  is None or g["floor"]  == ONLY_FLOOR)
    ]
    investors = {
        k: v for k, v in INVESTOR_CONFIGS.items()
        if ONLY_INVESTORS is None or k in ONLY_INVESTORS
    }
    total, run_n = len(grid) * len(investors), 0

    for g in grid:
        sleeve, floor = g["sleeve"], g["floor"]
        alloc_cfg_run = AllocationConfig(
            rebalance_frequency="ME",
            top_n_satellites=2,
            max_satellite_weight=sleeve,
            fixed_core_weights=BENCHMARK_WEIGHTS,
            long_only=True,
            no_leverage=True,
            transaction_cost_bps=5.0,
            turnover_limit=None,
            min_regime_obs=24,
            shrinkage_intensity=0.0,
            score_improvement_floor=floor,
            export_file="allocation_results.xlsx",
            equity_only_displacement=False,
            equity_ticker="^SP500TR",
            core_split_kappa=CORE_SPLIT_KAPPA,
            vol_signal_enabled=VOL_SIGNAL_ENABLED,
            vol_eta=VOL_ETA,
            vol_z_star=VOL_Z_STAR,
            hybrid_displacement=HYBRID_DISPLACEMENT,
            hybrid_variant=HYBRID_VARIANT,
            corr_blend_w=CORR_BLEND_W,
            corr_lambda=CORR_LAMBDA,
        )
        alloc_cfg_run.validate()

        sleeve_tag = f"{int(sleeve*100)}pct"
        floor_tag  = str(floor).replace("0.", "")
        equity_displ_key = "equity_only" if alloc_cfg_run.equity_only_displacement == True else "core_repl"
        vol_tag = f"_volA_e{VOL_ETA}_z{VOL_Z_STAR}".replace(".", "") if VOL_SIGNAL_ENABLED else ""
        ck_tag = "" if CORE_SPLIT_KAPPA == 0.2 else f"_ck{CORE_SPLIT_KAPPA:g}".replace(".", "")
        hyb_tag = f"_hyb{HYBRID_VARIANT}" if HYBRID_DISPLACEMENT else ""
        corr_tag = ((f"_rb{int(CORR_BLEND_W*100)}" if CORR_BLEND_W > 0 else "")
                    + (f"_lam{CORR_LAMBDA:g}".replace(".", "") if CORR_LAMBDA != 0 else ""))

        for inv_key, investor_cfg in investors.items():
            run_n += 1
            tag = f"EW_{sleeve_tag}_floor{floor_tag}_{inv_key}_{equity_displ_key}{vol_tag}{ck_tag}{hyb_tag}{corr_tag}"
            print(f"\n[{run_n}/{total}] === {tag} ===")

            bt_ew = run_expanding_window_backtest(
                res_core=res_core,
                allocation_df=allocation_df,
                hmm_cfg=HMM_CFG,
                alloc_cfg=alloc_cfg_run,
                investor_cfg=investor_cfg,
                satellite_specs=SECTOR_SPECS,
                benchmark_weights=BENCHMARK_WEIGHTS,
                ew_cfg=ew_cfg,
                signal_return_prefix="ExcessLog",
                realized_return_prefix="Log",
                periods_per_year=12,
                store_candidate_scores=False,
                cash_sleeve_cfg=CASH_SLEEVE,
                vol_z=vol_z,
                rho_realized=rho_realized,
            )
            print(bt_ew.performance_summary)

            if EXPORT_EXPANDING_WINDOW:
                export_allocation_backtest_to_excel(
                    backtest_res=bt_ew,
                    alloc_cfg=alloc_cfg_run,
                    investor_cfg=investor_cfg,
                    satellite_specs=SECTOR_SPECS,
                    res_core=res_core,
                    output_file=f"allocation_backtest_{tag}.xlsx",
                )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    df = clean_data(TICKERS_ALL, M_TICKERS, W_TICKERS)
    results = run_hmm_models(df)
    run_a1_backtest(results, df)
    run_sensitivity_grid(results, df)


if __name__ == "__main__":
    main()
