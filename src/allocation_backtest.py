import numpy as np
import pandas as pd

from src.allocation_regime import (
    fit_hmm_train_only_for_allocation,
    filter_test_probabilities_fixed_params,
    build_test_predictive_probability_panel,
)
from src.allocation_scoring import (
    build_candidate_library_from_train,
    select_best_tilt_at_date_from_library,
)
from src.allocation_config import BacktestResult


def _extract_return_panel(
    allocation_df: pd.DataFrame,
    assets: list[str],
    return_prefix: str,
) -> pd.DataFrame:
    cols = [f"{return_prefix}{a}" for a in assets]
    missing = [c for c in cols if c not in allocation_df.columns]
    if missing:
        raise KeyError(f"allocation_df is missing required return columns: {missing}")

    out = allocation_df[cols].copy()
    out.columns = assets
    return out


def _compute_turnover(prev_weights: dict[str, float], new_weights: dict[str, float], assets: list[str]) -> float:
    prev = pd.Series({a: prev_weights.get(a, 0.0) for a in assets}, dtype=float)
    new = pd.Series({a: new_weights.get(a, 0.0) for a in assets}, dtype=float)
    return float((new - prev).abs().sum())


def _apply_turnover_limit(
    prev_weights: dict[str, float],
    target_weights: dict[str, float],
    assets: list[str],
    turnover_limit: float | None,
) -> tuple[dict[str, float], float]:
    prev = pd.Series({a: prev_weights.get(a, 0.0) for a in assets}, dtype=float)
    target = pd.Series({a: target_weights.get(a, 0.0) for a in assets}, dtype=float)

    raw_turnover = float((target - prev).abs().sum())

    if turnover_limit is None or raw_turnover <= turnover_limit or raw_turnover == 0:
        return target.to_dict(), raw_turnover

    alpha = turnover_limit / raw_turnover
    new = prev + alpha * (target - prev)

    return new.to_dict(), float((new - prev).abs().sum())


def _decision_list_to_df(decisions: list) -> pd.DataFrame:
    rows = []
    for d in decisions:
        rows.append({
            "rebalance_date": d.rebalance_date,
            "investor_name": d.investor_name,
            "predictive_probabilities": d.predictive_probabilities,
            "baseline_score": d.baseline_score,
            "selected_satellites": ", ".join(d.selected_satellites) if d.selected_satellites else "NONE",
            "selected_weights": d.selected_weights,
            "total_portfolio_weights": d.total_portfolio_weights,
            "expected_return": d.expected_return,
            "variance": d.variance,
            "skewness": d.skewness,
            "kurtosis": d.kurtosis,
            "final_score": d.final_score,
            "score_improvement": d.score_improvement,
            "metadata": d.metadata,
        })
    return pd.DataFrame(rows)


def performance_summary_from_returns(
    strategy_simple_returns: pd.Series,
    benchmark_simple_returns: pd.Series | None = None,
    periods_per_year: int = 12,
    avg_turnover: float | None = None,
    label: str = "strategy",
) -> pd.DataFrame:
    def _stats(r: pd.Series, name: str) -> dict[str, float | str]:
        r = pd.to_numeric(r, errors="coerce").dropna()
        if len(r) == 0:
            return {
                "strategy": name,
                "n_periods": 0,
                "CAGR": np.nan,
                "Volatility": np.nan,
                "Sharpe": np.nan,
                "Max_Drawdown": np.nan,
                "Downside_Deviation": np.nan,
            }

        wealth = (1.0 + r).cumprod()
        cagr = wealth.iloc[-1] ** (periods_per_year / len(r)) - 1.0
        vol = r.std(ddof=1) * np.sqrt(periods_per_year)
        sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else np.nan

        running_max = wealth.cummax()
        drawdown = wealth / running_max - 1.0
        max_dd = drawdown.min()

        downside = r[r < 0]
        downside_dev = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan

        return {
            "strategy": name,
            "n_periods": len(r),
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Max_Drawdown": max_dd,
            "Downside_Deviation": downside_dev,
        }

    rows = [_stats(strategy_simple_returns, label)]
    if benchmark_simple_returns is not None:
        rows.append(_stats(benchmark_simple_returns, f"{label}_benchmark"))

    out = pd.DataFrame(rows)
    if avg_turnover is not None:
        out["Avg_Turnover"] = avg_turnover
    return out


# ---------------------------------------------------------------------
# 2) MAIN STAGE 6 BACKTEST RUNNER
# ---------------------------------------------------------------------

def run_regime_allocation_backtest(
    res: "ModelRunResult",
    allocation_df: pd.DataFrame,
    alloc_cfg,
    investor_cfg,
    satellite_specs: list,
    benchmark_weights: dict[str, float] | None = None,
    signal_return_prefix: str = "ExcessLog",
    realized_return_prefix: str = "Log",
    periods_per_year: int = 12,
    store_candidate_scores: bool = True,
) -> BacktestResult:
    """
    Run the prototype regime-aware allocation backtest.
 
    Prototype timing convention
    ---------------------------
    At date t:
      1) use filtered probabilities from the existing HMM result
      2) compute one-step predictive probabilities pi_{t+1|t}
      3) evaluate candidate tilts
      4) choose the best tilt
      5) apply chosen weights to realized return at t+1
 
    This is the correct Stage 6 prototype because:
    - decisions are next-period only
    - no smoothed probabilities are used
    - the logic stays wrapped around your current HMM pipeline
 
    Parameters
    ----------
    res : ModelRunResult
        Core regime-engine result from the current HMM pipeline.
 
    allocation_df : pd.DataFrame
        Return panel containing both:
        - signal columns (e.g. ExcessLog...)
        - realized columns (e.g. Log...)
 
    alloc_cfg : AllocationConfig
        Allocation settings.
 
    investor_cfg : InvestorPreferenceConfig
        Investor preference to score candidate tilts.
 
    satellite_specs : list[SatelliteSpec]
        Satellite candidate definitions.
 
    benchmark_weights : dict[str, float] | None
        Optional static benchmark portfolio over the same allocation universe.
 
    signal_return_prefix : str
        Prefix used by Stage 4/5 scoring, usually "ExcessLog".
 
    realized_return_prefix : str
        Prefix used for actual backtest PnL, usually "Log".
 
    periods_per_year : int
        12 for monthly, 52 for weekly.
 
    store_candidate_scores : bool
        If True, keep the full candidate-score table through time.
 
    Returns
    -------
    BacktestResult
    """
    # 1) predictive regime probabilities
    pred_probs = build_predictive_probability_panel(res, steps_ahead=1)
 
    # 2) build the tradable asset universe
    core_assets = list(alloc_cfg.fixed_core_weights.keys())
    sat_assets = [s.ticker for s in satellite_specs]
    all_assets = list(dict.fromkeys(core_assets + sat_assets))  # preserve order, unique
 
    realized_asset_returns = _extract_return_panel(
        allocation_df=allocation_df,
        assets=all_assets,
        return_prefix=realized_return_prefix,
    )
 
    # 3) align dates
    common_dates = pred_probs.index.intersection(realized_asset_returns.index)
    common_dates = common_dates.sort_values()
 
    if len(common_dates) < 2:
        raise ValueError("Not enough overlapping dates to run a next-period backtest.")
 
    decisions = []
    candidate_score_tables = []
 
    realized_rows = []
    weights_rows = []
 
    prev_applied_weights = alloc_cfg.fixed_core_weights.copy()
 
    # We decide at date t and realize at the next date t+1
    for i in range(len(common_dates) - 1):
        rebalance_date = common_dates[i]
        realized_date = common_dates[i + 1]
 
        decision, candidate_table = select_best_tilt_at_date(
            res=res,
            allocation_df=allocation_df,
            alloc_cfg=alloc_cfg,
            investor_cfg=investor_cfg,
            satellite_specs=satellite_specs,
            predictive_probability_panel=pred_probs,
            rebalance_date=rebalance_date,
            return_prefix=signal_return_prefix,
        )
 
        # turnover control (optional)
        applied_weights, realized_turnover = _apply_turnover_limit(
            prev_weights=prev_applied_weights,
            target_weights=decision.total_portfolio_weights,
            assets=all_assets,
            turnover_limit=alloc_cfg.turnover_limit,
        )
 
        # realized period return at t+1
        realized_log_row = realized_asset_returns.loc[realized_date]
        realized_log_return = float(
            np.dot(
                np.array([applied_weights.get(a, 0.0) for a in all_assets], dtype=float),
                realized_log_row.to_numpy(dtype=float),
            )
        )
 
        gross_simple_return = float(np.expm1(realized_log_return))
 
        # optional transaction costs
        transaction_cost = realized_turnover * (alloc_cfg.transaction_cost_bps / 10000.0)
        net_simple_return = gross_simple_return - transaction_cost
 
        # store decision and realized outcome
        decision.metadata = {
            **decision.metadata,
            "realized_date": realized_date,
            "applied_weights_after_turnover_limit": applied_weights,
            "turnover": realized_turnover,
            "transaction_cost": transaction_cost,
            "gross_simple_return": gross_simple_return,
            "net_simple_return": net_simple_return,
        }
        decisions.append(decision)
 
        # decision log / weight history / realized returns
        weights_rows.append(
            pd.Series(applied_weights, name=realized_date)
        )
 
        row = {
            "rebalance_date": rebalance_date,
            "realized_date": realized_date,
            "gross_simple_return": gross_simple_return,
            "net_simple_return": net_simple_return,
            "turnover": realized_turnover,
            "transaction_cost": transaction_cost,
        }
        realized_rows.append(pd.Series(row, name=realized_date))
 
        if store_candidate_scores:
            ct = candidate_table.copy()
            ct["realized_date"] = realized_date
            ct["model_label"] = res.spec.label
            candidate_score_tables.append(ct)
 
        prev_applied_weights = applied_weights
 
    # 4) assemble outputs
    decision_log = _decision_list_to_df(decisions)
    decision_log["realized_date"] = [d.metadata["realized_date"] for d in decisions]
    decision_log["turnover"] = [d.metadata["turnover"] for d in decisions]
    decision_log["transaction_cost"] = [d.metadata["transaction_cost"] for d in decisions]
    decision_log["gross_simple_return"] = [d.metadata["gross_simple_return"] for d in decisions]
    decision_log["net_simple_return"] = [d.metadata["net_simple_return"] for d in decisions]
 
    weights_df = pd.DataFrame(weights_rows).sort_index()
    realized_df = pd.DataFrame(realized_rows).sort_index()
 
    strategy_returns = realized_df["net_simple_return"].copy()
 
    cumulative_wealth = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    running_max = cumulative_wealth.cummax()
    drawdown = cumulative_wealth / running_max - 1.0
 
    benchmark_returns = None
    benchmark_wealth = None
    benchmark_drawdown = None
 
    if benchmark_weights is not None:
        bw = pd.Series({a: benchmark_weights.get(a, 0.0) for a in all_assets}, dtype=float)
        benchmark_log_return = (realized_asset_returns.loc[weights_df.index, all_assets] * bw.values).sum(axis=1)
        benchmark_returns = np.expm1(benchmark_log_return)
        benchmark_wealth = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
        benchmark_drawdown = benchmark_wealth / benchmark_wealth.cummax() - 1.0
 
    perf = performance_summary_from_returns(
        strategy_simple_returns=strategy_returns,
        benchmark_simple_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        avg_turnover=realized_df["turnover"].mean(),
        label=f"{res.spec.label} | {investor_cfg.name}",
    )
 
    candidate_scores = None
    if store_candidate_scores and candidate_score_tables:
        candidate_scores = pd.concat(candidate_score_tables, ignore_index=True)
 
    return BacktestResult(
        model_label=res.spec.label,
        investor_name=investor_cfg.name,
        decisions=decisions,
        decision_log=decision_log,
        weights=weights_df,
        asset_returns=realized_asset_returns.loc[weights_df.index, all_assets].copy(),
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        cumulative_wealth=cumulative_wealth,
        benchmark_wealth=benchmark_wealth,
        drawdown=drawdown,
        benchmark_drawdown=benchmark_drawdown,
        performance_summary=perf,
        candidate_scores=candidate_scores,
    )
 

def run_fixed_parameter_train_test_backtest(
    res_core,
    allocation_df: pd.DataFrame,
    hmm_cfg,
    tt_cfg,
    alloc_cfg,
    investor_cfg,
    satellite_specs: list,
    benchmark_weights: dict[str, float] | None = None,
    signal_return_prefix: str = "ExcessLog",
    realized_return_prefix: str = "Log",
    periods_per_year: int = 12,
    store_candidate_scores: bool = True,
) -> tuple[BacktestResult, object]:
    """
    Honest A1 regime-allocation backtest.
 
    Train
    -----
    - fit HMM on train only
    - freeze HMM parameters
    - estimate candidate moments on train only
    - freeze candidate library
 
    Test
    ----
    At each rebalance date t in test:
      - update filtered probability using frozen HMM and only data up to t
      - compute pi_{t+1|t}
      - score candidates using TRAIN-only moment library
      - choose best tilt
      - apply to realized return at t+1
    """
    # 1) fit HMM on train only and freeze parameters
    frozen_state = fit_hmm_train_only_for_allocation(
        res_core=res_core,
        hmm_cfg=hmm_cfg,
        tt_cfg=tt_cfg,
    )
 
    # 2) honest test-period filtered and predictive probabilities
    filtered_test = filter_test_probabilities_fixed_params(frozen_state)
    pred_test = build_test_predictive_probability_panel(frozen_state, steps_ahead=1)
 
    # 3) build train-sample allocation panel for candidate moments
    core_assets = list(alloc_cfg.fixed_core_weights.keys())
    sat_assets = [s.ticker for s in satellite_specs]
    all_assets = list(dict.fromkeys(core_assets + sat_assets))
 
    allocation_df_train = allocation_df.loc[frozen_state.train_x.index].copy()
 
    # 4) precompute candidate library using TRAIN only
    candidate_library = build_candidate_library_from_train(
        state_series_train=frozen_state.train_df_m["state"],
        regime_names=frozen_state.regime_names,
        allocation_df_train=allocation_df_train,
        alloc_cfg=alloc_cfg,
        satellite_specs=satellite_specs,
        return_prefix=signal_return_prefix,
    )
 
    # 5) realized asset return panel
    realized_asset_returns = _extract_return_panel(
        allocation_df=allocation_df,
        assets=all_assets,
        return_prefix=realized_return_prefix,
    )
 
    test_dates = pred_test.index.intersection(realized_asset_returns.index).sort_values()
    if len(test_dates) < 2:
        raise ValueError("Need at least 2 test dates for next-period realized backtest.")
 
    decisions = []
    candidate_score_tables = []
    realized_rows = []
    weights_rows = []
 
    prev_applied_weights = alloc_cfg.fixed_core_weights.copy()
 
    # decide at t, realize at t+1
    for i in range(len(test_dates) - 1):
        rebalance_date = test_dates[i]
        realized_date = test_dates[i + 1]
 
        pred_row = pred_test.loc[rebalance_date]
 
        decision, candidate_table = select_best_tilt_at_date_from_library(
            candidate_library=candidate_library,
            predictive_probabilities_row=pred_row,
            investor_cfg=investor_cfg,
            alloc_cfg=alloc_cfg,
            rebalance_date=rebalance_date,
            satellite_specs=satellite_specs,
        )
 
        applied_weights, realized_turnover = _apply_turnover_limit(
            prev_weights=prev_applied_weights,
            target_weights=decision.total_portfolio_weights,
            assets=all_assets,
            turnover_limit=alloc_cfg.turnover_limit,
        )
 
        realized_log_row = realized_asset_returns.loc[realized_date]
        realized_log_return = float(
            np.dot(
                np.array([applied_weights.get(a, 0.0) for a in all_assets], dtype=float),
                realized_log_row.to_numpy(dtype=float),
            )
        )
 
        gross_simple_return = float(np.expm1(realized_log_return))
        transaction_cost = realized_turnover * (alloc_cfg.transaction_cost_bps / 10000.0)
        net_simple_return = gross_simple_return - transaction_cost
 
        decision.metadata = {
            **decision.metadata,
            "realized_date": realized_date,
            "applied_weights_after_turnover_limit": applied_weights,
            "turnover": realized_turnover,
            "transaction_cost": transaction_cost,
            "gross_simple_return": gross_simple_return,
            "net_simple_return": net_simple_return,
        }
        decisions.append(decision)
 
        weights_rows.append(pd.Series(applied_weights, name=realized_date))
 
        row = {
            "rebalance_date": rebalance_date,
            "realized_date": realized_date,
            "gross_simple_return": gross_simple_return,
            "net_simple_return": net_simple_return,
            "turnover": realized_turnover,
            "transaction_cost": transaction_cost,
        }
        realized_rows.append(pd.Series(row, name=realized_date))
 
        if store_candidate_scores:
            ct = candidate_table.copy()
            ct["realized_date"] = realized_date
            ct["model_label"] = res_core.spec.label
            candidate_score_tables.append(ct)
 
        prev_applied_weights = applied_weights
 
    decision_log = _decision_list_to_df(decisions)
    decision_log["realized_date"] = [d.metadata["realized_date"] for d in decisions]
    decision_log["turnover"] = [d.metadata["turnover"] for d in decisions]
    decision_log["transaction_cost"] = [d.metadata["transaction_cost"] for d in decisions]
    decision_log["gross_simple_return"] = [d.metadata["gross_simple_return"] for d in decisions]
    decision_log["net_simple_return"] = [d.metadata["net_simple_return"] for d in decisions]
 
    weights_df = pd.DataFrame(weights_rows).sort_index()
    realized_df = pd.DataFrame(realized_rows).sort_index()
 
    strategy_returns = realized_df["net_simple_return"].copy()
    cumulative_wealth = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    drawdown = cumulative_wealth / cumulative_wealth.cummax() - 1.0
 
    benchmark_returns = None
    benchmark_wealth = None
    benchmark_drawdown = None
 
    if benchmark_weights is not None:
        bw = pd.Series({a: benchmark_weights.get(a, 0.0) for a in all_assets}, dtype=float)
        benchmark_log_return = (realized_asset_returns.loc[weights_df.index, all_assets] * bw.values).sum(axis=1)
        benchmark_returns = np.expm1(benchmark_log_return)
        benchmark_wealth = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
        benchmark_drawdown = benchmark_wealth / benchmark_wealth.cummax() - 1.0
 
    perf = performance_summary_from_returns(
        strategy_simple_returns=strategy_returns,
        benchmark_simple_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        avg_turnover=realized_df["turnover"].mean(),
        label=f"{res_core.spec.label} | {investor_cfg.name}",
    )
 
    candidate_scores = None
    if store_candidate_scores and candidate_score_tables:
        candidate_scores = pd.concat(candidate_score_tables, ignore_index=True)
 
    backtest_res = BacktestResult(
        model_label=res_core.spec.label,
        investor_name=investor_cfg.name,
        decisions=decisions,
        decision_log=decision_log,
        weights=weights_df,
        asset_returns=realized_asset_returns.loc[weights_df.index, all_assets].copy(),
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        cumulative_wealth=cumulative_wealth,
        benchmark_wealth=benchmark_wealth,
        drawdown=drawdown,
        benchmark_drawdown=benchmark_drawdown,
        performance_summary=perf,
        candidate_scores=candidate_scores,
    )
 
    return backtest_res, frozen_state


# =====================================================================
# EXPANDING WINDOW (IN-SAMPLE) BACKTEST
# =====================================================================
 
from dataclasses import dataclass as _dataclass
 
 
@_dataclass
class ExpandingWindowConfig:
    """
    Configuration for the expanding-window in-sample backtest.
 
    At each rebalance date t, the model uses ALL data up to and including
    t-1 to:
      1) re-fit the HMM from scratch (seed sweep)
      2) re-estimate candidate satellite moments
      3) compute one-step-ahead predictive probabilities
      4) pick the best tilt
      5) apply to the realized return at t
 
    This is the honest in-sample analogue of A1: no future data is ever
    used, but unlike A1 the model keeps learning as history accumulates.
 
    Philosophy
    ----------
    The key thesis insight this enables: compare the expanding-window
    result to A1 across two sub-periods:
      - Crisis period (2000–2009): many regime transitions, strategy earns
        the most — this is where regime awareness pays.
      - Bull period (2010–2026): few transitions, strategy adds little —
        this is the honest null result that A1 showed.
 
    Why not re-fit HMM every month in A1?
    --------------------------------------
    In A1 the candidate moment library is frozen at train-time.
    Re-fitting the HMM would give fresh probabilities but stale moments,
    so XLE would still always win (its commodity-supercycle 1999–2009
    train moments stay fixed). The expanding window fixes BOTH: fresh
    HMM AND fresh candidate moments at every step.
 
    Fields
    ------
    burn_in_periods:
        Minimum number of months needed before making the first decision.
        Must be >= alloc_cfg.min_regime_obs and >= hmm_cfg seeds need.
        Default 60 (5 years). Increasing this gives the HMM a more
        stable initial fit; decreasing it lets you start earlier.
 
    refit_every_n_periods:
        Re-fit the HMM every N rebalance dates instead of every month.
        Default 1 (monthly re-fit). Set to 3 for quarterly re-fit,
        12 for annual. Larger values are faster and reduce chatter from
        month-to-month HMM label switching.
        Recommendation: start with 1 for the thesis, note the runtime.
 
    verbose:
        Print progress every N periods. Useful for long runs.
    """
    burn_in_periods: int = 60
    refit_every_n_periods: int = 1
    verbose: bool = True
    verbose_every: int = 12
 
 

def run_expanding_window_backtest(
    res_core,
    allocation_df: pd.DataFrame,
    hmm_cfg,
    alloc_cfg,
    investor_cfg,
    satellite_specs: list,
    benchmark_weights: dict[str, float] | None = None,
    ew_cfg: "ExpandingWindowConfig | None" = None,
    signal_return_prefix: str = "ExcessLog",
    realized_return_prefix: str = "Log",
    periods_per_year: int = 12,
    store_candidate_scores: bool = False,
) -> "BacktestResult":
    """
    Expanding-window in-sample backtest.
 
    At each rebalance date t (after burn-in):
      - HMM is re-fit on ALL data from the start up to t-1
      - Candidate moment library is re-estimated on the same window
      - One-step-ahead predictive probability pi_{t|t-1} is computed
      - Best satellite tilt is selected
      - Realized return at t is applied
 
    This avoids lookahead bias while allowing the model and moment
    estimates to adapt as new data arrives — fixing the stale-library
    problem identified in the A1 frozen backtest.
 
    Parameters
    ----------
    res_core : ModelRunResult
        Full-sample HMM result. We use res_core.x (the full feature
        matrix) and res_core.spec (tickers, key_col, etc.).
        The fitted model inside res_core is NOT used directly — we
        re-fit from scratch at each window.
 
    allocation_df : pd.DataFrame
        Full return panel (ExcessLog... and Log... columns) for all
        assets including satellites. Must cover the full date range.
 
    hmm_cfg : GlobalRunConfig
        HMM settings (n_states, cov_type, seeds). The seed sweep runs
        at every re-fit step, so runtime scales with len(seeds) * T.
        Consider reducing seeds to range(1, 11) for faster iteration
        during development.
 
    alloc_cfg : AllocationConfig
        Allocation settings. fixed_core_weights, satellite universe,
        transaction costs, score_improvement_floor all apply as normal.
 
    investor_cfg : InvestorPreferenceConfig
        Investor preference (MV / MVS / MVK) with JR-calibrated params.
 
    satellite_specs : list[SatelliteSpec]
        Satellite candidate definitions.
 
    benchmark_weights : dict[str, float] | None
        Static benchmark portfolio for performance comparison.
 
    ew_cfg : ExpandingWindowConfig | None
        Expanding window settings. Uses defaults if None.
 
    signal_return_prefix : str
        Prefix for candidate scoring returns. Default "ExcessLog".
 
    realized_return_prefix : str
        Prefix for realized PnL returns. Default "Log".
 
    periods_per_year : int
        12 for monthly.
 
    store_candidate_scores : bool
        Whether to store the full candidate table at each date.
        Default False — the table is large and this is a long backtest.
 
    Returns
    -------
    BacktestResult
        Same structure as A1, with an additional "window_sizes" entry
        in decision metadata recording how many observations were used
        at each re-fit.
    """
    from src.allocation_regime import (
        fit_hmm_train_only_for_allocation,
        filter_test_probabilities_fixed_params,
        build_test_predictive_probability_panel,
        _extract_filtered_probabilities_from_dfm,
        _reorder_hmm_params,
        forward_filter_fixed_hmm,
        predictive_regime_probabilities_from_filtered,
    )
    from src.allocation_scoring import (
        build_candidate_library_from_train,
        select_best_tilt_at_date_from_library,
    )
    from src.hmm import hmm_sweep_seeds
    from src.postprocess import RegimePostProcessor
    from src.allocation_config import FrozenHMMState
 
    if ew_cfg is None:
        ew_cfg = ExpandingWindowConfig()
 
    # ── full feature matrix (core assets only, for HMM fitting) ──────
    x_full = res_core.x.sort_index().copy()
    all_dates = x_full.index
 
    # ── asset universe ────────────────────────────────────────────────
    core_assets = list(alloc_cfg.fixed_core_weights.keys())
    sat_assets  = [s.ticker for s in satellite_specs]
    all_assets  = list(dict.fromkeys(core_assets + sat_assets))
 
    realized_asset_returns = _extract_return_panel(
        allocation_df=allocation_df,
        assets=all_assets,
        return_prefix=realized_return_prefix,
    )
 
    # ── decision dates: every date after burn-in that also has
    #    realized returns at t+1 ────────────────────────────────────
    first_decision_idx = ew_cfg.burn_in_periods   # index into all_dates
    if first_decision_idx >= len(all_dates) - 1:
        raise ValueError(
            f"burn_in_periods={ew_cfg.burn_in_periods} leaves no decision dates. "
            f"Total periods available: {len(all_dates)}."
        )
 
    decision_dates = all_dates[first_decision_idx:-1]   # exclude last (no t+1)
    realized_dates = all_dates[first_decision_idx + 1:]
 
    # ── state tracking ────────────────────────────────────────────────
    decisions            = []
    candidate_score_tabs = []
    realized_rows        = []
    weights_rows         = []
    prev_weights         = alloc_cfg.fixed_core_weights.copy()
 
    # Cache: re-fit every N periods
    frozen: FrozenHMMState | None = None
    candidate_library             = None
    last_refit_idx                = -999
 
    for step, (rebalance_date, realized_date) in enumerate(
        zip(decision_dates, realized_dates)
    ):
        window_end_idx = first_decision_idx + step   # inclusive, in all_dates
 
        # ── re-fit if due ─────────────────────────────────────────────
        needs_refit = (step - last_refit_idx) >= ew_cfg.refit_every_n_periods
 
        if needs_refit:
            train_x = x_full.iloc[: window_end_idx + 1].copy()   # up to t-1 inclusive
 
            if ew_cfg.verbose and step % ew_cfg.verbose_every == 0:
                print(
                    f"  [EW step {step+1}/{len(decision_dates)}] "
                    f"rebalance={rebalance_date.date()} "
                    f"window={len(train_x)} obs — refitting HMM..."
                )
 
            try:
                sweep, best_seed, out_train, df_m_train, model_train = hmm_sweep_seeds(
                    df=train_x,
                    n_states=hmm_cfg.n_states,
                    cols=res_core.spec.tickers,
                    cov_type=hmm_cfg.cov_type,
                    seeds=hmm_cfg.seeds,
                    verbose=False,
                )
 
                # Guard: hmm_sweep_seeds returns the best non-collapsed run,
                # but with very few seeds some windows may still produce a
                # df_m_train where not all n_states are represented in the
                # p_state columns (collapsed run chosen as fallback).
                # Check now so we fail loudly here rather than inside
                # RegimePostProcessor with a cryptic length-mismatch error.
                p_cols = [c for c in df_m_train.columns if c.startswith("p_state")]
                if len(p_cols) != hmm_cfg.n_states:
                    raise ValueError(
                        f"HMM returned {len(p_cols)} p_state columns "
                        f"but n_states={hmm_cfg.n_states}. "
                        f"All seeds produced collapsed states — increase seeds or burn_in_periods."
                    )
 
            except Exception as e:
                if ew_cfg.verbose:
                    print(f"    HMM fit failed at step {step} ({rebalance_date.date()}): {e}"
                          f" — carrying forward previous fit.")
                if frozen is None:
                    # Nothing to carry forward yet — skip to next period
                    continue
                # else: frozen state from previous successful fit stays active
            else:
                # Successful fit — build frozen state
                pp = RegimePostProcessor(
                    model_name=f"{res_core.spec.label} | EW t={rebalance_date.date()}",
                    n_states=hmm_cfg.n_states,
                    key_col=res_core.spec.key_col,
                ).fit(df_m_train, out_train)
 
                filtered_train = _extract_filtered_probabilities_from_dfm(
                    pp.df_m, pp.regime_names
                )
 
                startprob, transmat, means, covars = _reorder_hmm_params(
                    model_train, pp.order_old, hmm_cfg.cov_type
                )
 
                frozen = FrozenHMMState(
                    model_label=res_core.spec.label,
                    model_code=res_core.spec.code,
                    n_states=hmm_cfg.n_states,
                    regime_names=pp.regime_names,
                    order_old=pp.order_old,
                    covariance_type=hmm_cfg.cov_type,
                    startprob=startprob,
                    transmat=transmat,
                    means=means,
                    covars=covars,
                    train_x=train_x,
                    test_x=pd.DataFrame(),        # not used in EW
                    train_df_m=pp.df_m.copy(),
                    train_out=out_train.copy(),
                    filtered_train=filtered_train,
                    filtered_test=None,
                )
 
                # Re-estimate candidate library on this window
                alloc_df_window = allocation_df.loc[
                    allocation_df.index.isin(train_x.index)
                ].copy()
 
                candidate_library = build_candidate_library_from_train(
                    state_series_train=pp.df_m["state"],
                    regime_names=pp.regime_names,
                    allocation_df_train=alloc_df_window,
                    alloc_cfg=alloc_cfg,
                    satellite_specs=satellite_specs,
                    return_prefix=signal_return_prefix,
                )
 
                last_refit_idx = step
 
        # ── predictive probability at t ───────────────────────────────
        # We already have filtered probs through t-1 (last row of
        # frozen.filtered_train). One-step ahead gives pi_{t|t-1}.
        last_filtered = frozen.filtered_train.iloc[-1].to_numpy(dtype=float)
 
        P = pd.DataFrame(
            frozen.transmat,
            index=frozen.regime_names,
            columns=frozen.regime_names,
        )
 
        pred_vec = last_filtered @ frozen.transmat
        pred_vec = pred_vec / pred_vec.sum()
 
        pred_row = pd.Series(pred_vec, index=frozen.regime_names)
 
        # ── tilt selection ────────────────────────────────────────────
        decision, candidate_table = select_best_tilt_at_date_from_library(
            candidate_library=candidate_library,
            predictive_probabilities_row=pred_row,
            investor_cfg=investor_cfg,
            alloc_cfg=alloc_cfg,
            rebalance_date=rebalance_date,
            satellite_specs=satellite_specs,
        )
 
        # ── apply turnover limit and compute realized return ──────────
        applied_weights, realized_turnover = _apply_turnover_limit(
            prev_weights=prev_weights,
            target_weights=decision.total_portfolio_weights,
            assets=all_assets,
            turnover_limit=alloc_cfg.turnover_limit,
        )
 
        realized_log_row    = realized_asset_returns.loc[realized_date]
        realized_log_return = float(
            np.dot(
                np.array([applied_weights.get(a, 0.0) for a in all_assets], dtype=float),
                realized_log_row.to_numpy(dtype=float),
            )
        )
 
        gross_simple_return = float(np.expm1(realized_log_return))
        transaction_cost    = realized_turnover * (alloc_cfg.transaction_cost_bps / 10000.0)
        net_simple_return   = gross_simple_return - transaction_cost
 
        decision.metadata = {
            **decision.metadata,
            "realized_date":                       realized_date,
            "applied_weights_after_turnover_limit": applied_weights,
            "turnover":                            realized_turnover,
            "transaction_cost":                    transaction_cost,
            "gross_simple_return":                 gross_simple_return,
            "net_simple_return":                   net_simple_return,
            "window_size":                         window_end_idx + 1,
        }
        decisions.append(decision)
 
        weights_rows.append(pd.Series(applied_weights, name=realized_date))
        realized_rows.append(pd.Series({
            "rebalance_date":     rebalance_date,
            "realized_date":      realized_date,
            "gross_simple_return": gross_simple_return,
            "net_simple_return":   net_simple_return,
            "turnover":            realized_turnover,
            "transaction_cost":    transaction_cost,
            "window_size":         window_end_idx + 1,
        }, name=realized_date))
 
        if store_candidate_scores:
            ct = candidate_table.copy()
            ct["realized_date"] = realized_date
            ct["model_label"]   = res_core.spec.label
            candidate_score_tabs.append(ct)
 
        prev_weights = applied_weights
 
    # ── assemble outputs ──────────────────────────────────────────────
    if not decisions:
        raise ValueError("No decisions were made — check burn_in_periods vs data length.")
 
    decision_log = _decision_list_to_df(decisions)
    decision_log["realized_date"]       = [d.metadata["realized_date"]       for d in decisions]
    decision_log["turnover"]            = [d.metadata["turnover"]            for d in decisions]
    decision_log["transaction_cost"]    = [d.metadata["transaction_cost"]    for d in decisions]
    decision_log["gross_simple_return"] = [d.metadata["gross_simple_return"] for d in decisions]
    decision_log["net_simple_return"]   = [d.metadata["net_simple_return"]   for d in decisions]
    decision_log["window_size"]         = [d.metadata["window_size"]         for d in decisions]
 
    weights_df   = pd.DataFrame(weights_rows).sort_index()
    realized_df  = pd.DataFrame(realized_rows).sort_index()
 
    strategy_returns  = realized_df["net_simple_return"].copy()
    cumulative_wealth = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    drawdown          = cumulative_wealth / cumulative_wealth.cummax() - 1.0
 
    benchmark_returns  = None
    benchmark_wealth   = None
    benchmark_drawdown = None
 
    if benchmark_weights is not None:
        bw = pd.Series({a: benchmark_weights.get(a, 0.0) for a in all_assets}, dtype=float)
        benchmark_log_return = (
            realized_asset_returns.loc[weights_df.index, all_assets] * bw.values
        ).sum(axis=1)
        benchmark_returns  = np.expm1(benchmark_log_return)
        benchmark_wealth   = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
        benchmark_drawdown = benchmark_wealth / benchmark_wealth.cummax() - 1.0
 
    perf = performance_summary_from_returns(
        strategy_simple_returns=strategy_returns,
        benchmark_simple_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        avg_turnover=realized_df["turnover"].mean(),
        label=f"{res_core.spec.label} | {investor_cfg.name} | EW",
    )
 
    candidate_scores = None
    if store_candidate_scores and candidate_score_tabs:
        candidate_scores = pd.concat(candidate_score_tabs, ignore_index=True)
 
    return BacktestResult(
        model_label=f"{res_core.spec.label} | EW",
        investor_name=investor_cfg.name,
        decisions=decisions,
        decision_log=decision_log,
        weights=weights_df,
        asset_returns=realized_asset_returns.loc[weights_df.index, all_assets].copy(),
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        cumulative_wealth=cumulative_wealth,
        benchmark_wealth=benchmark_wealth,
        drawdown=drawdown,
        benchmark_drawdown=benchmark_drawdown,
        performance_summary=perf,
        candidate_scores=candidate_scores,
    )