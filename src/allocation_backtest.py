from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from src.allocation_config import BacktestResult
from src.allocation_regime import build_predictive_probability_panel
from src.allocation_scoring import select_best_tilt_at_date

if TYPE_CHECKING:
    from src.runner import ModelRunResult


# ---------------------------------------------------------------------
# 1) HELPERS
# ---------------------------------------------------------------------

def _extract_return_panel(
    allocation_df: pd.DataFrame,
    assets: list[str],
    return_prefix: str,
) -> pd.DataFrame:
    """
    Extract a return panel with columns renamed to raw asset tickers.

    Example
    -------
    If assets = ["^SP500TR", "LT09TRUU", "XAU"] and return_prefix="Log",
    this expects columns:
        Log^SP500TR, LogLT09TRUU, LogXAU
    """
    cols = [f"{return_prefix}{a}" for a in assets]
    missing = [c for c in cols if c not in allocation_df.columns]
    if missing:
        raise KeyError(f"allocation_df is missing required return columns: {missing}")

    out = allocation_df[cols].copy()
    out.columns = assets
    return out


def _compute_turnover(prev_weights: dict[str, float], new_weights: dict[str, float], assets: list[str]) -> float:
    """
    Sum absolute changes in portfolio weights.
    """
    prev = pd.Series({a: prev_weights.get(a, 0.0) for a in assets}, dtype=float)
    new = pd.Series({a: new_weights.get(a, 0.0) for a in assets}, dtype=float)
    return float((new - prev).abs().sum())


def _apply_turnover_limit(
    prev_weights: dict[str, float],
    target_weights: dict[str, float],
    assets: list[str],
    turnover_limit: float | None,
) -> tuple[dict[str, float], float]:
    """
    Optionally cap turnover by moving only partway from prev_weights to target_weights.

    If turnover_limit is None:
        return target_weights unchanged

    If turnover_limit is binding:
        use a convex combination:
            new = prev + alpha * (target - prev)
        where alpha is chosen so total turnover hits the cap exactly.

    This is simple, long-only safe, and thesis-friendly.
    """
    prev = pd.Series({a: prev_weights.get(a, 0.0) for a in assets}, dtype=float)
    target = pd.Series({a: target_weights.get(a, 0.0) for a in assets}, dtype=float)

    raw_turnover = float((target - prev).abs().sum())

    if turnover_limit is None or raw_turnover <= turnover_limit or raw_turnover == 0:
        return target.to_dict(), raw_turnover

    alpha = turnover_limit / raw_turnover
    new = prev + alpha * (target - prev)

    return new.to_dict(), float((new - prev).abs().sum())


def _decision_list_to_df(decisions: list) -> pd.DataFrame:
    """
    Convert list[TiltDecision] into a tabular decision log.
    """
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
    """
    Build a simple performance summary table.

    Metrics
    -------
    - CAGR
    - volatility
    - Sharpe (rf assumed 0 here because returns can already be excess or total,
      depending on your chosen realized return convention)
    - max drawdown
    - downside deviation
    - optional average turnover
    """
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