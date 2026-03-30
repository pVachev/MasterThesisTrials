import pandas as pd
import numpy as np


def _config_to_df(alloc_cfg, investor_cfg, satellite_specs, res_core) -> pd.DataFrame:
    rows = []

    rows.append({"section": "core_model", "key": "model_label", "value": res_core.spec.label})
    rows.append({"section": "core_model", "key": "model_code", "value": res_core.spec.code})

    for k, v in alloc_cfg.__dict__.items():
        rows.append({"section": "allocation_config", "key": k, "value": str(v)})

    for k, v in investor_cfg.__dict__.items():
        rows.append({"section": "investor_config", "key": k, "value": str(v)})

    for i, sat in enumerate(satellite_specs):
        for k, v in sat.__dict__.items():
            rows.append({"section": f"satellite_{i}", "key": k, "value": str(v)})

    return pd.DataFrame(rows)


def _build_wealth_drawdown_table(backtest_res) -> pd.DataFrame:
    out = pd.DataFrame(index=backtest_res.strategy_returns.index)
    out["strategy_return"] = backtest_res.strategy_returns
    out["strategy_wealth"] = backtest_res.cumulative_wealth
    out["strategy_drawdown"] = backtest_res.drawdown

    if backtest_res.benchmark_returns is not None:
        out["benchmark_return"] = backtest_res.benchmark_returns
    if backtest_res.benchmark_wealth is not None:
        out["benchmark_wealth"] = backtest_res.benchmark_wealth
    if backtest_res.benchmark_drawdown is not None:
        out["benchmark_drawdown"] = backtest_res.benchmark_drawdown

    return out


def _extract_predictive_probabilities_from_decisions(backtest_res) -> pd.DataFrame:
    rows = []
    for d in backtest_res.decisions:
        row = {"rebalance_date": d.rebalance_date}
        row.update(d.predictive_probabilities)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    df = df.set_index("rebalance_date").sort_index()
    return df


def export_allocation_backtest_to_excel(
    backtest_res,
    alloc_cfg,
    investor_cfg,
    satellite_specs,
    res_core,
    output_file: str | None = None,
) -> None:
    """
    Export the allocation backtest output to a dedicated Excel workbook.

    Sheets
    ------
    Config
    Performance
    Decision_Log
    Weights
    Wealth_Drawdown
    Asset_Returns
    Predictive_Probabilities
    Candidate_Scores
    """
    output_file = output_file or alloc_cfg.export_file

    config_df = _config_to_df(
        alloc_cfg=alloc_cfg,
        investor_cfg=investor_cfg,
        satellite_specs=satellite_specs,
        res_core=res_core,
    )

    wealth_df = _build_wealth_drawdown_table(backtest_res)
    pred_probs_df = _extract_predictive_probabilities_from_decisions(backtest_res)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        config_df.to_excel(writer, sheet_name="Config", index=False)

        if backtest_res.performance_summary is not None:
            backtest_res.performance_summary.to_excel(writer, sheet_name="Performance", index=False)

        backtest_res.decision_log.to_excel(writer, sheet_name="Decision_Log", index=False)
        backtest_res.weights.to_excel(writer, sheet_name="Weights")
        wealth_df.to_excel(writer, sheet_name="Wealth_Drawdown")
        backtest_res.asset_returns.to_excel(writer, sheet_name="Asset_Returns")

        if not pred_probs_df.empty:
            pred_probs_df.to_excel(writer, sheet_name="Predictive_Probabilities")

        if backtest_res.candidate_scores is not None:
            backtest_res.candidate_scores.to_excel(writer, sheet_name="Candidate_Scores", index=False)

    print(f"Saved allocation backtest results to {output_file}")