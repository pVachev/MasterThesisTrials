import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def _build_selected_satellite_series(backtest_res) -> pd.Series:
    s = backtest_res.decision_log.copy()
    s["rebalance_date"] = pd.to_datetime(s["rebalance_date"])
    out = s.set_index("rebalance_date")["selected_satellites"].astype(str)
    return out


def plot_allocation_dashboard(backtest_res, figsize=(18, 16)):
    """
    5-panel dashboard:
      1) cumulative wealth
      2) drawdown
      3) weights through time
      4) predictive regime probabilities
      5) selected satellite timeline
    """
    pred_probs = _extract_predictive_probabilities_from_decisions(backtest_res)
    selected_sat = _build_selected_satellite_series(backtest_res)

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=False)

    # 1) wealth
    axes[0].plot(backtest_res.cumulative_wealth.index, backtest_res.cumulative_wealth, label="Strategy")
    if backtest_res.benchmark_wealth is not None:
        axes[0].plot(backtest_res.benchmark_wealth.index, backtest_res.benchmark_wealth, label="Benchmark")
    axes[0].set_title(f"{backtest_res.model_label} | {backtest_res.investor_name} — cumulative wealth")
    axes[0].legend()

    # 2) drawdown
    axes[1].plot(backtest_res.drawdown.index, backtest_res.drawdown, label="Strategy DD")
    if backtest_res.benchmark_drawdown is not None:
        axes[1].plot(backtest_res.benchmark_drawdown.index, backtest_res.benchmark_drawdown, label="Benchmark DD")
    axes[1].set_title("Drawdown")
    axes[1].legend()

    # 3) weights
    if backtest_res.weights is not None and not backtest_res.weights.empty:
        w = backtest_res.weights.fillna(0.0)
        axes[2].stackplot(w.index, w.T, labels=w.columns)
        axes[2].set_title("Portfolio weights")
        axes[2].legend(loc="upper left", ncol=min(len(w.columns), 4))

    # 4) predictive probabilities
    if not pred_probs.empty:
        axes[3].stackplot(pred_probs.index, pred_probs.T, labels=pred_probs.columns)
        axes[3].set_title("Predictive regime probabilities")
        axes[3].legend(loc="upper left", ncol=min(len(pred_probs.columns), 4))

    # 5) selected satellite timeline
    if not selected_sat.empty:
        cats = pd.Categorical(selected_sat)
        y = cats.codes
        axes[4].step(selected_sat.index, y, where="post")
        axes[4].set_title("Selected satellite")
        axes[4].set_yticks(range(len(cats.categories)))
        axes[4].set_yticklabels(list(cats.categories))

    plt.tight_layout()
    plt.show()


def plot_distribution_comparison(backtest_res, bins=40, figsize=(10, 6)):
    """
    Compare strategy and benchmark return distributions.
    """
    plt.figure(figsize=figsize)
    plt.hist(backtest_res.strategy_returns.dropna(), bins=bins, alpha=0.6, density=True, label="Strategy")

    if backtest_res.benchmark_returns is not None:
        plt.hist(backtest_res.benchmark_returns.dropna(), bins=bins, alpha=0.6, density=True, label="Benchmark")

    plt.title(f"{backtest_res.model_label} | {backtest_res.investor_name} — return distribution")
    plt.legend()
    plt.show()