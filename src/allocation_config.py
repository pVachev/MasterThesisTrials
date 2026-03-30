from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.runner import ModelRunResult



# ---------------------------------------------------------------------
# 1) INVESTOR PREFERENCE CONFIG
# ---------------------------------------------------------------------

@dataclass
class InvestorPreferenceConfig:
    """
    Stores the utility / scoring preference of one investor type.

    Supported investor types
    ------------------------
    1) Mean-Variance:
        Score = mu_p - lambda * sigma_p^2

    2) Mean-Variance-Skewness:
        Score = mu_p - lambda * sigma_p^2 + gamma * skew_p

    3) Mean-Variance-Kurtosis:
        Score = mu_p - lambda * sigma_p^2 - delta * kurt_p

    Notes
    -----
    - Keep lambda, gamma, delta configurable so you can test investor sensitivity.
    - `name` is user-facing and useful for plots / Excel tabs.
    - `investor_type` determines which score formula is applied later.
    """

    name: str
    investor_type: Literal["MV", "MVS", "MVK"]

    # Risk aversion / preference parameters
    lambda_: float = 3.0
    gamma: float = 0.0
    delta: float = 0.0


# ---------------------------------------------------------------------
# 2) SATELLITE SPEC
# ---------------------------------------------------------------------

@dataclass
class SatelliteSpec:
    """
    Defines one satellite asset candidate.

    Examples
    --------
    Gold:
        ticker="XAU", label="Gold"

    Oil:
        ticker="BCOMCOT", label="Brent"

    FX:
        ticker="EURUSD", label="EUR/USD"

    Fields
    ------
    ticker:
        Exact ticker name used in your return dataframe.

    label:
        Human-readable name for plots and logs.

    allowed_weights:
        Discrete set of candidate sleeve weights to test.
        Example: [0.0, 0.05, 0.10, 0.15, 0.20]

    group:
        Optional classification for later reporting.
        Example: "commodity", "fx", "equity_sector"
    """

    ticker: str
    label: str
    allowed_weights: list[float]
    group: str | None = None


# ---------------------------------------------------------------------
# 3) ALLOCATION CONFIG
# ---------------------------------------------------------------------

@dataclass
class AllocationConfig:
    """
    Global configuration for the regime-aware core-plus-satellite allocation layer.

    Philosophy
    ----------
    Keep the first version simple and thesis-friendly:
    - fixed core allocation
    - discrete satellite sleeve choices
    - long-only
    - no leverage
    - optional transaction costs
    - monthly rebalancing initially

    Main idea
    ---------
    At each rebalance date:
    1) use filtered probabilities + transition matrix
    2) compute one-step-ahead predictive probabilities
    3) evaluate candidate satellite tilts
    4) compare against no-satellite baseline
    5) pick the best conditional score improvement

    Fields
    ------
    rebalance_frequency:
        "ME" for monthly end, later extensible to "W-FRI"

    top_n_satellites:
        Number of satellites allowed in the sleeve:
        - 1 = top one only
        - 2 = top two allowed

    max_satellite_weight:
        Maximum total sleeve allocation to satellites.

    fixed_core_weights:
        Core portfolio weights before adding satellites.
        Example:
            {"^SP500TR": 0.60, "LT09TRUU": 0.40}

        These should sum to 1.0 before the satellite sleeve adjustment logic.

    transaction_cost_bps:
        Optional linear transaction cost in basis points.

    turnover_limit:
        Optional cap on turnover per rebalance period.

    min_regime_obs:
        Minimum number of observations required inside a regime before trusting
        raw higher-moment estimates. Later logic can shrink / fallback if not met.

    shrinkage_intensity:
        A placeholder parameter for later moment shrinkage.
        0.0 means no shrinkage.
        Higher values pull regime moments toward unconditional moments.

    score_improvement_floor:
        Minimum score gain required to activate a satellite tilt.
        This helps avoid tiny noisy tilts.

    export_file:
        Separate Excel output for allocation results.
    """

    rebalance_frequency: str = "ME"
    top_n_satellites: int = 1
    max_satellite_weight: float = 0.20

    fixed_core_weights: dict[str, float] = field(default_factory=dict)

    long_only: bool = True
    no_leverage: bool = True

    transaction_cost_bps: float = 0.0
    turnover_limit: float | None = None

    min_regime_obs: int = 24
    shrinkage_intensity: float = 0.0

    score_improvement_floor: float = 0.0

    export_file: str = "allocation_results.xlsx"

    def validate(self) -> None:
        """
        Basic guardrails so bad configs fail early.
        """
        if not self.fixed_core_weights:
            raise ValueError("fixed_core_weights cannot be empty.")

        total = sum(self.fixed_core_weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"fixed_core_weights must sum to 1.0, got {total:.6f}"
            )

        if self.top_n_satellites not in (1, 2):
            raise ValueError("top_n_satellites must be 1 or 2.")

        if not (0.0 <= self.max_satellite_weight <= 1.0):
            raise ValueError("max_satellite_weight must be between 0 and 1.")

        if self.turnover_limit is not None and self.turnover_limit < 0:
            raise ValueError("turnover_limit must be non-negative or None.")

        if self.min_regime_obs < 1:
            raise ValueError("min_regime_obs must be at least 1.")


# ---------------------------------------------------------------------
# 4) TILT DECISION
# ---------------------------------------------------------------------

@dataclass
class TiltDecision:
    """
    Stores the output of the tilt-selection engine at one rebalance date.

    This is the key 'decision log' object.

    Fields
    ------
    rebalance_date:
        Date at which the tilt decision is made.

    investor_name:
        Which investor preference generated the decision.

    predictive_probabilities:
        One-step-ahead predictive regime probabilities used for the decision.

    baseline_score:
        Score of the no-satellite baseline portfolio.

    selected_satellites:
        List of chosen satellite tickers.
        Empty list means no satellite tilt was taken.

    selected_weights:
        Final selected sleeve weights for satellites only.
        Example:
            {"XAU": 0.15}

    total_portfolio_weights:
        Final total portfolio weights after combining core + satellite sleeve.

    expected_return:
        Predictive aggregated portfolio mean used in the score.

    variance:
        Predictive aggregated portfolio variance.

    skewness:
        Predictive aggregated portfolio skewness.

    kurtosis:
        Predictive aggregated portfolio kurtosis.

    final_score:
        Score of the selected portfolio.

    score_improvement:
        final_score - baseline_score

    metadata:
        Optional free-form diagnostics for debugging / export.
    """

    rebalance_date: pd.Timestamp
    investor_name: str

    predictive_probabilities: dict[str, float]

    baseline_score: float

    selected_satellites: list[str]
    selected_weights: dict[str, float]
    total_portfolio_weights: dict[str, float]

    expected_return: float
    variance: float
    skewness: float
    kurtosis: float

    final_score: float
    score_improvement: float

    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------
# 5) BACKTEST RESULT
# ---------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Stores the full output of one allocation backtest.

    This object should be rich enough to support:
    - performance tables
    - Excel export
    - plotting
    - debugging the strategy logic

    Fields
    ------
    model_label:
        Name of the regime engine used.

    investor_name:
        Investor profile used in the tilt rule.

    decisions:
        One TiltDecision per rebalance date.

    decision_log:
        Tabular monthly decision history.

    weights:
        Time series of final portfolio weights.

    asset_returns:
        Time series of realized asset returns used in the allocation layer.

    strategy_returns:
        Realized strategy return series.

    benchmark_returns:
        Optional benchmark return series.

    cumulative_wealth:
        Cumulative wealth index for the strategy.

    benchmark_wealth:
        Optional cumulative wealth for the benchmark.

    drawdown:
        Strategy drawdown series.

    benchmark_drawdown:
        Optional benchmark drawdown series.

    performance_summary:
        Final performance table for export and reporting.

    candidate_scores:
        Optional full candidate-score table, useful for diagnostics and Excel export.
    """

    model_label: str
    investor_name: str

    decisions: list[TiltDecision]

    decision_log: pd.DataFrame
    weights: pd.DataFrame
    asset_returns: pd.DataFrame

    strategy_returns: pd.Series
    benchmark_returns: pd.Series | None = None

    cumulative_wealth: pd.Series | None = None
    benchmark_wealth: pd.Series | None = None

    drawdown: pd.Series | None = None
    benchmark_drawdown: pd.Series | None = None

    performance_summary: pd.DataFrame | None = None
    candidate_scores: pd.DataFrame | None = None

