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
 
    All three types are nested truncations of the same Taylor-expanded
    expected utility function (Jondeau & Rockinger, 2006):
 
        E[U(R)] ≈ μ_p - λ·σ²_p + (1/6)·γ·skew_p - (1/24)·δ·(kurt_p - 3)
 
    Supported investor types
    ------------------------
    MV  — Mean-Variance (2nd-order truncation):
        Score = μ_p - λ · σ²_p
 
    MVS — Mean-Variance-Skewness (3rd-order truncation):
        Score = μ_p - λ · σ²_p + (1/6) · γ · skew_p
 
    MVK — Mean-Variance-Kurtosis (4th-order truncation):
        Score = μ_p - λ · σ²_p + (1/6) · γ · skew_p - (1/24) · δ · (kurt_p - 3)
 
    Parameter guidance (monthly log excess returns, 60/40 core)
    -----------------------------------------------------------
    λ (lambda_):
        Risk aversion coefficient on variance.
        Recommended range: 1–5. Default 3.0 is well-calibrated for this
        dataset (variance term ≈ 114% of mean return at λ=3).
 
    γ (gamma):
        Skewness preference coefficient. Enters with factorial scaling
        (1/6) so it is directly comparable in magnitude to λ.
        Recommended range: 0.0006–0.0015.
        At γ=0.001 the skewness term is approximately 10–20% of the
        variance term — a reasonable secondary preference.
        Note: larger values make the investor increasingly a skewness
        maximiser, overriding the variance signal.
 
    δ (delta):
        Excess kurtosis aversion coefficient. Enters with factorial
        scaling (1/24). Recommended range: 0.0007–0.0016.
        Note: applied to (kurt - 3) so a normal portfolio receives
        zero penalty. Only fat-tail risk above the normal baseline
        is penalised.
 
    Sensitivity analysis
    --------------------
    Run three calibrations for robustness:
        Conservative : γ=0.0006, δ=0.0007   (≈10% of variance term)
        Moderate     : γ=0.0010, δ=0.0012   (≈17% of variance term)
        Aggressive   : γ=0.0015, δ=0.0016   (≈25% of variance term)
    Combined with λ ∈ {1, 3, 5} for a full 3×3 sensitivity grid.
 
    References
    ----------
    Jondeau, E. & Rockinger, M. (2006). Optimal Portfolio Allocation Under
        Higher Moments. European Financial Management, 12(1), 29–55.
 
    Guidolin, M. & Timmermann, A. (2008). International Asset Allocation
        under Regime Switching, Skew and Kurtosis Preferences.
        Review of Financial Studies, 21(2), 889–935.
    """
 
    name: str
    investor_type: Literal["MV", "MVS", "MVK"]
 
    # Risk aversion / preference parameters — JR (2006) calibrated
    lambda_: float = 3.0
    gamma: float   = 0.001   # was 0.01 — corrected for (1/6) factorial scaling
    delta: float   = 0.001   # was 0.01 — corrected for (1/24) factorial scaling


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
 
    # Regime conviction scaling style.
    # "cyclical"  — weight scaled by (1 - p_bear): full in bull, zero in bear.
    # "defensive" — weight scaled by p_bear:        full in bear, zero in bull.
    # This field is used by compute_regime_conviction_weights() in
    # allocation_scoring.py and does NOT affect the scoring / candidate
    # selection step — only the final applied weight.
    style: str = "cyclical"   # "cyclical" | "defensive"

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
 
    # Equity-only displacement mode
    # ---------------------------------------------------------------
    # When True, satellites displace ONLY the equity ticker (e.g. SP500),
    # leaving bond weights fixed. This isolates sector rotation within
    # the equity sleeve, keeping the bond allocation constant.
    #
    # Example with equity_ticker="^SP500TR", fixed_core={SP500:0.6, Bond:0.4}
    # and satellite={XLE:0.15}:
    #   Default (False): SP500=0.51, Bond=0.34, XLE=0.15  (both scaled down)
    #   Equity-only (True): SP500=0.45, Bond=0.40, XLE=0.15  (bond unchanged)
    equity_only_displacement: bool = False
    equity_ticker: str = "^SP500TR"
 
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
 
        if self.equity_only_displacement:
            if self.equity_ticker not in self.fixed_core_weights:
                raise ValueError(
                    f"equity_ticker '{self.equity_ticker}' not found in fixed_core_weights."
                )
            equity_w = self.fixed_core_weights[self.equity_ticker]
            if self.max_satellite_weight > equity_w + 1e-9:
                raise ValueError(
                    f"max_satellite_weight={self.max_satellite_weight:.2f} exceeds "
                    f"equity_ticker weight={equity_w:.2f}. Satellites cannot exceed "
                    f"the equity sleeve they are displacing."
                )
 
 
# ---------------------------------------------------------------------
# 3b) CASH SLEEVE CONFIG
# ---------------------------------------------------------------------

@dataclass
class CashSleeveConfig:
    """
    Post-scoring risk-management layer that moves portfolio weight to
    cash (the risk-free asset) when bear-regime conviction is high.

    Activation logic
    ----------------
    At each rebalance date, after the scoring engine has selected
    satellites and conviction scaling has been applied:

        if p_bear > activation_threshold:
            cash_weight = max_cash_weight
                          × (p_bear - activation_threshold)
                          / (1.0 - activation_threshold)
        else:
            cash_weight = 0.0

    All other portfolio weights (core + satellites) are scaled down
    proportionally:
        w_i_final = w_i × (1 - cash_weight)

    This ensures weights always sum to 1.0 and no leverage is introduced.

    When p_bear is below the threshold, the cash sleeve is inactive and
    the portfolio behaves exactly as before.

    Parameters
    ----------
    enabled : bool
        Master switch. If False, the cash sleeve is skipped entirely
        regardless of regime probabilities.

    activation_threshold : float
        Minimum p_bear required to trigger the cash sleeve.
        Recommended range: 0.50–0.65.
        Below 0.5 the model is not even majority-bear.

    max_cash_weight : float
        Maximum cash allocation when p_bear = 1.0.
        Recommended range: 0.15–0.30.

    rf_ticker : str
        Ticker name of the risk-free asset in the allocation dataframe.
        Must match the rf_col used in diff_data().

    Theoretical justification
    -------------------------
    Guidolin & Timmermann (2008) show that regime-aware investors
    optimally increase cash-equivalent holdings when the probability
    of transitioning into a high-volatility regime rises. The cash
    sleeve implements this as a linear function of bear probability,
    which is the simplest monotonic mapping consistent with their
    framework.
    """
    enabled: bool = True
    activation_threshold: float = 0.55
    max_cash_weight: float = 0.25
    rf_ticker: str = "RF"

    def compute_cash_weight(self, p_bear: float) -> float:
        """Return the cash allocation given the bear-regime probability."""
        if not self.enabled:
            return 0.0
        if p_bear <= self.activation_threshold:
            return 0.0
        return self.max_cash_weight * (
            (p_bear - self.activation_threshold)
            / (1.0 - self.activation_threshold)
        )


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


@dataclass
class TrainTestConfig:
    """
    Fixed-parameter train/test configuration for the honest A1 allocation pipeline.

    Philosophy
    ----------
    A1 = fit once on train, freeze HMM parameters and candidate moment estimates,
    then filter sequentially through test and make next-period decisions.

    Notes
    -----
    For satellite strategies, the training window must include the satellite
    universe. So if sector ETFs begin in late 1998, test cannot honestly begin
    before a meaningful post-1998 training window has been accumulated.
    """
    train_start: str | None = None
    train_end: str = ""
    test_start: str = ""
    test_end: str | None = None

    min_train_observations: int = 60
    freeze_hmm: bool = True
    freeze_candidate_moments: bool = True

    def validate(self, index: pd.Index | None = None) -> None:
        if not self.train_end:
            raise ValueError("train_end must be provided.")
        if not self.test_start:
            raise ValueError("test_start must be provided.")

        train_end_ts = pd.to_datetime(self.train_end)
        test_start_ts = pd.to_datetime(self.test_start)

        if not train_end_ts < test_start_ts:
            raise ValueError("train_end must be strictly earlier than test_start.")

        if self.test_end is not None:
            test_end_ts = pd.to_datetime(self.test_end)
            if not test_start_ts <= test_end_ts:
                raise ValueError("test_end must be on or after test_start.")

        if self.min_train_observations < 12:
            raise ValueError("min_train_observations should be at least 12 for monthly work.")

        if index is not None and len(index) > 0:
            idx = pd.Index(index).sort_values()
            lo = idx.min()
            hi = idx.max()

            train_start_ts = pd.to_datetime(self.train_start) if self.train_start is not None else lo
            if train_start_ts < lo:
                raise ValueError(f"train_start {train_start_ts} is earlier than data start {lo}.")
            if train_end_ts > hi:
                raise ValueError(f"train_end {train_end_ts} is later than data end {hi}.")
            if test_start_ts > hi:
                raise ValueError(f"test_start {test_start_ts} is later than data end {hi}.")


@dataclass
class FrozenHMMState:
    """
    Stores the train-only fitted HMM state for A1.

    This object is the bridge between:
    - your existing HMM training code
    - the honest out-of-sample allocation engine
    """
    model_label: str
    model_code: str
    n_states: int
    regime_names: list[str]
    order_old: list[int]
    covariance_type: str

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray

    train_x: pd.DataFrame
    test_x: pd.DataFrame

    train_df_m: pd.DataFrame
    train_out: pd.DataFrame

    filtered_train: pd.DataFrame
    filtered_test: pd.DataFrame | None = None