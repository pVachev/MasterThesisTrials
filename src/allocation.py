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




# ---------------------------------------------------------------------
# STAGE 3 — PREDICTIVE REGIME LOGIC
# ---------------------------------------------------------------------

def extract_filtered_probabilities(res: "ModelRunResult") -> pd.DataFrame:
    """
    Extract filtered regime probabilities from one fitted HMM result.

    Parameters
    ----------
    res : ModelRunResult
        Output of your current HMM pipeline.
        Expected:
        - res.pp.df_m exists
        - p_state0, p_state1, ... are present in res.pp.df_m
        - these p_state columns are already reordered by RegimePostProcessor.fit()

    Returns
    -------
    pd.DataFrame
        Date-indexed probability panel with columns equal to regime names.
        Each row is:
            pi_t = [P(S_t=0 | F_t), ..., P(S_t=K-1 | F_t)]

    Notes
    -----
    These are FILTERED probabilities from your currently fitted sample.
    They are acceptable for the prototype allocation layer as long as:
      - they are not smoothed
      - they are only used for next-period decisions
    """
    if res.pp.df_m is None:
        raise ValueError("res.pp.df_m is missing. Was RegimePostProcessor.fit() called?")

    df_m = res.pp.df_m.copy()
    prob_cols = [c for c in df_m.columns if c.startswith("p_state")]

    if not prob_cols:
        raise ValueError("No p_state* columns found in res.pp.df_m.")

    prob_cols = sorted(prob_cols, key=lambda c: int(c.replace("p_state", "")))

    if len(prob_cols) != len(res.pp.regime_names):
        raise ValueError(
            f"Mismatch between probability columns ({len(prob_cols)}) "
            f"and regime names ({len(res.pp.regime_names)})."
        )

    probs = df_m[prob_cols].copy()
    probs.columns = res.pp.regime_names

    # numerical hygiene
    probs = probs.div(probs.sum(axis=1), axis=0)

    return probs


def extract_reordered_transition_matrix(res: "ModelRunResult") -> pd.DataFrame:
    """
    Extract the HMM transition matrix reordered into the SAME regime order
    used by the postprocessed p_state columns.

    Why this matters
    ----------------
    Your RegimePostProcessor reorders:
      - the hard state labels
      - the p_state columns

    But model.transmat_ remains in the ORIGINAL HMM latent-state order.

    For predictive regime probabilities, filtered probabilities and transition
    matrix must refer to the SAME regime ordering. Therefore we use:
        res.pp.order_old

    Parameters
    ----------
    res : ModelRunResult

    Returns
    -------
    pd.DataFrame
        Transition matrix P in postprocessed regime order, labeled with:
            res.pp.regime_names

    Interpretation
    --------------
    Entry (i, j) is:
        P(S_{t+1}=j | S_t=i)
    """
    if res.model is None:
        raise ValueError("res.model is missing.")
    if res.pp.order_old is None:
        raise ValueError("res.pp.order_old is missing. Was RegimePostProcessor.fit() called?")

    T = np.asarray(res.model.transmat_, dtype=float)
    order_old = res.pp.order_old

    # reorder rows and columns into postprocessed regime order
    T_reordered = T[np.ix_(order_old, order_old)]

    P = pd.DataFrame(
        T_reordered,
        index=res.pp.regime_names,
        columns=res.pp.regime_names,
    )

    # numerical hygiene
    P = P.div(P.sum(axis=1), axis=0)

    return P


def predictive_regime_probabilities(
    filtered_probs: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    steps_ahead: int = 1,
) -> pd.DataFrame:
    """
    Compute predictive regime probabilities from filtered probabilities and
    a transition matrix.

    Formula
    -------
    One-step:
        pi_{t+1|t} = pi_t P

    Multi-step:
        pi_{t+h|t} = pi_t P^h

    Parameters
    ----------
    filtered_probs : pd.DataFrame
        Rows are dates t, columns are regimes.
        Each row is the filtered regime probability vector pi_t.

    transition_matrix : pd.DataFrame
        Transition matrix P with regime names as both index and columns.
        Must be aligned with filtered_probs columns.

    steps_ahead : int
        Number of steps ahead to project.
        - 1 is the main monthly prototype case
        - >1 is optional for later extensions

    Returns
    -------
    pd.DataFrame
        Predictive probability panel indexed by the same dates as filtered_probs.

        The row at date t contains:
            pi_{t+steps_ahead | t}

    Important interpretation
    ------------------------
    The row indexed by date t is information AVAILABLE at t and should be used
    for decisions applied to the NEXT period(s), not the same period.
    """
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be at least 1.")

    if list(filtered_probs.columns) != list(transition_matrix.columns):
        raise ValueError(
            "filtered_probs columns and transition_matrix columns must match exactly."
        )
    if list(transition_matrix.index) != list(transition_matrix.columns):
        raise ValueError("transition_matrix must have matching index and columns.")

    P = transition_matrix.to_numpy(dtype=float)
    P_h = np.linalg.matrix_power(P, steps_ahead)

    pi = filtered_probs.to_numpy(dtype=float)
    pred = pi @ P_h

    pred_df = pd.DataFrame(
        pred,
        index=filtered_probs.index,
        columns=filtered_probs.columns,
    )

    # numerical hygiene
    pred_df = pred_df.div(pred_df.sum(axis=1), axis=0)

    return pred_df


def build_predictive_probability_panel(
    res: "ModelRunResult",
    steps_ahead: int = 1,
) -> pd.DataFrame:
    """
    Convenience wrapper that goes directly from one ModelRunResult to
    predictive regime probabilities.

    Pipeline
    --------
    1) extract filtered probabilities from res.pp.df_m
    2) extract reordered transition matrix from res.model.transmat_
    3) compute predictive probabilities

    Parameters
    ----------
    res : ModelRunResult
        One fitted HMM result from your existing regime engine.

    steps_ahead : int
        Number of steps ahead to project.

    Returns
    -------
    pd.DataFrame
        Predictive regime probability panel.

    Example
    -------
    pred_probs = build_predictive_probability_panel(res_core, steps_ahead=1)

    Then at date t:
        pred_probs.loc[t]
    is your pi_{t+1|t} vector for next-period decisions.
    """
    filtered_probs = extract_filtered_probabilities(res)
    P = extract_reordered_transition_matrix(res)
    pred_probs = predictive_regime_probabilities(
        filtered_probs=filtered_probs,
        transition_matrix=P,
        steps_ahead=steps_ahead,
    )
    return pred_probs


# ---------------------------------------------------------------------
# STAGE 4 — SATELLITE CONDITIONAL MOMENT ENGINE
# ---------------------------------------------------------------------

def build_total_portfolio_weights(
    alloc_cfg,
    satellite_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Combine fixed core weights with a satellite sleeve.

    Logic
    -----
    Let the core weights sum to 1.0 in alloc_cfg.fixed_core_weights.

    If the total satellite sleeve weight is s, then:
        - the core is scaled down by (1 - s)
        - satellite weights are inserted on top
        - total portfolio weights sum to 1.0

    This keeps the implementation simple, long-only, and no-leverage.

    Parameters
    ----------
    alloc_cfg : AllocationConfig
        Allocation configuration.

    satellite_weights : dict[str, float] | None
        Example:
            {"XAU": 0.10}
        or:
            {"XAU": 0.10, "OIL": 0.10}

    Returns
    -------
    dict[str, float]
        Full portfolio weights across core + satellites.

    Notes
    -----
    This function enforces:
    - long-only
    - no leverage
    - max total satellite weight
    - top_n_satellites limit
    """
    satellite_weights = satellite_weights or {}

    if alloc_cfg.top_n_satellites == 1 and len([w for w in satellite_weights.values() if w > 0]) > 1:
        raise ValueError("top_n_satellites=1 but more than one satellite has positive weight.")

    if alloc_cfg.top_n_satellites == 2 and len([w for w in satellite_weights.values() if w > 0]) > 2:
        raise ValueError("top_n_satellites=2 but more than two satellites have positive weight.")

    if alloc_cfg.long_only:
        neg = {k: v for k, v in satellite_weights.items() if v < 0}
        if neg:
            raise ValueError(f"Negative satellite weights are not allowed: {neg}")

    sat_total = float(sum(satellite_weights.values()))
    if sat_total > alloc_cfg.max_satellite_weight + 1e-12:
        raise ValueError(
            f"Total satellite weight {sat_total:.4f} exceeds max_satellite_weight={alloc_cfg.max_satellite_weight:.4f}"
        )

    core_scale = 1.0 - sat_total
    if core_scale < -1e-12:
        raise ValueError("Satellite sleeve exceeds 100% of portfolio.")

    total_weights = {k: core_scale * v for k, v in alloc_cfg.fixed_core_weights.items()}

    for k, v in satellite_weights.items():
        total_weights[k] = total_weights.get(k, 0.0) + v

    total_sum = sum(total_weights.values())
    if alloc_cfg.no_leverage and not np.isclose(total_sum, 1.0):
        raise ValueError(f"Total portfolio weights must sum to 1.0, got {total_sum:.6f}")

    return total_weights


def _extract_asset_return_panel(
    allocation_df: pd.DataFrame,
    portfolio_assets: list[str],
    return_prefix: str = "ExcessLog",
) -> pd.DataFrame:
    """
    Extract the return panel needed for the allocation universe.

    Expected columns
    ----------------
    For asset ticker T, we expect:
        f"{return_prefix}{T}"

    Example:
        ExcessLog^SP500TR
        ExcessLogLT09TRUU
        ExcessLogXAU

    Returns
    -------
    pd.DataFrame
        Date-indexed return panel with columns renamed to the raw asset tickers.
    """
    cols = [f"{return_prefix}{a}" for a in portfolio_assets]
    missing = [c for c in cols if c not in allocation_df.columns]
    if missing:
        raise KeyError(f"allocation_df is missing required return columns: {missing}")

    out = allocation_df[cols].copy()
    out.columns = portfolio_assets
    return out


def _compute_raw_and_central_moments(r: pd.Series) -> dict[str, float]:
    """
    Compute raw moments and derived central moments for one return series.

    Returns
    -------
    dict with:
        n_obs
        m1, m2, m3, m4
        mean
        variance
        skewness
        kurtosis

    Notes
    -----
    - We compute raw moments because regime aggregation across probabilities
      is easiest and cleanest in raw-moment space.
    - Kurtosis returned here is regular kurtosis, not excess kurtosis.
    """
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = len(r)

    if n == 0:
        return {
            "n_obs": 0,
            "m1": np.nan, "m2": np.nan, "m3": np.nan, "m4": np.nan,
            "mean": np.nan, "variance": np.nan, "skewness": np.nan, "kurtosis": np.nan,
        }

    x = r.to_numpy(dtype=float)
    m1 = np.mean(x)
    m2 = np.mean(x**2)
    m3 = np.mean(x**3)
    m4 = np.mean(x**4)

    var = m2 - m1**2
    var = max(var, 0.0)

    if var <= 1e-16:
        skew = 0.0
        kurt = 3.0
    else:
        mu3 = m3 - 3*m1*m2 + 2*(m1**3)
        mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
        skew = mu3 / (var ** 1.5)
        kurt = mu4 / (var ** 2)

    return {
        "n_obs": n,
        "m1": float(m1),
        "m2": float(m2),
        "m3": float(m3),
        "m4": float(m4),
        "mean": float(m1),
        "variance": float(var),
        "skewness": float(skew),
        "kurtosis": float(kurt),
    }


def estimate_state_conditional_portfolio_moments(
    res: "ModelRunResult",
    allocation_df: pd.DataFrame,
    total_weights: dict[str, float],
    alloc_cfg,
    return_prefix: str = "ExcessLog",
) -> pd.DataFrame:
    """
    Estimate state-conditional moments of the resulting portfolio under one candidate tilt.

    Parameters
    ----------
    res : ModelRunResult
        Core regime model output.
        We use:
        - res.pp.df_m["state"] for hard regime membership
        - res.pp.regime_names for labeling

    allocation_df : pd.DataFrame
        Tradable return panel for all assets that may appear in the portfolio.

    total_weights : dict[str, float]
        Full portfolio weights across core + satellites.

    alloc_cfg : AllocationConfig
        Used for:
        - min_regime_obs
        - shrinkage_intensity

    return_prefix : str
        Usually "ExcessLog" for your current thesis setup.

    Returns
    -------
    pd.DataFrame
        One row per regime with:
            n_obs
            shrinkage_alpha
            m1, m2, m3, m4
            mean
            variance
            skewness
            kurtosis

    Method
    ------
    1) Build the portfolio return series:
           r_p,t = sum_i w_i r_i,t

    2) Split that portfolio return series by hard regime from res.pp.df_m["state"]

    3) Compute state-conditional moments

    4) Apply simple shrinkage / fallback toward unconditional moments when regime
       sample sizes are small

    Why this is thesis-friendly
    ---------------------------
    We avoid full co-skewness / co-kurtosis tensors and directly estimate the
    moments of the resulting candidate portfolio. This is robust and easy to explain.
    """
    if res.pp.df_m is None:
        raise ValueError("res.pp.df_m is missing.")
    if "state" not in res.pp.df_m.columns:
        raise KeyError("res.pp.df_m must contain 'state'.")

    portfolio_assets = list(total_weights.keys())
    asset_panel = _extract_asset_return_panel(
        allocation_df=allocation_df,
        portfolio_assets=portfolio_assets,
        return_prefix=return_prefix,
    )

    # align dates with the regime engine
    common_index = asset_panel.index.intersection(res.pp.df_m.index)
    asset_panel = asset_panel.loc[common_index].copy()
    states = res.pp.df_m.loc[common_index, "state"].copy()

    # portfolio return series
    w = pd.Series(total_weights, dtype=float)
    portfolio_r = (asset_panel * w.values).sum(axis=1)

    # unconditional moments as fallback / shrinkage anchor
    uncond = _compute_raw_and_central_moments(portfolio_r)

    rows = []
    for state_id in range(res.pp.n_states):
        r_s = portfolio_r.loc[states == state_id]
        raw = _compute_raw_and_central_moments(r_s)
        n = raw["n_obs"]

        # shrinkage alpha
        alpha = 1.0

        # minimum sample safeguard
        if n < alloc_cfg.min_regime_obs:
            alpha = min(alpha, n / alloc_cfg.min_regime_obs if alloc_cfg.min_regime_obs > 0 else 1.0)

        # optional additional shrinkage
        if alloc_cfg.shrinkage_intensity > 0:
            alpha = min(alpha, n / (n + alloc_cfg.shrinkage_intensity) if n > 0 else 0.0)

        # shrink raw moments toward unconditional raw moments
        shrunk = {}
        for k in ["m1", "m2", "m3", "m4"]:
            if np.isnan(raw[k]):
                shrunk[k] = uncond[k]
            else:
                shrunk[k] = alpha * raw[k] + (1.0 - alpha) * uncond[k]

        m1, m2, m3, m4 = shrunk["m1"], shrunk["m2"], shrunk["m3"], shrunk["m4"]

        var = m2 - m1**2
        var = max(var, 0.0)

        if var <= 1e-16:
            skew = 0.0
            kurt = 3.0
        else:
            mu3 = m3 - 3*m1*m2 + 2*(m1**3)
            mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
            skew = mu3 / (var ** 1.5)
            kurt = mu4 / (var ** 2)

        rows.append({
            "regime": res.pp.regime_names[state_id],
            "n_obs": n,
            "shrinkage_alpha": float(alpha),
            "m1": float(m1),
            "m2": float(m2),
            "m3": float(m3),
            "m4": float(m4),
            "mean": float(m1),
            "variance": float(var),
            "skewness": float(skew),
            "kurtosis": float(kurt),
        })

    return pd.DataFrame(rows)


def aggregate_predictive_portfolio_moments(
    state_moment_table: pd.DataFrame,
    predictive_probabilities_row: pd.Series,
) -> dict[str, float]:
    """
    Aggregate state-conditional portfolio moments using predictive regime probabilities.

    Parameters
    ----------
    state_moment_table : pd.DataFrame
        Output of estimate_state_conditional_portfolio_moments(...)

    predictive_probabilities_row : pd.Series
        One row of predictive regime probabilities:
            pi_{t+1|t}

        Index must be regime names:
            Regime 0, Regime 1, ...

    Returns
    -------
    dict with:
        expected_return
        variance
        skewness
        kurtosis
        plus raw moments M1..M4

    Formula
    -------
    We first aggregate raw moments:
        M_k = sum_s p_s m_{k,s}

    Then convert to central moments.

    This is the right practical way to combine regime-specific portfolio moments
    without using co-skewness / co-kurtosis tensors.
    """
    table = state_moment_table.set_index("regime").copy()

    if list(table.index) != list(predictive_probabilities_row.index):
        raise ValueError("Regime order mismatch between state_moment_table and predictive probabilities.")

    p = predictive_probabilities_row.to_numpy(dtype=float)

    M1 = float(np.dot(p, table["m1"].to_numpy(dtype=float)))
    M2 = float(np.dot(p, table["m2"].to_numpy(dtype=float)))
    M3 = float(np.dot(p, table["m3"].to_numpy(dtype=float)))
    M4 = float(np.dot(p, table["m4"].to_numpy(dtype=float)))

    var = M2 - M1**2
    var = max(var, 0.0)

    if var <= 1e-16:
        skew = 0.0
        kurt = 3.0
    else:
        mu3 = M3 - 3*M1*M2 + 2*(M1**3)
        mu4 = M4 - 4*M1*M3 + 6*(M1**2)*M2 - 3*(M1**4)
        skew = mu3 / (var ** 1.5)
        kurt = mu4 / (var ** 2)

    return {
        "expected_return": float(M1),
        "variance": float(var),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "M1": float(M1),
        "M2": float(M2),
        "M3": float(M3),
        "M4": float(M4),
    }


def evaluate_candidate_tilt_moments(
    res: "ModelRunResult",
    allocation_df: pd.DataFrame,
    alloc_cfg,
    predictive_probability_panel: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    satellite_weights: dict[str, float] | None = None,
    return_prefix: str = "ExcessLog",
) -> dict[str, object]:
    """
    Evaluate one candidate core-plus-satellite tilt at one rebalance date.

    Parameters
    ----------
    res : ModelRunResult
        Core regime model result.

    allocation_df : pd.DataFrame
        Tradable return panel containing the assets in the final portfolio.

    alloc_cfg : AllocationConfig
        Allocation configuration.

    predictive_probability_panel : pd.DataFrame
        Output of Stage 3:
            build_predictive_probability_panel(...)

    rebalance_date : pd.Timestamp
        Date t at which the decision is made.
        We use:
            pi_{t+1|t}
        from predictive_probability_panel.loc[rebalance_date]

    satellite_weights : dict[str, float] | None
        Candidate satellite sleeve weights.
        Example:
            {}            -> no-satellite baseline
            {"XAU": 0.10} -> 10% gold sleeve

    return_prefix : str
        Usually "ExcessLog"

    Returns
    -------
    dict with:
        total_weights
        predictive_probabilities
        state_moment_table
        aggregated_moments

    This function is the main Stage 4 workhorse and will be reused directly
    in Stage 5 for score comparison and tilt selection.
    """
    if rebalance_date not in predictive_probability_panel.index:
        raise KeyError(f"rebalance_date {rebalance_date} not found in predictive_probability_panel.")

    total_weights = build_total_portfolio_weights(
        alloc_cfg=alloc_cfg,
        satellite_weights=satellite_weights or {},
    )

    state_moment_table = estimate_state_conditional_portfolio_moments(
        res=res,
        allocation_df=allocation_df,
        total_weights=total_weights,
        alloc_cfg=alloc_cfg,
        return_prefix=return_prefix,
    )

    pred_row = predictive_probability_panel.loc[rebalance_date]
    aggregated = aggregate_predictive_portfolio_moments(
        state_moment_table=state_moment_table,
        predictive_probabilities_row=pred_row,
    )

    return {
        "rebalance_date": rebalance_date,
        "total_weights": total_weights,
        "predictive_probabilities": pred_row.to_dict(),
        "state_moment_table": state_moment_table,
        "aggregated_moments": aggregated,
    }