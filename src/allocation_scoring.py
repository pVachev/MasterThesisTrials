from itertools import combinations
import pandas as pd

from src.allocation_config import TiltDecision
from src.allocation_moments import (
    build_total_portfolio_weights,
    estimate_state_conditional_portfolio_moments_from_sample,
    aggregate_predictive_portfolio_moments,
)

# ---------------------------------------------------------------------
# 1) INVESTOR SCORE FUNCTIONS
# ---------------------------------------------------------------------
def compute_investor_score(
    investor_cfg,
    expected_return: float,
    variance: float,
    skewness: float,
    kurtosis: float,
) -> float:
    """
    Compute the investor's certainty-equivalent score for one candidate portfolio,
    using a Taylor expansion of expected utility around the portfolio return
    distribution (Jondeau & Rockinger, 2006, European Financial Management).
 
    The Taylor expansion of E[U(R)] around the mean gives:
 
        E[U(R)] ≈ U(μ) + (1/2) U''(μ) σ²
                         + (1/6) U'''(μ) μ₃
                         + (1/24) U''''(μ) μ₄  + ...
 
    For an investor with constant absolute risk aversion (CARA), the
    derivatives map to the signed moment preferences as follows:
 
        U(μ)       = μ                          (return, 1st moment)
        (1/2)U''   = -λ                         (variance aversion)
        (1/6)U'''  = +(1/6) γ                   (skewness preference)
        (1/24)U'''' = -(1/24) δ                 (excess kurtosis aversion)
 
    This gives the three investor types:
 
    MV  (truncate at 2nd order):
        Score = μ_p - λ · σ²_p
 
    MVS (truncate at 3rd order):
        Score = μ_p - λ · σ²_p + (1/6) · γ · skew_p
 
    MVK (truncate at 4th order):
        Score = μ_p - λ · σ²_p + (1/6) · γ · skew_p - (1/24) · δ · (kurt_p - 3)
 
    Notes
    -----
    - The (1/6) and (1/24) factorial denominators are the key correction
      versus a naive linear weighting. Without them, γ and δ are 6× and
      24× too large relative to λ, causing the higher-moment terms to
      dominate the score entirely (verified empirically on this dataset).
 
    - kurtosis is entered as raw kurtosis (not excess). The subtraction
      of 3 in the MVK formula removes the normal baseline, so only the
      *excess* kurtosis — the fat-tail component — penalises the score.
      A normally distributed portfolio receives zero kurtosis penalty.
 
    - Parameters λ, γ, δ are all expressed in the same utility scale.
      Recommended calibrated values for this project (monthly log excess
      returns, 60/40 core):
          λ = 3.0   (well-calibrated: variance term ≈ 114% of mean return)
          γ = 0.001  (skewness term ≈ 10–25% of variance term)
          δ = 0.001  (excess kurtosis term ≈ 10–25% of variance term)
 
    References
    ----------
    Jondeau, E. & Rockinger, M. (2006). Optimal Portfolio Allocation Under
        Higher Moments. European Financial Management, 12(1), 29–55.
 
    Guidolin, M. & Timmermann, A. (2008). International Asset Allocation
        under Regime Switching, Skew and Kurtosis Preferences.
        Review of Financial Studies, 21(2), 889–935.
    """
    if investor_cfg.investor_type == "MV":
        return (
            expected_return
            - investor_cfg.lambda_ * variance
        )
 
    elif investor_cfg.investor_type == "MVS":
        return (
            expected_return
            - investor_cfg.lambda_ * variance
            + (1.0 / 6.0) * investor_cfg.gamma * skewness
        )
 
    elif investor_cfg.investor_type == "MVK":
        excess_kurt = kurtosis - 3.0
        return (
            expected_return
            - investor_cfg.lambda_ * variance
            + (1.0 / 6.0) * investor_cfg.gamma * skewness
            - (1.0 / 24.0) * investor_cfg.delta * excess_kurt
        )
 
    else:
        raise ValueError(f"Unsupported investor_type: {investor_cfg.investor_type}")


# ---------------------------------------------------------------------
# 2) CANDIDATE ENUMERATION
# ---------------------------------------------------------------------

def enumerate_candidate_tilts(satellite_specs: list, alloc_cfg) -> list[dict[str, float]]:
    candidates = [{}]

    for sat in satellite_specs:
        for w in sat.allowed_weights:
            if w <= 0:
                continue
            if w <= alloc_cfg.max_satellite_weight + 1e-12:
                candidates.append({sat.ticker: float(w)})

    if alloc_cfg.top_n_satellites >= 2:
        for sat_a, sat_b in combinations(satellite_specs, 2):
            for w_a in sat_a.allowed_weights:
                for w_b in sat_b.allowed_weights:
                    if w_a <= 0 or w_b <= 0:
                        continue
                    total = float(w_a + w_b)
                    if total <= alloc_cfg.max_satellite_weight + 1e-12:
                        candidates.append({sat_a.ticker: float(w_a), sat_b.ticker: float(w_b)})

    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    return unique_candidates


# ---------------------------------------------------------------------
# 3) CANDIDATE EVALUATION
# ---------------------------------------------------------------------

def evaluate_candidate_scores_at_date(
    res,
    allocation_df: pd.DataFrame,
    alloc_cfg,
    investor_cfg,
    satellite_specs: list,
    predictive_probability_panel: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    return_prefix: str = "ExcessLog",
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate all candidate satellite tilts at one rebalance date.

    Returns
    -------
    candidate_table : pd.DataFrame
        One row per candidate with:
        - candidate_id
        - selected_satellites
        - satellite_weights
        - total_portfolio_weights
        - expected_return
        - variance
        - skewness
        - kurtosis
        - score
        - baseline_score
        - score_improvement

    baseline_payload : dict
        Full Stage 4 evaluation object for the no-satellite baseline.
    """
    candidates = enumerate_candidate_tilts(satellite_specs, alloc_cfg)

    # Evaluate baseline first
    baseline_payload = evaluate_candidate_tilt_moments(
        res=res,
        allocation_df=allocation_df,
        alloc_cfg=alloc_cfg,
        predictive_probability_panel=predictive_probability_panel,
        rebalance_date=rebalance_date,
        satellite_weights={},
        return_prefix=return_prefix,
    )

    base_m = baseline_payload["aggregated_moments"]
    baseline_score = compute_investor_score(
        investor_cfg=investor_cfg,
        expected_return=base_m["expected_return"],
        variance=base_m["variance"],
        skewness=base_m["skewness"],
        kurtosis=base_m["kurtosis"],
    )

    rows = []

    for i, sat_weights in enumerate(candidates):
        payload = evaluate_candidate_tilt_moments(
            res=res,
            allocation_df=allocation_df,
            alloc_cfg=alloc_cfg,
            predictive_probability_panel=predictive_probability_panel,
            rebalance_date=rebalance_date,
            satellite_weights=sat_weights,
            return_prefix=return_prefix,
        )

        m = payload["aggregated_moments"]
        score = compute_investor_score(
            investor_cfg=investor_cfg,
            expected_return=m["expected_return"],
            variance=m["variance"],
            skewness=m["skewness"],
            kurtosis=m["kurtosis"],
        )

        score_improvement = score - baseline_score

        selected_satellites = list(sat_weights.keys())
        selected_satellites_str = ", ".join(selected_satellites) if selected_satellites else "NONE"

        rows.append({
            "rebalance_date": rebalance_date,
            "investor_name": investor_cfg.name,
            "candidate_id": i,
            "selected_satellites": selected_satellites_str,
            "satellite_weights": sat_weights,
            "total_portfolio_weights": payload["total_weights"],
            "expected_return": m["expected_return"],
            "variance": m["variance"],
            "skewness": m["skewness"],
            "kurtosis": m["kurtosis"],
            "score": score,
            "baseline_score": baseline_score,
            "score_improvement": score_improvement,
        })

    candidate_table = pd.DataFrame(rows)

    # sort best to worst
    candidate_table = candidate_table.sort_values(
        ["score_improvement", "score"],
        ascending=False,
    ).reset_index(drop=True)

    return candidate_table, baseline_payload


# ---------------------------------------------------------------------
# 4) BEST TILT SELECTION
# ---------------------------------------------------------------------

def select_best_tilt_at_date(
    res,
    allocation_df: pd.DataFrame,
    alloc_cfg,
    investor_cfg,
    satellite_specs: list,
    predictive_probability_panel: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    return_prefix: str = "ExcessLog",
) -> tuple[TiltDecision, pd.DataFrame]:
    """
    Select the best tilt at one rebalance date.

    Decision rule
    -------------
    1) Evaluate the no-satellite baseline and all satellite candidates
    2) Compute investor-specific score for each
    3) Compare score improvement relative to baseline
    4) Pick the best candidate
    5) If the best score improvement is below score_improvement_floor,
       keep the baseline (no satellite)

    Returns
    -------
    decision : TiltDecision
        The chosen tilt decision at the rebalance date.

    candidate_table : pd.DataFrame
        Full candidate score table for diagnostics / export.
    """
    candidate_table, baseline_payload = evaluate_candidate_scores_at_date(
        res=res,
        allocation_df=allocation_df,
        alloc_cfg=alloc_cfg,
        investor_cfg=investor_cfg,
        satellite_specs=satellite_specs,
        predictive_probability_panel=predictive_probability_panel,
        rebalance_date=rebalance_date,
        return_prefix=return_prefix,
    )

    best_row = candidate_table.iloc[0].copy()

    # if best improvement is too small, force baseline
    if best_row["score_improvement"] < alloc_cfg.score_improvement_floor:
        baseline_row = candidate_table.loc[
            candidate_table["selected_satellites"] == "NONE"
        ].iloc[0].copy()
        chosen = baseline_row
    else:
        chosen = best_row

    selected_satellites = [] if chosen["selected_satellites"] == "NONE" else [
        s.strip() for s in chosen["selected_satellites"].split(",")
    ]

    selected_weights = {}
    if selected_satellites:
        selected_weights = dict(chosen["satellite_weights"])

    decision = TiltDecision(
        rebalance_date=pd.Timestamp(rebalance_date),
        investor_name=investor_cfg.name,
        predictive_probabilities=dict(
            predictive_probability_panel.loc[rebalance_date].to_dict()
        ),
        baseline_score=float(chosen["baseline_score"]),
        selected_satellites=selected_satellites,
        selected_weights=selected_weights,
        total_portfolio_weights=dict(chosen["total_portfolio_weights"]),
        expected_return=float(chosen["expected_return"]),
        variance=float(chosen["variance"]),
        skewness=float(chosen["skewness"]),
        kurtosis=float(chosen["kurtosis"]),
        final_score=float(chosen["score"]),
        score_improvement=float(chosen["score_improvement"]),
        metadata={
            "selected_satellites_str": chosen["selected_satellites"],
            "candidate_id": int(chosen["candidate_id"]),
        },
    )

    return decision, candidate_table


def build_candidate_library_from_train(
    state_series_train: pd.Series,
    regime_names: list[str],
    allocation_df_train: pd.DataFrame,
    alloc_cfg,
    satellite_specs: list,
    return_prefix: str = "ExcessLog",
) -> list[dict]:
    """
    Precompute state-conditional candidate moment tables from TRAIN sample only.

    This is the key A1 efficiency and honesty feature:
    - candidate moments are fixed once using train only
    - test-time decisions only update predictive regime probabilities
    """
    candidates = enumerate_candidate_tilts(satellite_specs, alloc_cfg)
    library = []

    for i, sat_weights in enumerate(candidates):
        total_weights = build_total_portfolio_weights(
            alloc_cfg=alloc_cfg,
            satellite_weights=sat_weights,
        )

        state_moment_table = estimate_state_conditional_portfolio_moments_from_sample(
            state_series=state_series_train,
            regime_names=regime_names,
            allocation_df_sample=allocation_df_train,
            total_weights=total_weights,
            alloc_cfg=alloc_cfg,
            return_prefix=return_prefix,
        )

        library.append({
            "candidate_id": i,
            "satellite_weights": sat_weights,
            "selected_satellites_str": ", ".join(sat_weights.keys()) if sat_weights else "NONE",
            "total_portfolio_weights": total_weights,
            "state_moment_table": state_moment_table,
        })

    return library

def compute_regime_conviction_weights(
    base_satellite_weights: dict[str, float],
    satellite_specs: list,
    predictive_probabilities_row: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Scale satellite weights by regime conviction.
 
    The bear regime is always the first entry in predictive_probabilities_row
    (index[0]) because RegimePostProcessor sorts regimes by ascending
    mean return on the key column — so Regime 0 is always the bear.
 
    Scaling rules
    -------------
    cyclical satellites  (XLE, XLK, XLI, XLB, XLY, XLF, ...):
        w_applied = w_base × (1 − p_bear)
        → full weight in a confirmed bull, zero in a confirmed bear.
 
    defensive satellites (XLP, XLU, XLV, ...):
        w_applied = w_base × p_bear
        → full weight in a confirmed bear, zero in a confirmed bull.
 
    When the model is uncertain (p_bear ≈ 0.5), both types get half
    weight and the portfolio drifts back toward the 60/40 core.
    This naturally implements a regime-uncertainty discount without
    any additional parameters.
 
    Parameters
    ----------
    base_satellite_weights : dict[str, float]
        The satellite weights as selected by the scoring engine (pre-scaling).
 
    satellite_specs : list[SatelliteSpec]
        Full satellite spec list. Used to look up each ticker's style.
 
    predictive_probabilities_row : pd.Series
        One-step-ahead predictive probabilities indexed by regime name.
        The first index entry is treated as the bear regime.
 
    Returns
    -------
    scaled_weights : dict[str, float]
        Satellite weights after conviction scaling. May be smaller than
        base weights. The sum may be less than the original satellite
        sleeve total — the core 60/40 fills the gap automatically in
        build_total_portfolio_weights().
 
    conviction_scalars : dict[str, float]
        The scalar applied to each ticker, useful for diagnostics /
        Excel export.
    """
    # Bear regime = lowest mean return = first entry after RegimePostProcessor
    bear_regime = predictive_probabilities_row.index[0]
    p_bear = float(predictive_probabilities_row.iloc[0])
 
    style_map = {s.ticker: getattr(s, "style", "cyclical") for s in satellite_specs}
 
    scaled_weights = {}
    conviction_scalars = {}
 
    for ticker, base_w in base_satellite_weights.items():
        style = style_map.get(ticker, "cyclical")
        if style == "defensive":
            scalar = p_bear
        else:
            scalar = 1.0 - p_bear
        scaled_weights[ticker] = base_w * scalar
        conviction_scalars[ticker] = scalar
 
    return scaled_weights, conviction_scalars


def apply_cash_sleeve(
    portfolio_weights: dict[str, float],
    cash_sleeve_cfg,
    predictive_probabilities_row: pd.Series,
) -> tuple[dict[str, float], float]:
    """
    Post-scoring risk-management: move weight to cash when bear conviction is high.

    Parameters
    ----------
    portfolio_weights : dict[str, float]
        Final portfolio weights AFTER conviction scaling (core + satellites).
        These should sum to 1.0.

    cash_sleeve_cfg : CashSleeveConfig
        Cash sleeve configuration.

    predictive_probabilities_row : pd.Series
        One-step-ahead predictive probabilities indexed by regime name.
        The first index entry is treated as the bear regime (lowest mean return).

    Returns
    -------
    final_weights : dict[str, float]
        Portfolio weights after applying the cash sleeve.
        Includes the rf_ticker key if cash_weight > 0.

    cash_weight : float
        The cash allocation applied (0.0 if sleeve is inactive).
    """
    if cash_sleeve_cfg is None or not cash_sleeve_cfg.enabled:
        return portfolio_weights, 0.0

    p_bear = float(predictive_probabilities_row.iloc[0])
    cash_weight = cash_sleeve_cfg.compute_cash_weight(p_bear)

    if cash_weight <= 0.0:
        return portfolio_weights, 0.0

    scale = 1.0 - cash_weight
    final_weights = {asset: w * scale for asset, w in portfolio_weights.items()}
    final_weights[cash_sleeve_cfg.rf_ticker] = (
        final_weights.get(cash_sleeve_cfg.rf_ticker, 0.0) + cash_weight
    )

    return final_weights, cash_weight


def select_best_tilt_at_date_from_library(
    candidate_library: list[dict],
    predictive_probabilities_row: pd.Series,
    investor_cfg,
    alloc_cfg,
    rebalance_date: pd.Timestamp,
    satellite_specs: list | None = None,
) -> tuple[TiltDecision, pd.DataFrame]:
    """
    Honest A1 / EW selector with optional regime conviction scaling.
 
    Step 1 — Score all candidates (unchanged):
        Uses precomputed train-window moment tables aggregated by
        predictive regime probabilities. Picks the highest-scoring
        satellite combination.
 
    Step 2 — Regime conviction scaling (new):
        If satellite_specs is provided and any spec has a style field,
        the *applied* satellite weight is scaled by regime conviction:
 
            cyclical  satellites: w_applied = w_base × (1 − p_bear)
            defensive satellites: w_applied = w_base × p_bear
 
        where p_bear is the predictive probability of the lowest-mean
        regime (always regime_names[0] after RegimePostProcessor sorting).
 
        Rationale (Guidolin & Timmermann 2008):
            The scoring step tells us *which* satellite dominates under
            the current predictive distribution. The conviction scaling
            tells us *how much* to hold, proportional to how strongly
            the model believes we are in the appropriate regime for that
            satellite type. This separates selection (driven by moments)
            from sizing (driven by regime confidence), and prevents the
            model from holding full cyclical exposure during ambiguous
            or transitional regimes.
 
        The core 60/40 weights scale up automatically to fill the gap
        as satellite weights shrink — no leverage is ever introduced.
 
        If satellite_specs is None, no scaling is applied (original
        behaviour preserved for backward compatibility).
    """
    rows = []
    baseline_score = None
 
    for candidate in candidate_library:
        agg = aggregate_predictive_portfolio_moments(
            state_moment_table=candidate["state_moment_table"],
            predictive_probabilities_row=predictive_probabilities_row,
        )
 
        score = compute_investor_score(
            investor_cfg=investor_cfg,
            expected_return=agg["expected_return"],
            variance=agg["variance"],
            skewness=agg["skewness"],
            kurtosis=agg["kurtosis"],
        )
 
        if candidate["selected_satellites_str"] == "NONE":
            baseline_score = score
 
        rows.append({
            "rebalance_date": rebalance_date,
            "investor_name": investor_cfg.name,
            "candidate_id": candidate["candidate_id"],
            "selected_satellites": candidate["selected_satellites_str"],
            "satellite_weights": candidate["satellite_weights"],
            "total_portfolio_weights": candidate["total_portfolio_weights"],
            "expected_return": agg["expected_return"],
            "variance": agg["variance"],
            "skewness": agg["skewness"],
            "kurtosis": agg["kurtosis"],
            "score": score,
        })
 
    if baseline_score is None:
        raise ValueError("Baseline candidate {} was not found in candidate_library.")
 
    candidate_table = pd.DataFrame(rows)
    candidate_table["baseline_score"] = baseline_score
    candidate_table["score_improvement"] = candidate_table["score"] - baseline_score
    candidate_table = candidate_table.sort_values(
        ["score_improvement", "score"], ascending=False
    ).reset_index(drop=True)
 
    best_row = candidate_table.iloc[0].copy()
 
    if best_row["score_improvement"] < alloc_cfg.score_improvement_floor:
        chosen = candidate_table.loc[
            candidate_table["selected_satellites"] == "NONE"
        ].iloc[0].copy()
    else:
        chosen = best_row
 
    selected_satellites = (
        []
        if chosen["selected_satellites"] == "NONE"
        else [s.strip() for s in chosen["selected_satellites"].split(",")]
    )
    base_satellite_weights = (
        {}
        if chosen["selected_satellites"] == "NONE"
        else dict(chosen["satellite_weights"])
    )
 
    # ── regime conviction scaling ────────────────────────────────────
    conviction_scalar = None
    if satellite_specs is not None and base_satellite_weights:
        scaled_weights, conviction_scalar = compute_regime_conviction_weights(
            base_satellite_weights=base_satellite_weights,
            satellite_specs=satellite_specs,
            predictive_probabilities_row=predictive_probabilities_row,
        )
    else:
        scaled_weights = base_satellite_weights
 
    # Rebuild total portfolio weights with (possibly scaled) satellite weights
    final_total_weights = build_total_portfolio_weights(
        alloc_cfg=alloc_cfg,
        satellite_weights=scaled_weights,
    )
 
    decision = TiltDecision(
        rebalance_date=pd.Timestamp(rebalance_date),
        investor_name=investor_cfg.name,
        predictive_probabilities=dict(predictive_probabilities_row.to_dict()),
        baseline_score=float(chosen["baseline_score"]),
        selected_satellites=selected_satellites,
        selected_weights=scaled_weights,
        total_portfolio_weights=final_total_weights,
        expected_return=float(chosen["expected_return"]),
        variance=float(chosen["variance"]),
        skewness=float(chosen["skewness"]),
        kurtosis=float(chosen["kurtosis"]),
        final_score=float(chosen["score"]),
        score_improvement=float(chosen["score_improvement"]),
        metadata={
            "selected_satellites_str": chosen["selected_satellites"],
            "candidate_id": int(chosen["candidate_id"]),
            "base_satellite_weights": base_satellite_weights,
            "conviction_scalar": conviction_scalar,
        },
    )
 
    return decision, candidate_table