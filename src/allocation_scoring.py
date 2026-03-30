from itertools import combinations
import numpy as np
import pandas as pd

from src.allocation_config import TiltDecision
from src.allocation_moments import evaluate_candidate_tilt_moments


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
    Compute the investor-specific score for one candidate portfolio.

    Supported types
    ---------------
    MV:
        Score = mu_p - lambda * sigma_p^2

    MVS:
        Score = mu_p - lambda * sigma_p^2 + gamma * skew_p

    MVK:
        Score = mu_p - lambda * sigma_p^2 - delta * kurt_p

    Notes
    -----
    - We use variance directly, not volatility.
    - Kurtosis is regular kurtosis, not excess kurtosis.
    """
    if investor_cfg.investor_type == "MV":
        return expected_return - investor_cfg.lambda_ * variance

    elif investor_cfg.investor_type == "MVS":
        return (
            expected_return
            - investor_cfg.lambda_ * variance
            + investor_cfg.gamma * skewness
        )

    elif investor_cfg.investor_type == "MVK":
        return (
            expected_return
            - investor_cfg.lambda_ * variance
            - investor_cfg.delta * kurtosis
        )

    else:
        raise ValueError(f"Unsupported investor_type: {investor_cfg.investor_type}")


# ---------------------------------------------------------------------
# 2) CANDIDATE ENUMERATION
# ---------------------------------------------------------------------

def enumerate_candidate_tilts(
    satellite_specs: list,
    alloc_cfg,
) -> list[dict[str, float]]:
    """
    Build the discrete set of candidate satellite sleeve weights.

    Always includes:
        {}

    meaning:
        no-satellite baseline

    If top_n_satellites = 1:
        generates:
            {sat1: w}
            {sat2: w}
            ...

    If top_n_satellites = 2:
        also generates pair combinations:
            {sat1: w1, sat2: w2}

    Constraints enforced
    --------------------
    - long-only
    - total satellite sleeve <= max_satellite_weight
    - no empty duplicate candidates
    """
    candidates = [{}]

    # 1-satellite candidates
    for sat in satellite_specs:
        for w in sat.allowed_weights:
            if w <= 0:
                continue
            if w <= alloc_cfg.max_satellite_weight + 1e-12:
                candidates.append({sat.ticker: float(w)})

    # 2-satellite candidates
    if alloc_cfg.top_n_satellites >= 2:
        for sat_a, sat_b in combinations(satellite_specs, 2):
            for w_a in sat_a.allowed_weights:
                for w_b in sat_b.allowed_weights:
                    if w_a <= 0 or w_b <= 0:
                        continue

                    total = float(w_a + w_b)
                    if total <= alloc_cfg.max_satellite_weight + 1e-12:
                        candidates.append({
                            sat_a.ticker: float(w_a),
                            sat_b.ticker: float(w_b),
                        })

    # deduplicate
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