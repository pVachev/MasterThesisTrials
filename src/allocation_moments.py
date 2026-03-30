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