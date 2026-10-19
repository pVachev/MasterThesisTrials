import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# STAGE 4 — SATELLITE CONDITIONAL MOMENT ENGINE
# ---------------------------------------------------------------------

def build_total_portfolio_weights(
    alloc_cfg,
    satellite_weights: dict[str, float] | None = None,
    core_weights_override: dict[str, float] | None = None,   # NEW
    satellite_styles: dict[str, str] | None = None,          # required for hybrid
) -> dict[str, float]:
    satellite_weights = satellite_weights or {}

    # NEW: drop dust positions below the per-satellite minimum (weight returns to core)
    min_sat = getattr(alloc_cfg, "min_satellite_weight", 0.0) or 0.0
    if min_sat > 0.0:
        satellite_weights = {k: v for k, v in satellite_weights.items() if v >= min_sat}
 
    sat_total = float(sum(satellite_weights.values()))
    if sat_total > alloc_cfg.max_satellite_weight + 1e-12:
        raise ValueError(
            f"Total satellite weight {sat_total:.4f} exceeds max_satellite_weight={alloc_cfg.max_satellite_weight:.4f}"
        )
 
    if alloc_cfg.long_only:
        neg = {k: v for k, v in satellite_weights.items() if v < 0}
        if neg:
            raise ValueError(f"Negative satellite weights are not allowed: {neg}")
 
    if getattr(alloc_cfg, 'equity_only_displacement', False):
        # Satellites displace ONLY the equity ticker; bond weights stay fixed.
        equity_tk = alloc_cfg.equity_ticker
        equity_w = alloc_cfg.fixed_core_weights[equity_tk] - sat_total
        if equity_w < -1e-9:
            raise ValueError(
                f"Satellite total {sat_total:.4f} exceeds equity weight "
                f"{alloc_cfg.fixed_core_weights[equity_tk]:.4f} for '{equity_tk}'."
            )
        total_weights = dict(alloc_cfg.fixed_core_weights)
        total_weights[equity_tk] = max(equity_w, 0.0)
    elif getattr(alloc_cfg, 'hybrid_displacement', False):
        # Hybrid funding: cyclical legs displace EQUITY only; defensive legs
        # displace the core PRO-RATA. Invariant: the bond weight equals
        # (1 - defensive_total) * core_bond — bonds are touched only by
        # defensive legs, never by cyclicals.
        if satellite_styles is None:
            raise ValueError(
                "hybrid_displacement=True requires satellite_styles "
                "(ticker -> 'cyclical'|'defensive')."
            )
        unknown = [k for k in satellite_weights if k not in satellite_styles]
        if unknown:
            raise ValueError(f"hybrid_displacement: no style for {unknown}")
        core = core_weights_override if core_weights_override is not None else alloc_cfg.fixed_core_weights
        equity_tk = alloc_cfg.equity_ticker
        cyc_total = float(sum(v for k, v in satellite_weights.items()
                              if satellite_styles[k] == "cyclical"))
        def_total = sat_total - cyc_total
        variant = str(getattr(alloc_cfg, "hybrid_variant", "A")).upper()
        if variant == "B":
            # B: cyclicals PRO-RATA (beta engine intact); defensives EQUITY-ONLY
            # (bond block untouched in risk-off months). Invariant:
            # bond weight = (1 - cyclical_total) * core_bond.
            pro_rata_total, equity_funded = cyc_total, def_total
        else:
            # A: cyclicals EQUITY-ONLY; defensives PRO-RATA. Invariant:
            # bond weight = (1 - defensive_total) * core_bond.
            pro_rata_total, equity_funded = def_total, cyc_total
        total_weights = {k: (1.0 - pro_rata_total) * v for k, v in core.items()}
        eq_after = total_weights.get(equity_tk, 0.0) - equity_funded
        if eq_after < -1e-9:
            # Boundary spillover: the corr-split / lambda tilt can cap the
            # core bond weight at its upper bound (armed, deeply negative-
            # rho months), leaving core equity = 1 - hi. Any equity-funded
            # sleeve above that (e.g. all-defensive at sleeve > 0.40 with
            # bounds (0.20, 0.60)) overshoots. The sleeve conviction
            # OUTRANKS the core tilt: fund the shortfall pro-rata from the
            # remaining core leg(s) and floor equity at zero. Long-only and
            # sum-to-one preserved; bonds still end far above base in the
            # trigger case (e.g. 0.55 vs 0.40).
            shortfall = -eq_after
            others = {k: v for k, v in total_weights.items() if k != equity_tk}
            other_sum = float(sum(others.values()))
            if shortfall > other_sum + 1e-9:
                raise ValueError(
                    f"hybrid_displacement({variant}): satellite total "
                    f"{sat_total:.4f} infeasible against the core."
                )
            for k in others:
                total_weights[k] -= shortfall * (others[k] / other_sum)
            eq_after = 0.0
        total_weights[equity_tk] = max(eq_after, 0.0)
    else:
            # Default: scale all core weights down proportionally.
        core = core_weights_override if core_weights_override is not None else alloc_cfg.fixed_core_weights
        core_scale = 1.0 - sat_total
        total_weights = {k: core_scale * v for k, v in core.items()}
 
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
    cols = [f"{return_prefix}{a}" for a in portfolio_assets]
    missing = [c for c in cols if c not in allocation_df.columns]
    if missing:
        raise KeyError(f"allocation_df is missing required return columns: {missing}")

    out = allocation_df[cols].copy()
    out.columns = portfolio_assets
    return out


def _compute_raw_and_central_moments(r: pd.Series) -> dict[str, float]:
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

    var = max(m2 - m1**2, 0.0)

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


def aggregate_predictive_portfolio_moments(
    state_moment_table: pd.DataFrame,
    predictive_probabilities_row: pd.Series,
) -> dict[str, float]:
    table = state_moment_table.set_index("regime").copy()

    if list(table.index) != list(predictive_probabilities_row.index):
        raise ValueError("Regime order mismatch between state_moment_table and predictive probabilities.")

    p = predictive_probabilities_row.to_numpy(dtype=float)

    M1 = float(np.dot(p, table["m1"].to_numpy(dtype=float)))
    M2 = float(np.dot(p, table["m2"].to_numpy(dtype=float)))
    M3 = float(np.dot(p, table["m3"].to_numpy(dtype=float)))
    M4 = float(np.dot(p, table["m4"].to_numpy(dtype=float)))

    var = max(M2 - M1**2, 0.0)

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

def estimate_state_conditional_portfolio_moments_from_sample(
    state_series: pd.Series,
    regime_names: list[str],
    allocation_df_sample: pd.DataFrame,
    total_weights: dict[str, float],
    alloc_cfg,
    return_prefix: str = "ExcessLog",
) -> pd.DataFrame:
    """
    Honest A1 version:
    estimate candidate portfolio moments using a SAMPLE you explicitly pass in
    (for A1: the training sample only).
    """
    portfolio_assets = list(total_weights.keys())

    asset_panel = _extract_asset_return_panel(
        allocation_df=allocation_df_sample,
        portfolio_assets=portfolio_assets,
        return_prefix=return_prefix,
    )

    common_index = asset_panel.index.intersection(state_series.index)
    asset_panel = asset_panel.loc[common_index].copy()
    states = state_series.loc[common_index].copy()

    if len(common_index) == 0:
        raise ValueError("No overlap between state_series and allocation_df_sample.")

    w = pd.Series(total_weights, dtype=float)
    portfolio_r = (np.expm1(asset_panel) * w.values).sum(axis=1)

    uncond = _compute_raw_and_central_moments(portfolio_r)

    rows = []
    for state_id in range(len(regime_names)):
        r_s = portfolio_r.loc[states == state_id]
        raw = _compute_raw_and_central_moments(r_s)
        n = raw["n_obs"]

        alpha = 1.0
        if n < alloc_cfg.min_regime_obs:
            alpha = min(alpha, n / alloc_cfg.min_regime_obs if alloc_cfg.min_regime_obs > 0 else 1.0)

        if alloc_cfg.shrinkage_intensity > 0:
            alpha = min(alpha, n / (n + alloc_cfg.shrinkage_intensity) if n > 0 else 0.0)

        shrunk = {}
        for k in ["m1", "m2", "m3", "m4"]:
            if np.isnan(raw[k]):
                shrunk[k] = uncond[k]
            else:
                shrunk[k] = alpha * raw[k] + (1.0 - alpha) * uncond[k]

        m1, m2, m3, m4 = shrunk["m1"], shrunk["m2"], shrunk["m3"], shrunk["m4"]

        var = max(m2 - m1**2, 0.0)

        if var <= 1e-16:
            skew = 0.0
            kurt = 3.0
        else:
            mu3 = m3 - 3*m1*m2 + 2*(m1**3)
            mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)
            skew = mu3 / (var ** 1.5)
            kurt = mu4 / (var ** 2)

        rows.append({
            "regime": regime_names[state_id],
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


def calibrate_investor_params(
    benchmark_returns: pd.Series,
    lambda_: float = 3.0,
    skewness_target_pct: float = 0.15,
    kurtosis_target_pct: float = 0.20,
) -> dict:
    """
    Calibrate gamma and delta from unconditional benchmark moments such that
    skewness and kurtosis terms each contribute a target percentage of the
    variance term at benchmark moment values.

    Without factorial scalings:
        gamma = target_pct * lambda * var_bm / |skew_bm|
        delta = target_pct * lambda * var_bm / |ekurt_bm|

    Parameters are fixed once from the benchmark and held constant —
    they represent stable investor preferences, not time-varying estimates.
    """
    

    var   = benchmark_returns.var()
    skew  = abs(stats.skew(benchmark_returns))
    ekurt = abs(stats.kurtosis(benchmark_returns, fisher=True))
    return {
        "lambda_": lambda_,
        "gamma":   round(skewness_target_pct * lambda_ * var / skew, 8),
        "delta":   round(kurtosis_target_pct * lambda_ * var / ekurt, 8),
    }