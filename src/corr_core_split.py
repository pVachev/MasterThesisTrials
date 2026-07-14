"""
corr_core_split.py
==================
Correlation-conditioned core split for the core-replacement configuration.

Two pure, side-effect-free functions:

1. predictive_bond_equity_corr(...)
   Extracts the predictive bond-equity correlation rho_{t+1|t} from the
   regime-mixture covariance, using the SAME hard regime labels the moment
   engine uses (res.pp.df_m["state"]). No look-ahead: Sigma_k and mu_k are
   estimated on the training window, pi is the one-step-ahead predictive prob.

       Cov_mix = sum_k pi_k * Sigma_k
                 + sum_k pi_k * (mu_k - mu_bar)(mu_k - mu_bar)'
       rho     = Cov_mix[E,B] / sqrt(Cov_mix[E,E] * Cov_mix[B,B])

   (within-regime covariance weighted by predictive probs, plus the
    between-regime mean dispersion — the standard mixture-of-Gaussians
    covariance identity.)

2. corr_adjusted_core_split(...)
   Tilts the residual core's INTERNAL equity:bond split as a monotonic
   function of rho. This is a SIZING rule on the core; it does NOT touch
   selection and does NOT overlay the p_bear satellite conviction scalar.

       bond_w = clip( base_bond - kappa * rho , bond_bounds )
       eq_w   = 1 - bond_w

   kappa = 0.0 reproduces the fixed base split exactly (backward compatible).
   rho < 0 (bonds hedge)      -> more bonds  (drawdown protection)
   rho > 0 (2022 rate shock)  -> more equity (avoid dead bond exposure)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def predictive_bond_equity_corr(
    state_series: pd.Series,
    eq_ret: pd.Series,
    bond_ret: pd.Series,
    predictive_probabilities_row: pd.Series,
    min_obs: int = 12,
) -> float:
    """Predictive bond-equity correlation from the regime-mixture covariance.

    Parameters
    ----------
    state_series : pd.Series
        Hard regime label per date over the estimation window
        (frozen.train_df_m["state"]). Its index defines the window.
    eq_ret, bond_ret : pd.Series
        Equity / bond return series (same return space as the signal, i.e.
        ExcessLog). Need not be pre-trimmed: the inner join on state_series
        restricts to the training window.
    predictive_probabilities_row : pd.Series
        pi_{t+1|t}, indexed by regime name in the canonical regime order.
    min_obs : int
        Regimes with fewer than this many window observations fall back to the
        pooled (all-window) covariance/mean, to avoid singular 2x2 estimates
        from a thinly-populated regime.

    Returns
    -------
    float
        rho in [-1, 1], or np.nan if undefined (degenerate variance).
    """
    df = pd.DataFrame({"state": state_series, "E": eq_ret, "B": bond_ret}).dropna()
    if len(df) < 2:
        return float("nan")

    regimes = list(predictive_probabilities_row.index)
    pi = predictive_probabilities_row.to_numpy(dtype=float)

    pooled_mu = df[["E", "B"]].mean().to_numpy()
    pooled_cov = np.cov(df[["E", "B"]].to_numpy().T, ddof=1)

    mus, covs = [], []
    for rg in regimes:
        sub = df.loc[df["state"] == rg, ["E", "B"]]
        if len(sub) >= min_obs:
            mus.append(sub.mean().to_numpy())
            covs.append(np.cov(sub.to_numpy().T, ddof=1))
        else:
            mus.append(pooled_mu.copy())
            covs.append(pooled_cov.copy())

    mus = np.asarray(mus)            # K x 2
    covs = np.asarray(covs)          # K x 2 x 2
    mu_bar = (pi[:, None] * mus).sum(axis=0)              # 2
    within = (pi[:, None, None] * covs).sum(axis=0)       # 2 x 2
    dev = mus - mu_bar                                     # K x 2
    between = np.einsum("k,ki,kj->ij", pi, dev, dev)       # 2 x 2
    cov_mix = within + between

    var_e, var_b = cov_mix[0, 0], cov_mix[1, 1]
    if var_e <= 1e-18 or var_b <= 1e-18:
        return float("nan")
    return float(cov_mix[0, 1] / np.sqrt(var_e * var_b))


def corr_adjusted_core_split(
    rho: float | None,
    base_core_weights: dict[str, float],
    equity_ticker: str,
    bond_ticker: str,
    kappa: float = 0.2,
    bond_bounds: tuple[float, float] = (0.20, 0.60),
    rho_realized: float | None = None,
    blend_w: float = 0.0,
    lam: float = 0.0,
    z: float | None = None,
    z_star: float = 2.0,
) -> dict[str, float]:
    """Return the rho-adjusted INTERNAL core split {equity: w, bond: 1-w} (sums to 1).

    kappa = 0.0  -> returns the fixed base split (no-op).
    """
    base_bond = float(base_core_weights[bond_ticker])

    # effective rho for the standing kappa term: optional shrink of the
    # mixture rho toward the realized (horizon-matched) estimate.
    rho_mix = float("nan") if rho is None else float(rho)
    rho_rlz = float("nan") if rho_realized is None else float(rho_realized)
    if blend_w and np.isfinite(rho_rlz) and np.isfinite(rho_mix):
        rho_eff = blend_w * rho_rlz + (1.0 - blend_w) * rho_mix
    elif blend_w and np.isfinite(rho_rlz):
        rho_eff = rho_rlz
    else:
        rho_eff = rho_mix

    # UNIFIED stress slope (one estimator, one lever): the armed state
    # raises the aggressiveness of the SAME tilt, kappa_eff = kappa + lam.
    #   w_B = clip(base - (kappa + lam*1[z>=z*]) * rho_eff, bounds)
    # Under sign disagreement between rho_mix and rho_rlz, the blend
    # shrinks rho_eff toward zero, so aggressive tilts require agreement.
    # Note: with kappa = lam, this reproduces the legacy dual form
    # (-kappa*rho_mix - lam*1[armed]*rho_rlz) EXACTLY in armed months at
    # blend_w = lam/(kappa+lam); the legacy form is retired from code and
    # survives only in pre-Jul-2026 _lam workbooks (spec-5 ladder row).
    armed = (lam != 0.0 and z is not None and np.isfinite(z) and z >= z_star)
    kappa_eff = kappa + (lam if armed else 0.0)

    if not kappa_eff or not np.isfinite(rho_eff):
        bond_w = base_bond
    else:
        bond_w = base_bond - kappa_eff * rho_eff
        lo, hi = bond_bounds
        bond_w = min(max(bond_w, lo), hi)
    return {equity_ticker: 1.0 - bond_w, bond_ticker: bond_w}


# --------------------------------------------------------------------------
# Unit tests (synthetic, no backtest). Run: python -m src.corr_core_split
# --------------------------------------------------------------------------
def _test():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=600, freq="ME")

    # Build two regimes with KNOWN within-regime correlations:
    #   regime A: strong NEGATIVE eq-bond corr (-0.6)
    #   regime B: POSITIVE eq-bond corr (+0.5)
    def draw(n, corr, sd_e, sd_b, mu_e, mu_b):
        cov = np.array([[sd_e**2, corr*sd_e*sd_b], [corr*sd_e*sd_b, sd_b**2]])
        return rng.multivariate_normal([mu_e, mu_b], cov, size=n)

    nA, nB = 300, 300
    A = draw(nA, -0.6, 0.04, 0.02, 0.004, 0.002)
    B = draw(nB, +0.5, 0.07, 0.03, -0.002, -0.001)
    data = np.vstack([A, B])
    state = pd.Series(["RA"] * nA + ["RB"] * nB, index=idx)
    eq = pd.Series(data[:, 0], index=idx)
    bd = pd.Series(data[:, 1], index=idx)

    # If predictive prob is all on A -> rho should be close to -0.6
    pA = pd.Series([1.0, 0.0], index=["RA", "RB"])
    rhoA = predictive_bond_equity_corr(state, eq, bd, pA)
    assert -0.7 < rhoA < -0.5, rhoA

    # All on B -> close to +0.5
    pB = pd.Series([0.0, 1.0], index=["RA", "RB"])
    rhoB = predictive_bond_equity_corr(state, eq, bd, pB)
    assert 0.4 < rhoB < 0.6, rhoB

    # Mixed 50/50 -> between the two, and the between-regime mean dispersion
    # (means differ across regimes) pulls it; just check it is between.
    pM = pd.Series([0.5, 0.5], index=["RA", "RB"])
    rhoM = predictive_bond_equity_corr(state, eq, bd, pM)
    assert rhoA < rhoM < rhoB + 0.2, (rhoA, rhoM, rhoB)

    # --- split function ---
    base = {"^SP500TR": 0.60, "LT09TRUU": 0.40}
    # kappa=0 -> no-op
    s0 = corr_adjusted_core_split(-0.5, base, "^SP500TR", "LT09TRUU", kappa=0.0)
    assert abs(s0["LT09TRUU"] - 0.40) < 1e-12 and abs(sum(s0.values()) - 1) < 1e-12

    # negative rho -> more bonds
    s1 = corr_adjusted_core_split(-0.5, base, "^SP500TR", "LT09TRUU", kappa=0.20)
    assert abs(s1["LT09TRUU"] - 0.50) < 1e-9, s1   # 0.40 - 0.20*(-0.5) = 0.50

    # positive rho -> fewer bonds
    s2 = corr_adjusted_core_split(+0.25, base, "^SP500TR", "LT09TRUU", kappa=0.20)
    assert abs(s2["LT09TRUU"] - 0.35) < 1e-9, s2   # 0.40 - 0.20*0.25 = 0.35

    # clipping at the bound
    s3 = corr_adjusted_core_split(-2.0, base, "^SP500TR", "LT09TRUU", kappa=0.20, bond_bounds=(0.20, 0.60))
    assert abs(s3["LT09TRUU"] - 0.60) < 1e-12, s3  # clipped to upper bound

    print("rho(all-A) =", round(rhoA, 3), " rho(all-B) =", round(rhoB, 3), " rho(50/50) =", round(rhoM, 3))
    print("split(rho=-0.5,k=.2) =", {k: round(v, 3) for k, v in s1.items()})
    print("split(rho=+0.25,k=.2) =", {k: round(v, 3) for k, v in s2.items()})
    # ── blend + armed amplifier ──
    # blend: 0.5 between -0.5 (mix) and +0.3 (realized) -> rho_eff=-0.1 -> bond 0.42
    s4 = corr_adjusted_core_split(-0.5, base, "^SP500TR", "LT09TRUU", kappa=0.20,
                                  rho_realized=0.3, blend_w=0.5)
    assert abs(s4["LT09TRUU"] - 0.42) < 1e-9, s4
    # armed amplifier: z=3>=2, lam=0.2, rho_rlz=+0.4 -> bond 0.40 -0.2*0.4 -0.2*0.4 = 0.24
    s5 = corr_adjusted_core_split(0.4, base, "^SP500TR", "LT09TRUU", kappa=0.20,
                                  rho_realized=0.4, lam=0.2, z=3.0, z_star=2.0)
    assert abs(s5["LT09TRUU"] - 0.24) < 1e-9, s5
    # not armed (z below gate): amplifier silent
    s6 = corr_adjusted_core_split(0.4, base, "^SP500TR", "LT09TRUU", kappa=0.20,
                                  rho_realized=0.4, lam=0.2, z=1.0, z_star=2.0)
    assert abs(s6["LT09TRUU"] - 0.32) < 1e-9, s6
    # unified semantics under sign DISAGREEMENT: armed, blend 0.5,
    # rho_mix=-0.283, rho_rlz=+0.329 -> rho_eff=+0.023 -> bond 0.40-0.4*0.023=0.3908
    s8 = corr_adjusted_core_split(-0.283, base, "^SP500TR", "LT09TRUU", kappa=0.20,
                                  rho_realized=0.329, blend_w=0.5, lam=0.2, z=2.2, z_star=2.0)
    assert abs(s8["LT09TRUU"] - (0.40 - 0.4 * 0.023)) < 1e-9, s8
    # defaults byte-identical to legacy
    s7 = corr_adjusted_core_split(-0.5, base, "^SP500TR", "LT09TRUU", kappa=0.20)
    assert abs(s7["LT09TRUU"] - s1["LT09TRUU"]) < 1e-12
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _test()
