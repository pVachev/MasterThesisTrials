"""
model_selection.py
──────────────────
Fits the HMM for K=2, 3, 5 states on the Model A asset set
(^SP500TR, LT09TRUU) and prints log-likelihood, AIC, BIC,
average regime duration, and minimum regime fraction.

Run from the project root:
    python model_selection.py

The numbers go into Table (tab:model_selection) in the thesis.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm

from src.transform import clean_data
from src.load import diff_data
from src.runner import GlobalRunConfig


def n_params(k: int) -> int:
    """
    Free parameters for a K-state full-covariance bivariate Gaussian HMM:
      - (K^2 - K) transition probabilities (rows sum to 1)
      - K * 2    state-conditional means (2 assets)
      - K * 3    unique entries per (2x2) full covariance (var1, var2, cov12)
      - (K - 1)  initial state distribution
    """
    return (k**2 - k) + k * 2 + k * 3 + (k - 1)


def fit_best(X: np.ndarray, k: int, n_seeds: int = 25) -> tuple[hmm.GaussianHMM, float]:
    best_model, best_ll = None, -np.inf
    for seed in range(1, n_seeds + 1):
        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=2000,
            tol=1e-6,
            random_state=seed,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except Exception:
            continue
    return best_model, best_ll


def regime_stats(model: hmm.GaussianHMM, X: np.ndarray):
    """Return (avg expected duration in months, min regime fraction)."""
    T = len(X)
    p_diag = np.diag(model.transmat_)
    durations = 1.0 / (1.0 - p_diag + 1e-12)

    # Viterbi hard states for regime fractions
    states = model.predict(X)
    fracs = [np.sum(states == k) / T for k in range(model.n_components)]

    return durations.mean(), min(fracs)


def main():
    # ── Replicate Model A data prep from main.py ─────────────────────────────
    tickers_all = [
        "SPY", "WFBIX", "^IRX", "LBUSTRUU", "LT09TRUU", "^SP500TR", "G1BM",
        "RF", "XAU", "USGG3M", "LT01TRUU", "LT12TRUU", "LT13TRUU", "DEMUSD",
        "Oil COMP", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV",
        "XLY", "LT09TRUUW", "RFW", "EEM",
    ]
    m_tickers = [
        "LBUSTRUU", "LT09TRUU", "LT01TRUU", "LT12TRUU", "XAU",
        "USGG3M", "RF", "LT13TRUU", "DEMUSD", "Oil COMP",
    ]
    w_tickers = ["LT09TRUUW", "RFW"]

    cfg = GlobalRunConfig(
        n_states=3,
        cov_type="full",
        seeds=range(1, 26),
        rf_col="RF",
        rf_mode="simple_return_monthly_decimal",
        freq="ME",
        start_date="1999-01-31",
        end_date="2026-03-31",
        output_file="hmm_regime_results_monthly.xlsx",
    )

    print("Loading data...")
    df = clean_data(tickers_all, m_tickers, w_tickers)
    df_diff = diff_data(
        df,
        cols=["^SP500TR", "LT09TRUU"],
        rf_col=cfg.rf_col,
        freq=cfg.freq,
        monthly_cols=m_tickers,
        weekly_cols=w_tickers,
        rf_mode=cfg.rf_mode,
    )
    df_diff = df_diff.loc[
        (df_diff.index >= pd.Timestamp(cfg.start_date)) &
        (df_diff.index <= pd.Timestamp(cfg.end_date))
    ]
    X = df_diff[["ExcessLog^SP500TR", "ExcessLogLT09TRUU"]].dropna().values
    T = len(X)
    print(f"Sample size T = {T}\n")

    # ── Fit and report ────────────────────────────────────────────────────────
    header = f"{'K':>3}  {'LogL':>10}  {'Params':>7}  {'AIC':>10}  {'BIC':>10}  {'AvgDur(mo)':>12}  {'MinFrac':>9}"
    print(header)
    print("-" * len(header))

    for k in [2, 3, 5]:
        model, ll = fit_best(X, k)
        p      = n_params(k)
        aic    = -2 * ll + 2 * p
        bic    = -2 * ll + p * np.log(T)
        avg_d, min_f = regime_stats(model, X)
        print(f"{k:>3}  {ll:>10.2f}  {p:>7}  {aic:>10.2f}  {bic:>10.2f}  {avg_d:>12.1f}  {min_f:>9.3f}")

   


if __name__ == "__main__":
    main()