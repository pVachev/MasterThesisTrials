import numpy as np
import pandas as pd



def extract_filtered_probabilities_from_dfm(df_m: pd.DataFrame, regime_names: list[str]) -> pd.DataFrame:
    prob_cols = [c for c in df_m.columns if c.startswith("p_state")]
    if not prob_cols:
        raise ValueError("No p_state* columns found in df_m.")
    prob_cols = sorted(prob_cols, key=lambda c: int(c.replace("p_state", "")))

    probs = df_m[prob_cols].copy()
    probs.columns = regime_names
    probs = probs.div(probs.sum(axis=1), axis=0)
    return probs


def reorder_hmm_params(model, order_old: list[int], covariance_type: str):
    """
    Reorder frozen HMM parameters into the postprocessed regime order.
    """
    startprob = np.asarray(model.startprob_, dtype=float)[order_old]
    transmat = np.asarray(model.transmat_, dtype=float)[np.ix_(order_old, order_old)]
    means = np.asarray(model.means_, dtype=float)[order_old]

    covars_raw = np.asarray(model.covars_, dtype=float)
    if covariance_type == "full":
        covars = covars_raw[order_old]
    elif covariance_type == "diag":
        covars = covars_raw[order_old]
    else:
        raise ValueError(f"Unsupported covariance_type for A1: {covariance_type}")

    # numerical hygiene
    startprob = startprob / startprob.sum()
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    return startprob, transmat, means, covars


