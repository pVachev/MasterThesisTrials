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