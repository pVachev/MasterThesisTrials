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

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.hmm import hmm_sweep_seeds
from src.postprocess import RegimePostProcessor
from src.allocation_config import FrozenHMMState


def _extract_filtered_probabilities_from_dfm(df_m: pd.DataFrame, regime_names: list[str]) -> pd.DataFrame:
    prob_cols = [c for c in df_m.columns if c.startswith("p_state")]
    if not prob_cols:
        raise ValueError("No p_state* columns found in df_m.")
    prob_cols = sorted(prob_cols, key=lambda c: int(c.replace("p_state", "")))

    probs = df_m[prob_cols].copy()
    probs.columns = regime_names
    probs = probs.div(probs.sum(axis=1), axis=0)
    return probs


def _reorder_hmm_params(model, order_old: list[int], covariance_type: str):
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


def fit_hmm_train_only_for_allocation(
    res_core,
    hmm_cfg,
    tt_cfg,
) -> FrozenHMMState:
    """
    Fit the HMM on the TRAIN sample only and freeze parameters for A1.

    Parameters
    ----------
    res_core : ModelRunResult
        One full-sample HMM result chosen as the core regime engine template.
        We reuse:
        - res_core.x   (full prepared core excess-return panel)
        - res_core.spec metadata

    hmm_cfg : GlobalRunConfig
        Existing HMM run config containing n_states, cov_type, seeds.

    tt_cfg : TrainTestConfig
        Train/test split definition.

    Returns
    -------
    FrozenHMMState
    """
    x_full = res_core.x.sort_index().copy()
    tt_cfg.validate(x_full.index)

    train_start = pd.to_datetime(tt_cfg.train_start) if tt_cfg.train_start is not None else x_full.index.min()
    train_end = pd.to_datetime(tt_cfg.train_end)
    test_start = pd.to_datetime(tt_cfg.test_start)
    test_end = pd.to_datetime(tt_cfg.test_end) if tt_cfg.test_end is not None else x_full.index.max()

    train_x = x_full.loc[(x_full.index >= train_start) & (x_full.index <= train_end)].copy()
    test_x = x_full.loc[(x_full.index >= test_start) & (x_full.index <= test_end)].copy()

    if len(train_x) < tt_cfg.min_train_observations:
        raise ValueError(
            f"Training sample has only {len(train_x)} observations; "
            f"need at least {tt_cfg.min_train_observations}."
        )
    if len(test_x) < 2:
        raise ValueError("Test sample must contain at least 2 observations for next-period backtesting.")

    # Train-only fit
    sweep, best_seed, out_train, df_m_train, model_train = hmm_sweep_seeds(
        df=train_x,
        n_states=hmm_cfg.n_states,
        cols=res_core.spec.tickers,
        cov_type=hmm_cfg.cov_type,
        seeds=hmm_cfg.seeds,
    )

    pp_train = RegimePostProcessor(
        model_name=f"{res_core.spec.label} | train_only",
        n_states=hmm_cfg.n_states,
        key_col=res_core.spec.key_col,
    ).fit(df_m_train, out_train)

    filtered_train = _extract_filtered_probabilities_from_dfm(
        pp_train.df_m,
        pp_train.regime_names,
    )

    startprob, transmat, means, covars = _reorder_hmm_params(
        model_train,
        pp_train.order_old,
        hmm_cfg.cov_type,
    )

    return FrozenHMMState(
        model_label=res_core.spec.label,
        model_code=res_core.spec.code,
        n_states=hmm_cfg.n_states,
        regime_names=pp_train.regime_names,
        order_old=pp_train.order_old,
        covariance_type=hmm_cfg.cov_type,
        startprob=startprob,
        transmat=transmat,
        means=means,
        covars=covars,
        train_x=train_x,
        test_x=test_x,
        train_df_m=pp_train.df_m.copy(),
        train_out=out_train.copy(),
        filtered_train=filtered_train,
        filtered_test=None,
    )


def _state_log_emission_prob(x: np.ndarray, means: np.ndarray, covars: np.ndarray, covariance_type: str) -> np.ndarray:
    """
    Log emission probability of one observation under each state.
    """
    k = means.shape[0]
    out = np.empty(k, dtype=float)

    for s in range(k):
        mu = means[s]

        if covariance_type == "full":
            cov = covars[s]
        elif covariance_type == "diag":
            cov = np.diag(covars[s])
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        out[s] = multivariate_normal.logpdf(x, mean=mu, cov=cov, allow_singular=True)

    return out


def forward_filter_fixed_hmm(
    x_panel: pd.DataFrame,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str,
    regime_names: list[str],
    init_filtered: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sequential forward filtering under fixed HMM parameters.

    If init_filtered is None:
        start from startprob (used for train filtering if needed)

    If init_filtered is provided:
        use init_filtered @ P as the predictive prior for the first observation
        (used for test filtering after the train sample).
    """
    X = x_panel.to_numpy(dtype=float)
    T = len(X)
    k = len(regime_names)

    filtered = np.zeros((T, k), dtype=float)

    if init_filtered is None:
        pred = startprob.copy()
    else:
        pred = np.asarray(init_filtered, dtype=float) @ transmat
        pred = pred / pred.sum()

    for t in range(T):
        log_em = _state_log_emission_prob(
            X[t],
            means=means,
            covars=covars,
            covariance_type=covariance_type,
        )

        log_prior = np.log(np.clip(pred, 1e-300, None))
        log_num = log_prior + log_em
        log_den = logsumexp(log_num)

        alpha = np.exp(log_num - log_den)
        alpha = alpha / alpha.sum()

        filtered[t, :] = alpha

        # predictive prior for the next observation
        pred = alpha @ transmat
        pred = pred / pred.sum()

    return pd.DataFrame(filtered, index=x_panel.index, columns=regime_names)


def filter_test_probabilities_fixed_params(frozen_state: FrozenHMMState) -> pd.DataFrame:
    """
    Filter the TEST sample sequentially using only:
    - frozen train-fitted parameters
    - the last filtered probability from the training sample
    """
    last_train_filtered = frozen_state.filtered_train.iloc[-1].to_numpy(dtype=float)

    filtered_test = forward_filter_fixed_hmm(
        x_panel=frozen_state.test_x,
        startprob=frozen_state.startprob,
        transmat=frozen_state.transmat,
        means=frozen_state.means,
        covars=frozen_state.covars,
        covariance_type=frozen_state.covariance_type,
        regime_names=frozen_state.regime_names,
        init_filtered=last_train_filtered,
    )

    frozen_state.filtered_test = filtered_test.copy()
    return filtered_test


def predictive_regime_probabilities_from_filtered(
    filtered_probs: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    steps_ahead: int = 1,
) -> pd.DataFrame:
    """
    Same math as Stage 3, but now fed by honest train/test filtered probabilities.
    """
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be at least 1.")

    if list(filtered_probs.columns) != list(transition_matrix.columns):
        raise ValueError("filtered_probs columns and transition_matrix columns must match exactly.")
    if list(transition_matrix.index) != list(transition_matrix.columns):
        raise ValueError("transition_matrix must have matching index and columns.")

    P = transition_matrix.to_numpy(dtype=float)
    P_h = np.linalg.matrix_power(P, steps_ahead)

    pi = filtered_probs.to_numpy(dtype=float)
    pred = pi @ P_h

    pred_df = pd.DataFrame(pred, index=filtered_probs.index, columns=filtered_probs.columns)
    pred_df = pred_df.div(pred_df.sum(axis=1), axis=0)
    return pred_df


def build_test_predictive_probability_panel(frozen_state: FrozenHMMState, steps_ahead: int = 1) -> pd.DataFrame:
    """
    Honest A1 predictive probability panel for the TEST sample.
    """
    if frozen_state.filtered_test is None:
        _ = filter_test_probabilities_fixed_params(frozen_state)

    P = pd.DataFrame(
        frozen_state.transmat,
        index=frozen_state.regime_names,
        columns=frozen_state.regime_names,
    )

    pred_test = predictive_regime_probabilities_from_filtered(
        filtered_probs=frozen_state.filtered_test,
        transition_matrix=P,
        steps_ahead=steps_ahead,
    )
    return pred_test