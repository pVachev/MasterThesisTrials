import pandas as pd
from hmmlearn import hmm 
import numpy as np   
from src.load import diff_data, prepare_data



def hmm_converge(
    df: pd.DataFrame,
    n_states: int,
    cols: list[str],
    cov_type: str,
    seed: int = 7,
    n_iter: int = 300,
    rf_col: str = "^IRX",
    verbose: bool = True,
    return_details: bool = False,
    diff_kwargs: dict | None = None,   # pass monthly_cols/freq/etc if needed
):
    # ignore rf if accidentally included
    cols = [c for c in cols if c != rf_col]

    # always target ExcessLog columns
    ex_cols = [c if c.startswith("ExcessLog") else f"ExcessLog{c}" for c in cols]

    if all(c in df.columns for c in ex_cols):
        df_m = df.sort_index()[ex_cols].dropna().copy()
    else:
        diff_kwargs = diff_kwargs or {}
        df2 = diff_data(df, cols=cols, rf_col=rf_col, **diff_kwargs)
        df_m = prepare_data(df2, cols=cols, rf_col=rf_col).sort_index().dropna().copy()

    X = df_m.to_numpy(dtype=float)

    model = hmm.GaussianHMM(
        n_components=n_states,
        n_iter=n_iter,
        covariance_type=cov_type,
        random_state=seed,
    ).fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)

    df_m = df_m.copy()
    df_m["state"] = states
    if return_details:
        for k in range(n_states):
            df_m[f"p_state{k}"] = probs[:, k]

    g = df_m.groupby("state")[ex_cols]
    means = (100 * g.mean()).add_prefix("mean_%_")
    stds  = (100 * g.std(ddof=1)).add_prefix("std_%_")
    skew  = g.apply(lambda x: x.skew()).add_prefix("skew_")
    kex   = g.apply(lambda x: x.kurt()).add_prefix("kurt_excess_")
    kurt  = (g.apply(lambda x: x.kurt()) + 3).add_prefix("kurt_")
    n_obs = df_m.groupby("state").size().rename("n_obs")

    out = pd.concat([means, stds, skew, kurt, kex, n_obs], axis=1).T
    out.columns = [f"Regime {c}" for c in out.columns]

    if verbose:
        conv = getattr(model.monitor_, "converged", None)
        iters = getattr(model.monitor_, "iter", None)
        if conv is not None:
            print(f"Converged: {conv}" + (f" | iters: {iters}" if iters is not None else ""))

    return (out, df_m, model) if return_details else out