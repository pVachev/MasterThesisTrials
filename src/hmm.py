import pandas as pd
from hmmlearn import hmm 
import numpy as np   
from src.load import diff_data, prepare_data





# def hmm_converge(df: pd.DataFrame, 
#                  n_states: int,
#                  cols: list[str],
#                  cov_type: str,
#                  seed = 7,
#                  verbose=True, 
#                  ):

#     model = hmm.GaussianHMM(n_components=n_states, 
#                             n_iter=300, 
#                             covariance_type=cov_type,
#                             random_state=seed)
    
#     df_m = df
#     if verbose: 
#         df_m = df_m.resample("ME").sum()[cols].dropna()
#     df_m = df_m[cols]


#     print(df_m["ExcessLogBND"].mean())
#     print(df_m["ExcessLogSPY"].mean())
#     print(df)
#     print(df_m)

#     X = df_m.to_numpy()


#     model.fit(X)

#     states = model.predict(X)
#     probs = model.predict_proba(X)
#     df_m["state"] = states


#     g = df_m.groupby(by=["state"])[cols]
#     out = pd.concat(
#         [
#             (100* g.mean()).add_prefix("mean_%_"),
#             (100 * g.std(ddof=1)).add_prefix("std_%_"),
#             df_m.groupby(["state"]).size().rename("n_obs"),
#     ],
#     axis=1)

#     out.sort_values(f"std_%_{cols[0]}", ascending=False)

#     return out 



def hmm_converge(
    df: pd.DataFrame,
    n_states: int,
    cols: list[str],          # ideally your tickers, e.g. ["BND","SPY"] (can also be ["ExcessLogBND", ...])
    cov_type: str,
    seed: int = 7,
    n_iter: int = 300,
    rf_col: str = "^IRX",
    resample_me: bool = True,
    verbose: bool = True,
    return_details: bool = False,
):
    """
    Works with your diff_data / prepare_data framework:
    - If df doesn't already have the needed ExcessLog columns, it will call diff_data() then prepare_data().
    - Ignores rf_col if you accidentally include it in cols.
    - Adds skewness + kurtosis by regime.
    - Returns regimes as COLUMNS (metrics as rows).
    """

    # 1) clean the user list (ignore ^IRX if included)
    base_cols = [c for c in cols if c != rf_col]

    # 2) decide whether we already have the columns or need to compute them
    #    (supports passing either tickers ["BND","SPY"] or already-prepared names ["ExcessLogBND",...])
    if all(c in df.columns for c in base_cols):
        # user passed actual existing columns (could already be ExcessLog*)
        df_ex = df.sort_index()[base_cols].dropna().copy()
        used_cols = base_cols
    else:
        mapped_excess = [
            c if c.startswith("ExcessLog") else f"ExcessLog{c}"
            for c in base_cols
        ]
        if all(c in df.columns for c in mapped_excess):
            # df already has ExcessLog{ticker} columns; just keep them
            df_ex = df.sort_index()[mapped_excess].dropna().copy()
            used_cols = mapped_excess
        else:
            # compute everything from raw prices + rf
            df2 = diff_data(df, cols=base_cols, rf_col=rf_col)
            df_ex = prepare_data(df2, cols=base_cols, rf_col=rf_col)  # returns only ExcessLog{ticker}
            df_ex = df_ex.sort_index().dropna().copy()
            used_cols = df_ex.columns.tolist()

    # 3) monthly resample if requested (sum of log returns is consistent)
    df_m = df_ex
    if resample_me:
        df_m = df_m.resample("ME").sum(min_count=1).dropna()

    # 4) fit HMM
    X = df_m.to_numpy(dtype=float)

    model = hmm.GaussianHMM(
        n_components=n_states,
        n_iter=n_iter,
        covariance_type=cov_type,
        random_state=seed,
    )
    model.fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)

    df_m = df_m.copy()
    df_m["state"] = states

    # (optional) keep regime probabilities for inspection
    if return_details:
        for k in range(n_states):
            df_m[f"p_state{k}"] = probs[:, k]

    # 5) regime stats (mean/std in %, skew & kurt unitless)
    g = df_m.groupby("state")[used_cols]

    means = (100 * g.mean()).add_prefix("mean_%_")
    stds  = (100 * g.std(ddof=1)).add_prefix("std_%_")

    skew_raw = g.apply(lambda x: x.skew())
    kurt_excess_raw = g.apply(lambda x: x.kurt()) 


    skew  = skew_raw.add_prefix("skew_")
    kurt_excess = kurt_excess_raw.add_prefix("kurt_excess_")   # pandas = Fisher/excess kurtosis
    kurt  = (kurt_excess_raw + 3).add_prefix("kurt_")          # “plain” kurtosis
    n_obs = df_m.groupby("state").size().rename("n_obs")

    out_rows = pd.concat([means, stds, skew, kurt, kurt_excess, n_obs], axis=1)

    # 6) regimes as COLUMNS
    out = out_rows.T

    # sort regimes by vol of first series (descending), if available
    sort_row = f"std_%_{used_cols[0]}"
    if sort_row in out.index:
        out = out.loc[:, out.loc[sort_row].sort_values(ascending=False).index]

    # nicer regime labels
    out.columns = [f"Regime {c}" for c in out.columns]

    if verbose:
        conv = getattr(model.monitor_, "converged", None)
        iters = getattr(model.monitor_, "iter", None)
        if conv is not None:
            print(f"Converged: {conv}" + (f" | iters: {iters}" if iters is not None else ""))
        print(out)

    if return_details:
        return out, df_m, model

    return out