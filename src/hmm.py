import pandas as pd
from hmmlearn import hmm 
import numpy as np   
from src.load import diff_data, prepare_data
import warnings


def make_sticky_transmat(k: int, stay_prob: float) -> np.ndarray:
    off = (1.0 - stay_prob) / (k - 1)
    T = np.full((k, k), off, dtype=float)
    np.fill_diagonal(T, stay_prob)
    return T

def hmm_converge(
    df: pd.DataFrame,
    n_states: int,
    cols: list[str],
    cov_type: str,
    seed: int = 7,
    n_iter: int = 300,
    rf_col: str = "RFW",
    verbose: bool = True,
    return_details: bool = False,
    diff_kwargs: dict | None = None,
    sticky: bool = False,
    stay_prob: float = 0.95,
    min_covar: float = 1e-4,  # helps full cov stability
):
    cols = [c for c in cols if c != rf_col]
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
        min_covar=min_covar,
    )

    if sticky:
        model.startprob_ = np.full(n_states, 1.0 / n_states)
        model.transmat_ = make_sticky_transmat(n_states, stay_prob)
        # Don't re-initialize startprob/transmat randomly
        model.init_params = "mc"   # initialize only means/covars
        # Allow EM to update everything (including transmat/startprob)
        model.params = "stmc"

    model.fit(X)

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


def hmm_sweep_seeds(
    df: pd.DataFrame,          # xA / xB / xC (already monthly ExcessLog cols)
    n_states: int,
    cols: list[str],           # tickers list e.g. ["SPY","WFBIX","^IRX"] (ok)
    cov_type: str,
    seeds=range(1, 41),
    n_iter: int = 300,
    min_state_frac: float = 0.05,
    verbose: bool = False,
):
    rows = []
    best_any = None
    best_stable = None

    for seed in seeds:
        try: 
            out, df_m, model = hmm_converge(
                df=df,
                n_states=n_states,
                cols=cols,
                cov_type=cov_type,
                seed=seed,
                n_iter=n_iter,
                verbose=False,
                return_details=True,
            )
        except Exception as e:
            rows.append({
                "seed": seed,
                "logL": np.nan,
                "n_states_used": 0,
                "min_state_frac": 0.0,
                "collapsed": True,
                "error": type(e).__name__,
            })
            continue


        feat_cols = [c for c in df_m.columns if c.startswith("ExcessLog")]
        X = df_m[feat_cols].to_numpy(dtype=float)

        logL = float(model.score(X))

        occ = df_m["state"].value_counts(normalize=True).to_dict()
        occ_full = {k: float(occ.get(k, 0.0)) for k in range(n_states)}
        min_frac = min(occ_full.values())
        n_used = sum(v > 0 for v in occ_full.values())
        collapsed = (n_used < n_states) or (min_frac < min_state_frac)

        row = {
            "seed": seed,
            "logL": logL,
            "n_states_used": n_used,
            "min_state_frac": min_frac,
            "collapsed": collapsed,
        }
        for k in range(n_states):
            row[f"frac_state{k}"] = occ_full[k]
        rows.append(row)

        pack = {"seed": seed, "logL": logL, "out": out, "df_m": df_m, "model": model, "collapsed": collapsed}

        if best_any is None or logL > best_any["logL"]:
            best_any = pack

        if (not collapsed) and (best_stable is None or logL > best_stable["logL"]):
            best_stable = pack

    summary = pd.DataFrame(rows).sort_values("logL", ascending=False).reset_index(drop=True)
    chosen = best_stable if best_stable is not None else best_any

    if verbose:
        print(summary.head(10))
        if best_stable is None:
            print(f"No seed passed stability filter (min_state_frac >= {min_state_frac}). Using best logL overall.")
        print(f"Chosen seed: {chosen['seed']} | logL: {chosen['logL']:.3f} | collapsed: {chosen['collapsed']}")

    return summary, chosen["seed"], chosen["out"], chosen["df_m"], chosen["model"]