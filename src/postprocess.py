import numpy as np
import pandas as pd

class RegimePostProcessor:
    """
    Post-processes a chosen HMM run:
      - relabels regimes deterministically (by mean of key_col)
      - reorders p_state* columns accordingly
      - optionally renames out columns to regime names
      - provides compact regime summary + long-format table helpers
    """

    def __init__(
        self,
        model_name: str,
        n_states: int,
        key_col: str = "ExcessLog^SP500TR",
        regime_names: list[str] | None = None,
    ):
        self.model_name = model_name
        self.n_states = n_states
        self.key_col = key_col
        self.regime_names = regime_names or [f"Regime {i}" for i in range(n_states)]

        self.mapping_old_to_new: dict[int, int] | None = None
        self.order_old: list[int] | None = None

        self.df_m: pd.DataFrame | None = None
        self.out: pd.DataFrame | None = None

    def fit(self, df_m: pd.DataFrame, out: pd.DataFrame | None = None) -> "RegimePostProcessor":
        if "state" not in df_m.columns:
            raise KeyError("df_m must contain a 'state' column.")
        if self.key_col not in df_m.columns:
            raise KeyError(f"'{self.key_col}' not in df_m columns.")

        # 1) Determine ordering by mean(key_col)
        order_old = (
            df_m.groupby("state")[self.key_col]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        mapping = {old: new for new, old in enumerate(order_old)}

        df2 = df_m.copy()
        df2["state"] = df2["state"].map(mapping)

        # 2) Reorder probability columns if present
        old_prob_cols = [f"p_state{old}" for old in order_old]
        if all(c in df2.columns for c in old_prob_cols):
            tmp = df2[old_prob_cols].copy()
            tmp.columns = [f"p_state{k}" for k in range(self.n_states)]
            drop_cols = [c for c in df2.columns if c.startswith("p_state")]
            df2.drop(columns=drop_cols, inplace=True, errors="ignore")
            df2 = pd.concat([df2, tmp], axis=1)

        # 3) Store
        self.mapping_old_to_new = mapping
        self.order_old = order_old
        self.df_m = df2

        # 4) Optionally rename out columns
        if out is not None:
            out2 = out.copy()
            # assume out columns are "Regime <state>" or similar; we just overwrite by deterministic names
            out2.columns = self.regime_names
            self.out = out2

        return self

    def regime_summary(self, asset_col: str) -> pd.DataFrame:
        """
        Compact per-regime table:
          n_obs, frac_obs, mean/std (in %) for SPY and asset, and hard-state correlation.
        """
        if self.df_m is None:
            raise ValueError("Call .fit(df_m, out) first.")
        df_m = self.df_m

        if asset_col not in df_m.columns:
            raise KeyError(f"{asset_col} not found in df_m.")
        if self.key_col not in df_m.columns:
            raise KeyError(f"{self.key_col} not found in df_m.")

        g = df_m.groupby("state")
        n_obs = g.size()
        frac = n_obs / len(df_m)

        mean_key = 100 * g[self.key_col].mean()
        std_key  = 100 * g[self.key_col].std(ddof=1)

        mean_ast = 100 * g[asset_col].mean()
        std_ast  = 100 * g[asset_col].std(ddof=1)

        corr = g.apply(lambda d: d[self.key_col].corr(d[asset_col]))

        res = pd.DataFrame({
            "model": self.model_name,
            "regime": [self.regime_names[i] for i in range(self.n_states)],
            "n_obs": [int(n_obs.get(i, 0)) for i in range(self.n_states)],
            "frac_obs": [float(frac.get(i, 0.0)) for i in range(self.n_states)],
            "mean_%_SPY": [float(mean_key.get(i, np.nan)) for i in range(self.n_states)],
            "std_%_SPY":  [float(std_key.get(i, np.nan)) for i in range(self.n_states)],
            f"mean_%_{asset_col}": [float(mean_ast.get(i, np.nan)) for i in range(self.n_states)],
            f"std_%_{asset_col}":  [float(std_ast.get(i, np.nan)) for i in range(self.n_states)],
            f"corr_{self.key_col}_{asset_col}": [float(corr.get(i, np.nan)) for i in range(self.n_states)],
        })
        return res
    
    def regime_correlation_table(self, asset_cols: list[str]) -> pd.DataFrame:
        """
        Build a lower-triangular correlation table per regime for up to 4 assets.
        """
        if self.df_m is None:
            raise ValueError("Call .fit(df_m, out) first.")
        if not asset_cols:
            raise ValueError("asset_cols cannot be empty.")
        if len(asset_cols) > 4:
            raise ValueError("regime_correlation_table currently supports up to 4 assets.")

        df_m = self.df_m
        missing = [c for c in asset_cols if c not in df_m.columns]
        if missing:
            raise KeyError(f"Missing asset column(s) in df_m: {missing}")

        blocks = []
        for state in range(self.n_states):
            d = df_m.loc[df_m["state"] == state, asset_cols].copy()
            corr = d.corr()

            rows = []
            for i, row_name in enumerate(asset_cols):
                row = {
                    "model": self.model_name,
                    "regime": self.regime_names[state],
                    "matrix_row": row_name,
                }
                for j, col_name in enumerate(asset_cols):
                    row[col_name] = float(corr.loc[row_name, col_name]) if j <= i else np.nan
                rows.append(row)

            blocks.append(pd.DataFrame(rows))

        return pd.concat(blocks, ignore_index=True)

    def regime_correlation_table(self, asset_cols: list[str]) -> pd.DataFrame:
        """
        Build a lower-triangular correlation table per regime for up to 4 assets.

        Parameters
        ----------
        asset_cols : list[str]
            Excess-log return columns to include, e.g.
            ["ExcessLog^SP500TR", "ExcessLogLT09TRUU", "ExcessLogXAU"]

        Returns
        -------
        pd.DataFrame
            Long-but-readable table with one block per regime and one row per
            lower-triangular matrix row.
        """
        if self.df_m is None:
            raise ValueError("Call .fit(df_m, out) first.")
        if not asset_cols:
            raise ValueError("asset_cols cannot be empty.")
        # if len(asset_cols) > 4:
        #     raise ValueError("regime_correlation_table currently supports up to 4 assets.")

        df_m = self.df_m
        missing = [c for c in asset_cols if c not in df_m.columns]
        if missing:
            raise KeyError(f"Missing asset column(s) in df_m: {missing}")

        blocks = []
        for state in range(self.n_states):
            d = df_m.loc[df_m["state"] == state, asset_cols].copy()
            corr = d.corr()

            rows = []
            for i, row_name in enumerate(asset_cols):
                row = {
                    "model": self.model_name,
                    "regime": self.regime_names[state],
                    "matrix_row": row_name,
                }
                for j, col_name in enumerate(asset_cols):
                    row[col_name] = float(corr.loc[row_name, col_name]) if j <= i else np.nan
                rows.append(row)

            block = pd.DataFrame(rows)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)


    def out_long(self) -> pd.DataFrame:
        """
        Converts `out` (metrics x regimes) into tidy long format:
          model | metric | regime | value
        """
        if self.out is None:
            raise ValueError("No `out` stored. Call .fit(df_m, out=out) first.")
        tmp = self.out.reset_index().rename(columns={"index": "metric"})
        long = tmp.melt(id_vars="metric", var_name="regime", value_name="value")
        long.insert(0, "model", self.model_name)
        return long
    



def hmm_persistence_report(model, order_old: list[int] | None = None, regime_names: list[str] | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - p_ii: self-transition prob
      - expected_duration: 1 / (1 - p_ii)  (in periods; here: months)
    Also returns the transition matrix as a DataFrame.

    If `order_old` is provided, rows/columns are reordered to match the
    post-processed regime order. If `regime_names` is provided, those names are
    used as index/column labels.
    """
    T = np.asarray(model.transmat_, dtype=float)
    k = T.shape[0]

    if order_old is not None:
        T = T[np.ix_(order_old, order_old)]

    labels = regime_names if regime_names is not None else [f"state{i}" for i in range(k)]

    trans = pd.DataFrame(T, index=labels, columns=labels)

    p_ii = np.diag(T)
    # guard against division by 0 when p_ii ~ 1
    expected = np.where(1 - p_ii > 1e-12, 1.0 / (1.0 - p_ii), np.inf)

    summary = pd.DataFrame({
        "p_ii": p_ii,
        "expected_duration_months": expected,
    }, index=labels)

    return trans, summary



def realized_chatter_stats(df_m: pd.DataFrame) -> dict:
    """
    Uses the hard 'state' sequence in df_m to compute:
      - number of switches
      - switch rate
      - average run length (months)
      - median run length
    """
    if "state" not in df_m.columns:
        raise KeyError("df_m must contain 'state' column.")

    s = df_m["state"].astype(int).to_numpy()
    if len(s) < 2:
        return {"n_obs": len(s), "n_switches": 0, "switch_rate": np.nan, "avg_run": np.nan, "median_run": np.nan}

    switches = np.sum(s[1:] != s[:-1])
    switch_rate = switches / (len(s) - 1)

    # run lengths
    runs = []
    run_len = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)

    runs = np.array(runs, dtype=int)

    return {
        "n_obs": len(s),
        "n_switches": int(switches),
        "switch_rate": float(switch_rate),
        "avg_run": float(runs.mean()),
        "median_run": float(np.median(runs)),
        "min_run": int(runs.min()),
        "max_run": int(runs.max()),
    }

def diagnose_hmm(
    name: str,
    model,
    df_m: pd.DataFrame,
    return_tables: bool = False,
    order_old: list[int] | None = None,
    regime_names: list[str] | None = None,
):
    trans, summ = hmm_persistence_report(model, order_old=order_old, regime_names=regime_names)
    chat = realized_chatter_stats(df_m)

    print(f"=== {name} ===")
    print("Transition matrix (transmat_):")
    print(trans)
    print("Expected duration (months): 1/(1 - p_ii)")
    print(summ)

    print("Realized state-sequence chatter stats:")
    for k, v in chat.items():
        print(f"  {k}: {v}")

    if return_tables:
        chat_df = pd.DataFrame([chat])
        return trans, summ, chat_df
