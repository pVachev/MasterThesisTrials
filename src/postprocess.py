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
        key_col: str = "ExcessLogSPY",
        regime_names: list[str] | None = None,
    ):
        self.model_name = model_name
        self.n_states = n_states
        self.key_col = key_col
        self.regime_names = regime_names or (["Bear", "Neutral", "Bull"] if n_states == 3 else [f"Regime {i}" for i in range(n_states)])

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