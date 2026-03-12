import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from scipy import stats 



def plot_regime_dashboard_stack(
    models: list[tuple[str, "pd.DataFrame"]],
    *,
    figsize=(22, 18),
):
    """
    2 rows per model:
      - stacked regime probabilities (p_state*)
      - hard state (steps)
    """
    n = len(models)
    fig, axes = plt.subplots(2 * n, 1, figsize=figsize, sharex=True)

    if n == 1:
        axes = np.array([axes]).reshape(-1)

    for i, (title, df_m) in enumerate(models):
        ax_p = axes[2 * i]
        ax_s = axes[2 * i + 1]

        prob_cols = [c for c in df_m.columns if c.startswith("p_state")]
        if not prob_cols:
            raise ValueError(f"{title}: no p_state* columns found.")
        if "state" not in df_m.columns:
            raise ValueError(f"{title}: no 'state' column found.")

        probs = df_m[prob_cols].to_numpy().T  # shape (K, T)

        # stacked probabilities
        ax_p.stackplot(df_m.index, probs, labels=prob_cols)
        ax_p.set_title(f"{title} — regime probabilities (stacked)")
        ax_p.set_ylabel("Probability")
        ax_p.set_ylim(0, 1)
        ax_p.legend(loc="upper right", ncol=min(len(prob_cols), 4))

        # hard state
        ax_s.plot(df_m.index, df_m["state"], drawstyle="steps-post")
        ax_s.set_title(f"{title} — regime state")
        ax_s.set_ylabel("state")

    plt.tight_layout()
    plt.show()



def plot_regime_distribution_grid(
    panels: list[tuple[str, pd.DataFrame]],
    value_col: str = "ExcessLog^SP500TR",
    bins: int = 30,
    figsize: tuple[int, int] = (20, 14),
    density: bool = True,
    sharex: bool | str = "col",
    sharey: bool = False,
    add_kde: bool = True,
):
    """
    Dashboard-style grid of per-regime distributions.

    Parameters
    ----------
    panels : list of (model_name, df_m)
    value_col : str
        Column whose within-regime distribution is plotted.
    bins : int
        If int, adaptive binning is used:
          - Sturges for small samples
          - Freedman-Diaconis for larger samples
        If array-like, passed directly to matplotlib.
    add_kde : bool
        If True, overlays a Gaussian KDE curve when enough observations exist.
    sharex : bool | str
        Use "col" to share x-axis within each regime column.
    """
    if not panels:
        raise ValueError("`panels` cannot be empty.")

    n_models = len(panels)
    states_per_model: list[list[int]] = []
    max_states = 0

    for _, df_m in panels:
        if value_col not in df_m.columns:
            raise KeyError(f"'{value_col}' not found in one of the panel dataframes.")
        if "state" not in df_m.columns:
            raise KeyError("Each dataframe must contain a 'state' column.")

        states = sorted(pd.Series(df_m["state"]).dropna().astype(int).unique().tolist())
        states_per_model.append(states)
        max_states = max(max_states, len(states))

    fig, axes = plt.subplots(
        n_models,
        max_states,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    for i, (model_name, df_m) in enumerate(panels):
        states = states_per_model[i]

        for j in range(max_states):
            ax = axes[i, j]

            if j >= len(states):
                ax.axis("off")
                continue

            state_id = states[j]
            s = pd.to_numeric(
                df_m.loc[df_m["state"] == state_id, value_col],
                errors="coerce",
            ).dropna()

            if s.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"Regime {state_id}")
                continue

            if isinstance(bins, int):
                hist_bins = np.histogram_bin_edges(
                    s,
                    bins=10 if len(s) < 40 else 20 if len(s) <= 100 else 50,
                    # bins="fd"
                )
            else:
                hist_bins = bins

            ax.hist(
                s,
                bins=hist_bins,
                density=density,
                alpha=0.75,
                color="skyblue",
                edgecolor="black",
            )

            if add_kde and len(s) >= 3 and s.nunique() > 1:
                x_grid = np.linspace(s.min(), s.max(), 250)
                kde = stats.gaussian_kde(s)
                ax.plot(x_grid, kde(x_grid), linewidth=2)

            ax.axvline(s.mean(), linestyle="--", linewidth=1)
            ax.axvline(0.0, linestyle=":", linewidth=1)

            if i == 0:
                ax.set_title(f"Regime {state_id}")

            if j == 0:
                ax.set_ylabel(model_name)

            stats_text = (
                f"n={len(s)}\n"
                f"mean={100 * s.mean():.2f}%\n"
                f"std={100 * s.std(ddof=1):.2f}%\n"
                f"skew={s.skew():.2f}\n"
                f"kurt={s.kurt() + 3:.2f}"
            )
            ax.text(
                0.03,
                0.97,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", alpha=0.15),
                fontsize=9,
            )

            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

    fig.suptitle(f"Distribution grid by model and regime: {value_col}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()
    return fig, axes


def plot_results_dashboard(results) -> None:
    panels = [(res.spec.label, res.pp.df_m) for res in results]
    plot_regime_dashboard_stack(panels, figsize=(26, 6 * max(len(results), 1)))


def plot_requested_distributions(results) -> None:
    asset_to_panels = {}

    for res in results:
        for asset_col in res.spec.dist_assets:
            asset_to_panels.setdefault(asset_col, []).append((res.spec.label, res.pp.df_m))

    for asset_col, panels in asset_to_panels.items():
        plot_regime_distribution_grid(
            panels,
            value_col=asset_col,
            bins=70,
            figsize=(20, 14),
            add_kde=True,
        )