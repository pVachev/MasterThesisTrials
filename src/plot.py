import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


    