from dataclasses import dataclass
import pandas as pd

from src.load import diff_data, prepare_data
from src.hmm import hmm_sweep_seeds
from src.postprocess import RegimePostProcessor, diagnose_hmm

@dataclass
class GlobalRunConfig:
    n_states: int = 3
    cov_type: str = "full"
    seeds: range = range(1, 41)
    rf_col: str = "RF"
    rf_mode: str = "simple_return_monthly_decimal"
    start_date: str | None = None
    make_dashboard: bool = True
    make_distribution_plots: bool = True
    export_excel: bool = True
    output_file: str = "hmm_regime_results.xlsx"


@dataclass
class ModelSpec:
    code: str
    risky_assets: list[str]
    rf_col: str

    @property
    def tickers(self) -> list[str]:
        return self.risky_assets + [self.rf_col]

    @property
    def label(self) -> str:
        if len(self.risky_assets) > 1:
            return f"Model {self.code} ({' + '.join(self.risky_assets[1:])})"
        return f"Model {self.code} ({self.risky_assets[0]})"

    @property
    def key_col(self) -> str:
        return f"ExcessLog{self.risky_assets[0]}"

    @property
    def summary_asset(self) -> str | None:
        if len(self.risky_assets) >= 2:
            return f"ExcessLog{self.risky_assets[1]}"
        return None

    @property
    def dist_assets(self) -> list[str]:
        return [f"ExcessLog{t}" for t in self.risky_assets]

    @property
    def corr_assets(self) -> list[str]:
        return [f"ExcessLog{t}" for t in self.risky_assets[:4]]


@dataclass
class ModelRunResult:
    spec: ModelSpec
    x: pd.DataFrame
    sweep: pd.DataFrame
    best_seed: int
    out: pd.DataFrame
    df_m: pd.DataFrame
    model: object
    pp: RegimePostProcessor
    regime_summary: pd.DataFrame | None
    trans: pd.DataFrame
    duration: pd.DataFrame
    chatter: pd.DataFrame
    corr_table: pd.DataFrame | None = None


def build_model_specs(model_asset_sets: list[list[str]], rf_col: str) -> list[ModelSpec]:
    specs = []
    for i, risky_assets in enumerate(model_asset_sets):
        code = chr(ord("A") + i)
        specs.append(ModelSpec(code=code, risky_assets=risky_assets, rf_col=rf_col))
    return specs


def build_model_input(
    raw_df: pd.DataFrame,
    spec: ModelSpec,
    monthly_tickers: list[str],
    rf_mode: str,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_model = diff_data(
        raw_df,
        spec.tickers,
        rf_col=spec.rf_col,
        monthly_cols=monthly_tickers,
        rf_mode=rf_mode,
    )
    x_model = prepare_data(
    df_model,
    spec.tickers,
    rf_col=spec.rf_col,
    start_date=start_date,
)
    return df_model, x_model


def run_one_model(
    spec: ModelSpec,
    x_model: pd.DataFrame,
    cfg: GlobalRunConfig,
) -> ModelRunResult:
    sweep, best_seed, out, df_m, model = hmm_sweep_seeds(
        x_model,
        cfg.n_states,
        spec.tickers,
        cfg.cov_type,
        seeds=cfg.seeds,
    )

    pp = RegimePostProcessor(spec.label, cfg.n_states, key_col=spec.key_col).fit(df_m, out)

    regime_summary = None
    if spec.summary_asset is not None:
        regime_summary = pp.regime_summary(spec.summary_asset)

    trans, duration, chatter = diagnose_hmm(
        spec.label,
        model,
        pp.df_m,
        return_tables=True,
        order_old=pp.order_old,
        regime_names=pp.regime_names,
    )

    corr_table = None
    if spec.corr_assets:
        corr_table = pp.regime_correlation_table(spec.corr_assets)

    return ModelRunResult(
        spec=spec,
        x=x_model,
        sweep=sweep,
        best_seed=best_seed,
        out=out,
        df_m=df_m,
        model=model,
        pp=pp,
        regime_summary=regime_summary,
        trans=trans,
        duration=duration,
        chatter=chatter,
        corr_table=corr_table,
    )