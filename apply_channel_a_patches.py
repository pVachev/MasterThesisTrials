"""
apply_channel_a_patches.py — surgical Channel A wiring, CRLF-preserving.

Two-phase and transactional:
  1) every file is auto-located (repo root or src/, regardless of where the
     script sits or is run from) and every anchor is asserted to occur
     exactly once;
  2) files are written ONLY after all edits across all five files validate.
A failure at any anchor therefore leaves the repo completely untouched.

Files are read/written as UTF-8 explicitly (locale-independent).
Run from the repo root:   python src/apply_channel_a_patches.py
(or from inside src/ — both work). Refuses already-patched files, so it
cannot double-apply.
"""
import os
from pathlib import Path

N = "\r\n"

_HERE = Path(__file__).resolve().parent
_CANDIDATE_DIRS: list[Path] = []
for _d in (Path.cwd(), Path.cwd() / "src", _HERE, _HERE / "src",
           _HERE.parent, _HERE.parent / "src"):
    _r = _d.resolve()
    if _r.is_dir() and _r not in _CANDIDATE_DIRS:
        _CANDIDATE_DIRS.append(_r)

_STAGED: list[tuple[Path, str, int]] = []


def _locate(fname: str) -> Path:
    hits: list[Path] = []
    for d in _CANDIDATE_DIRS:
        p = d / fname
        if p.is_file() and p not in hits:
            hits.append(p)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        searched = "\n  ".join(str(d / fname) for d in _CANDIDATE_DIRS)
        raise FileNotFoundError(f"{fname} not found; searched:\n  {searched}")
    dup = "\n  ".join(str(h) for h in hits)
    raise RuntimeError(
        f"{fname} found in MULTIPLE locations — remove/rename the stale copy first:\n  {dup}"
    )


def patch(fname: str, edits: list[tuple[str, str]]) -> None:
    """Validate and stage edits; nothing is written until every file passes."""
    p = _locate(fname)
    with open(p, "r", encoding="utf-8", newline="") as fh:
        content = fh.read()
    for old, new in edits:
        cnt = content.count(old)
        assert cnt == 1, f"{p}: anchor found {cnt}x (expected 1):\n{old[:90]!r}"
        content = content.replace(old, new)
    _STAGED.append((p, content, len(edits)))
    print(f"validated {p} ({len(edits)} edits)")


# ── allocation_config.py ─────────────────────────────────────────────
old = "    core_split_bond_bounds: tuple[float, float] = (0.20, 0.60)" + N
new = old + N.join([
    "",
    "    # ── Channel A: vol-sharpened bear probability (sizing-only overlay) ──",
    "    # p_eff = min(vol_p_cap, p_bear + vol_eta * max(0, z_t - vol_z_star)),",
    "    # z_t = month-t downside realized vol / trailing 60m median (src/vol_signal.py).",
    "    # Consumed ONLY by compute_regime_conviction_weights via p_bear_override;",
    "    # Stage-1 selection and the predictive probability row are untouched.",
    "    vol_signal_enabled: bool = False",
    "    vol_eta: float = 0.3",
    "    vol_z_star: float = 2.0",
    "    vol_p_cap: float = 1.0",
    "",
])
patch("allocation_config.py", [(old, new)])

# ── allocation_scoring.py ────────────────────────────────────────────
edits = []

old = ("    satellite_specs: list," + N +
       "    predictive_probabilities_row: pd.Series," + N +
       ") -> tuple[dict[str, float], dict[str, float]]:")
new = ("    satellite_specs: list," + N +
       "    predictive_probabilities_row: pd.Series," + N +
       "    p_bear_override: float | None = None," + N +
       ") -> tuple[dict[str, float], dict[str, float]]:")
edits.append((old, new))

old = ("    bear_regime = predictive_probabilities_row.index[0]" + N +
       "    p_bear = float(predictive_probabilities_row.iloc[0])" + N)
new = (old +
       "    if p_bear_override is not None:" + N +
       "        # Channel A: vol-sharpened effective bear probability (sizing only)." + N +
       "        p_bear = float(p_bear_override)" + N)
edits.append((old, new))

old = ("    satellite_specs: list | None = None," + N +
       "    core_weights_override: dict[str, float] | None = None," + N +
       ") -> tuple[TiltDecision, pd.DataFrame]:")
new = ("    satellite_specs: list | None = None," + N +
       "    core_weights_override: dict[str, float] | None = None," + N +
       "    p_bear_override: float | None = None," + N +
       ") -> tuple[TiltDecision, pd.DataFrame]:")
edits.append((old, new))

old = ("        scaled_weights, conviction_scalar = compute_regime_conviction_weights(" + N +
       "            base_satellite_weights=base_satellite_weights," + N +
       "            satellite_specs=satellite_specs," + N +
       "            predictive_probabilities_row=predictive_probabilities_row," + N +
       "        )")
new = ("        scaled_weights, conviction_scalar = compute_regime_conviction_weights(" + N +
       "            base_satellite_weights=base_satellite_weights," + N +
       "            satellite_specs=satellite_specs," + N +
       "            predictive_probabilities_row=predictive_probabilities_row," + N +
       "            p_bear_override=p_bear_override," + N +
       "        )")
edits.append((old, new))

old = ('            "base_satellite_weights": base_satellite_weights,' + N +
       '            "conviction_scalar": conviction_scalar,')
new = (old + N +
       '            "p_bear_effective": (' + N +
       '                float(p_bear_override)' + N +
       '                if p_bear_override is not None' + N +
       '                else float(predictive_probabilities_row.iloc[0])' + N +
       '            ),')
edits.append((old, new))

patch("allocation_scoring.py", edits)

# ── allocation_backtest.py ───────────────────────────────────────────
edits = []

old = "from src.corr_core_split import predictive_bond_equity_corr, corr_adjusted_core_split" + N
new = old + "from src.vol_signal import effective_bear_probability" + N
edits.append((old, new))

# EW signature
old = ("    store_candidate_scores: bool = False," + N +
       "    cash_sleeve_cfg=None," + N +
       ') -> "BacktestResult":')
new = ("    store_candidate_scores: bool = False," + N +
       "    cash_sleeve_cfg=None," + N +
       '    vol_z: "pd.Series | None" = None,' + N +
       ') -> "BacktestResult":')
edits.append((old, new))

# A1 signature
old = ("    store_candidate_scores: bool = True," + N +
       "    cash_sleeve_cfg=None," + N +
       ") -> tuple[BacktestResult, object]:")
new = ("    store_candidate_scores: bool = True," + N +
       "    cash_sleeve_cfg=None," + N +
       '    vol_z: "pd.Series | None" = None,' + N +
       ") -> tuple[BacktestResult, object]:")
edits.append((old, new))

VOL_BLOCK = N.join([
    "",
    "        # ── Channel A: vol-sharpened bear probability (sizing-only) ──",
    "        # p_eff feeds compute_regime_conviction_weights via p_bear_override.",
    "        # Stage-1 scoring consumes the untouched pred_row.",
    "        p_bear_override = None",
    '        z_t = float("nan")',
    '        if getattr(alloc_cfg, "vol_signal_enabled", False) and vol_z is not None:',
    '            z_t = float(vol_z.get(rebalance_date, float("nan")))',
    "            p_bear_override = effective_bear_probability(",
    "                float(pred_row.iloc[0]), z_t,",
    "                alloc_cfg.vol_eta, alloc_cfg.vol_z_star, alloc_cfg.vol_p_cap,",
    "            )",
    "",
])

# EW: after pred_row construction
old = "        pred_row = pd.Series(pred_vec, index=frozen.regime_names)" + N
new = old + VOL_BLOCK
edits.append((old, new))

# A1: after pred_row lookup
old = "        pred_row = pred_test.loc[rebalance_date]" + N
new = old + VOL_BLOCK
edits.append((old, new))

# EW selector call (unique via core_weights_override)
old = ("            satellite_specs=satellite_specs," + N +
       "            core_weights_override=core_override," + N +
       "        )")
new = ("            satellite_specs=satellite_specs," + N +
       "            core_weights_override=core_override," + N +
       "            p_bear_override=p_bear_override," + N +
       "        )")
edits.append((old, new))

# A1 selector call (unique: no core override, followed by cash sleeve block)
old = ("            rebalance_date=rebalance_date," + N +
       "            satellite_specs=satellite_specs," + N +
       "        )")
new = ("            rebalance_date=rebalance_date," + N +
       "            satellite_specs=satellite_specs," + N +
       "            p_bear_override=p_bear_override," + N +
       "        )")
edits.append((old, new))

# EW metadata: log z_t (wide spacing unique to EW block)
old = '            "realized_date":                       realized_date,'
new = old + N + '            "vol_z":                               z_t,'
edits.append((old, new))

patch("allocation_backtest.py", edits)

# ── main.py ──────────────────────────────────────────────────────────
edits = []

old = "from src.allocation_export import export_allocation_backtest_to_excel, export_model_results_to_excel" + N
new = old + "from src.vol_signal import build_monthly_vol_signal" + N
edits.append((old, new))

old = ("CASH_SLEEVE = CashSleeveConfig(" + N +
       "    enabled=False," + N +
       "    activation_threshold=0.55," + N +
       "    max_cash_weight=0.40," + N +
       '    rf_ticker="RF",' + N +
       ")" + N)
new = old + N.join([
    "",
    "# ── Channel A: vol-sharpened bear probability (sizing-only overlay) ────────",
    "# Flip VOL_SIGNAL_ENABLED only after vol_signal_study.py validates locally.",
    "# Empirical z quantiles 2004+: 1.5 ~ 74th pct, 2.0 ~ 85th pct (Z_Distribution).",
    "VOL_SIGNAL_ENABLED = False",
    "VOL_ETA            = 0.3",
    "VOL_Z_STAR         = 2.0",
    'DAILY_EQUITY_CSV   = "data/raw/^SP500TR.csv"',
    "",
])
edits.append((old, new))

old = ("    ew_cfg = ExpandingWindowConfig(" + N +
       "        burn_in_periods=60," + N +
       "        refit_every_n_periods=1," + N +
       "        verbose=True," + N +
       "        verbose_every=12," + N +
       "        rolling_window=60," + N +
       "    )" + N)
new = old + N.join([
    "",
    "    vol_z = None",
    "    if VOL_SIGNAL_ENABLED:",
    '        vol_z = build_monthly_vol_signal(DAILY_EQUITY_CSV)["z"]',
    '        print(f"vol signal: {vol_z.notna().sum()} usable months, "',
    '              f"eta={VOL_ETA}, z*={VOL_Z_STAR}")',
    "",
])
edits.append((old, new))

old = ("            core_split_kappa=0.2," + N +
       "        )")
new = ("            core_split_kappa=0.2," + N +
       "            vol_signal_enabled=VOL_SIGNAL_ENABLED," + N +
       "            vol_eta=VOL_ETA," + N +
       "            vol_z_star=VOL_Z_STAR," + N +
       "        )")
edits.append((old, new))

old = '        equity_displ_key = "equity_only" if alloc_cfg_run.equity_only_displacement == True else "core_repl"'
new = (old + N +
       '        vol_tag = f"_volA_e{VOL_ETA}_z{VOL_Z_STAR}".replace(".", "") if VOL_SIGNAL_ENABLED else ""')
edits.append((old, new))

old = '            tag = f"EW_{sleeve_tag}_floor{floor_tag}_{inv_key}_{equity_displ_key}"'
new = '            tag = f"EW_{sleeve_tag}_floor{floor_tag}_{inv_key}_{equity_displ_key}{vol_tag}"'
edits.append((old, new))

old = ("                store_candidate_scores=False," + N +
       "                cash_sleeve_cfg=CASH_SLEEVE," + N +
       "            )")
new = ("                store_candidate_scores=False," + N +
       "                cash_sleeve_cfg=CASH_SLEEVE," + N +
       "                vol_z=vol_z," + N +
       "            )")
edits.append((old, new))

patch("main.py", edits)

# ── engineering_metrics.py ───────────────────────────────────────────
edits = []

old = '    eq_only= str(cfg[("allocation_config","equity_only_displacement")]).strip().lower()=="true"' + N
new = (old +
       '    vol_on = str(cfg.get(("allocation_config","vol_signal_enabled"), "False")).strip().lower()=="true"' + N +
       '    vol_lab = (f"e{cfg[(\'allocation_config\',\'vol_eta\')]}/z{cfg[(\'allocation_config\',\'vol_z_star\')]}"' + N +
       '               if vol_on else "off")' + N)
edits.append((old, new))

old = "    return dict(path=path, inv=inv, sleeve=sleeve, floor=floor, tcbps=tcbps, eq_only=eq_only,"
new = "    return dict(path=path, inv=inv, sleeve=sleeve, floor=floor, tcbps=tcbps, eq_only=eq_only, vol=vol_lab,"
edits.append((old, new))

old = '        "displacement":"core-repl" if not r["eq_only"] else "equity-only",'
new = old + N + '        "vol":r["vol"],'
edits.append((old, new))

old = 'show=["investor","displacement",'
new = 'show=["investor","displacement","vol",'
edits.append((old, new))

old = "print(f\"  parsed {f.split('/')[-1]}: {row['investor']} {row['displacement']}"
new = "print(f\"  parsed {f.split('/')[-1]}: {row['investor']} {row['displacement']} vol={row['vol']}"
edits.append((old, new))

patch("engineering_metrics.py", edits)

# Commit phase: write every file to a sibling .tmp first, then atomically
# replace. A failure at ANY point leaves every original file intact.
_tmps: list[tuple[Path, Path, int]] = []
try:
    for _p, _content, _n in _STAGED:
        _tmp = _p.with_name(_p.name + ".channelA.tmp")
        with open(_tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(_content)
        _tmps.append((_tmp, _p, _n))
except BaseException:
    for _tmp, _p, _n in _tmps:
        _tmp.unlink(missing_ok=True)
    raise

for _tmp, _p, _n in _tmps:
    os.replace(_tmp, _p)
    print(f"wrote {_p} ({_n} edits)")

print("ALL PATCHES APPLIED")
