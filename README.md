# A Regime-Switching Framework for Tactical Sector Tilts under Higher-Moment Investor Preferences

**Preslav Vachev | emlyon business school | Master Thesis, May 2026**
**Supervisor: Professor Bertrand Tavin**

## Overview

This codebase implements a rolling Hidden Markov Model (HMM) that conditions tactical
sector ETF and gold allocation on detected macro regimes. A three-state HMM is estimated
on monthly excess log returns of the S&P 500 TR and Bloomberg US Treasury 7-10 Year index.
Regime probabilities are used to score candidate satellite tilts via a Taylor-expanded
expected utility function, accommodating MV, MVS, and MVK investor types. The strategy
operates as a core-plus-satellite overlay on a fixed 60/40 benchmark.

## Repository Structure

```
main.py                      — entry point: HMM estimation, A1 backtest, sensitivity grid
generate_thesis_figures_1.py — generates all thesis figures from Excel outputs
model_selection.py           — K=2,3,5 model selection diagnostic (run once)
requirements.txt             — Python dependencies
ThesisDoc/                   — LaTeX thesis source and figures

src/
├── allocation_backtest.py   — rolling window backtest runners + performance analytics
├── allocation_config.py     — dataclasses: InvestorPreferenceConfig, SatelliteSpec, etc.
├── allocation_export.py     — Excel export for backtest and HMM results
├── allocation_moments.py    — regime-conditional moment computation + calibration utils
├── allocation_plot.py       — allocation dashboards and distribution plots
├── allocation_regime.py     — HMM probability extraction and regime ordering
├── allocation_scoring.py    — candidate enumeration and tilt selection
├── hmm.py                   — GaussianHMM wrapper
├── load.py                  — data loading and differencing
├── plot.py                  — HMM diagnostics and price level plots
├── postprocess.py           — regime post-processing utilities
├── runner.py                — model run orchestration
└── transform.py             — data cleaning and feature construction

data/
├── getData.py               — Bloomberg/Yahoo data download scripts
└── raw/                     — raw CSV files (RF rate, bond weights)
```

## Getting Started

```bash
pip install -r requirements.txt
python main.py
```

## Run Flags (top of main.py)

| Flag | Default | Description |
|---|---|---|
| `RUN_ALLOCATION` | `True` | Run A1 honest train/test backtest |
| `EXPORT_ALLOCATION` | `True` | Export backtest results to Excel |
| `RUN_EXPANDING_WINDOW` | `False` | Run full sensitivity grid |
| `ONLY_SLEEVE` | `None` | Filter grid to one sleeve size |
| `ONLY_FLOOR` | `None` | Filter grid to one floor value |
| `ONLY_INVESTORS` | `None` | Filter grid to subset of investor types |

## Primary Configuration

- **Model:** 3-state HMM, full covariance, rolling 60-month window
- **Satellite universe:** 9 SPDR sector ETFs + Gold (XAU)
- **Primary config:** 45% sleeve, score improvement floor = 0.001
- **Investor types:** MV, MVS_cons, MVS, MVK
- **Benchmark:** 60/40 S&P 500 TR / LT09TRUU
- **Backtest period:** February 2004 – January 2026 (264 months)
- **Transaction costs:** 5 bps per side
