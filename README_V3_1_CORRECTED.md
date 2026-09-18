# Macro Sentinel V3.1 — Corrected Integration Package

## Replace existing files
- `main.py`
- `app.py`
- `regime_config.json`
- `adaptive_strategy_layer.py`
- `.github/workflows/sentinel_check.yml`

## Add new files
- `predictive_risk_engine.py`
- `predictive_audit.py`

## Leave unchanged
- `regime_engine.py`
- `backtest_regimes.py`
- `backtest_report.md`
- `find_better_strategies.py`
- `stress_test_analysis.py`
- `stress_test_results.json`
- `strategy_research.py`
- `strategy_stress_test.py`
- `requirements.txt`
- `cms_history.csv`

There is exactly one workflow file: `.github/workflows/sentinel_check.yml`.
Do not add `sentinel_check_predictive.yml` as a second workflow. The predictive workflow content is already installed under the canonical filename above.

V3.1 remains fail-closed and point-in-time. Predictive adjustments remain in WARMUP until the configured matured-observation and bin-sample thresholds are met.
