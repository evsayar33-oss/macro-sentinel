# Macro Sentinel V3 — Adaptive Cross-Asset Strategy Layer

This package integrates the V3 strategy layer into the existing Macro Sentinel without replacing the existing regime/event engine.

## Replace / add

Replace existing:
- `main.py`
- `regime_config.json`
- `app.py`
- `.github/workflows/sentinel_check.yml`

Add to repository root:
- `adaptive_strategy_layer.py`
- `strategy_research.py`
- `strategy_stress_test.py`

Do not upload any `mnt/` folder.

## Production flow

`regime_engine.py` continues to classify the macro regime and event state.
`adaptive_strategy_layer.py` now owns production cross-asset risk budgeting and capital preservation.

The layer:
- scores equity, gold, bond, commodity and crypto on trailing momentum/trend/drawdown/volatility quality;
- scales total deployed risk by regime and forecast volatility;
- detects synchronized cross-asset selloffs independently of VIX spikes;
- moves to at least 90% cash in hard synchronized stress;
- compresses risk during broad stress;
- stages re-entry after a stress episode rather than instantly returning to full risk;
- does not boost commodity on oil pressure alone; only confirmed oil events can change commodity priority;
- never uses leverage;
- always normalizes to 100%.

## Research

`strategy_research.py` is research-only and is not called by the daily production workflow. It uses a 5-session research rebalance cadence to remain computationally tractable and requires materially longer real history before it reports real-data walk-forward candidates.

## Validation

The daily GitHub Actions workflow compiles the new strategy module and executes `strategy_stress_test.py` before validating and committing the trusted history update.
