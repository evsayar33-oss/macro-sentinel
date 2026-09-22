# Macro Sentinel V3.2 — Installation / Change Map

This package is the V3.2 upgrade focused on the reported problem: data and event
states were not refreshing often enough and the strategy did not have a persistent,
multi-horizon market risk-appetite state.

## Replace / add

Replace these existing files:

- `main.py`
- `app.py`
- `adaptive_strategy_layer.py`
- `regime_config.json`
- `.github/workflows/sentinel_check.yml`
- `predictive_risk_engine.py`
- `predictive_audit.py`

Add this new file:

- `risk_appetite_engine.py`

## Leave unchanged

Do not replace:

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

## V3.2 behaviour

1. Four scheduled workflow refreshes per day, including weekends as monitor runs.
2. Weekend runs refresh 24/7 market/risk-appetite telemetry but keep the last trusted weekday allocation.
3. Streamlit first reads the latest public `cms_history.csv` directly from GitHub, with a local fallback.
4. Risk appetite combines 5/20/60/120/252/756-observation horizons and observable proxies for credit, volatility, dollar, real rates, liquidity, carry, growth-sensitive assets and breadth.
5. Persistent risk-on/off states require multi-day persistence; single-day noise does not automatically become a long-lived regime.
6. Large multi-factor / multi-horizon shifts can produce a major-event state.
7. Risk-on states can increase the bounded risk budget and tilt toward growth-sensitive assets; risk-off/tightening states compress risk and increase defensive weight. Hard stress, regime emergency and unknown-event guards remain authoritative.
8. Predictive V3.1 logic is retained with the dailyized 5/20 trading-day point-in-time fix.
9. No leverage is introduced and allocations continue to normalize to 100%.

## Important runtime note

The new workflow does not guarantee that GitHub Actions itself will start exactly on
schedule; GitHub can delay scheduled runs. The increased cadence and the explicit
freshness/age checks are intended to make missed refreshes visible rather than silent.

## Final V3.2 corrections
- CRISIS risk-appetite multiplier now remains 0.30 end-to-end instead of being clipped by the strategy-layer lower bound.
- Major-event detection now uses genuine underlying factor shock Z-scores; bounded [-1,1] appetite factors are no longer incorrectly compared with a Z threshold.
- The configured `major_change_z` threshold is now used directly by the multi-horizon shift detector.
