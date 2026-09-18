# Macro Sentinel V3.1 — Predictive Risk / Opportunity Layer

## Files
- `predictive_risk_engine.py` — point-in-time conditional risk/opportunity estimator.
- `predictive_audit.py` — daily matured-outcome audit and CSV persistence.
- `adaptive_strategy_layer.py` — drop-in replacement for the current strategy layer.
- `sentinel_check_predictive.yml` — full replacement for `.github/workflows/sentinel_check.yml`.
- `predictive_config_fragment.json` — optional config block; defaults are embedded in code, but adding this block to `regime_config.json` makes the policy explicit.

## Operational contract
1. No future outcome may change today's allocation.
2. Predictive adaptation remains in WARMUP until at least 60 matured observations and 20 observations in both the active risk and opportunity score bins.
3. Bayesian shrinkage keeps probabilities conservative during the early calibrated period.
4. The predictive layer can reduce risk, increase risk only within a small bound, or impose an immediate deterioration cap.
5. Hard/broad cross-asset stress and cash-preservation rules still take precedence.
6. `predictive_audit.py` updates only the latest history row with diagnostics after the live decision has already been made.

## What gets measured automatically
- 5-day probability of broad cross-asset loss.
- Conditional expected 5-day median asset return.
- 20-day probability that at least one available risk sleeve exceeds the opportunity threshold.
- Conditional expected 20-day top-sleeve return.
- Trend-deterioration score.
- Sample size and calibration confidence.

Until the warm-up thresholds are reached, the predictive metrics are diagnostic only and do not alter the live allocation.
