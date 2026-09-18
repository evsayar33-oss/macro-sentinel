"""
Research / comparison utility for Macro Sentinel.

Important:
- Uses the live engine's configuration as the only allocation source of truth.
- Does not contain a second, hard-coded copy of regime allocations.
- Results are descriptive research outputs, not guarantees of future performance.
"""

import numpy as np
import pandas as pd

from backtest_regimes import generate_synthetic_macro_history
from regime_engine import MacroRegimeEngine


engine = MacroRegimeEngine()
raw_data = generate_synthetic_macro_history()
prepared = engine.prepare_indicators(raw_data)
classified = engine.run_time_series(prepared)


# Asset return proxies used by the synthetic research dataset.
spx_ret = classified["spx"].pct_change().fillna(0.0)
bond_ret = classified["ust10y"].pct_change().fillna(0.0)
cash_ret = (classified["dgs2"] / 100.0) / 252.0
gold_ret = classified["gold"].pct_change().fillna(0.0)
oil_ret = classified["oil"].pct_change().fillna(0.0)
btc_ret = classified["btc"].pct_change().fillna(0.0)

ann_factor = 252.0
n_years = len(classified) / ann_factor


def evaluate_ret_series(ret_series: pd.Series, name: str) -> dict:
    cum = (1.0 + ret_series).cumprod()
    total = (cum.iloc[-1] - 1.0) * 100.0
    annual = (cum.iloc[-1] ** (1.0 / max(n_years, 0.1)) - 1.0) * 100.0
    vol = ret_series.std() * np.sqrt(ann_factor) * 100.0
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = abs(dd.min()) * 100.0
    sharpe = (annual - 3.0) / max(vol, 0.01)
    calmar = annual / max(max_dd, 0.01)
    return {
        "Name": name,
        "Annual Return (%)": round(float(annual), 2),
        "Annual Vol (%)": round(float(vol), 2),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Max Drawdown (%)": round(float(max_dd), 2),
        "Calmar Ratio": round(float(calmar), 2),
        "Total Return (%)": round(float(total), 2),
    }


# Current Macro Sentinel allocation path. No duplicate regime map exists here;
# classified regime weights already include confirmed event overlays.
w_csh = (classified["regime_cash_weight"] / 100.0).shift(1).fillna(0.35)
w_gld = (classified["regime_gold_weight"] / 100.0).shift(1).fillna(0.20)
w_bnd = (classified["regime_bond_weight"] / 100.0).shift(1).fillna(0.20)
w_eq = (classified["regime_eq_weight"] / 100.0).shift(1).fillna(0.15)
w_cmd = (classified["regime_commodity_weight"] / 100.0).shift(1).fillna(0.05)
w_crp = (classified["regime_crypto_weight"] / 100.0).shift(1).fillna(0.05)
strat_current = (
    w_csh * cash_ret
    + w_gld * gold_ret
    + w_bnd * bond_ret
    + w_eq * spx_ret
    + w_cmd * oil_ret
    + w_crp * btc_ret
)

results = [evaluate_ret_series(strat_current, "Macro Sentinel Dynamic")]

# Reference portfolios are deliberately static and explicitly labeled as such.
taleb = 0.85 * cash_ret + 0.10 * gold_ret + 0.05 * btc_ret
results.append(evaluate_ret_series(taleb, "Reference: 85/10/5 Cash-Gold-BTC"))

defensive = 0.35 * cash_ret + 0.20 * gold_ret + 0.20 * bond_ret + 0.15 * spx_ret + 0.05 * oil_ret + 0.05 * btc_ret
results.append(evaluate_ret_series(defensive, "Reference: Defensive Shield"))

artemis = 0.25 * spx_ret + 0.25 * cash_ret + 0.20 * gold_ret + 0.15 * bond_ret + 0.10 * oil_ret + 0.05 * btc_ret
results.append(evaluate_ret_series(artemis, "Reference: Artemis Dragon"))

summary = pd.DataFrame(results)
cols = [
    "Name", "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio",
    "Max Drawdown (%)", "Calmar Ratio", "Total Return (%)"
]
print(summary[cols].to_string(index=False))

# Basic invariant check for the engine path.
weights = classified[[
    "regime_cash_weight", "regime_gold_weight", "regime_bond_weight",
    "regime_eq_weight", "regime_commodity_weight", "regime_crypto_weight"
]].sum(axis=1)
assert np.allclose(weights.values, 100.0, atol=0.05)
print("\nAllocation invariant: PASS")
