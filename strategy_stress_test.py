"""Targeted regression tests for the v3 adaptive strategy layer."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from adaptive_strategy_layer import AdaptiveStrategyLayer


def make_base(n=900, seed=42):
    rng=np.random.default_rng(seed); idx=pd.date_range("2022-01-03",periods=n,freq="B")
    def p(mu,vol): return 100*np.cumprod(1+mu+rng.normal(0,vol,n))
    return pd.DataFrame({
        "spx":p(.0006,.0015), "gold":p(.00045,.0012),
        "ust10y":4.0+np.cumsum(rng.normal(-.003,.01,n)),
        "oil":p(.00045,.0020), "brent":p(.00042,.0020),
        "copper":p(.0004,.0018), "silver":p(.00035,.0020),
        "btc":p(.0008,.0040),
        "vix_percentile_252":np.clip(25+rng.normal(0,6,n),5,55),
        "ndl_z":rng.normal(.15,.25,n),
    })


def main():
    layer=AdaptiveStrategyLayer()
    normal=make_base()
    normal["confirmed_regime_id"]=5
    normal["oil_event_active"]=False
    normal["oil_pressure_score"]=0.0
    normal_result=layer.allocate(normal)

    crash=normal.copy()
    # Synchronized multi-asset liquidation without relying on a VIX spike.
    for c in ["spx","gold","oil","brent","copper","silver","btc"]:
        crash.loc[crash.index[-45:],c] *= np.linspace(1.0,0.55,45)
    crash["vix_percentile_252"]=50.0
    crash["ndl_z"]=0.0
    crash["confirmed_regime_id"]=0
    crash_result=layer.allocate(crash)

    for result in (normal_result, crash_result):
        total=sum(result["weights"].values())
        assert abs(total-100.0)<1e-8, total
        assert min(result["weights"].values())>=-1e-9

    assert crash_result["stress_broad"] or crash_result["stress_hard"]
    assert crash_result["weights"]["cash"] >= 70.0
    assert crash_result["risk_budget_final"] <= 0.30
    print("normal:", normal_result["strategy_mode"], normal_result["weights"])
    print("crash:", crash_result["strategy_mode"], crash_result["weights"])
    print("PASS")

if __name__ == "__main__":
    main()
