"""
Macro Sentinel V3.1 — daily predictive audit persistence.

Reads cms_history.csv, canonicalizes intraday history to one observation per
calendar day inside the predictive engine, evaluates only matured 5/20 trading-
day outcomes, and writes predictive diagnostics onto the current last raw row.
It never changes today's allocation and never feeds future outcomes backward
into a past decision.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from predictive_risk_engine import PredictiveRiskEngine

HISTORY_FILE = "cms_history.csv"
CONFIG_FILE = "regime_config.json"


def _load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"predictive_audit=config_error={str(exc)[:160]}")
        return {}


def _num(v: Any, default=np.nan) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def run_audit() -> Dict[str, Any]:
    if not os.path.exists(HISTORY_FILE):
        print("predictive_audit=WARMUP reason=cms_history.csv not found")
        return {"status": "NO_HISTORY"}

    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        print(f"predictive_audit=WARMUP reason=history_read_error:{str(exc)[:160]}")
        return {"status": "NO_HISTORY"}

    if df.empty:
        print("predictive_audit=WARMUP reason=empty history")
        return {"status": "NO_HISTORY"}

    predictor = PredictiveRiskEngine(_load_config())
    latest = df.iloc[-1]
    risk_score = _num(latest.get("strategy_stress_score"), 0.5)
    opp_score = _num(latest.get("strategy_opportunity_score"), 0.5)
    current_timestamp = latest.get("date")

    # Exclude the current raw row from training/forward-outcome history. The
    # engine then collapses any earlier intraday rows into one daily observation.
    past = df.iloc[:-1].copy()
    estimate = predictor.evaluate(
        past,
        risk_score,
        opp_score,
        current_timestamp=current_timestamp,
    )

    fields = {
        "predictive_status": estimate.status,
        "predictive_confidence": round(float(estimate.confidence), 6),
        "predictive_matured_observations": int(estimate.matured_observations),
        "predictive_risk_bin_observations": int(estimate.risk_bin_observations),
        "predictive_opportunity_bin_observations": int(estimate.opportunity_bin_observations),
        "predictive_risk_probability_5d": round(float(estimate.risk_probability_5d), 6),
        "predictive_expected_median_return_5d": round(float(estimate.expected_median_return_5d), 6),
        "predictive_expected_loss_5d": round(float(estimate.expected_loss_5d), 6),
        "predictive_opportunity_probability_20d": round(float(estimate.opportunity_probability_20d), 6),
        "predictive_expected_opportunity_return_20d": round(float(estimate.expected_opportunity_return_20d), 6),
        "predictive_deterioration_score": round(float(estimate.deterioration_score), 6),
        "predictive_audit_reason": str(estimate.reason),
    }

    # Rebuild instead of scalar assignment. This avoids pandas dtype-upcast
    # problems in legacy CSV columns containing booleans, ints and floats.
    rows = df.to_dict(orient="records")
    rows[-1].update(fields)
    out = pd.DataFrame(rows)
    out.to_csv(HISTORY_FILE, index=False)

    print(f"predictive_status={estimate.status}")
    print(f"predictive_confidence={estimate.confidence:.3f}")
    print(f"predictive_matured_observations={estimate.matured_observations}")
    print(f"predictive_risk_probability_5d={estimate.risk_probability_5d:.3f}")
    print(f"predictive_opportunity_probability_20d={estimate.opportunity_probability_20d:.3f}")
    print(f"predictive_deterioration_score={estimate.deterioration_score:.3f}")
    return fields


if __name__ == "__main__":
    run_audit()
