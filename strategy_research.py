"""
Macro Sentinel v3.0 — Walk-Forward Strategy Research

Searches for robust risk-budget configurations rather than optimizing one
historical path. This script deliberately refuses to call synthetic or short
history results "validated".

Usage
-----
python strategy_research.py --history cms_history.csv --out strategy_research_results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from adaptive_strategy_layer import AdaptiveStrategyLayer


@dataclass
class Metrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    max_drawdown: float
    calmar: float
    sharpe: float
    observations: int


def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").drop_duplicates("date").set_index("date")

    # Macro Sentinel persists point-in-time prices under explicit *_price names.
    # Map them into the strategy layer's canonical fields without inventing data.
    aliases = {
        "spx": "spx_price",
        "gold": "gold_price",
        "ust10y": "ust10y_yield",
        "oil": "oil_price",
        "brent": "brent_price",
        "copper": "copper_price",
        "silver": "silver_price",
        "btc": "btc_price",
    }
    for target, source in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = pd.to_numeric(df[source], errors="coerce")
    if "confirmed_regime_id" not in df.columns and "regime_id" in df.columns:
        df["confirmed_regime_id"] = pd.to_numeric(df["regime_id"], errors="coerce").fillna(0)
    return df


def portfolio_returns(df: pd.DataFrame, layer: AdaptiveStrategyLayer, rebalance_step: int = 5) -> pd.Series:
    """Evaluate strategy returns with a bounded rebalancing cadence.

    A 5-session rebalance prevents the research utility from recomputing large
    rolling windows at every single bar and keeps the research process tractable.
    Allocation at t applies to the next available session(s) until the next rebalance.
    """
    ret = layer._build_asset_returns(df)
    rows = []
    dates = []
    window = 180
    step = max(1, int(rebalance_step))
    for i in range(126, len(df) - 1, step):
        start = max(0, i + 1 - window)
        sub = df.iloc[start:i + 1]
        row = sub.iloc[-1]
        classified = pd.DataFrame([{
            "confirmed_regime_id": int(float(row.get("confirmed_regime_id", 0) or 0)),
            "oil_event_active": row.get("oil_event_active", False),
            "oil_pressure_score": row.get("oil_pressure_score", 0.0),
            "unknown_event_active": row.get("unknown_event_active", False),
        }], index=[sub.index[-1]])
        history_tail = df.iloc[max(0, start - 5):i]
        alloc = layer.allocate(sub, classified_df=classified, history=history_tail)
        w = alloc["weights"]

        for j in range(i + 1, min(i + 1 + step, len(df))):
            r = ret.iloc[j]
            pr = 0.0
            for k in ("equity", "gold", "bond", "commodity", "crypto"):
                rv = r.get(k, np.nan)
                if np.isfinite(rv):
                    pr += (w[k] / 100.0) * rv
            rows.append(pr)
            dates.append(df.index[j])
    return pd.Series(rows, index=dates, dtype=float)


def metrics(r: pd.Series) -> Metrics:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return Metrics(*(np.nan,) * 6, observations=len(r))
    nav = (1.0 + r).cumprod()
    total = float(nav.iloc[-1] - 1.0)
    years = max(len(r) / 252.0, 1.0 / 252.0)
    ann = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(252.0))
    sharpe = float((r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(252.0))
    peak = nav.cummax()
    dd = nav / peak - 1.0
    mdd = float(abs(dd.min()))
    calmar = float(ann / mdd) if mdd > 1e-12 else np.nan
    return Metrics(total, ann, vol, mdd, calmar, sharpe, len(r))


def parameter_grid() -> List[Dict[str, float]]:
    candidates = []
    for target_vol in (0.065, 0.075, 0.085, 0.095, 0.105):
        for hard_cap in (0.08, 0.10, 0.12):
            for broad_mult in (0.45, 0.55, 0.65):
                for soft_temp in (0.9, 1.15, 1.4):
                    candidates.append({
                        "target_annual_vol": target_vol,
                        "hard_risk_cap": hard_cap,
                        "broad_risk_multiplier": broad_mult,
                        "softmax_temperature": soft_temp,
                    })
    return candidates


def make_layer(params: Dict[str, float]) -> AdaptiveStrategyLayer:
    cfg = {
        "research_strategies": {
            "production_risk_budget_by_regime": {
                "0": 0.60, "1": 0.35, "2": 0.08,
                "3": 0.45, "4": 0.25, "5": 0.82,
            },
            "production_min_cash_by_regime": {
                "0": 0.10, "1": 0.15, "2": 0.80,
                "3": 0.20, "4": 0.25, "5": 0.05,
            },
            "adaptive_layer": params,
        }
    }
    return AdaptiveStrategyLayer(cfg)


def walk_forward_score(df: pd.DataFrame, params: Dict[str, float], n_splits: int = 4) -> Dict[str, object]:
    n = len(df)
    folds = []
    min_train = max(252, int(n * 0.40))
    if n < min_train + 4 * 30:
        return {"status": "INSUFFICIENT_REAL_HISTORY", "params": params}
    test_size = max(60, int((n - min_train) / n_splits))
    layer = make_layer(params)
    test_metrics = []
    for k in range(n_splits):
        train_end = min_train + k * test_size
        test_end = min(n, train_end + test_size)
        if test_end - train_end < 30:
            continue
        # Strategy is re-run from the start of the sample, but we score only the
        # fold's forward period. This prevents using future fold observations.
        r = portfolio_returns(df.iloc[:test_end], layer, rebalance_step=5)
        r = r[r.index > df.index[train_end - 1]]
        if len(r) < 30:
            continue
        m = metrics(r)
        test_metrics.append(asdict(m))
        folds.append({"fold": k + 1, "train_end": str(df.index[train_end - 1]), "test_end": str(df.index[test_end - 1]), "metrics": asdict(m)})
    if not test_metrics:
        return {"status": "INSUFFICIENT_REAL_HISTORY", "params": params}
    calmar = float(np.nanmedian([x["calmar"] for x in test_metrics]))
    ann = float(np.nanmedian([x["annualized_return"] for x in test_metrics]))
    mdd = float(np.nanmedian([x["max_drawdown"] for x in test_metrics]))
    sharpe = float(np.nanmedian([x["sharpe"] for x in test_metrics]))
    return {
        "status": "RESEARCH_ONLY",
        "params": params,
        "median_annualized_return": ann,
        "median_max_drawdown": mdd,
        "median_calmar": calmar,
        "median_sharpe": sharpe,
        "folds": folds,
    }


def pareto_frontier(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    valid = [r for r in results if r.get("status") == "RESEARCH_ONLY" and np.isfinite(r.get("median_calmar", np.nan))]
    frontier = []
    for r in valid:
        dominated = False
        for q in valid:
            if q is r:
                continue
            better_or_equal = (
                q["median_annualized_return"] >= r["median_annualized_return"]
                and q["median_max_drawdown"] <= r["median_max_drawdown"]
            )
            strictly_better = (
                q["median_annualized_return"] > r["median_annualized_return"]
                or q["median_max_drawdown"] < r["median_max_drawdown"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda x: (-x["median_calmar"], -x["median_annualized_return"]))
    return frontier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="cms_history.csv")
    ap.add_argument("--out", default="strategy_research_results.json")
    ap.add_argument("--rebalance-step", type=int, default=5, help="Rebalance every N observations for research speed.")
    args = ap.parse_args()

    df = load_history(args.history)
    required = {"spx_price", "gold_weight"}
    # We do not require the full set here because real-history CSV schemas vary;
    # price columns are checked by the allocator itself.
    if len(df) < 400:
        payload = {
            "status": "INSUFFICIENT_REAL_HISTORY",
            "observations": len(df),
            "required_minimum": 400,
            "message": "Need a materially longer real market history before strategy selection.",
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps(payload, ensure_ascii=False))
        return

    results = []
    for params in parameter_grid():
        result = walk_forward_score(df, params, n_splits=4)
        results.append(result)
    frontier = pareto_frontier(results)
    payload = {
        "status": "RESEARCH_ONLY",
        "method": "walk_forward_pareto_search",
        "rebalance_step": 5,
        "observations": len(df),
        "candidate_count": len(results),
        "pareto_candidate_count": len(frontier),
        "pareto_frontier": frontier[:20],
        "warning": "No candidate is promoted automatically. Real-data out-of-sample validation is required before production deployment.",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
