"""
Macro Sentinel strategy audit.

This script compares the production allocation architecture with research
candidates across five independent synthetic seeds and reports a return/
drawdown Pareto frontier. It does not mutate live parameters and does not
claim synthetic results as live-market evidence.
"""

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest_regimes import build_research_strategy_returns, generate_synthetic_macro_history
from regime_engine import MacroRegimeEngine

SEEDS = (7, 19, 42, 71, 101)


def metrics(ret: pd.Series) -> Dict[str, float]:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    cum = (1.0 + r).cumprod()
    years = len(r) / 252.0
    ann = (cum.iloc[-1] ** (1.0 / max(years, 0.1)) - 1.0) * 100.0
    vol = r.std() * np.sqrt(252.0) * 100.0
    dd = abs(((cum / cum.cummax()) - 1.0).min()) * 100.0
    return {
        "annualized_return": float(ann),
        "max_drawdown": float(dd),
        "calmar_ratio": float(ann / max(dd, 0.01)),
        "sharpe_ratio": float((ann - 3.0) / max(vol, 0.01)),
        "total_return": float((cum.iloc[-1] - 1.0) * 100.0),
    }


def candidate_returns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    production = (
        (df["regime_cash_weight"] / 100.0).shift(1).fillna(0.35) * (df["dgs2"] / 100.0 / 252.0)
        + (df["regime_gold_weight"] / 100.0).shift(1).fillna(0.20) * df["gold"].pct_change().fillna(0.0)
        + (df["regime_bond_weight"] / 100.0).shift(1).fillna(0.20) * df["ust10y"].pct_change().fillna(0.0)
        + (df["regime_eq_weight"] / 100.0).shift(1).fillna(0.15) * df["spx"].pct_change().fillna(0.0)
        + (df["regime_commodity_weight"] / 100.0).shift(1).fillna(0.05) * df["oil"].pct_change().fillna(0.0)
        + (df["regime_crypto_weight"] / 100.0).shift(1).fillna(0.05) * df["btc"].pct_change().fillna(0.0)
    )
    return {
        "Macro Sentinel Dynamic (Production)": production,
        "Research: Adaptive Opportunity": build_research_strategy_returns(df, "adaptive_opportunity"),
        "Research: Adaptive Opportunity + BTC": build_research_strategy_returns(df, "adaptive_opportunity_btc"),
        "Research: Balanced Trend": build_research_strategy_returns(df, "balanced_trend"),
    }


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    keep: List[int] = []
    for i, row in summary.iterrows():
        dominated = False
        for j, other in summary.iterrows():
            if i == j:
                continue
            better_return = other["median_annualized_return"] >= row["median_annualized_return"]
            lower_dd = other["median_max_drawdown"] <= row["median_max_drawdown"]
            strict = (
                other["median_annualized_return"] > row["median_annualized_return"]
                or other["median_max_drawdown"] < row["median_max_drawdown"]
            )
            if better_return and lower_dd and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return summary.loc[keep].sort_values("median_annualized_return", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    rows = []
    for seed in seeds:
        engine = MacroRegimeEngine()
        raw = generate_synthetic_macro_history(seed=seed)
        classified = engine.run_time_series(engine.prepare_indicators(raw))
        for name, ret in candidate_returns(classified).items():
            rows.append({"seed": seed, "strategy": name, **metrics(ret)})

    df = pd.DataFrame(rows)
    summary = df.groupby("strategy").agg(
        median_annualized_return=("annualized_return", "median"),
        p25_annualized_return=("annualized_return", lambda s: float(s.quantile(0.25))),
        median_max_drawdown=("max_drawdown", "median"),
        p75_max_drawdown=("max_drawdown", lambda s: float(s.quantile(0.75))),
        median_calmar=("calmar_ratio", "median"),
        median_sharpe=("sharpe_ratio", "median"),
        median_total_return=("total_return", "median"),
    ).reset_index()

    print("\nMulti-seed strategy audit (synthetic only):")
    print(summary.round(2).to_string(index=False))
    print("\nReturn / drawdown Pareto frontier:")
    print(pareto_frontier(summary).round(2).to_string(index=False))

    print("\nProduction strategy by seed:")
    prod = df[df["strategy"] == "Macro Sentinel Dynamic (Production)"]
    print(prod[["seed", "annualized_return", "max_drawdown", "calmar_ratio", "sharpe_ratio", "total_return"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
