"""
Macro Sentinel research strategy suite.

Important:
- These candidates are research-only.
- Synthetic results are not live-performance evidence.
- No candidate is auto-deployed into the live allocation engine.
"""

import argparse
import pandas as pd
from backtest_regimes import generate_synthetic_macro_history, research_strategy_suite, multi_seed_research_suite
from regime_engine import MacroRegimeEngine


def run_single(seed: int = 42) -> pd.DataFrame:
    engine = MacroRegimeEngine()
    raw = generate_synthetic_macro_history(seed=seed)
    classified = engine.run_time_series(engine.prepare_indicators(raw))
    result = research_strategy_suite(classified)
    return pd.DataFrame.from_dict(result, orient="index").reset_index(names="strategy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true")
    args = parser.parse_args()

    single = run_single(args.seed)
    print("\nSingle-seed research results (NOT live allocation):")
    print(single.to_string(index=False))

    if args.multi_seed:
        multi = multi_seed_research_suite()
        summary = multi.groupby("strategy").agg(
            median_return=("annualized_return", "median"),
            median_max_dd=("max_drawdown", "median"),
            median_sharpe=("sharpe_ratio", "median"),
            p25_return=("annualized_return", lambda s: float(s.quantile(0.25))),
            p75_return=("annualized_return", lambda s: float(s.quantile(0.75))),
        ).round(2)
        print("\nMulti-seed research summary:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
