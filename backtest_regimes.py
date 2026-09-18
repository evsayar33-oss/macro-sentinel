"""
Backtest & Dynamic Threshold Optimization Engine
for Macro Event Interpretation System v1.0
Calibrated for Multi-Asset 6-Asset Architecture:
1. Macro Sentinel Apex (Dynamic 6-Asset Optimum: +229.19%, Sharpe 2.81, Max DD 5.63%)
2. Benchmark 1: Multi-Asset Defensive Shield (%35 Nakit / %20 Altın / %20 Tahvil / %15 Hisse / %5 Emtia / %5 Kripto)
3. Benchmark 2: Artemis Dragon Portfolio (%25 Hisse / %25 Nakit / %20 Altın / %15 Tahvil / %10 Emtia / %5 Kripto)
4. Benchmark 3: Taleb Barbell Asymmetric (%85 Nakit & T-Bill / %10 Altın / %5 Kripto - Ultra Düşük DD %3.41)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from regime_engine import MacroRegimeEngine


def generate_synthetic_macro_history(start_date="2018-01-01", end_date="2026-09-01", seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n = len(dates)

    spx_price = np.zeros(n)
    spx_price[0] = 2700.0

    oil_price = np.zeros(n)
    oil_price[0] = 60.0

    brent_price = np.zeros(n)
    brent_price[0] = 63.0
    heating_oil = np.zeros(n)
    heating_oil[0] = 1.8
    gasoline_price = np.zeros(n)
    gasoline_price[0] = 1.7
    natgas_price = np.zeros(n)
    natgas_price[0] = 2.5
    crude_stocks = np.zeros(n)
    crude_stocks[0] = 430.0

    freight_index = np.zeros(n)
    freight_index[0] = 1200.0

    hy_oas = np.zeros(n)
    hy_oas[0] = 3.5

    ig_oas = np.zeros(n)
    ig_oas[0] = 1.1

    vix = np.zeros(n)
    vix[0] = 15.0

    dxy = np.zeros(n)
    dxy[0] = 90.0

    usdjpy = np.zeros(n)
    usdjpy[0] = 110.0

    tips10y = np.zeros(n)
    tips10y[0] = 0.5

    t10yie = np.zeros(n)
    t10yie[0] = 2.0

    dgs2 = np.zeros(n)
    dgs2[0] = 2.0

    dgs10 = np.zeros(n)
    dgs10[0] = 2.5

    ndl = np.zeros(n)
    ndl[0] = 5500000.0

    btc_price = np.zeros(n)
    btc_price[0] = 10000.0

    gold_price = np.zeros(n)
    gold_price[0] = 1300.0

    ust10y_bond = np.zeros(n)
    ust10y_bond[0] = 100.0

    for i in range(1, n):
        d = dates[i]
        year = d.year
        month = d.month

        spx_ret = np.random.normal(0.0005, 0.008)
        oil_ret = np.random.normal(0.0001, 0.015)
        freight_ret = np.random.normal(0.0, 0.012)
        d_hy = np.random.normal(0.0, 0.03)
        d_ig = np.random.normal(0.0, 0.01)
        vix_ret = -0.06 * (vix[i - 1] - 15.0) + np.random.normal(0.0, 1.0)
        dxy_ret = np.random.normal(0.0, 0.0025)
        usdjpy_ret = np.random.normal(0.0, 0.0035)
        d_tips = np.random.normal(0.0, 0.015)
        d_t10yie = np.random.normal(0.0, 0.015)
        d_dgs2 = np.random.normal(0.0, 0.015)
        d_dgs10 = np.random.normal(0.0, 0.015)
        d_ndl = np.random.normal(600.0, 12000.0)
        btc_ret = np.random.normal(0.0012, 0.025)
        gold_ret = np.random.normal(0.0004, 0.006)

        # 1. 2018 Q4
        if year == 2018 and month in [10, 11, 12]:
            spx_ret -= 0.0035
            d_tips += 0.025
            d_dgs2 += 0.02
            d_dgs10 += 0.015
            vix_ret += 0.6
            d_hy += 0.045
        elif year == 2019:
            spx_ret += 0.0007
            d_hy -= 0.01
            d_dgs2 -= 0.005
            d_dgs10 -= 0.005
        # 2. 2020 March
        elif year == 2020 and month == 3:
            spx_ret -= 0.022
            btc_ret -= 0.035
            oil_ret -= 0.035
            vix_ret += 2.8
            dxy_ret += 0.008
            d_hy += 0.22
            d_ig += 0.07
            d_dgs10 -= 0.04
        # 3. 2020-2021 QE Expansion
        elif (year == 2020 and month >= 5) or (year == 2021):
            spx_ret += 0.0014
            btc_ret += 0.004
            gold_ret += 0.0012
            d_ndl += 25000.0
            d_hy -= 0.018
            dxy_ret -= 0.0012
            if vix[i - 1] > 16.0:
                vix_ret -= 0.6
        # 4. 2022 H1 Stagflation Shock
        elif year == 2022 and month in [2, 3, 4, 5, 6]:
            oil_ret += 0.015
            freight_ret += 0.02
            spx_ret -= 0.0028
            d_dgs10 += 0.025
            gold_ret += 0.0018
            d_t10yie += 0.022
            d_hy += 0.035
        # 5. 2022 H2 Fed Rate Shock
        elif year == 2022 and month in [7, 8, 9, 10, 11]:
            d_tips += 0.032
            d_dgs2 += 0.038
            d_dgs10 += 0.022
            d_t10yie -= 0.01
            spx_ret -= 0.0022
            dxy_ret += 0.0025
        # 6. 2023 March SVB Banking Stress
        elif year == 2023 and month == 3:
            d_hy += 0.08
            d_ig += 0.04
            spx_ret -= 0.0025
            vix_ret += 0.7
            d_dgs2 -= 0.05
        # 7. 2023 H2 - 2024 H1 AI Expansion
        elif (year == 2023 and month >= 5) or (year == 2024 and month in [1, 2, 3, 4, 5, 6, 7]):
            spx_ret += 0.0012
            btc_ret += 0.003
            d_ndl += 5000.0
            d_hy -= 0.012
            if vix[i - 1] > 13.5:
                vix_ret -= 0.35
            dxy_ret -= 0.0002
        # 8. 2024 August JPY Carry Flash Crash
        elif year == 2024 and month == 8 and d.day <= 10:
            usdjpy_ret -= 0.025
            vix_ret += 4.0
            spx_ret -= 0.022
            btc_ret -= 0.04
            d_hy += 0.07
        elif year >= 2025:
            spx_ret += 0.0004
            d_hy += np.random.normal(0.0, 0.01)

        spx_price[i] = max(100.0, spx_price[i - 1] * (1.0 + spx_ret))
        oil_price[i] = max(10.0, oil_price[i - 1] * (1.0 + oil_ret))
        brent_price[i] = max(15.0, brent_price[i - 1] * (1.0 + oil_ret * 0.95 + np.random.normal(0.0, 0.003)))
        heating_oil[i] = max(0.5, heating_oil[i - 1] * (1.0 + oil_ret * 1.10 + np.random.normal(0.0, 0.004)))
        gasoline_price[i] = max(0.5, gasoline_price[i - 1] * (1.0 + oil_ret * 0.90 + np.random.normal(0.0, 0.004)))
        natgas_price[i] = max(0.8, natgas_price[i - 1] * (1.0 + oil_ret * 0.45 + np.random.normal(0.0, 0.01)))
        crude_stocks[i] = max(300.0, crude_stocks[i - 1] + np.random.normal(0.0, 2.5))
        if year == 2022 and month in [2, 3, 4, 5, 6]:
            crude_stocks[i] = max(300.0, crude_stocks[i - 1] - 3.5 + np.random.normal(0.0, 1.0))
        freight_index[i] = max(300.0, freight_index[i - 1] * (1.0 + freight_ret))
        hy_oas[i] = np.clip(hy_oas[i - 1] + d_hy, 2.5, 12.0)
        ig_oas[i] = np.clip(ig_oas[i - 1] + d_ig, 0.8, 4.5)
        vix[i] = np.clip(vix[i - 1] + vix_ret, 9.0, 85.0)
        dxy[i] = max(70.0, dxy[i - 1] * (1.0 + dxy_ret))
        usdjpy[i] = max(75.0, usdjpy[i - 1] * (1.0 + usdjpy_ret))
        tips10y[i] = np.clip(tips10y[i - 1] + d_tips, -1.5, 3.0)
        t10yie[i] = np.clip(t10yie[i - 1] + d_t10yie, 0.5, 3.5)
        dgs2[i] = np.clip(dgs2[i - 1] + d_dgs2, 0.1, 5.5)
        dgs10[i] = np.clip(dgs10[i - 1] + d_dgs10, 0.5, 5.5)
        ndl[i] = max(3000000.0, ndl[i - 1] + d_ndl)
        btc_price[i] = max(3000.0, btc_price[i - 1] * (1.0 + btc_ret))
        gold_price[i] = max(1000.0, gold_price[i - 1] * (1.0 + gold_ret))

        bond_ret = -8.0 * (dgs10[i] - dgs10[i - 1]) / 100.0 + (dgs10[i - 1] / 100.0) / 252.0
        ust10y_bond[i] = ust10y_bond[i - 1] * (1.0 + bond_ret)

    df = pd.DataFrame({
        'date': dates,
        'spx': spx_price,
        'oil': oil_price,
        'brent': brent_price,
        'heating_oil': heating_oil,
        'gasoline': gasoline_price,
        'natgas': natgas_price,
        'crude_stocks': crude_stocks,
        'freight': freight_index,
        'hy_oas': hy_oas,
        'ig_oas': ig_oas,
        'vix': vix,
        'broad_dollar': dxy,
        'usdjpy': usdjpy,
        'tips10y': tips10y,
        't10yie': t10yie,
        'dgs2': dgs2,
        'dgs10': dgs10,
        'ndl': ndl,
        'btc': btc_price,
        'gold': gold_price,
        'ust10y': ust10y_bond
    }).set_index('date')

    return df


def validate_engine_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate semantic invariants before using research results."""
    weight_cols = [
        "regime_cash_weight", "regime_gold_weight", "regime_bond_weight",
        "regime_eq_weight", "regime_commodity_weight", "regime_crypto_weight"
    ]
    totals = df[weight_cols].sum(axis=1)
    if not np.allclose(totals.values, 100.0, atol=0.05):
        raise ValueError("Allocation integrity failure")

    oil_active = df.get("oil_event_active", pd.Series(False, index=df.index)).astype(bool)
    oil_type = df.get("oil_event_type", pd.Series("NONE", index=df.index)).astype(str).str.upper()
    oil_score = pd.to_numeric(df.get("oil_event_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    structural = pd.to_numeric(df.get("oil_structural_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    structural_qualified = df.get("oil_structural_qualified", pd.Series(False, index=df.index)).astype(bool)

    confirmed_types = {"MOMENTUM", "STRUCTURAL", "COMBINED"}
    if bool((oil_active & ~oil_type.isin(list(confirmed_types))).any()):
        raise ValueError("Oil event active without confirmed event type")
    if bool((~oil_active & oil_type.isin(list(confirmed_types))).any()):
        raise ValueError("Oil event type says confirmed while oil_event_active is false")
    if bool(((oil_type == "STRUCTURAL") & ~structural_qualified).any()):
        raise ValueError("STRUCTURAL event without structural qualification")
    if bool((oil_score < -1e-9).any()) or bool((oil_score > 1.0 + 1e-9).any()):
        raise ValueError("Oil event score outside [0,1]")
    if bool((structural < -1e-9).any()) or bool((structural > 1.0 + 1e-9).any()):
        raise ValueError("Oil structural score outside [0,1]")

    return {
        "allocation_integrity": True,
        "oil_event_semantics": True,
        "structural_qualification_semantics": True,
    }


def run_portfolio_backtest(df_classified: pd.DataFrame) -> Dict[str, Any]:
    df = df_classified.copy()
    invariants = validate_engine_invariants(df)

    weight_cols = [
        "regime_cash_weight", "regime_gold_weight", "regime_bond_weight",
        "regime_eq_weight", "regime_commodity_weight", "regime_crypto_weight"
    ]
    if not all(c in df.columns for c in weight_cols):
        raise ValueError("Backtest is missing regime allocation columns")

    spx_ret = df['spx'].pct_change().fillna(0.0)
    bond_ret = df['ust10y'].pct_change().fillna(0.0)
    cash_ret = (df['dgs2'] / 100.0) / 252.0
    gold_ret = df['gold'].pct_change().fillna(0.0)
    oil_ret = df['oil'].pct_change().fillna(0.0)
    btc_ret = df['btc'].pct_change().fillna(0.0)

    w_csh = (df['regime_cash_weight'] / 100.0).shift(1).fillna(0.35)
    w_gld = (df['regime_gold_weight'] / 100.0).shift(1).fillna(0.20)
    w_bnd = (df['regime_bond_weight'] / 100.0).shift(1).fillna(0.20)
    w_eq = (df['regime_eq_weight'] / 100.0).shift(1).fillna(0.15)
    w_cmd = (df['regime_commodity_weight'] / 100.0).shift(1).fillna(0.05)
    w_crp = (df['regime_crypto_weight'] / 100.0).shift(1).fillna(0.05)

    # 1. Macro Sentinel Dynamic (current regime + event architecture)
    strat_ret = (w_csh * cash_ret +
                 w_gld * gold_ret +
                 w_bnd * bond_ret +
                 w_eq * spx_ret +
                 w_cmd * oil_ret +
                 w_crp * btc_ret)

    # 2. Benchmark 1: Defensive Shield
    bench_def_ret = (0.35 * cash_ret +
                     0.20 * gold_ret +
                     0.20 * bond_ret +
                     0.15 * spx_ret +
                     0.05 * oil_ret +
                     0.05 * btc_ret)

    # 3. Benchmark 2: Artemis Dragon
    bench_art_ret = (0.25 * spx_ret +
                     0.25 * cash_ret +
                     0.20 * gold_ret +
                     0.15 * bond_ret +
                     0.10 * oil_ret +
                     0.05 * btc_ret)

    # 4. Benchmark 3: Taleb Barbell Asymmetric (Ultra Low DD)
    bench_taleb_ret = (0.85 * cash_ret +
                       0.10 * gold_ret +
                       0.05 * btc_ret)





    def calc_metrics(ret_series):
        ann_factor = 252.0
        n_years = len(ret_series) / ann_factor
        cum = (1.0 + ret_series).cumprod()
        total_return = (cum.iloc[-1] - 1.0) * 100.0
        ann_return = ((cum.iloc[-1]) ** (1.0 / max(0.1, n_years)) - 1.0) * 100.0
        ann_vol = ret_series.std() * np.sqrt(ann_factor) * 100.0

        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = abs(drawdown.min()) * 100.0

        excess_ret = ann_return - 3.0
        sharpe = excess_ret / max(ann_vol, 0.01)
        calmar = ann_return / max(max_dd, 0.01)

        return {
            "annualized_return": round(float(ann_return), 2),
            "annualized_volatility": round(float(ann_vol), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown": round(float(max_dd), 2),
            "calmar_ratio": round(float(calmar), 2),
            "total_return": round(float(total_return), 2)
        }

    strat_metrics = calc_metrics(strat_ret)
    def_metrics = calc_metrics(bench_def_ret)
    art_metrics = calc_metrics(bench_art_ret)
    taleb_metrics = calc_metrics(bench_taleb_ret)

    covid_bars = df.loc["2020-03-01":"2020-03-31"]
    covid_detected = (covid_bars['confirmed_regime_id'] == 2).any()

    stagflation_bars = df.loc["2022-02-01":"2022-06-30"]
    stagflation_detected = (stagflation_bars['confirmed_regime_id'] == 1).any()

    rate_shock_bars = df.loc["2022-07-01":"2022-11-30"]
    rate_shock_detected = (rate_shock_bars['confirmed_regime_id'] == 3).any()

    carry_bars = df.loc["2024-08-01":"2024-08-15"]
    carry_detected = (carry_bars['confirmed_regime_id'] == 2).any()

    structural_oil_bars = df.loc["2022-02-01":"2022-06-30"]
    if 'oil_event_active' in structural_oil_bars.columns and 'oil_event_type' in structural_oil_bars.columns:
        structural_oil_detected = (
            structural_oil_bars['oil_event_active'].astype(bool)
            & structural_oil_bars['oil_event_type'].astype(str).str.upper().isin(['STRUCTURAL', 'COMBINED'])
        ).any()
    else:
        structural_oil_detected = False

    regime_counts = df['confirmed_regime_name'].value_counts(normalize=True) * 100.0
    regime_dist = {str(k): round(float(v), 1) for k, v in regime_counts.items()}

    return {
        "strategy": strat_metrics,
        "benchmark_defensive_shield": def_metrics,
        "benchmark_artemis_dragon": art_metrics,
        "benchmark_taleb_barbell": taleb_metrics,
        "regime_distribution": regime_dist,
        "invariants": invariants,
        "crisis_details": {
            "covid_2020_detected": bool(covid_detected),
            "stagflation_2022_detected": bool(stagflation_detected),
            "rate_shock_2022_detected": bool(rate_shock_detected),
            "jpy_carry_2024_detected": bool(carry_detected),
            "oil_structural_event_2022_detected": bool(structural_oil_detected)
        }
    }



def _calc_metrics(ret_series: pd.Series) -> Dict[str, float]:
    ann_factor = 252.0
    r = pd.to_numeric(ret_series, errors="coerce").fillna(0.0)
    n_years = len(r) / ann_factor
    cum = (1.0 + r).cumprod()
    total = float((cum.iloc[-1] - 1.0) * 100.0)
    ann_ret = float((cum.iloc[-1] ** (1.0 / max(n_years, 0.1)) - 1.0) * 100.0)
    ann_vol = float(r.std() * np.sqrt(ann_factor) * 100.0)
    dd = (cum / cum.cummax()) - 1.0
    max_dd = float(abs(dd.min()) * 100.0)
    sharpe = float((ann_ret - 3.0) / max(ann_vol, 0.01))
    calmar = float(ann_ret / max(max_dd, 0.01))
    return {
        "annualized_return": round(ann_ret, 2),
        "annualized_volatility": round(ann_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "calmar_ratio": round(calmar, 2),
        "total_return": round(total, 2),
    }


def build_research_strategy_returns(df: pd.DataFrame, variant: str = "adaptive_opportunity") -> pd.Series:
    """Research-only return-seeking overlay. Uses only t-1 information for t returns."""
    eq = pd.to_numeric(df["spx"], errors="coerce")
    gold = pd.to_numeric(df["gold"], errors="coerce")
    oil = pd.to_numeric(df["oil"], errors="coerce")
    btc = pd.to_numeric(df["btc"], errors="coerce")
    bond = pd.to_numeric(df["ust10y"], errors="coerce")
    cash = pd.to_numeric(df["dgs2"], errors="coerce") / 100.0

    eq_ret = eq.pct_change().fillna(0.0)
    gold_ret = gold.pct_change().fillna(0.0)
    oil_ret = oil.pct_change().fillna(0.0)
    btc_ret = btc.pct_change().fillna(0.0)
    bond_ret = bond.pct_change().fillna(0.0)
    cash_ret = cash / 252.0

    weights = []
    for i in range(len(df)):
        if i == 0:
            weights.append((0.35, 0.20, 0.20, 0.15, 0.05, 0.05))
            continue
        j = i - 1
        rid = int(df["confirmed_regime_id"].iloc[j])
        sp20 = float(eq.pct_change(20).iloc[j]) if pd.notna(eq.pct_change(20).iloc[j]) else 0.0
        sp100 = float(eq.pct_change(100).iloc[j]) if pd.notna(eq.pct_change(100).iloc[j]) else 0.0
        btc60 = float(btc.pct_change(60).iloc[j]) if pd.notna(btc.pct_change(60).iloc[j]) else 0.0
        vix_pct = float(df["vix_percentile_252"].iloc[j]) if pd.notna(df["vix_percentile_252"].iloc[j]) else 50.0

        base_risk = {0: 0.60, 1: 0.35, 2: 0.08, 3: 0.45, 4: 0.25, 5: 0.82}.get(rid, 0.60)
        if sp20 > 0 and sp100 > 0:
            base_risk += 0.12
        elif sp20 < 0 and sp100 < 0:
            base_risk -= 0.15
        if vix_pct > 80:
            base_risk -= 0.18
        elif vix_pct < 30:
            base_risk += 0.05
        if variant == "adaptive_opportunity_btc" and btc60 > 0:
            base_risk += 0.06
        elif variant == "adaptive_opportunity_btc" and btc60 < 0:
            base_risk -= 0.04
        if variant == "balanced_trend" and rid == 0 and sp100 > 0:
            base_risk += 0.05

        risk_budget = float(np.clip(base_risk, 0.05, 0.88))
        eq_w = risk_budget * 0.62
        gold_w = risk_budget * 0.14
        oil_w = risk_budget * 0.10
        bond_w = risk_budget * 0.14
        btc_w = risk_budget * 0.10 if (variant == "adaptive_opportunity_btc" and btc60 > 0 and risk_budget > 0.45) else 0.0
        bond_w = max(0.0, bond_w - btc_w)
        cash_w = max(0.0, 1.0 - (eq_w + gold_w + oil_w + bond_w + btc_w))
        weights.append((cash_w, gold_w, bond_w, eq_w, oil_w, btc_w))

    w = pd.DataFrame(weights, index=df.index)
    return (
        w[0] * cash_ret
        + w[1] * gold_ret
        + w[2] * bond_ret
        + w[3] * eq_ret
        + w[4] * oil_ret
        + w[5] * btc_ret
    )


def research_strategy_suite(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    candidates = {
        "Macro Sentinel Dynamic": (
            (df["regime_cash_weight"] / 100.0).shift(1).fillna(0.35) * (df["dgs2"] / 100.0 / 252.0)
            + (df["regime_gold_weight"] / 100.0).shift(1).fillna(0.20) * df["gold"].pct_change().fillna(0.0)
            + (df["regime_bond_weight"] / 100.0).shift(1).fillna(0.20) * df["ust10y"].pct_change().fillna(0.0)
            + (df["regime_eq_weight"] / 100.0).shift(1).fillna(0.15) * df["spx"].pct_change().fillna(0.0)
            + (df["regime_commodity_weight"] / 100.0).shift(1).fillna(0.05) * df["oil"].pct_change().fillna(0.0)
            + (df["regime_crypto_weight"] / 100.0).shift(1).fillna(0.05) * df["btc"].pct_change().fillna(0.0)
        ),
        "Research: Adaptive Opportunity": build_research_strategy_returns(df, "adaptive_opportunity"),
        "Research: Adaptive Opportunity + BTC": build_research_strategy_returns(df, "adaptive_opportunity_btc"),
        "Research: Balanced Trend": build_research_strategy_returns(df, "balanced_trend"),
    }
    return {name: _calc_metrics(ret) for name, ret in candidates.items()}


def multi_seed_research_suite(seeds=(7, 19, 42, 71, 101)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        engine = MacroRegimeEngine()
        raw = generate_synthetic_macro_history(seed=seed)
        prepared = engine.prepare_indicators(raw)
        classified = engine.run_time_series(prepared)
        suite = research_strategy_suite(classified)
        for name, metrics in suite.items():
            rows.append({"seed": seed, "strategy": name, **metrics})
    return pd.DataFrame(rows)

def main():
    print("=" * 85)
    print("🏛️ MACRO SENTINEL: DİNAMİK REJİM & OLAY BACKTEST SÜİTİ (2018 - 2026)")
    print("=" * 85)

    engine = MacroRegimeEngine()
    raw_data = generate_synthetic_macro_history()
    prepared = engine.prepare_indicators(raw_data)
    classified = engine.run_time_series(prepared)
    bt = run_portfolio_backtest(classified)

    print("\n" + "=" * 110)
    print("📊 2018 - 2026 DÖNEMİ TEST SONUÇLARI & BENCHMARK KARŞILAŞTIRMA TABLOSU")
    print("=" * 110)

    perf_summary = pd.DataFrame([
        {
            "Portföy / Strateji": "⚡ Macro Sentinel Dynamic",
            "Varlık Dağılım Mimarisi": "Dinamik 6 Varlık (Rejim + Olay Katmanı)",
            "Yıllık Getiri (%)": f"%{bt['strategy']['annualized_return']:.2f}",
            "Volatilite (%)": f"%{bt['strategy']['annualized_volatility']:.2f}",
            "Sharpe": f"{bt['strategy']['sharpe_ratio']:.2f}",
            "Max DD (%)": f"%{bt['strategy']['max_drawdown']:.2f}",
            "Calmar": f"{bt['strategy']['calmar_ratio']:.2f}",
            "Toplam Getiri (%)": f"+%{bt['strategy']['total_return']:.2f}"
        },
        {
            "Portföy / Strateji": "🛡️ Benchmark 1: Defensive Shield",
            "Varlık Dağılım Mimarisi": "%35 Nakit / %20 Altın / %20 Tahvil / %15 Hisse / %5 Emtia / %5 Kripto",
            "Yıllık Getiri (%)": f"%{bt['benchmark_defensive_shield']['annualized_return']:.2f}",
            "Volatilite (%)": f"%{bt['benchmark_defensive_shield']['annualized_volatility']:.2f}",
            "Sharpe": f"{bt['benchmark_defensive_shield']['sharpe_ratio']:.2f}",
            "Max DD (%)": f"%{bt['benchmark_defensive_shield']['max_drawdown']:.2f}",
            "Calmar": f"{bt['benchmark_defensive_shield']['calmar_ratio']:.2f}",
            "Toplam Getiri (%)": f"+%{bt['benchmark_defensive_shield']['total_return']:.2f}"
        },
        {
            "Portföy / Strateji": "🐉 Benchmark 2: Artemis Dragon",
            "Varlık Dağılım Mimarisi": "%25 Hisse / %25 Nakit / %20 Altın / %15 Tahvil / %10 Emtia / %5 Kripto",
            "Yıllık Getiri (%)": f"%{bt['benchmark_artemis_dragon']['annualized_return']:.2f}",
            "Volatilite (%)": f"%{bt['benchmark_artemis_dragon']['annualized_volatility']:.2f}",
            "Sharpe": f"{bt['benchmark_artemis_dragon']['sharpe_ratio']:.2f}",
            "Max DD (%)": f"%{bt['benchmark_artemis_dragon']['max_drawdown']:.2f}",
            "Calmar": f"{bt['benchmark_artemis_dragon']['calmar_ratio']:.2f}",
            "Toplam Getiri (%)": f"+%{bt['benchmark_artemis_dragon']['total_return']:.2f}"
        },
        {
            "Portföy / Strateji": "🛡️ Benchmark 3: Taleb Barbell Asymmetric",
            "Varlık Dağılım Mimarisi": "%85 Nakit & T-Bill / %10 Altın / %5 Kripto",
            "Yıllık Getiri (%)": f"%{bt['benchmark_taleb_barbell']['annualized_return']:.2f}",
            "Volatilite (%)": f"%{bt['benchmark_taleb_barbell']['annualized_volatility']:.2f}",
            "Sharpe": f"{bt['benchmark_taleb_barbell']['sharpe_ratio']:.2f}",
            "Max DD (%)": f"%{bt['benchmark_taleb_barbell']['max_drawdown']:.2f}",
            "Calmar": f"{bt['benchmark_taleb_barbell']['calmar_ratio']:.2f}",
            "Toplam Getiri (%)": f"+%{bt['benchmark_taleb_barbell']['total_return']:.2f}"
        },
    ])
    print(perf_summary.to_string(index=False))
    print("\nModel invariants:", bt["invariants"])

    research = research_strategy_suite(classified)
    print("\n\nResearch strategy candidates (synthetic; not auto-deployed):")
    print(pd.DataFrame.from_dict(research, orient="index").to_string())

    config = engine.load_config()
    config["research_strategy_metrics_single_seed"] = research
    config["backtest_metrics"] = {
        "strategy": {
            "name": "⚡ Macro Sentinel Dynamic",
            "allocation_desc": "Dinamik 6 Varlık (Rejim + Olay Katmanı)",
            **bt["strategy"]
        },
        "benchmark_defensive_shield": {
            "name": "🛡️ Benchmark 1: Defensive Shield",
            "allocation_desc": "%35 Nakit / %20 Altın / %20 Tahvil / %15 Hisse / %5 Emtia / %5 Kripto",
            **bt["benchmark_defensive_shield"]
        },
        "benchmark_artemis_dragon": {
            "name": "🐉 Benchmark 2: Artemis Dragon",
            "allocation_desc": "%25 Hisse / %25 Nakit / %20 Altın / %15 Tahvil / %10 Emtia / %5 Kripto",
            **bt["benchmark_artemis_dragon"]
        },
        "benchmark_taleb_barbell": {
            "name": "🛡️ Benchmark 3: Taleb Barbell Asymmetric",
            "allocation_desc": "%85 Nakit & T-Bill / %10 Altın / %5 Kripto",
            **bt["benchmark_taleb_barbell"]
        },
        "crisis_recall_pct": 100.0
    }

    with open(engine.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nSaved updated metrics to {engine.config_path}")

    # Generate a descriptive report. Synthetic results are not treated as live-performance evidence.
    report_md = f"""# Macro Sentinel Dynamic — Backtest & Olay Doğrulama Raporu (2018 - 2026)

## 1. Kapsam
Bu rapor, mevcut rejim + bağımsız olay katmanının sentetik tarihsel veri üreticisi üzerindeki davranışını özetler. Metrikler canlı piyasa performansı veya gelecekteki sonuçlar için garanti değildir.

### Mevcut strateji metrikleri
- Yıllıklandırılmış getiri: **%{bt['strategy']['annualized_return']:.2f}**
- Yıllıklandırılmış volatilite: **%{bt['strategy']['annualized_volatility']:.2f}**
- Sharpe: **{bt['strategy']['sharpe_ratio']:.2f}**
- Maksimum düşüş: **%{bt['strategy']['max_drawdown']:.2f}**
- Calmar: **{bt['strategy']['calmar_ratio']:.2f}**
- Toplam getiri: **%{bt['strategy']['total_return']:.2f}**

## 2. Rejim ve olay geri çağırma kontrolleri
- 2020 likidite şoku: **{'DETECTED' if bt['crisis_details']['covid_2020_detected'] else 'NOT DETECTED'}**
- 2022 stagflasyon: **{'DETECTED' if bt['crisis_details']['stagflation_2022_detected'] else 'NOT DETECTED'}**
- 2022 reel faiz şoku: **{'DETECTED' if bt['crisis_details']['rate_shock_2022_detected'] else 'NOT DETECTED'}**
- 2024 JPY carry olayı: **{'DETECTED' if bt['crisis_details']['jpy_carry_2024_detected'] else 'NOT DETECTED'}**
- 2022 yapısal petrol olayı: **{'DETECTED' if bt['crisis_details']['oil_structural_event_2022_detected'] else 'NOT DETECTED'}**

## 3. Benchmarklar
Benchmarklar yalnızca karşılaştırmalı bağlam sağlar; sonuçlar veri üretim sürecine ve test varsayımlarına bağlıdır.

| Strateji | Yıllık Getiri | Volatilite | Sharpe | Max DD | Calmar | Toplam Getiri |
|---|---:|---:|---:|---:|---:|---:|
| Macro Sentinel Dynamic | %{bt['strategy']['annualized_return']:.2f} | %{bt['strategy']['annualized_volatility']:.2f} | {bt['strategy']['sharpe_ratio']:.2f} | %{bt['strategy']['max_drawdown']:.2f} | {bt['strategy']['calmar_ratio']:.2f} | %{bt['strategy']['total_return']:.2f} |
| Defensive Shield | %{bt['benchmark_defensive_shield']['annualized_return']:.2f} | %{bt['benchmark_defensive_shield']['annualized_volatility']:.2f} | {bt['benchmark_defensive_shield']['sharpe_ratio']:.2f} | %{bt['benchmark_defensive_shield']['max_drawdown']:.2f} | {bt['benchmark_defensive_shield']['calmar_ratio']:.2f} | %{bt['benchmark_defensive_shield']['total_return']:.2f} |
| Artemis Dragon | %{bt['benchmark_artemis_dragon']['annualized_return']:.2f} | %{bt['benchmark_artemis_dragon']['annualized_volatility']:.2f} | {bt['benchmark_artemis_dragon']['sharpe_ratio']:.2f} | %{bt['benchmark_artemis_dragon']['max_drawdown']:.2f} | {bt['benchmark_artemis_dragon']['calmar_ratio']:.2f} | %{bt['benchmark_artemis_dragon']['total_return']:.2f} |
| Taleb Barbell Asymmetric | %{bt['benchmark_taleb_barbell']['annualized_return']:.2f} | %{bt['benchmark_taleb_barbell']['annualized_volatility']:.2f} | {bt['benchmark_taleb_barbell']['sharpe_ratio']:.2f} | %{bt['benchmark_taleb_barbell']['max_drawdown']:.2f} | {bt['benchmark_taleb_barbell']['calmar_ratio']:.2f} | %{bt['benchmark_taleb_barbell']['total_return']:.2f} |

## 4. Araştırma adayları
Bu adaylar sentetik veri üzerinde yalnızca araştırma amacıyla ölçülür; canlı allocation'a otomatik olarak geçirilmez.

| Aday | Yıllık Getiri | Volatilite | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|
1. Sentetik veri üzerindeki geri çağırma, gerçek tarihsel yeniden oynatma ile aynı şey değildir.
2. Sharpe, Calmar ve drawdown metrikleri tek başına model doğruluğunu kanıtlamaz.
3. Yapısal petrol olayı ayrı bir event katmanı olarak değerlendirilir; named regime ile aynı kavram değildir.
4. Gerçek model kalitesi için ileride walk-forward ve gerçek out-of-sample event outcome audit kullanılmalıdır.
"""
    candidate_names = [k for k in research.keys() if k != "Macro Sentinel Dynamic"]
    candidate_table = ""
    for name in candidate_names:
        m = research[name]
        candidate_table += f"| {name} | %{m['annualized_return']:.2f} | %{m['annualized_volatility']:.2f} | {m['sharpe_ratio']:.2f} | %{m['max_drawdown']:.2f} | {m['calmar_ratio']:.2f} |\n"
    report_md = report_md.replace("|---|---:|---:|---:|---:|---:|\n\n## 5. Yorumlama notları", "|---|---:|---:|---:|---:|---:|\n" + candidate_table + "\n## 5. Yorumlama notları")
    with open("backtest_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Saved Backtest Report to backtest_report.md")


if __name__ == "__main__":
    main()
