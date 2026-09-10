"""
Deep Quantitative Strategy Search & Optimization Engine
Evaluates if higher Sharpe / Calmar or lower Max DD is mathematically possible:
1. Macro Sentinel Apex (Dynamic Regime with Calibrated Optimums: Stagflation Oil 30%, Risk-On BTC 10%)
2. All-Weather Convex Shield (Bridgewater style upgraded with BTC convexity)
3. Tail-Risk Asymmetric Parity (Nassim Taleb Barbell style)
4. Dynamic Volatility-Targeted Momentum (Vol-Targeting overlay)
"""

import sys
sys.path.append('/tmp/macro-sentinel')
import numpy as np
import pandas as pd
from backtest_regimes import generate_synthetic_macro_history
from regime_engine import MacroRegimeEngine

engine = MacroRegimeEngine()
raw_data = generate_synthetic_macro_history()
prepared = engine.prepare_indicators(raw_data)
classified = engine.run_time_series(prepared)

spx_ret = classified['spx'].pct_change().fillna(0.0)
bond_ret = classified['ust10y'].pct_change().fillna(0.0)
cash_ret = (classified['dgs2'] / 100.0) / 252.0
gold_ret = classified['gold'].pct_change().fillna(0.0)
oil_ret = classified['oil'].pct_change().fillna(0.0)
btc_ret = classified['btc'].pct_change().fillna(0.0)

ann_factor = 252.0
n_years = len(classified) / ann_factor

def evaluate_ret_series(ret_series, name="Strategy"):
    cum = (1.0 + ret_series).cumprod()
    tot_ret = (cum.iloc[-1] - 1.0) * 100.0
    ann_ret = (cum.iloc[-1] ** (1.0 / n_years) - 1.0) * 100.0
    ann_vol = ret_series.std() * np.sqrt(ann_factor) * 100.0
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = abs(dd.min()) * 100.0
    sharpe = (ann_ret - 3.0) / max(ann_vol, 0.01)
    calmar = ann_ret / max(max_dd, 0.01)
    return {
        "Name": name,
        "Annual Return (%)": round(float(ann_ret), 2),
        "Annual Vol (%)": round(float(ann_vol), 2),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Max Drawdown (%)": round(float(max_dd), 2),
        "Calmar Ratio": round(float(calmar), 2),
        "Total Return (%)": round(float(tot_ret), 2),
        "cum": cum,
        "ret": ret_series
    }

# 1. Macro Sentinel Standard (Current in Repo)
w_csh = (classified['regime_cash_weight'] / 100.0).shift(1).fillna(0.35)
w_gld = (classified['regime_gold_weight'] / 100.0).shift(1).fillna(0.20)
w_bnd = (classified['regime_bond_weight'] / 100.0).shift(1).fillna(0.20)
w_eq = (classified['regime_eq_weight'] / 100.0).shift(1).fillna(0.15)
w_cmd = (classified['regime_commodity_weight'] / 100.0).shift(1).fillna(0.05)
w_crp = (classified['regime_crypto_weight'] / 100.0).shift(1).fillna(0.05)
strat_current = w_csh*cash_ret + w_gld*gold_ret + w_bnd*bond_ret + w_eq*spx_ret + w_cmd*oil_ret + w_crp*btc_ret
res_current = evaluate_ret_series(strat_current, "🏛️ Macro Sentinel (Mevcut)")

# 2. Macro Sentinel Apex (Hassasiyet Testi Optimizasyonu: Stagflasyonda %30 Petrol, Risk-On'da %10 Kripto)
apex_r = []
for idx in range(len(classified)):
    if idx == 0:
        apex_r.append(0.0)
        continue
    prev_reg = classified['confirmed_regime_id'].iloc[idx-1]
    if prev_reg == 1: # Stagflasyon: %40 Nakit, %25 Altın, %30 Petrol, %5 Tahvil
        w_c, w_g, w_b, w_e, w_o, w_k = 0.40, 0.25, 0.05, 0.0, 0.30, 0.0
    elif prev_reg == 2: # Likidite Şoku: %95 Nakit, %5 Tahvil
        w_c, w_g, w_b, w_e, w_o, w_k = 0.95, 0.0, 0.05, 0.0, 0.0, 0.0
    elif prev_reg == 3: # Reel Faiz Şoku
        w_c, w_g, w_b, w_e, w_o, w_k = 0.6726, 0.11, 0.10, 0.088, 0.0294, 0.0
    elif prev_reg == 4: # Kredi Temerrüt
        w_c, w_g, w_b, w_e, w_o, w_k = 0.65, 0.20, 0.15, 0.0, 0.0, 0.0
    elif prev_reg == 5: # Risk On: %10 Nakit, %10 Altın, %0 Tahvil, %70 Hisse, %10 Kripto
        w_c, w_g, w_b, w_e, w_o, w_k = 0.10, 0.10, 0.0, 0.70, 0.0, 0.10
    else: # Neutral
        w_c, w_g, w_b, w_e, w_o, w_k = 0.35, 0.20, 0.20, 0.15, 0.05, 0.05
    r = w_c*cash_ret.iloc[idx] + w_g*gold_ret.iloc[idx] + w_b*bond_ret.iloc[idx] + w_e*spx_ret.iloc[idx] + w_o*oil_ret.iloc[idx] + w_k*btc_ret.iloc[idx]
    apex_r.append(r)
strat_apex = pd.Series(apex_r, index=classified.index)
res_apex = evaluate_ret_series(strat_apex, "⚡ Macro Sentinel Apex (Optimize)")

# 3. Taleb Barbell Asymmetric (%85 Ultra-Safe Nakit/T-Bill + %10 Altın + %5 Konveks Kripto)
taleb_r = 0.85*cash_ret + 0.10*gold_ret + 0.05*btc_ret
res_taleb = evaluate_ret_series(taleb_r, "🛡️ Barbell Asymmetric (Taleb Style)")

# 4. Bridgewater All-Weather Plus (%40 Tahvil, %30 Hisse, %15 Altın, %10 Emtia, %5 Kripto)
allweather_r = 0.40*bond_ret + 0.30*spx_ret + 0.15*gold_ret + 0.10*oil_ret + 0.05*btc_ret
res_allweather = evaluate_ret_series(allweather_r, "🌐 All-Weather Plus (Ray Dalio)")

# 5. Mevcut Benchmarklar
def_r = 0.35*cash_ret + 0.20*gold_ret + 0.20*bond_ret + 0.15*spx_ret + 0.05*oil_ret + 0.05*btc_ret
art_r = 0.25*spx_ret + 0.25*cash_ret + 0.20*gold_ret + 0.15*bond_ret + 0.10*oil_ret + 0.05*btc_ret
res_def = evaluate_ret_series(def_r, "🛡️ Benchmark 1: Defensive Shield")
res_art = evaluate_ret_series(art_r, "🐉 Benchmark 2: Artemis Dragon")

summary = pd.DataFrame([res_apex, res_current, res_def, res_art, res_allweather, res_taleb])
cols = ["Name", "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio", "Total Return (%)"]
print(summary[cols].to_string(index=False))
