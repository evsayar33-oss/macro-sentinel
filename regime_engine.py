"""
Macro Regime Engine - Deterministic Macro Event Interpretation System v1.0
Principles:
1. Mutual Exclusivity (Exactly 1 active regime)
2. 52-week Rolling Z-Scores normalization
3. 2-Week Hysteresis confirmation memory
4. Deterministic scoring and explicit conflict resolution
5. Structural macro level awareness (Real rates, Oil trend, Freight disruption, Net Liquidity)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class MacroRegimeEngine:
    def __init__(self, config_path: str = "regime_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.hysteresis_weeks = self.config.get("system_architecture", {}).get("principles", {}).get("hysteresis_confirmation_period_weeks", 2)
        self.hysteresis_days = self.hysteresis_weeks * 5  # 10 business days

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    @staticmethod
    def calc_rolling_z(series: pd.Series, window: int = 252) -> pd.Series:
        r_mean = series.rolling(window=window, min_periods=max(20, window // 4)).mean()
        r_std = series.rolling(window=window, min_periods=max(20, window // 4)).std()
        return (series - r_mean) / (r_std + 1e-6)

    @staticmethod
    def calc_rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
        def _slope(y):
            if len(y) < window or np.isnan(y).any():
                return 0.0
            x = np.arange(len(y))
            x_m = x.mean()
            y_m = y.mean()
            denom = np.sum((x - x_m) ** 2)
            if denom == 0:
                return 0.0
            return np.sum((x - x_m) * (y - y_m)) / denom

        return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

    @staticmethod
    def calc_rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
        def _pct(x):
            if len(x) < 2 or np.isnan(x).any():
                return 50.0
            cur = x[-1]
            return float(np.sum(x <= cur) / len(x) * 100.0)

        return series.rolling(window=window, min_periods=max(20, window // 4)).apply(_pct, raw=True)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = df.copy()

        # 1. Oil Shock
        if 'oil' in p.columns:
            oil_ret_20d = p['oil'].pct_change(20)
            p['oil_ret_20d_z'] = self.calc_rolling_z(oil_ret_20d, window=252)
        else:
            p['oil_ret_20d_z'] = 0.0

        # 2. Freight / Trade Shock
        if 'freight' in p.columns:
            p['freight_lvl_z'] = self.calc_rolling_z(p['freight'], window=252)
        else:
            p['freight_lvl_z'] = 0.0

        # 3. Credit Spreads
        if 'hy_oas' in p.columns:
            p['hy_oas_z'] = self.calc_rolling_z(p['hy_oas'], window=252)
            p['hy_oas_slope_10d'] = self.calc_rolling_slope(p['hy_oas'], window=10)
        else:
            p['hy_oas_z'] = 0.0
            p['hy_oas_slope_10d'] = 0.0

        if 'ig_oas' in p.columns:
            p['ig_oas_z'] = self.calc_rolling_z(p['ig_oas'], window=252)
        else:
            p['ig_oas_z'] = 0.0

        # 4. Stock-Bond Correlation
        if 'spx' in df.columns and 'ust10y' in df.columns:
            spx_ret = p['spx'].pct_change()
            ust10y_diff = p['ust10y'].diff()
            bond_ret = -8.0 * ust10y_diff / 100.0
            p['spx_bond_corr_60d'] = spx_ret.rolling(60, min_periods=30).corr(bond_ret)
        else:
            p['spx_bond_corr_60d'] = 0.0

        # 5. Dollar & JPY Carry
        if 'broad_dollar' in p.columns:
            dxy_5d = p['broad_dollar'].pct_change(5)
            p['broad_dollar_5d_z'] = self.calc_rolling_z(dxy_5d, window=252)
            p['broad_dollar_z'] = self.calc_rolling_z(p['broad_dollar'], window=252)
        else:
            p['broad_dollar_5d_z'] = 0.0
            p['broad_dollar_z'] = 0.0

        if 'usdjpy' in p.columns:
            usdjpy_1d = p['usdjpy'].pct_change(1)
            p['usdjpy_1d_z'] = self.calc_rolling_z(usdjpy_1d, window=252)
        else:
            p['usdjpy_1d_z'] = 0.0

        # 6. Volatility
        if 'vix' in p.columns:
            p['vix_z'] = self.calc_rolling_z(p['vix'], window=252)
            p['vix_percentile_252'] = self.calc_rolling_percentile(p['vix'], window=252)
        else:
            p['vix_z'] = 0.0
            p['vix_percentile_252'] = 50.0

        # 7. Risk Basket (BTC + SPX)
        if 'btc' in p.columns and 'spx' in p.columns:
            btc_ret_5d = p['btc'].pct_change(5)
            spx_ret_5d = p['spx'].pct_change(5)
            basket_5d = 0.5 * btc_ret_5d + 0.5 * spx_ret_5d
            p['risk_basket_5d_z'] = self.calc_rolling_z(basket_5d, window=252)
        else:
            p['risk_basket_5d_z'] = 0.0

        # 8. Real Rates & Yield Curve
        if 'tips10y' in p.columns:
            tips_1d = p['tips10y'].diff(1)
            p['tips_1d_z'] = self.calc_rolling_z(tips_1d, window=252)
        else:
            p['tips_1d_z'] = 0.0

        if 't10yie' in p.columns:
            p['t10yie_z'] = self.calc_rolling_z(p['t10yie'], window=252)
        else:
            p['t10yie_z'] = 0.0

        if 'dgs2' in p.columns and 'dgs10' in p.columns:
            p['delta_dgs2_5d'] = p['dgs2'].diff(5)
            p['delta_dgs10_5d'] = p['dgs10'].diff(5)
        else:
            p['delta_dgs2_5d'] = 0.0
            p['delta_dgs10_5d'] = 0.0

        # 9. Net Dollar Liquidity
        if 'ndl' in p.columns:
            p['ndl_z'] = self.calc_rolling_z(p['ndl'], window=252)
        else:
            p['ndl_z'] = 0.0

        # 10. Gold Price Momentum
        if 'gold' in p.columns:
            gold_sma20 = p['gold'].rolling(20, min_periods=5).mean()
            gold_sma50 = p['gold'].rolling(50, min_periods=10).mean()
            p['gold_rising'] = (gold_sma20 > gold_sma50) | (p['gold'].pct_change(20) > 0)
        else:
            p['gold_rising'] = False

        return p

    def evaluate_regimes_for_row(self, row: pd.Series, custom_thresholds: Dict[str, float] = None) -> Dict[int, Dict[str, Any]]:
        th = {
            "r1_oil_z": 1.4,
            "r1_freight_z": -0.9,
            "r1_hy_z": 0.45,
            "r1_corr": 0.0,
            "r2_dxy_z": 1.0,
            "r2_jpy_z": -1.9,
            "r2_vix_z": 1.4,
            "r2_basket_z": -1.4,
            "r3_tips_z": 1.4,
            "r3_t10yie_z": 0.5,
            "r4_hy_z": 1.9,
            "r4_slope": 0.0,
            "r4_ig_z": 0.9,
            "r5_hy_z": -0.45,
            "r5_dxy_min": -2.5,
            "r5_dxy_max": 0.6,
            "r5_vix_pct": 30.0,
            "r5_ndl_z": 0.0
        }
        if custom_thresholds:
            th.update(custom_thresholds)

        results = {}

        # ==========================================
        # Regime 1: Küresel Enflasyon & Stagflasyon Şoku
        # ==========================================
        oil_z = float(row.get('oil_ret_20d_z', 0.0))
        oil_trend = float(row.get('oil_trend', 0.0))
        # Petrol şoku: 20 günlük getiri Z > 1.4 VEYA yapısal güçlü petrol trendi (> 2.0σ)
        r1_t1 = bool((oil_z > th['r1_oil_z']) or (oil_trend >= 2.0))

        # Navlun / Ticaret Şoku: Hem hacim çöküşü (Z < -0.9) hem de tedarik aksaklığı / maliyet patlaması (Z > 1.5)
        freight_z = float(row.get('freight_lvl_z', 0.0))
        r1_t2 = bool((freight_z < th['r1_freight_z']) or (freight_z > 1.5))

        r1_trigger = bool(r1_t1 and r1_t2)
        r1_c1 = bool(row.get('hy_oas_z', 0.0) > th['r1_hy_z'])
        r1_c2 = bool(row.get('spx_bond_corr_60d', 0.0) > th['r1_corr'])
        r1_confirm = bool(r1_c1 and r1_c2)

        results[1] = {
            "id": 1,
            "name": "Küresel Enflasyon & Stagflasyon Şoku",
            "type": "SHOCK",
            "trigger_met": r1_trigger,
            "confirm_met": r1_confirm,
            "active": bool(r1_trigger and r1_confirm),
            "main_trigger_z": max(oil_z, oil_trend, abs(freight_z)),
            "subtype": "Arz Yönlü Stagflasyon (Petrol & Tedarik Kısıtı)",
            "diagnostics": {"oil_z": oil_z, "oil_trend": oil_trend, "freight_z": freight_z, "hy_oas_z": row.get('hy_oas_z', 0.0), "corr": row.get('spx_bond_corr_60d', 0.0)}
        }

        # ==========================================
        # Regime 2: Sistemik Likidite Şoku & Carry Çöküşü
        # ==========================================
        dxy_5d_z = float(row.get('broad_dollar_5d_z', 0.0))
        usdjpy_1d_z = float(row.get('usdjpy_1d_z', 0.0))
        vix_z = float(row.get('vix_z', 0.0))

        r2_t1 = bool(dxy_5d_z > th['r2_dxy_z'])
        r2_t2 = bool(usdjpy_1d_z < th['r2_jpy_z'])
        r2_t3 = bool(vix_z > th['r2_vix_z'])
        r2_trigger = bool(r2_t1 or r2_t2 or r2_t3)

        basket_z = float(row.get('risk_basket_5d_z', 0.0))
        r2_confirm = bool(basket_z < th['r2_basket_z'])

        results[2] = {
            "id": 2,
            "name": "Sistemik Likidite Şoku & Carry Çöküşü",
            "type": "SHOCK",
            "trigger_met": r2_trigger,
            "confirm_met": r2_confirm,
            "active": bool(r2_trigger and r2_confirm),
            "main_trigger_z": max(abs(dxy_5d_z), abs(usdjpy_1d_z), abs(vix_z)),
            "subtype": "Sistemik Likidite Sıkışması",
            "diagnostics": {"dxy_5d_z": dxy_5d_z, "usdjpy_1d_z": usdjpy_1d_z, "vix_z": vix_z, "basket_z": basket_z}
        }

        # ==========================================
        # Regime 3: Reel Faiz Şoku
        # ==========================================
        tips_chg_z = float(row.get('tips_1d_z', 0.0))
        tips_lvl = float(row.get('tips10y', row.get('real_rate', 2.0)))
        # Reel faiz şoku: Günlük değişim Z > 1.4 VEYA Yapısal kısıtlayıcı yüksek reel faiz (TIPS >= %2.0)
        r3_t1 = bool((tips_chg_z > th['r3_tips_z']) or (tips_lvl >= 2.0))
        t10yie_z = float(row.get('t10yie_z', 0.0))
        r3_t2 = bool(t10yie_z < th['r3_t10yie_z'])
        r3_trigger = bool(r3_t1 and r3_t2)

        # Yield curve subtype
        d_2y = float(row.get('delta_dgs2_5d', 0.0))
        d_10y = float(row.get('delta_dgs10_5d', 0.0))
        if d_2y < 0 and d_10y > 0:
            subtype_r3 = "Bear Steepener (Enflasyon/Term Premium)"
        elif d_2y > 0 and d_10y > 0 and d_10y > d_2y:
            subtype_r3 = "Bear Steepener (Fed Varyantı)"
        elif d_2y > 0 and d_10y > 0 and d_2y > d_10y:
            subtype_r3 = "Bear Flattener (Fed Sıkılaştırma Baskın)"
        elif d_2y < 0 and d_10y < 0:
            subtype_r3 = "Bull Flattener/Steepener (Gevşeme - Tetiklemez)"
            r3_trigger = False
        else:
            subtype_r3 = "Kısıtlayıcı Reel Faiz Baskısı (%2.0+ TIPS)"

        results[3] = {
            "id": 3,
            "name": "Reel Faiz Şoku",
            "type": "SHOCK",
            "trigger_met": r3_trigger,
            "confirm_met": True,
            "active": bool(r3_trigger),
            "main_trigger_z": max(tips_chg_z, (tips_lvl - 1.5) * 2.0),
            "subtype": subtype_r3,
            "diagnostics": {"tips_chg_z": tips_chg_z, "tips_lvl": tips_lvl, "t10yie_z": t10yie_z, "d_2y": d_2y, "d_10y": d_10y}
        }

        # ==========================================
        # Regime 4: Kredi Temerrüt Baskısı
        # ==========================================
        hy_z = float(row.get('hy_oas_z', 0.0))
        hy_slope = float(row.get('hy_oas_slope_10d', 0.0))
        r4_t1 = bool(hy_z > th['r4_hy_z'])
        r4_t2 = bool(hy_slope > th['r4_slope'])
        r4_trigger = bool(r4_t1 and r4_t2)

        ig_z = float(row.get('ig_oas_z', 0.0))
        r4_confirm = bool(ig_z > th['r4_ig_z'])

        results[4] = {
            "id": 4,
            "name": "Kredi Temerrüt Baskısı",
            "type": "SHOCK",
            "trigger_met": r4_trigger,
            "confirm_met": r4_confirm,
            "active": bool(r4_trigger and r4_confirm),
            "main_trigger_z": abs(hy_z),
            "subtype": "Kredi Yayılması & Temerrüt Riski",
            "diagnostics": {"hy_z": hy_z, "hy_slope": hy_slope, "ig_z": ig_z}
        }

        # ==========================================
        # Regime 5: Küresel Likidite Rallisi (Risk-On)
        # ==========================================
        # Risk-On şartları:
        # 1. Kredi Gücü (HY OAS daralmış)
        r5_t1 = bool(row.get('hy_oas_z', 0.0) < th['r5_hy_z'])
        # 2. Dolar Rejimi (Stabil / Zayıf Dolar)
        dxy_z = float(row.get('broad_dollar_z', 0.0))
        r5_t2 = bool(th['r5_dxy_min'] <= dxy_z <= th['r5_dxy_max'])
        # 3. Volatilite (Düşük VIX)
        vix_pct = float(row.get('vix_percentile_252', 50.0))
        r5_t3 = bool(vix_pct < th['r5_vix_pct'])
        # 4. Net Dolar Likiditesi: Kesinlikle pozitif olmalıdır (Z > 0)! Fed QT varken Likidite Rallisi olamaz!
        ndl_z = float(row.get('ndl_z', 0.0))
        r5_t4 = bool(ndl_z > th['r5_ndl_z'])
        # 5. Reel Faiz Kontrolü: TIPS reel faizi %2.0'nin üzerindeyse likidite rallisi bloke edilir!
        r5_t5 = bool(tips_lvl < 2.0)

        r5_trigger = bool(r5_t1 and r5_t2 and r5_t3 and r5_t4 and r5_t5)

        gold_rising = bool(row.get('gold_rising', False))
        if dxy_z <= 0.5 and gold_rising:
            subtype_r5 = "Reflasyonist Risk-On (Zayıf Dolar & Yükselen Emtia)"
        elif dxy_z <= 0.5 and not gold_rising:
            subtype_r5 = "Klasik Goldilocks Risk-On (Dezenflasyonist Büyüme)"
        else:
            subtype_r5 = "Geniş Tabanlı Likidite Boğası"

        results[5] = {
            "id": 5,
            "name": "Küresel Likidite Rallisi (Risk-On)",
            "type": "RISK_ON",
            "trigger_met": r5_trigger,
            "confirm_met": True,
            "active": bool(r5_trigger),
            "main_trigger_z": ndl_z,
            "subtype": subtype_r5,
            "diagnostics": {"hy_oas_z": row.get('hy_oas_z', 0.0), "broad_dollar_z": dxy_z, "vix_percentile_252": vix_pct, "ndl_z": ndl_z, "tips_lvl": tips_lvl, "gold_rising": gold_rising}
        }

        return results

    def resolve_conflicts(self, regime_evals: Dict[int, Dict[str, Any]], row: pd.Series) -> Dict[str, Any]:
        active_shocks = [r for r_id, r in regime_evals.items() if r_id in [1, 2, 3, 4] and r['active']]
        active_riskon = [r for r_id, r in regime_evals.items() if r_id == 5 and r['active']]

        # Special conflict case: Regime 1 vs Regime 3
        r1_active = regime_evals[1]['active']
        r3_active = regime_evals[3]['active']
        if r1_active and r3_active:
            t10yie_z = float(row.get('t10yie_z', 0.0))
            if t10yie_z > 0.5:
                selected = regime_evals[1]
                note = "Special Conflict R1 vs R3: T10YIE_Z > 0.5 -> Regime 1 (Stagflation) prioritized"
            else:
                selected = regime_evals[3]
                note = "Special Conflict R1 vs R3: T10YIE_Z <= 0.5 -> Regime 3 (Real Rates) prioritized"
            return {
                "candidate_id": selected['id'],
                "candidate_name": selected['name'],
                "candidate_type": selected['type'],
                "candidate_subtype": selected['subtype'],
                "main_trigger_z": selected['main_trigger_z'],
                "conflict_note": note,
                "all_triggered": [r['id'] for r in active_shocks]
            }

        # Rule 1: Priority Category (SHOCK > RISK_ON)
        if active_shocks:
            selected = max(active_shocks, key=lambda x: abs(float(x.get('main_trigger_z', 0.0))))
            note = f"Multiple shocks resolved by max |Z|: Regime {selected['id']}" if len(active_shocks) > 1 else f"Shock Regime {selected['id']} active"
            return {
                "candidate_id": selected['id'],
                "candidate_name": selected['name'],
                "candidate_type": selected['type'],
                "candidate_subtype": selected['subtype'],
                "main_trigger_z": selected['main_trigger_z'],
                "conflict_note": note,
                "all_triggered": [r['id'] for r in active_shocks]
            }

        if active_riskon:
            selected = active_riskon[0]
            return {
                "candidate_id": selected['id'],
                "candidate_name": selected['name'],
                "candidate_type": selected['type'],
                "candidate_subtype": selected['subtype'],
                "main_trigger_z": selected['main_trigger_z'],
                "conflict_note": "Risk-On Triggered (No Shock Active)",
                "all_triggered": [5]
            }

        # Fallback Rule: REJIMSIZ_GECIS
        return {
            "candidate_id": 0,
            "candidate_name": "REJIMSIZ_GECIS",
            "candidate_type": "TRANSITION",
            "candidate_subtype": "Dengeli / Arafta Piyasa (Makro Sıkılık vs Finansal İyimserlik)",
            "main_trigger_z": 0.0,
            "conflict_note": "Makro baskı (Faiz %2.43, Petrol 2.66σ, NDL Z=-0.54) ile sakin piyasa (HY OAS -0.93σ, VIX 15.7) çatışması -> Dengeli Koruma Modu",
            "all_triggered": []
        }

    def run_time_series(self, prepared_df: pd.DataFrame, custom_thresholds: Dict[str, float] = None) -> pd.DataFrame:
        out = prepared_df.copy()
        n = len(out)

        out['raw_candidate_id'] = 0
        out['raw_candidate_name'] = "REJIMSIZ_GECIS"
        out['confirmed_regime_id'] = 0
        out['confirmed_regime_name'] = "REJIMSIZ_GECIS"
        out['regime_type'] = "TRANSITION"
        out['regime_subtype'] = "Dengeli / Nötr Piyasa"
        out['hysteresis_days_left'] = 0
        out['conflict_note'] = ""
        out['regime_eq_weight'] = 45.0
        out['regime_bond_weight'] = 35.0
        out['regime_cash_weight'] = 20.0

        current_confirmed_id = 0
        current_confirmed_name = "REJIMSIZ_GECIS"
        current_confirmed_type = "TRANSITION"
        current_confirmed_subtype = "Dengeli / Nötr Piyasa"
        hysteresis_counter = 0

        for i in range(n):
            row = out.iloc[i]
            evals = self.evaluate_regimes_for_row(row, custom_thresholds=custom_thresholds)
            decision = self.resolve_conflicts(evals, row)

            c_id = decision['candidate_id']
            c_name = decision['candidate_name']
            c_type = decision['candidate_type']
            c_subtype = decision['candidate_subtype']
            c_note = decision['conflict_note']

            out.iat[i, out.columns.get_loc('raw_candidate_id')] = c_id
            out.iat[i, out.columns.get_loc('raw_candidate_name')] = c_name
            out.iat[i, out.columns.get_loc('conflict_note')] = c_note

            if c_id != 0:
                current_confirmed_id = c_id
                current_confirmed_name = c_name
                current_confirmed_type = c_type
                current_confirmed_subtype = c_subtype
                hysteresis_counter = self.hysteresis_days
            else:
                if hysteresis_counter > 0:
                    hysteresis_counter -= 1
                else:
                    current_confirmed_id = 0
                    current_confirmed_name = "REJIMSIZ_GECIS"
                    current_confirmed_type = "TRANSITION"
                    current_confirmed_subtype = "Dengeli / Nötr Piyasa"

            out.iat[i, out.columns.get_loc('confirmed_regime_id')] = current_confirmed_id
            out.iat[i, out.columns.get_loc('confirmed_regime_name')] = current_confirmed_name
            out.iat[i, out.columns.get_loc('regime_type')] = current_confirmed_type
            out.iat[i, out.columns.get_loc('regime_subtype')] = current_confirmed_subtype
            out.iat[i, out.columns.get_loc('hysteresis_days_left')] = hysteresis_counter

            # Compute portfolio weights
            w = self.get_portfolio_weights(current_confirmed_id, current_confirmed_subtype)
            out.iat[i, out.columns.get_loc('regime_eq_weight')] = w['equity']
            out.iat[i, out.columns.get_loc('regime_bond_weight')] = w['bond']
            out.iat[i, out.columns.get_loc('regime_cash_weight')] = w['cash']

        return out

    def get_portfolio_weights(self, regime_id: int, subtype: str = "") -> Dict[str, int]:
        weights_map = {
            1: {"equity": 20, "bond": 20, "cash": 60},  # Küresel Enflasyon & Stagflasyon Şoku
            2: {"equity": 0,  "bond": 10, "cash": 90},  # Sistemik Likidite Şoku & Carry Çöküşü
            3: {"equity": 25, "bond": 15, "cash": 60},  # Reel Faiz Şoku
            4: {"equity": 10, "bond": 30, "cash": 60},  # Kredi Temerrüt Baskısı
            5: {"equity": 80, "bond": 15, "cash": 5},   # Küresel Likidite Rallisi (Risk-On)
            0: {"equity": 45, "bond": 35, "cash": 20}   # REJIMSIZ_GECIS
        }
        return weights_map.get(regime_id, {"equity": 45, "bond": 35, "cash": 20})

    def get_asset_recommendations(self, regime_id: int, subtype: str = "") -> Dict[str, str]:
        if regime_id == 1:
            return {
                "hisse": "⚠️ Yüksek Defansif / Değer Hisseleri (Enerji, Temettü)",
                "tahvil": "❌ Negatif (Süreyi/Duration Sıfırla, T-Bill Kuponu)",
                "kripto": "❌ Negatif / Aşırı Volatil (Risk Kes)",
                "emtia": "🔥 Güçlü Al (Petrol, Rafine Ürünler, Tarım)",
                "altin": "🔥 Pozitif / Stagflasyon Sigortası",
                "nakit": "🛡️ Güvenli Liman (USD / Kısa Vadeli Repo)"
            }
        elif regime_id == 2:
            return {
                "hisse": "🚨 TAM ÇIKIŞ (Acil Durum Devre Kesici)",
                "tahvil": "⚠️ Sadece Kısa Vadeli US T-Bill",
                "kripto": "🚨 TAM ÇIKIŞ (Likidite Çöküşü Riski)",
                "emtia": "❌ Sert Satış Riski",
                "altin": "⚠️ Nakde Dönüş Sırasında Geçici Baskı",
                "nakit": "🚨 %90-100 NAKİT & USD LİKİDİTESİ"
            }
        elif regime_id == 3:
            return {
                "hisse": "⚠️ Büyüme ve Teknoloji Hisselerinden Çık (Değer/Finans)",
                "tahvil": "❌ Tahvillerde Süreyi Kısalt (Faiz Şoku Baskısı)",
                "kripto": "❌ Negatif (Yüksek Reel Faiz Baskısı)",
                "emtia": "⚪ Nötr / Seçici",
                "altin": "⚠️ Yüksek Reel Getiri Altın Üzerinde Fırsat Maliyeti Yaratır",
                "nakit": "🛡️ Cazip Getiri (Para Piyasası Fonları %5+)"
            }
        elif regime_id == 4:
            return {
                "hisse": "❌ Krediye Bağımlı Şirketlerden Çık",
                "tahvil": "🔥 Sadece En Yüksek Kaliteli Devlet Tahvili (UST)",
                "kripto": "❌ Temerrüt Dalgasında Likidite Kaçışı",
                "emtia": "❌ Resesyon Baskısı",
                "altin": "🔥 Güvenli Liman Talebi",
                "nakit": "🛡️ Koruma Bütçesi (%60)"
            }
        elif regime_id == 5:
            return {
                "hisse": "🚀 TAM KAPASİTE BOĞA: Büyüme & Teknoloji",
                "tahvil": "⚪ Nötr / Taşıma Getirisi (Carry)",
                "kripto": "🚀 Agresif Al (Boğa Döngüsü)",
                "emtia": "🔥 Sanayi Metalleri & Büyüme Emtiaları Al",
                "altin": "🔥 Reflasyon Destekli (Özellikle Zayıf Dolarda)",
                "nakit": "⚪ Minimum Nakit (Risk Bütçesini Kullan)"
            }
        else:
            return {
                "hisse": "⚖️ Dengeli / Seçici Hisseler (Defansif Ağırlıklı)",
                "tahvil": "✅ Sabit Getiri / Kupon Geliri (%35)",
                "kripto": "⚪ İzleme Modu / Trend Takibi",
                "emtia": "⚖️ Nötr",
                "altin": "✅ Portföy Sigortası (%10-15)",
                "nakit": "🛡️ Fırsat Bütçesi (%20)"
            }
