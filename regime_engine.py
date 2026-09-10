"""
macro-event-interpretation-system
Version: 1.0
Autonomous deterministic macro regime classification and dynamic asset allocation engine.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "regime_config.json")


class MacroRegimeEngine:
    """
    Implements the Macro Event Interpretation System v1.0.
    Calculates 52-week rolling z-scores, deterministic triggers, confirmations,
    conflict resolution, 2-week hysteresis tracking, and dynamic portfolio weights.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self.hysteresis_days = int(self.config["system_architecture"]["principles"].get("hysteresis_confirmation_period_weeks", 2) * 5)
        self.window_52w = 252 # Trading days in ~52 weeks

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

    @staticmethod
    def calc_rolling_z(series: pd.Series, window: int = 252) -> pd.Series:
        """Calculates 52-week rolling Z-score."""
        mean = series.rolling(window=window, min_periods=max(20, window // 5)).mean()
        std = series.rolling(window=window, min_periods=max(20, window // 5)).std()
        return (series - mean) / (std + 1e-6)

    @staticmethod
    def calc_rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
        """Calculates rolling linear regression slope over n days."""
        x = np.arange(window)
        x_dev = x - x.mean()
        x_var = (x_dev ** 2).sum()

        def _slope(y_vals):
            if len(y_vals) < window or np.isnan(y_vals).any():
                return 0.0
            y_dev = y_vals - np.mean(y_vals)
            return np.sum(x_dev * y_dev) / (x_var + 1e-6)

        return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

    @staticmethod
    def calc_rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
        """Calculates rolling percentile rank (0 to 100)."""
        def _pct_rank(vals):
            last = vals[-1]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                return 50.0
            return (np.sum(valid <= last) / len(valid)) * 100.0

        return series.rolling(window=window, min_periods=max(20, window // 5)).apply(_pct_rank, raw=True)

    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares all macroeconomic and market indicators required for the 5 regimes.
        Expected columns or proxies in `data`:
        - 'oil': Brent or WTI Spot (e.g. CL=F or BZ=F)
        - 'freight': Baltic Dry Index (BDI or proxy BDRY)
        - 'hy_oas': FRED:BAMLH0A0HYM2
        - 'ig_oas': FRED:BAMLC0A0CM
        - 'spx': S&P 500 (ES=F or ^GSPC)
        - 'ust10y': 10Y Treasury yield (DGS10) or bond ETF (TLT)
        - 'broad_dollar': FRED:DTWEXBGS (Nominal Broad Dollar)
        - 'usdjpy': USD/JPY Spot (USDJPY=X or JPY=X)
        - 'vix': FRED:VIXCLS or ^VIX
        - 'btc': BTC-USD
        - 'tips10y': FRED:DFII10 (10Y TIPS Real Rate)
        - 't10yie': FRED:T10YIE (10Y Breakeven Inflation Rate)
        - 'dgs2': FRED:DGS2 (2Y Treasury Constant Maturity)
        - 'dgs10': FRED:DGS10 (10Y Treasury Constant Maturity)
        - 'ndl': Net Dollar Liquidity (WALCL - WTREGEN - RRPONTSYD*1000)
        - 'gold': Gold Spot (GC=F)
        """
        df = data.copy()

        # Regime 1 Indicators
        if 'oil' in df.columns:
            oil_ret_20d = df['oil'].pct_change(20)
            df['oil_ret_20d_z'] = self.calc_rolling_z(oil_ret_20d, self.window_52w)
        else:
            df['oil_ret_20d_z'] = 0.0

        if 'freight' in df.columns:
            df['freight_lvl_z'] = self.calc_rolling_z(df['freight'], self.window_52w)
        else:
            df['freight_lvl_z'] = 0.0

        if 'hy_oas' in df.columns:
            df['hy_oas_z'] = self.calc_rolling_z(df['hy_oas'], self.window_52w)
            df['hy_oas_slope_10d'] = self.calc_rolling_slope(df['hy_oas'], window=10)
        else:
            df['hy_oas_z'] = 0.0
            df['hy_oas_slope_10d'] = 0.0

        if 'spx' in df.columns and ('ust10y' in df.columns or 'tlt' in df.columns):
            spx_ret = df['spx'].pct_change()
            bond_ret = df['ust10y'].pct_change() if 'ust10y' in df.columns else df['tlt'].pct_change()
            df['spx_bond_corr_60d'] = spx_ret.rolling(60, min_periods=20).corr(bond_ret)
        else:
            df['spx_bond_corr_60d'] = 0.0

        # Regime 2 Indicators
        if 'broad_dollar' in df.columns:
            dxy_chg_5d = df['broad_dollar'].pct_change(5)
            df['broad_dollar_5d_z'] = self.calc_rolling_z(dxy_chg_5d, self.window_52w)
            df['broad_dollar_z'] = self.calc_rolling_z(df['broad_dollar'], self.window_52w)
        else:
            df['broad_dollar_5d_z'] = 0.0
            df['broad_dollar_z'] = 0.0

        if 'usdjpy' in df.columns:
            usdjpy_chg_1d = df['usdjpy'].pct_change(1)
            df['usdjpy_1d_z'] = self.calc_rolling_z(usdjpy_chg_1d, self.window_52w)
        else:
            df['usdjpy_1d_z'] = 0.0

        if 'vix' in df.columns:
            df['vix_z'] = self.calc_rolling_z(df['vix'], self.window_52w)
            df['vix_percentile_252'] = self.calc_rolling_percentile(df['vix'], self.window_52w)
        else:
            df['vix_z'] = 0.0
            df['vix_percentile_252'] = 50.0

        # Confirmation 2: Risk asset basket (BTC + SPX equal-weighted)
        if 'btc' in df.columns and 'spx' in df.columns:
            basket_ret_5d = 0.5 * df['btc'].pct_change(5) + 0.5 * df['spx'].pct_change(5)
            df['risk_basket_5d_z'] = self.calc_rolling_z(basket_ret_5d, self.window_52w)
        elif 'spx' in df.columns:
            basket_ret_5d = df['spx'].pct_change(5)
            df['risk_basket_5d_z'] = self.calc_rolling_z(basket_ret_5d, self.window_52w)
        else:
            df['risk_basket_5d_z'] = 0.0

        # Regime 3 Indicators
        if 'tips10y' in df.columns:
            tips_chg_1d = df['tips10y'].diff(1)
            df['tips_1d_z'] = self.calc_rolling_z(tips_chg_1d, self.window_52w)
        else:
            df['tips_1d_z'] = 0.0

        if 't10yie' in df.columns:
            df['t10yie_z'] = self.calc_rolling_z(df['t10yie'], self.window_52w)
        else:
            df['t10yie_z'] = 0.0

        if 'dgs2' in df.columns and 'dgs10' in df.columns:
            df['delta_dgs2'] = df['dgs2'].diff(5)
            df['delta_dgs10'] = df['dgs10'].diff(5)
        else:
            df['delta_dgs2'] = 0.0
            df['delta_dgs10'] = 0.0

        # Regime 4 Indicators
        if 'ig_oas' in df.columns:
            df['ig_oas_z'] = self.calc_rolling_z(df['ig_oas'], self.window_52w)
        else:
            df['ig_oas_z'] = 0.0

        # Regime 5 Indicators
        if 'ndl' in df.columns:
            df['ndl_z'] = self.calc_rolling_z(df['ndl'], self.window_52w)
        else:
            df['ndl_z'] = 0.0

        if 'gold' in df.columns:
            gold_sma20 = df['gold'].rolling(20).mean()
            df['gold_rising'] = df['gold'] > gold_sma20
        else:
            df['gold_rising'] = False

        return df

    def evaluate_regimes_for_row(self, row: pd.Series, custom_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluates trigger and confirmation conditions for all 5 regimes on a single bar/row.
        Returns evaluation dict with statuses, sub-types, main trigger z-scores, and raw signals.
        """
        th = {
            "r1_oil_z": 1.5,
            "r1_freight_z": -1.0,
            "r1_hy_z": 0.5,
            "r1_corr": 0.0,
            "r2_dxy_z": 1.0,
            "r2_jpy_z": -2.0,
            "r2_vix_z": 1.5,
            "r2_basket_z": -1.5,
            "r3_tips_z": 1.5,
            "r3_t10yie_z": 0.5,
            "r4_hy_z": 2.0,
            "r4_slope": 0.0,
            "r4_ig_z": 1.0,
            "r5_hy_z": -0.45,
            "r5_dxy_min": -2.5,
            "r5_dxy_max": 0.6,
            "r5_vix_pct": 35.0,
            "r5_ndl_z": 0.0
        }
        if custom_thresholds:
            th.update(custom_thresholds)

        results = {}

        # ==========================================
        # Regime 1: Küresel Enflasyon & Stagflasyon Şoku
        # ==========================================
        r1_t1 = row.get('oil_ret_20d_z', 0.0) > th['r1_oil_z']
        r1_t2 = row.get('freight_lvl_z', 0.0) < th['r1_freight_z']
        r1_trigger = bool(r1_t1 and r1_t2)

        r1_c1 = row.get('hy_oas_z', 0.0) > th['r1_hy_z']
        r1_c2 = row.get('spx_bond_corr_60d', 0.0) > th['r1_corr']
        r1_confirm = bool(r1_c1 and r1_c2)

        r1_active = bool(r1_trigger and r1_confirm)
        r1_main_z = float(row.get('oil_ret_20d_z', 0.0))

        results[1] = {
            "id": 1,
            "name": "Küresel Enflasyon & Stagflasyon Şoku",
            "type": "SHOCK",
            "trigger_met": r1_trigger,
            "confirm_met": r1_confirm,
            "active": r1_active,
            "main_trigger_z": r1_main_z,
            "subtype": "Stagflasyon / Enflasyon Şoku",
            "diagnostics": {
                "oil_ret_20d_z": float(row.get('oil_ret_20d_z', 0.0)),
                "freight_lvl_z": float(row.get('freight_lvl_z', 0.0)),
                "hy_oas_z": float(row.get('hy_oas_z', 0.0)),
                "spx_bond_corr_60d": float(row.get('spx_bond_corr_60d', 0.0))
            }
        }

        # ==========================================
        # Regime 2: Sistemik Likidite Şoku & Carry Çöküşü
        # ==========================================
        r2_t1 = row.get('broad_dollar_5d_z', 0.0) > th['r2_dxy_z']
        r2_t2 = row.get('usdjpy_1d_z', 0.0) < th['r2_jpy_z']
        r2_t3 = row.get('vix_z', 0.0) > th['r2_vix_z']
        r2_trigger = bool(r2_t1 or r2_t2 or r2_t3)

        r2_confirm = bool(row.get('risk_basket_5d_z', 0.0) < th['r2_basket_z'])
        r2_active = bool(r2_trigger and r2_confirm)

        # Main trigger Z: highest absolute Z among triggers
        r2_z_candidates = [
            abs(float(row.get('broad_dollar_5d_z', 0.0))),
            abs(float(row.get('usdjpy_1d_z', 0.0))),
            abs(float(row.get('vix_z', 0.0)))
        ]
        r2_main_z = float(max(r2_z_candidates))

        # Identify sub-flavor
        r2_sub = []
        if r2_t2: r2_sub.append("JPY Carry Çöküşü")
        if r2_t3: r2_sub.append("Volatilite Patlaması")
        if r2_t1: r2_sub.append("Geniş Dolar Sıkışması")
        subtype_r2 = " & ".join(r2_sub) if r2_sub else "Sistemik Likidite Şoku"

        results[2] = {
            "id": 2,
            "name": "Sistemik Likidite Şoku & Carry Çöküşü",
            "type": "SHOCK",
            "trigger_met": r2_trigger,
            "confirm_met": r2_confirm,
            "active": r2_active,
            "main_trigger_z": r2_main_z,
            "subtype": subtype_r2,
            "diagnostics": {
                "broad_dollar_5d_z": float(row.get('broad_dollar_5d_z', 0.0)),
                "usdjpy_1d_z": float(row.get('usdjpy_1d_z', 0.0)),
                "vix_z": float(row.get('vix_z', 0.0)),
                "risk_basket_5d_z": float(row.get('risk_basket_5d_z', 0.0))
            }
        }

        # ==========================================
        # Regime 3: Reel Faiz Şoku
        # ==========================================
        # Reel faiz şoku hem ani değişim (Z > 1.4) hem de kısıtlayıcı yüksek seviye (TIPS > %2.0 ve Z > 1.0) ile tetiklenebilir
        tips_lvl = float(row.get('tips10y', row.get('real_rate', 2.0)))
        tips_chg_z = float(row.get('tips_1d_z', 0.0))
        r3_t1 = bool((tips_chg_z > th['r3_tips_z']) or (tips_lvl >= 2.0 and tips_chg_z >= -0.2))
        r3_t2 = row.get('t10yie_z', 0.0) < th['r3_t10yie_z']
        r3_trigger = bool(r3_t1 and r3_t2)
        r3_confirm = True # Specification states trigger contains the differentiator
        r3_active = bool(r3_trigger and r3_confirm)
        r3_main_z = float(row.get('tips_1d_z', 0.0))

        # Sub-types:
        d_dgs2 = float(row.get('delta_dgs2', 0.0))
        d_dgs10 = float(row.get('delta_dgs10', 0.0))
        if d_dgs2 < 0 and d_dgs10 > 0:
            subtype_r3 = "Bear Steepener (Enflasyon/Term Premium)"
        elif d_dgs2 > 0 and d_dgs10 > 0 and d_dgs10 > d_dgs2:
            subtype_r3 = "Bear Steepener (Fed Varyantı)"
        elif d_dgs2 > 0 and d_dgs10 > 0 and d_dgs2 > d_dgs10:
            subtype_r3 = "Bear Flattener (Fed Sıkılaştırma Baskın)"
        elif d_dgs2 < 0 and d_dgs10 < 0:
            subtype_r3 = "Bull Flattener/Steepener (Gevşeme - Tetiklemez)"
            # Note: per rule, bull easing does not confirm real rate shock
            if subtype_r3.startswith("Bull"):
                r3_active = False
        else:
            subtype_r3 = "Reel Faiz Artışı (Eğri Nötr)"

        results[3] = {
            "id": 3,
            "name": "Reel Faiz Şoku",
            "type": "SHOCK",
            "trigger_met": r3_trigger,
            "confirm_met": r3_confirm,
            "active": r3_active,
            "main_trigger_z": r3_main_z,
            "subtype": subtype_r3,
            "diagnostics": {
                "tips_1d_z": float(row.get('tips_1d_z', 0.0)),
                "t10yie_z": float(row.get('t10yie_z', 0.0)),
                "delta_dgs2": d_dgs2,
                "delta_dgs10": d_dgs10
            }
        }

        # ==========================================
        # Regime 4: Kredi Temerrüt Baskısı
        # ==========================================
        r4_t1 = row.get('hy_oas_z', 0.0) > th['r4_hy_z']
        r4_t2 = row.get('hy_oas_slope_10d', 0.0) > th['r4_slope']
        r4_trigger = bool(r4_t1 and r4_t2)

        r4_c1 = row.get('ig_oas_z', 0.0) > th['r4_ig_z']
        r4_confirm = bool(r4_c1)

        r4_active = bool(r4_trigger and r4_confirm)
        r4_main_z = float(row.get('hy_oas_z', 0.0))

        results[4] = {
            "id": 4,
            "name": "Kredi Temerrüt Baskısı",
            "type": "SHOCK",
            "trigger_met": r4_trigger,
            "confirm_met": r4_confirm,
            "active": r4_active,
            "main_trigger_z": r4_main_z,
            "subtype": "Kredi / Yayılma Stresi (Credit Crunch)",
            "diagnostics": {
                "hy_oas_z": float(row.get('hy_oas_z', 0.0)),
                "hy_oas_slope_10d": float(row.get('hy_oas_slope_10d', 0.0)),
                "ig_oas_z": float(row.get('ig_oas_z', 0.0))
            }
        }

        # ==========================================
        # Regime 5: Küresel Likidite Rallisi (Risk-On)
        # ==========================================
        r5_t1 = row.get('hy_oas_z', 0.0) < th['r5_hy_z']
        dxy_z = float(row.get('broad_dollar_z', 0.0))
        r5_t2 = bool(th['r5_dxy_min'] <= dxy_z <= th['r5_dxy_max'])
        vix_pct = float(row.get('vix_percentile_252', 50.0))
        r5_t3 = bool(vix_pct < th['r5_vix_pct'])
        r5_t4 = bool(row.get('ndl_z', 0.0) > th['r5_ndl_z'])

        r5_trigger = bool(r5_t1 and r5_t2 and r5_t3 and r5_t4)
        r5_confirm = True
        r5_active = bool(r5_trigger)
        r5_main_z = float(row.get('ndl_z', 0.0))

        # Post-hoc Sub-types:
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
            "confirm_met": r5_confirm,
            "active": r5_active,
            "main_trigger_z": r5_main_z,
            "subtype": subtype_r5,
            "diagnostics": {
                "hy_oas_z": float(row.get('hy_oas_z', 0.0)),
                "broad_dollar_z": dxy_z,
                "vix_percentile_252": vix_pct,
                "ndl_z": float(row.get('ndl_z', 0.0)),
                "gold_rising": gold_rising
            }
        }

        return results

    def resolve_conflicts(self, regime_results: Dict[int, Dict[str, Any]], row: pd.Series) -> Dict[str, Any]:
        """
        Applies deterministic priority rules and conflict resolution:
        1. Category Priority: Shock Regimes (1, 2, 3, 4) > Risk-On (5)
        2. If multiple shock regimes trigger:
           - Special conflict case: R1 vs R3: IF T10YIE 52w_Z > +0.5 THEN R1 ELSE R3
           - Otherwise: select regime with highest absolute Z-score of its main trigger indicator.
        3. If only Regime 5 triggers: select Regime 5.
        4. Fallback: 'REJIMSIZ_GECIS'
        """
        active_shocks = [r for r_id, r in regime_results.items() if r['type'] == 'SHOCK' and r['active']]
        active_risk_on = [r for r_id, r in regime_results.items() if r['type'] == 'RISK_ON' and r['active']]

        if active_shocks:
            # Check special case: Regime 1 vs Regime 3
            shock_ids = [s['id'] for s in active_shocks]
            if 1 in shock_ids and 3 in shock_ids:
                t10yie_z = float(row.get('t10yie_z', 0.0))
                if t10yie_z > 0.5:
                    chosen = regime_results[1]
                    conflict_reason = f"Special Conflict R1 vs R3: T10YIE 52w_Z ({t10yie_z:.2f}) > 0.5 -> Selected Regime 1 (Enflasyon Şoku)"
                else:
                    chosen = regime_results[3]
                    conflict_reason = f"Special Conflict R1 vs R3: T10YIE 52w_Z ({t10yie_z:.2f}) <= 0.5 -> Selected Regime 3 (Reel Faiz Şoku)"
            else:
                # Select the shock regime with highest absolute Z-score of main trigger
                chosen = max(active_shocks, key=lambda s: abs(s['main_trigger_z']))
                conflict_reason = f"Multiple Shocks: Selected Regime {chosen['id']} with max |Z| = {abs(chosen['main_trigger_z']):.2f}"

            return {
                "candidate_id": chosen['id'],
                "candidate_name": chosen['name'],
                "candidate_type": chosen['type'],
                "candidate_subtype": chosen['subtype'],
                "main_trigger_z": chosen['main_trigger_z'],
                "conflict_note": conflict_reason,
                "all_triggered": [s['id'] for s in active_shocks]
            }

        elif active_risk_on:
            chosen = active_risk_on[0]
            return {
                "candidate_id": chosen['id'],
                "candidate_name": chosen['name'],
                "candidate_type": chosen['type'],
                "candidate_subtype": chosen['subtype'],
                "main_trigger_z": chosen['main_trigger_z'],
                "conflict_note": "Risk-On Triggered (No Shock Active)",
                "all_triggered": [5]
            }

        else:
            return {
                "candidate_id": 0,
                "candidate_name": "REJIMSIZ_GECIS",
                "candidate_type": "TRANSITION",
                "candidate_subtype": "Dengeli / Nötr Piyasa",
                "main_trigger_z": 0.0,
                "conflict_note": "No threshold met -> Fallback rule applied",
                "all_triggered": []
            }

    def run_time_series(self, prepared_df: pd.DataFrame, custom_thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Executes regime evaluation across the entire time series,
        enforcing mutual exclusivity, conflict resolution, and 2-week hysteresis tracking.
        """
        df = prepared_df.copy()

        regime_ids = []
        regime_names = []
        regime_types = []
        regime_subtypes = []
        confirmed_ids = []
        confirmed_names = []
        hysteresis_days_left = []
        conflict_notes = []
        eq_weights = []
        bond_weights = []
        cash_weights = []

        # State machine variables
        current_confirmed_id = 0
        current_confirmed_name = "REJIMSIZ_GECIS"
        current_confirmed_type = "TRANSITION"
        current_confirmed_subtype = "Dengeli / Nötr Piyasa"
        days_in_hysteresis = 0

        for idx, row in df.iterrows():
            regime_results = self.evaluate_regimes_for_row(row, custom_thresholds)
            decision = self.resolve_conflicts(regime_results, row)
            candidate_id = decision['candidate_id']

            # Hysteresis Logic:
            # - If candidate is a SHOCK or RISK_ON regime: switch immediately or confirm
            # - If candidate is REJIMSIZ_GECIS (0): retain previous confirmed regime for up to hysteresis_days
            if candidate_id != 0:
                # Active signal present
                current_confirmed_id = candidate_id
                current_confirmed_name = decision['candidate_name']
                current_confirmed_type = decision['candidate_type']
                current_confirmed_subtype = decision['candidate_subtype']
                days_in_hysteresis = 0
                h_left = self.hysteresis_days
            else:
                # Candidate is REJIMSIZ_GECIS
                if current_confirmed_id != 0:
                    days_in_hysteresis += 1
                    if days_in_hysteresis <= self.hysteresis_days:
                        # Retain previous confirmed regime under hysteresis
                        h_left = self.hysteresis_days - days_in_hysteresis
                    else:
                        # Hysteresis expired, transition to REJIMSIZ_GECIS
                        current_confirmed_id = 0
                        current_confirmed_name = "REJIMSIZ_GECIS"
                        current_confirmed_type = "TRANSITION"
                        current_confirmed_subtype = "Dengeli / Nötr Piyasa"
                        h_left = 0
                else:
                    h_left = 0

            # Determine portfolio weights based on confirmed regime
            weights = self.get_portfolio_weights(current_confirmed_id, current_confirmed_subtype)

            regime_ids.append(candidate_id)
            regime_names.append(decision['candidate_name'])
            regime_types.append(decision['candidate_type'])
            regime_subtypes.append(decision['candidate_subtype'])
            confirmed_ids.append(current_confirmed_id)
            confirmed_names.append(current_confirmed_name)
            hysteresis_days_left.append(h_left)
            conflict_notes.append(decision['conflict_note'])
            eq_weights.append(weights['equity'])
            bond_weights.append(weights['bond'])
            cash_weights.append(weights['cash'])

        df['raw_regime_id'] = regime_ids
        df['raw_regime_name'] = regime_names
        df['regime_type'] = regime_types
        df['regime_subtype'] = regime_subtypes
        df['confirmed_regime_id'] = confirmed_ids
        df['confirmed_regime_name'] = confirmed_names
        df['hysteresis_days_left'] = hysteresis_days_left
        df['conflict_note'] = conflict_notes
        df['regime_eq_weight'] = eq_weights
        df['regime_bond_weight'] = bond_weights
        df['regime_cash_weight'] = cash_weights

        return df

    @staticmethod
    def get_portfolio_weights(regime_id: int, subtype: str = "") -> Dict[str, int]:
        """
        Deterministic asset allocation weights based on the active regime:
        - Regime 1: Küresel Enflasyon & Stagflasyon Şoku -> Defensive / Energy Hedge
        - Regime 2: Sistemik Likidite Şoku & Carry Çöküşü -> 100% Cash / Siyah Kuğu Savunması
        - Regime 3: Reel Faiz Şoku -> Short Duration / High Cash
        - Regime 4: Kredi Temerrüt Baskısı -> Flight to Quality (Sovereign Bonds & Cash)
        - Regime 5: Küresel Likidite Rallisi (Risk-On) -> High Equity & Risk Assets
        - Regime 0: REJIMSIZ_GECIS -> 60/40 or Balanced Risk Parity
        """
        if regime_id == 1:
            # Stagflation: Equities hurt, bonds hurt, energy/commodities and cash hedge
            return {"equity": 15, "bond": 10, "cash": 75, "commodity_note": "Aşırı Ağırlık Petrol / Emtia"}
        elif regime_id == 2:
            # Systemic liquidity / carry shock / margin calls: Liquidate all risk assets
            return {"equity": 0, "bond": 10, "cash": 90, "commodity_note": "Nakit / Günlük Repo / USD"}
        elif regime_id == 3:
            # Real rate shock: duration hurts, tech hurts
            if "Bear Flattener" in subtype:
                return {"equity": 15, "bond": 15, "cash": 70, "commodity_note": "Kısa Vade Hazine / Nakit"}
            return {"equity": 20, "bond": 20, "cash": 60, "commodity_note": "Kısa Vadeli TIPS / Nakit"}
        elif regime_id == 4:
            # Credit default stress: Avoid HY/IG credit, flight to safe US Treasuries & Cash
            return {"equity": 15, "bond": 40, "cash": 45, "commodity_note": "Sadece Kaliteli Devlet Tahvili (UST)"}
        elif regime_id == 5:
            # Global liquidity rally (Risk-On)
            if "Reflasyonist" in subtype:
                return {"equity": 75, "bond": 10, "cash": 15, "commodity_note": "Hisse, Kripto, Altın ve Bakır"}
            elif "Goldilocks" in subtype:
                return {"equity": 80, "bond": 15, "cash": 5, "commodity_note": "Mega-Cap Tech, Kripto, Büyüme"}
            return {"equity": 75, "bond": 15, "cash": 10, "commodity_note": "Geniş Hisseler ve Büyüme"}
        else:
            # REJIMSIZ_GECIS (Balanced baseline)
            return {"equity": 45, "bond": 35, "cash": 20, "commodity_note": "Dengeli Makro Portföy"}

    @staticmethod
    def get_asset_recommendations(regime_id: int, subtype: str = "") -> Dict[str, str]:
        """
        Provides granular asset-class analysis for Streamlit dashboard and reports.
        """
        if regime_id == 1:
            return {
                "hisse": "🚨 Satış / Aşırı Defansif (Maliyet Baskısı)",
                "kripto": "🚨 Likidite Azalması - Düşüş Riski",
                "tahvil": "⚠️ Sat / Süre Riskini Sıfırla (Faizler Artıyor)",
                "altin": "🔥 Güçlü Koruyucu / Stagflasyon Kalkanı",
                "emtia": "🚀 Agresif Al (Petrol, Enerji, Tarım)",
                "nakit": "✅ Yüksek Koruma (Dolar / Para Piyasası)"
            }
        elif regime_id == 2:
            return {
                "hisse": "🚨 DEVRE KESİCİ: TAM SATIŞ (Margin Call)",
                "kripto": "🚨 ÇÖKÜŞ ALARMI: Likidite Kuruyor",
                "tahvil": "⚠️ Temkinli (Tahviller de Satılabilir)",
                "altin": "⚪ Geçici Likidite Satışı (Sonra Güçlenir)",
                "emtia": "📉 Talep Çöküşü - Sat",
                "nakit": "🚨 %100 GÜVENLİ LİMAN: Sadece USD & Repo"
            }
        elif regime_id == 3:
            return {
                "hisse": "📉 Büyüme ve Teknoloji Hisselerini Azalt",
                "kripto": "📉 Reel Faiz Baskısı - Uzak Dur",
                "tahvil": "⚠️ Uzun Vadeli Tahvilleri Kes (Faiz Şoku)",
                "altin": "⚠️ Baskı Altında (Reel Getiri Alternatifi)",
                "emtia": "⚪ Nötr / Dalgalı",
                "nakit": "🔥 Kısa Vadeli T-Bill / Yüksek Getirili Nakit"
            }
        elif regime_id == 4:
            return {
                "hisse": "📉 Kredi Duyarlı Hisseleri Azalt",
                "kripto": "📉 Kredi Daralması Baskısı",
                "tahvil": "🔥 Güvenli Liman: Yalnızca ABD Hazinesi (UST)",
                "altin": "✅ Kredi Riski Hedge (Pozitif)",
                "emtia": "📉 Resesyon Riskiyle Baskılı",
                "nakit": "✅ Yüksek Nakit / Repo Ağırlığı"
            }
        elif regime_id == 5:
            return {
                "hisse": "🚀 TAM KAPASİTE BOĞA: Büyüme & Teknoloji",
                "kripto": "🚀 Agresif Al (Boğa Döngüsü)",
                "tahvil": "⚪ Nötr / Taşıma Getirisi (Carry)",
                "altin": "🔥 Reflasyon Destekli (Özellikle Zayıf Dolarda)",
                "emtia": "🔥 Sanayi Metalleri & Büyüme Emtiaları Al",
                "nakit": "⚪ Minimum Nakit (Risk Bütçesini Kullan)"
            }
        else:
            return {
                "hisse": "✅ Dengeli / Seçici Hisseler",
                "kripto": "⚪ İzleme Modu / Trend Takibi",
                "tahvil": "✅ Sabit Getiri / Kupon Geliri",
                "altin": "✅ Portföy Sigortası (%10-15)",
                "emtia": "⚖️ Nötr",
                "nakit": "✅ Fırsat Bütçesi (%20)"
            }
