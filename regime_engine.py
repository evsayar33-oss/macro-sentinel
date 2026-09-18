"""
Macro Sentinel Regime Engine v2.0

Design goals
------------
* Point-in-time / no-lookahead calculations.
* Deterministic regime classification with bounded hysteresis.
* Independent event layer for commodity/oil shocks and unknown anomalies.
* Dynamic, distribution-aware event thresholds without unbounded self-optimization.
* Portfolio weights always normalize to exactly 100%.
* Data-quality failures never manufacture market data.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class MacroRegimeEngine:
    def __init__(self, config_path: str = "regime_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        principles = self.config.get("system_architecture", {}).get("principles", {})
        self.hysteresis_weeks = int(principles.get("hysteresis_confirmation_period_weeks", 2))
        self.hysteresis_days = max(1, self.hysteresis_weeks * 5)
        self.switch_confirmation_observations = max(1, int(principles.get("switch_confirmation_observations", 2)))
        self.minimum_history = int(principles.get("minimum_history_days", 126))

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def calc_rolling_z(series: pd.Series, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if min_periods is None:
            min_periods = max(40, window // 4)
        mean = s.rolling(window=window, min_periods=min_periods).mean()
        std = s.rolling(window=window, min_periods=min_periods).std()
        return (s - mean) / (std.replace(0.0, np.nan) + 1e-9)

    @staticmethod
    def calc_trailing_z(series: pd.Series, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """Z-score current observation against prior observations only."""
        s = pd.to_numeric(series, errors="coerce")
        if min_periods is None:
            min_periods = max(40, window // 4)
        ref = s.shift(1)
        mean = ref.rolling(window=window, min_periods=min_periods).mean()
        std = ref.rolling(window=window, min_periods=min_periods).std()
        return (s - mean) / (std.replace(0.0, np.nan) + 1e-9)

    @staticmethod
    def calc_rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")

        def _slope(y: np.ndarray) -> float:
            if len(y) < window or np.isnan(y).any():
                return np.nan
            x = np.arange(len(y), dtype=float)
            x_m = x.mean()
            y_m = y.mean()
            denom = np.sum((x - x_m) ** 2)
            if denom <= 0:
                return np.nan
            return float(np.sum((x - x_m) * (y - y_m)) / denom)

        return s.rolling(window=window, min_periods=window).apply(_slope, raw=True)

    @staticmethod
    def calc_rolling_percentile(series: pd.Series, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if min_periods is None:
            min_periods = max(40, window // 4)

        def _pct(x: np.ndarray) -> float:
            clean = x[np.isfinite(x)]
            if len(clean) < 2:
                return np.nan
            return float(np.mean(clean <= clean[-1]) * 100.0)

        return s.rolling(window=window, min_periods=min_periods).apply(_pct, raw=True)

    def calc_trailing_percentile(self, series: pd.Series, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """Percentile of current value versus the previous window only."""
        s = pd.to_numeric(series, errors="coerce")
        if min_periods is None:
            min_periods = max(40, window // 4)

        def _pct(x: np.ndarray) -> float:
            current = x[-1]
            hist = x[:-1]
            clean = hist[np.isfinite(hist)]
            if not np.isfinite(current) or len(clean) < 2:
                return np.nan
            return float(np.mean(clean <= current) * 100.0)

        return s.rolling(window=window + 1, min_periods=min_periods + 1).apply(_pct, raw=True)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all point-in-time indicators. No future observations are used."""
        p = df.copy().sort_index()

        # Oil / commodity event layer.
        if "oil" in p.columns:
            oil = pd.to_numeric(p["oil"], errors="coerce")
            oil_ret_1d = oil.pct_change(1)
            oil_ret_5d = oil.pct_change(5)
            oil_ret_20d = oil.pct_change(20)
            oil_vol_20d = oil_ret_1d.rolling(20, min_periods=10).std()

            p["oil_ret_5d_z"] = self.calc_trailing_z(oil_ret_5d, 252)
            p["oil_ret_20d_z"] = self.calc_trailing_z(oil_ret_20d, 252)
            p["oil_vol_20d_z"] = self.calc_trailing_z(oil_vol_20d, 252)

            trend_raw = oil_ret_20d / (oil_vol_20d * np.sqrt(20.0) + 1e-9)
            p["oil_trend_strength"] = self.calc_trailing_z(trend_raw, 252)

            # Adaptive percentile: how unusual is today's absolute 5d move
            # relative to the prior ~1y distribution? This is point-in-time.
            abs_5d = oil_ret_5d.abs()
            p["oil_abs_5d_percentile"] = self.calc_trailing_percentile(abs_5d, 252, min_periods=63)

            # Structural oil-price layer: maintain a true 252-session view and a
            # slower ~3-year view. Structural qualification uses the 756-session
            # distribution; the 252-session view remains a separate dashboard sensor.
            p["oil_level_z_252"] = self.calc_trailing_z(oil, 252, min_periods=63)
            p["oil_level_percentile_252"] = self.calc_trailing_percentile(oil, 252, min_periods=63)
            p["oil_level_z_756"] = self.calc_trailing_z(oil, 756, min_periods=126)
            p["oil_level_percentile_756"] = self.calc_trailing_percentile(oil, 756, min_periods=126)
            level_pct = p["oil_level_percentile_756"]
            p["oil_high_level_persistence_60d"] = (level_pct >= 85.0).astype(float).rolling(60, min_periods=20).mean()
        else:
            for col in [
                "oil_ret_5d_z", "oil_ret_20d_z", "oil_vol_20d_z",
                "oil_trend_strength", "oil_abs_5d_percentile",
                "oil_level_z_756", "oil_level_percentile_756",
                "oil_level_z_252", "oil_level_percentile_252",
                "oil_high_level_persistence_60d",
            ]:
                p[col] = np.nan

        # Second oil benchmark / energy-complex confirmation.  Brent-vs-WTI
        # dislocation and broad energy participation make the structural event
        # less dependent on a single ticker.
        if "brent" in p.columns:
            brent = pd.to_numeric(p["brent"], errors="coerce")
            p["brent_level_z_252"] = self.calc_trailing_z(brent, 252, min_periods=63)
            p["brent_level_percentile_252"] = self.calc_trailing_percentile(brent, 252, min_periods=63)
        else:
            p["brent_level_z_252"] = np.nan
            p["brent_level_percentile_252"] = np.nan

        if "brent" in p.columns and "oil" in p.columns:
            spread = pd.to_numeric(p["brent"], errors="coerce") - pd.to_numeric(p["oil"], errors="coerce")
            p["brent_wti_spread"] = spread
            p["brent_wti_spread_z_252"] = self.calc_trailing_z(spread, 252, min_periods=63)
            p["brent_wti_spread_percentile_252"] = self.calc_trailing_percentile(spread, 252, min_periods=63)
        else:
            p["brent_wti_spread"] = np.nan
            p["brent_wti_spread_z_252"] = np.nan
            p["brent_wti_spread_percentile_252"] = np.nan

        energy_breadth_inputs = []
        for asset in ("oil", "brent", "heating_oil", "gasoline", "natgas"):
            if asset in p.columns:
                energy_breadth_inputs.append(pd.to_numeric(p[asset], errors="coerce").pct_change(20) > 0)
        if energy_breadth_inputs:
            p["energy_breadth_20d"] = pd.concat(energy_breadth_inputs, axis=1).mean(axis=1)
        else:
            p["energy_breadth_20d"] = np.nan

        # EIA weekly crude inventories (when supplied by main.py). Draws are
        # converted to a positive stress score because inventory contraction is
        # consistent with tighter supply/demand balance.
        if "crude_stocks" in p.columns:
            stocks = pd.to_numeric(p["crude_stocks"], errors="coerce")
            draw = -stocks.diff(1)
            p["crude_inventory_draw_z"] = self.calc_trailing_z(draw, 52, min_periods=20)
        else:
            p["crude_inventory_draw_z"] = np.nan

        # Freight / trade.
        if "freight" in p.columns:
            freight = pd.to_numeric(p["freight"], errors="coerce")
            p["freight_lvl_z"] = self.calc_rolling_z(freight, 252)
            p["freight_20d_z"] = self.calc_rolling_z(freight.pct_change(20), 252)
        else:
            p["freight_lvl_z"] = 0.0
            p["freight_20d_z"] = 0.0

        # Credit spreads.
        if "hy_oas" in p.columns:
            hy = pd.to_numeric(p["hy_oas"], errors="coerce")
            p["hy_oas_z"] = self.calc_rolling_z(hy, 252)
            p["hy_oas_slope_10d"] = self.calc_rolling_slope(hy, 10)
        else:
            p["hy_oas_z"] = 0.0
            p["hy_oas_slope_10d"] = 0.0

        if "ig_oas" in p.columns:
            p["ig_oas_z"] = self.calc_rolling_z(pd.to_numeric(p["ig_oas"], errors="coerce"), 252)
        else:
            p["ig_oas_z"] = 0.0

        # Stock/bond correlation; contemporaneous historical returns only.
        if "spx" in p.columns and "ust10y" in p.columns:
            spx_ret = pd.to_numeric(p["spx"], errors="coerce").pct_change()
            yield_diff = pd.to_numeric(p["ust10y"], errors="coerce").diff()
            bond_proxy_ret = -8.0 * yield_diff / 100.0
            p["spx_bond_corr_60d"] = spx_ret.rolling(60, min_periods=30).corr(bond_proxy_ret)
        else:
            p["spx_bond_corr_60d"] = 0.0

        # Dollar / carry.
        if "broad_dollar" in p.columns:
            dxy = pd.to_numeric(p["broad_dollar"], errors="coerce")
            p["broad_dollar_5d_z"] = self.calc_rolling_z(dxy.pct_change(5), 252)
            p["broad_dollar_z"] = self.calc_rolling_z(dxy, 252)
        else:
            p["broad_dollar_5d_z"] = 0.0
            p["broad_dollar_z"] = 0.0

        if "usdjpy" in p.columns:
            usdjpy = pd.to_numeric(p["usdjpy"], errors="coerce")
            p["usdjpy_1d_z"] = self.calc_rolling_z(usdjpy.pct_change(1), 252)
        else:
            p["usdjpy_1d_z"] = 0.0

        # Volatility.
        if "vix" in p.columns:
            vix = pd.to_numeric(p["vix"], errors="coerce")
            p["vix_z"] = self.calc_rolling_z(vix, 252)
            p["vix_percentile_252"] = self.calc_rolling_percentile(vix, 252)
        else:
            p["vix_z"] = 0.0
            p["vix_percentile_252"] = 50.0

        # Opportunity-state features. These are point-in-time and are consumed
        # by the bounded production strategy layer.
        if "spx" in p.columns:
            spx = pd.to_numeric(p["spx"], errors="coerce")
            p["spx_ret_20d"] = spx.pct_change(20)
            p["spx_ret_100d"] = spx.pct_change(100)
        else:
            p["spx_ret_20d"] = np.nan
            p["spx_ret_100d"] = np.nan
        if "btc" in p.columns:
            btc_series = pd.to_numeric(p["btc"], errors="coerce")
            p["btc_ret_60d"] = btc_series.pct_change(60)
        else:
            p["btc_ret_60d"] = np.nan

        # Risk basket.
        if "btc" in p.columns and "spx" in p.columns:
            basket = (
                0.5 * pd.to_numeric(p["btc"], errors="coerce").pct_change(5)
                + 0.5 * pd.to_numeric(p["spx"], errors="coerce").pct_change(5)
            )
            p["risk_basket_5d_z"] = self.calc_rolling_z(basket, 252)
        else:
            p["risk_basket_5d_z"] = 0.0

        # Real rates / breakeven / curve.
        if "tips10y" in p.columns:
            tips = pd.to_numeric(p["tips10y"], errors="coerce")
            p["tips_1d_z"] = self.calc_trailing_z(tips.diff(1), 252)
            p["tips_level_z"] = self.calc_trailing_z(tips, 252)
            p["tips_level_percentile_252"] = self.calc_trailing_percentile(tips, 252)
        else:
            p["tips_1d_z"] = 0.0
            p["tips_level_z"] = 0.0
            p["tips_level_percentile_252"] = 50.0

        if "t10yie" in p.columns:
            p["t10yie_z"] = self.calc_rolling_z(pd.to_numeric(p["t10yie"], errors="coerce"), 252)
        else:
            p["t10yie_z"] = 0.0

        if "dgs2" in p.columns and "dgs10" in p.columns:
            p["delta_dgs2_5d"] = pd.to_numeric(p["dgs2"], errors="coerce").diff(5)
            p["delta_dgs10_5d"] = pd.to_numeric(p["dgs10"], errors="coerce").diff(5)
        else:
            p["delta_dgs2_5d"] = 0.0
            p["delta_dgs10_5d"] = 0.0

        # Net dollar liquidity.
        if "ndl" in p.columns:
            p["ndl_z"] = self.calc_rolling_z(pd.to_numeric(p["ndl"], errors="coerce"), 252)
        else:
            p["ndl_z"] = 0.0

        # Gold / industrial-metals breadth.
        if "gold" in p.columns:
            gold = pd.to_numeric(p["gold"], errors="coerce")
            p["gold_rising"] = (gold.rolling(20, min_periods=10).mean() > gold.rolling(50, min_periods=20).mean()) | (gold.pct_change(20) > 0)
            p["gold_ret_20d"] = gold.pct_change(20)
        else:
            p["gold_rising"] = False
            p["gold_ret_20d"] = 0.0

        breadth_inputs = []
        for asset in ("copper", "silver", "gold"):
            if asset in p.columns:
                breadth_inputs.append(pd.to_numeric(p[asset], errors="coerce").pct_change(20) > 0)
        if breadth_inputs:
            p["commodity_breadth_20d"] = pd.concat(breadth_inputs, axis=1).mean(axis=1)
        else:
            p["commodity_breadth_20d"] = 0.0

        # Bounded historical event quality. At date t this uses only events
        # that happened sufficiently far in the past to have a realized 5d move.
        if "oil" in p.columns:
            oil5 = pd.to_numeric(p["oil"], errors="coerce").pct_change(5)
            prior_extreme = (p["oil_ret_5d_z"].shift(5) > 1.0).astype(float)
            realized_positive = (oil5 > 0).astype(float)
            valid = prior_extreme.rolling(60, min_periods=10).sum()
            hit = (prior_extreme * realized_positive).rolling(60, min_periods=10).sum()
            p["oil_event_quality_60d"] = (hit / valid.replace(0.0, np.nan)).fillna(0.5)
        else:
            p["oil_event_quality_60d"] = 0.5

        return p

    def evaluate_regimes_for_row(self, row: pd.Series, custom_thresholds: Optional[Dict[str, float]] = None) -> Dict[int, Dict[str, Any]]:
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
            "r5_ndl_z": 0.0,
        }
        # The config file is the single source of truth for calibrated thresholds.
        configured = self.config.get("calibrated_thresholds", {})
        if isinstance(configured, dict):
            th.update({k: float(v) for k, v in configured.items() if k in th})
        if custom_thresholds:
            th.update({k: float(v) for k, v in custom_thresholds.items() if k in th})

        oil_z = self._safe_float(row.get("oil_ret_20d_z"))
        oil_trend = self._safe_float(row.get("oil_trend_strength", row.get("oil_trend", 0.0)))
        freight_z = self._safe_float(row.get("freight_lvl_z"))
        hy_z = self._safe_float(row.get("hy_oas_z"))
        corr = self._safe_float(row.get("spx_bond_corr_60d"))

        # Regime 1: require independent supply evidence, but the independent
        # oil event overlay can react even when the full stagflation regime is
        # not yet confirmed.
        r1_oil = oil_z > th["r1_oil_z"] or oil_trend >= 2.0
        r1_freight = freight_z < th["r1_freight_z"] or freight_z > 1.5
        r1_confirm = hy_z > th["r1_hy_z"] and corr > th["r1_corr"]
        r1_active = bool(r1_oil and r1_freight and r1_confirm)

        usdjpy_z = self._safe_float(row.get("usdjpy_1d_z"))
        dxy5_z = self._safe_float(row.get("broad_dollar_5d_z"))
        vix_z = self._safe_float(row.get("vix_z"))
        basket_z = self._safe_float(row.get("risk_basket_5d_z"))
        r2_trigger = dxy5_z > th["r2_dxy_z"] or usdjpy_z < th["r2_jpy_z"] or vix_z > th["r2_vix_z"]
        r2_confirm = basket_z < th["r2_basket_z"]
        r2_active = bool(r2_trigger and r2_confirm)

        tips_chg_z = self._safe_float(row.get("tips_1d_z"))
        tips_level_z = self._safe_float(row.get("tips_level_z"))
        tips_level_pct = self._safe_float(row.get("tips_level_percentile_252"), 50.0)
        tips_lvl = self._safe_float(row.get("tips10y"), 2.0)
        t10yie_z = self._safe_float(row.get("t10yie_z"))

        semantics = self.config.get("regime_semantics", {})
        high_real_floor = float(semantics.get("r3_high_real_rate_floor", 2.0))
        high_real_pct_floor = float(semantics.get("r3_high_real_rate_percentile_floor", 75.0))
        r3_daily_shock = tips_chg_z > th["r3_tips_z"]
        r3_persistent = (
            tips_lvl >= high_real_floor
            or tips_level_z > th.get("r3_tips_level_z", 1.0)
            or tips_level_pct >= high_real_pct_floor
        )
        r3_t2 = t10yie_z < th["r3_t10yie_z"]
        r3_active = bool((r3_daily_shock or r3_persistent) and r3_t2)

        d2 = self._safe_float(row.get("delta_dgs2_5d"))
        d10 = self._safe_float(row.get("delta_dgs10_5d"))
        if d2 < 0 and d10 > 0:
            subtype_r3 = "Bear Steepener (Enflasyon/Term Premium)"
        elif d2 > 0 and d10 > 0 and d10 > d2:
            subtype_r3 = "Bear Steepener (Fed Varyantı)"
        elif d2 > 0 and d10 > 0 and d2 > d10:
            subtype_r3 = "Bear Flattener (Fed Sıkılaştırma Baskın)"
        elif d2 < 0 and d10 < 0:
            subtype_r3 = "Bull Flattener/Steepener (Gevşeme - Tetiklemez)"
            r3_active = False
        else:
            subtype_r3 = "Kısıtlayıcı Reel Faiz Baskısı"

        hy_slope = self._safe_float(row.get("hy_oas_slope_10d"))
        ig_z = self._safe_float(row.get("ig_oas_z"))
        r4_trigger = hy_z > th["r4_hy_z"] and hy_slope > th["r4_slope"]
        r4_confirm = ig_z > th["r4_ig_z"]
        r4_active = bool(r4_trigger and r4_confirm)

        dxy_z = self._safe_float(row.get("broad_dollar_z"))
        vix_pct = self._safe_float(row.get("vix_percentile_252"), 50.0)
        ndl_z = self._safe_float(row.get("ndl_z"))
        gold_rising = bool(row.get("gold_rising", False))
        r5_active = bool(
            hy_z < th["r5_hy_z"]
            and th["r5_dxy_min"] <= dxy_z <= th["r5_dxy_max"]
            and vix_pct < th["r5_vix_pct"]
            and ndl_z > th["r5_ndl_z"]
            and tips_lvl < 2.0
        )
        subtype_r5 = (
            "Reflasyonist Risk-On (Zayıf Dolar & Yükselen Emtia)"
            if dxy_z <= 0.5 and gold_rising
            else "Klasik Goldilocks Risk-On"
            if dxy_z <= 0.5
            else "Geniş Tabanlı Likidite Boğası"
        )

        return {
            1: {
                "id": 1,
                "name": "Küresel Enflasyon & Stagflasyon Şoku",
                "type": "SHOCK",
                "trigger_met": bool(r1_oil and r1_freight),
                "confirm_met": bool(r1_confirm),
                "active": r1_active,
                "main_trigger_z": max(oil_z, oil_trend, abs(freight_z)),
                "subtype": "Arz Yönlü Stagflasyon (Petrol & Tedarik Kısıtı)",
                "diagnostics": {"oil_z": oil_z, "oil_trend": oil_trend, "freight_z": freight_z, "hy_oas_z": hy_z, "corr": corr},
            },
            2: {
                "id": 2,
                "name": "Sistemik Likidite Şoku & Carry Çöküşü",
                "type": "SHOCK",
                "trigger_met": bool(r2_trigger),
                "confirm_met": bool(r2_confirm),
                "active": r2_active,
                "main_trigger_z": max(abs(dxy5_z), abs(usdjpy_z), abs(vix_z)),
                "subtype": "Sistemik Likidite Sıkışması",
                "diagnostics": {"dxy_5d_z": dxy5_z, "usdjpy_1d_z": usdjpy_z, "vix_z": vix_z, "basket_z": basket_z},
            },
            3: {
                "id": 3,
                "name": "Reel Faiz Şoku" if r3_daily_shock else "Reel Faiz Kısıtlayıcı Rejimi",
                "type": "SHOCK" if r3_daily_shock else "CONSTRAINT",
                "trigger_met": bool((r3_daily_shock or r3_persistent) and r3_t2),
                "confirm_met": bool(r3_t2),
                "active": r3_active,
                "main_trigger_z": max(abs(tips_chg_z), abs(tips_level_z)),
                "severity_source": "DAILY_SHOCK" if r3_daily_shock else "PERSISTENT_LEVEL" if r3_persistent else "NONE",
                "subtype": subtype_r3,
                "diagnostics": {"tips_chg_z": tips_chg_z, "tips_level_z": tips_level_z, "tips_level_pct": tips_level_pct, "tips_lvl": tips_lvl, "t10yie_z": t10yie_z, "d2": d2, "d10": d10},
            },
            4: {
                "id": 4,
                "name": "Kredi Temerrüt Baskısı",
                "type": "SHOCK",
                "trigger_met": bool(r4_trigger),
                "confirm_met": bool(r4_confirm),
                "active": r4_active,
                "main_trigger_z": abs(hy_z),
                "subtype": "Kredi Yayılması & Temerrüt Riski",
                "diagnostics": {"hy_z": hy_z, "hy_slope": hy_slope, "ig_z": ig_z},
            },
            5: {
                "id": 5,
                "name": "Küresel Likidite Rallisi (Risk-On)",
                "type": "RISK_ON",
                "trigger_met": r5_active,
                "confirm_met": True,
                "active": r5_active,
                "main_trigger_z": ndl_z,
                "subtype": subtype_r5,
                "diagnostics": {"hy_oas_z": hy_z, "dxy_z": dxy_z, "vix_pct": vix_pct, "ndl_z": ndl_z, "tips_lvl": tips_lvl},
            },
        }

    def resolve_conflicts(self, regime_evals: Dict[int, Dict[str, Any]], row: pd.Series) -> Dict[str, Any]:
        active_hard_shocks = [r for rid, r in regime_evals.items() if rid in (1, 2, 4) and r["active"]]
        if regime_evals[3]["active"] and regime_evals[3].get("type") == "SHOCK":
            active_hard_shocks.append(regime_evals[3])

        if active_hard_shocks:
            selected = max(active_hard_shocks, key=lambda x: abs(self._safe_float(x.get("main_trigger_z"))))
            return {
                "candidate_id": selected["id"],
                "candidate_name": selected["name"],
                "candidate_type": selected["type"],
                "candidate_subtype": selected["subtype"],
                "main_trigger_z": selected["main_trigger_z"],
                "severity_source": selected.get("severity_source", "NONE"),
                "conflict_note": f"Multiple hard risk states resolved by trigger magnitude: R{selected['id']}.",
                "all_triggered": [r["id"] for r in active_hard_shocks],
            }

        if regime_evals[3]["active"]:
            selected = regime_evals[3]
            return {
                "candidate_id": selected["id"],
                "candidate_name": selected["name"],
                "candidate_type": selected["type"],
                "candidate_subtype": selected["subtype"],
                "main_trigger_z": selected["main_trigger_z"],
                "severity_source": selected.get("severity_source", "PERSISTENT_LEVEL"),
                "conflict_note": "Persistent restrictive real-rate state active; no hard shock regime confirmed.",
                "all_triggered": [3],
            }

        if regime_evals[5]["active"]:
            r = regime_evals[5]
            return {
                "candidate_id": 5,
                "candidate_name": r["name"],
                "candidate_type": r["type"],
                "candidate_subtype": r["subtype"],
                "main_trigger_z": r["main_trigger_z"],
                "conflict_note": "Risk-On active with no shock regime active.",
                "all_triggered": [5],
            }

        oil_z = self._safe_float(row.get("oil_ret_20d_z"))
        vix_z = self._safe_float(row.get("vix_z"))
        hy_z = self._safe_float(row.get("hy_oas_z"))
        ndl_z = self._safe_float(row.get("ndl_z"))
        tips = self._safe_float(row.get("tips10y"))
        note = (
            "No named regime currently satisfies its full confirmation rules. "
            f"Live snapshot: Oil20D Z={oil_z:.2f}, VIX Z={vix_z:.2f}, "
            f"HY OAS Z={hy_z:.2f}, NDL Z={ndl_z:.2f}, TIPS={tips:.2f}%."
        )
        return {
            "candidate_id": 0,
            "candidate_name": "REJIMSIZ_GECIS",
            "candidate_type": "TRANSITION",
            "candidate_subtype": "Dengeli / Nötr Piyasa",
            "main_trigger_z": max(abs(oil_z), abs(vix_z), abs(hy_z), abs(ndl_z)),
            "conflict_note": note,
            "all_triggered": [],
        }

    def run_time_series(self, prepared_df: pd.DataFrame, custom_thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        out = prepared_df.copy().sort_index()
        defaults = {
            "raw_candidate_id": 0,
            "raw_candidate_name": "REJIMSIZ_GECIS",
            "confirmed_regime_id": 0,
            "confirmed_regime_name": "REJIMSIZ_GECIS",
            "regime_type": "TRANSITION",
            "regime_subtype": "Dengeli / Nötr Piyasa",
            "regime_severity_source": "NONE",
            "pending_regime_id": 0,
            "pending_regime_count": 0,
            "hysteresis_days_left": 0,
            "conflict_note": "",
            "regime_cash_weight": 35.0,
            "regime_gold_weight": 20.0,
            "regime_bond_weight": 20.0,
            "regime_eq_weight": 15.0,
            "regime_commodity_weight": 5.0,
            "regime_crypto_weight": 5.0,
            "oil_pressure_score": 0.0,
            "oil_event_score": 0.0,
            "oil_momentum_score": 0.0,
            "oil_structural_score": 0.0,
            "oil_structural_qualified": False,
            "oil_momentum_confirmed": False,
            "oil_event_type": "NONE",
            "oil_event_active": False,
            "commodity_event_active": False,
            "commodity_event_reason": "",
            "unknown_event_score": 0.0,
            "unknown_event_active": False,
            "unknown_guard_active": False,
            "strategy_mode": "STATIC_REGIME",
        }
        for col, val in defaults.items():
            out[col] = val

        current_id = 0
        current_name = "REJIMSIZ_GECIS"
        current_type = "TRANSITION"
        current_subtype = "Dengeli / Nötr Piyasa"
        current_severity_source = "NONE"
        pending_id = 0
        pending_count = 0
        hysteresis = 0

        for i in range(len(out)):
            row = out.iloc[i]
            evals = self.evaluate_regimes_for_row(row, custom_thresholds=custom_thresholds)
            decision = self.resolve_conflicts(evals, row)
            candidate_id = int(decision["candidate_id"])
            candidate_hard = bool(
                candidate_id in (1, 2, 4)
                or (candidate_id == 3 and decision.get("severity_source") == "DAILY_SHOCK")
            )
            hard_unknown = self._hard_event_override(row)

            out.at[out.index[i], "raw_candidate_id"] = candidate_id
            out.at[out.index[i], "raw_candidate_name"] = decision["candidate_name"]
            out.at[out.index[i], "conflict_note"] = decision["conflict_note"]

            if candidate_hard:
                current_id = candidate_id
                current_name = decision["candidate_name"]
                current_type = decision["candidate_type"]
                current_subtype = decision["candidate_subtype"]
                current_severity_source = decision.get("severity_source", "NONE")
                pending_id = 0
                pending_count = 0
                hysteresis = self.hysteresis_days
            elif candidate_id == current_id and candidate_id != 0:
                current_name = decision["candidate_name"]
                current_type = decision["candidate_type"]
                current_subtype = decision["candidate_subtype"]
                current_severity_source = decision.get("severity_source", current_severity_source)
                pending_id = 0
                pending_count = 0
                hysteresis = self.hysteresis_days
            elif candidate_id != 0:
                if candidate_id == pending_id:
                    pending_count += 1
                else:
                    pending_id = candidate_id
                    pending_count = 1
                if pending_count >= self.switch_confirmation_observations:
                    current_id = candidate_id
                    current_name = decision["candidate_name"]
                    current_type = decision["candidate_type"]
                    current_subtype = decision["candidate_subtype"]
                    current_severity_source = decision.get("severity_source", "NONE")
                    pending_id = 0
                    pending_count = 0
                    hysteresis = self.hysteresis_days
                elif current_id != 0:
                    hysteresis = max(hysteresis - 1, 0)
            elif hard_unknown:
                current_id = 0
                current_name = "REJIMSIZ_GECIS"
                current_type = "TRANSITION"
                current_subtype = "Extreme Unclassified Event"
                current_severity_source = "UNKNOWN_EVENT"
                pending_id = 0
                pending_count = 0
                hysteresis = 0
            elif current_id != 0 and hysteresis > 0:
                pending_id = 0
                pending_count = 0
                hysteresis -= 1
            else:
                current_id = 0
                current_name = "REJIMSIZ_GECIS"
                current_type = "TRANSITION"
                current_subtype = "Dengeli / Nötr Piyasa"
                current_severity_source = "NONE"
                pending_id = 0
                pending_count = 0
                hysteresis = 0

            idx = out.index[i]
            out.at[idx, "confirmed_regime_id"] = current_id
            out.at[idx, "confirmed_regime_name"] = current_name
            out.at[idx, "regime_type"] = current_type
            out.at[idx, "regime_subtype"] = current_subtype
            out.at[idx, "regime_severity_source"] = current_severity_source
            out.at[idx, "pending_regime_id"] = pending_id
            out.at[idx, "pending_regime_count"] = pending_count
            out.at[idx, "hysteresis_days_left"] = hysteresis

            weights = self.get_portfolio_weights(current_id, current_subtype, row=row)
            for source, target in (
                ("cash", "regime_cash_weight"),
                ("gold", "regime_gold_weight"),
                ("bond", "regime_bond_weight"),
                ("equity", "regime_eq_weight"),
                ("commodity", "regime_commodity_weight"),
                ("crypto", "regime_crypto_weight"),
            ):
                out.at[idx, target] = float(weights[source])
            for key in (
                "oil_pressure_score", "oil_event_score", "oil_momentum_score", "oil_structural_score",
                "oil_structural_qualified", "oil_momentum_confirmed", "oil_event_type", "oil_event_active",
                "commodity_event_active", "commodity_event_reason", "unknown_event_score",
                "unknown_event_active", "unknown_guard_active", "strategy_mode",
            ):
                out.at[idx, key] = weights.get(key, defaults[key])

        return out

    def _hard_event_override(self, row: pd.Series) -> bool:
        cfg = self.config.get("event_overlay", {}).get("unknown_event", {})
        if not bool(cfg.get("enabled", True)):
            return False
        threshold = float(cfg.get("hard_anomaly_z", 3.5))
        sensors = [
            abs(self._safe_float(row.get("vix_z"))),
            abs(self._safe_float(row.get("hy_oas_z"))),
            abs(self._safe_float(row.get("broad_dollar_5d_z"))),
            abs(self._safe_float(row.get("usdjpy_1d_z"))),
            abs(self._safe_float(row.get("tips_1d_z"))),
        ]
        return max(sensors, default=0.0) >= threshold

    def _unknown_event_score(self, row: pd.Series) -> float:
        cfg = self.config.get("event_overlay", {}).get("unknown_event", {})
        if not bool(cfg.get("enabled", True)):
            return 0.0
        soft = float(cfg.get("soft_anomaly_z", 2.5))
        hard = max(soft + 0.1, float(cfg.get("hard_anomaly_z", 3.5)))
        peak = max(
            abs(self._safe_float(row.get("vix_z"))),
            abs(self._safe_float(row.get("hy_oas_z"))),
            abs(self._safe_float(row.get("broad_dollar_5d_z"))),
            abs(self._safe_float(row.get("usdjpy_1d_z"))),
            abs(self._safe_float(row.get("tips_1d_z"))),
        )
        if not np.isfinite(peak) or peak <= soft:
            return 0.0
        return float(np.clip((peak - soft) / (hard - soft), 0.0, 1.0))

    def _oil_momentum_score(self, row: pd.Series) -> float:
        cfg = self.config.get("event_overlay", {}).get("oil", {})
        base_z = float(cfg.get("activation_z", 0.75))
        saturation_z = max(base_z + 0.1, float(cfg.get("saturation_z", 3.5)))
        percentile_gate = float(cfg.get("adaptive_percentile_gate", 80.0))
        shock_z = max(
            self._safe_float(row.get("oil_ret_5d_z")),
            self._safe_float(row.get("oil_ret_20d_z")),
            self._safe_float(row.get("oil_trend_strength", row.get("oil_trend", 0.0))),
        )
        move_pct = self._safe_float(row.get("oil_abs_5d_percentile"), np.nan)
        if not np.isfinite(shock_z) or (shock_z <= base_z and (not np.isfinite(move_pct) or move_pct < percentile_gate)):
            return 0.0
        intensity = np.clip((max(shock_z, base_z) - base_z) / (saturation_z - base_z), 0.0, 1.0)
        percentile_boost = 0.10 if np.isfinite(move_pct) and move_pct >= percentile_gate else 0.0
        breadth = self._safe_float(row.get("commodity_breadth_20d"), np.nan)
        breadth_boost = 0.10 if np.isfinite(breadth) and breadth >= 0.50 else 0.0
        quality = float(np.clip(self._safe_float(row.get("oil_event_quality_60d"), 0.5), 0.5, 1.0))
        score = float(np.clip(intensity + percentile_boost + breadth_boost, 0.0, 1.0))
        return float(np.clip(score * (0.80 + 0.20 * quality), 0.0, 1.0))

    def _oil_structural_score(self, row: pd.Series) -> Dict[str, Any]:
        cfg = self.config.get("event_overlay", {}).get("oil", {}).get("structural", {})
        if not bool(cfg.get("enabled", True)):
            return {"score": 0.0, "qualified": False, "qualification_reason": "Structural oil layer disabled."}
        level_pct = self._safe_float(row.get("oil_level_percentile_756"), np.nan)
        if not np.isfinite(level_pct):
            level_pct = self._safe_float(row.get("oil_level_percentile_252"), np.nan)
        persistence = self._safe_float(row.get("oil_high_level_persistence_60d"), np.nan)
        breadth = self._safe_float(row.get("energy_breadth_20d"), np.nan)
        inventory_draw = self._safe_float(row.get("crude_inventory_draw_z"), np.nan)
        spread_pct = self._safe_float(row.get("brent_wti_spread_percentile_252"), np.nan)
        brent_pct = self._safe_float(row.get("brent_level_percentile_252"), np.nan)
        components, weights = [], []
        if np.isfinite(level_pct):
            soft = float(cfg.get("level_soft_percentile", 80.0)); hard = max(soft + 1.0, float(cfg.get("level_hard_percentile", 95.0)))
            components.append(float(np.clip((level_pct - soft) / (hard - soft), 0.0, 1.0))); weights.append(float(cfg.get("level_weight", 0.35)))
        if np.isfinite(brent_pct):
            soft = float(cfg.get("brent_soft_percentile", 80.0)); hard = max(soft + 1.0, float(cfg.get("brent_hard_percentile", 95.0)))
            components.append(float(np.clip((brent_pct - soft) / (hard - soft), 0.0, 1.0))); weights.append(float(cfg.get("brent_weight", 0.15)))
        if np.isfinite(persistence):
            gate = float(cfg.get("persistence_gate", 0.40))
            components.append(float(np.clip((persistence - gate) / max(1.0 - gate, 1e-6), 0.0, 1.0))); weights.append(float(cfg.get("persistence_weight", 0.25)))
        if np.isfinite(breadth):
            gate = float(cfg.get("breadth_gate", 0.60))
            components.append(float(np.clip((breadth - gate) / max(1.0 - gate, 1e-6), 0.0, 1.0))); weights.append(float(cfg.get("breadth_weight", 0.10)))
        if np.isfinite(inventory_draw):
            components.append(float(np.clip(inventory_draw / 2.0, 0.0, 1.0))); weights.append(float(cfg.get("inventory_weight", 0.10)))
        if np.isfinite(spread_pct):
            soft = float(cfg.get("spread_soft_percentile", 70.0))
            components.append(float(np.clip((spread_pct - soft) / max(100.0 - soft, 1.0), 0.0, 1.0))); weights.append(float(cfg.get("spread_weight", 0.05)))
        if not components or sum(weights) <= 0:
            return {"score": 0.0, "qualified": False, "qualification_reason": "Insufficient structural data."}
        score = float(np.average(components, weights=weights))
        high_level = np.isfinite(level_pct) and level_pct >= float(cfg.get("qualification_level_percentile", 85.0))
        persistent = np.isfinite(persistence) and persistence >= float(cfg.get("qualification_persistence", 0.40))
        inventory_stress = np.isfinite(inventory_draw) and inventory_draw >= float(cfg.get("qualification_inventory_draw_z", 1.0))
        spread_stress = np.isfinite(spread_pct) and spread_pct >= float(cfg.get("qualification_spread_percentile", 85.0))
        qualified = bool(
            (high_level and persistent)
            or (high_level and (inventory_stress or spread_stress))
            or (np.isfinite(level_pct) and level_pct >= 90.0 and inventory_stress)
        )
        supply_confirmed = bool(inventory_stress or spread_stress or (np.isfinite(breadth) and breadth >= float(cfg.get("qualification_breadth", 0.60))))
        if qualified:
            reason = "Structural qualification confirmed by high price level plus persistence/supply confirmation."
        elif np.isfinite(level_pct) and level_pct >= float(cfg.get("pressure_level_percentile", 75.0)):
            reason = "Structural pressure present, but confirmed-event qualification is not met."
        else:
            reason = "Structural pressure below the confirmed-event qualification zone."
        return {
            "score": score,
            "qualified": qualified,
            "high_level": bool(high_level),
            "persistent": bool(persistent),
            "inventory_stress": bool(inventory_stress),
            "spread_stress": bool(spread_stress),
            "breadth_confirmed": bool(np.isfinite(breadth) and breadth >= float(cfg.get("qualification_breadth", 0.60))),
            "supply_confirmed": supply_confirmed,
            "qualification_reason": reason,
        }

    def _oil_event_snapshot(self, row: pd.Series) -> Dict[str, Any]:
        cfg = self.config.get("event_overlay", {}).get("oil", {})
        momentum = self._oil_momentum_score(row)
        structural_info = self._oil_structural_score(row)
        structural = float(structural_info["score"])
        # Qualification is derived from the raw structural evidence, never from
        # a previously persisted flag. This prevents stale history values from
        # contaminating the live event state.
        structural_qualified = bool(structural_info["qualified"])
        momentum_activation = float(cfg.get("momentum_activation_score", cfg.get("activation_score", 0.20)))
        structural_activation = float(cfg.get("structural_activation_score", cfg.get("activation_score", 0.20)))
        momentum_confirmed = momentum >= momentum_activation
        structural_confirmed = structural_qualified and structural >= structural_activation
        if momentum_confirmed and structural_confirmed:
            event_type = "COMBINED"; event_score = max(momentum, structural)
        elif momentum_confirmed:
            event_type = "MOMENTUM"; event_score = momentum
        elif structural_confirmed:
            event_type = "STRUCTURAL"; event_score = structural
        else:
            event_type = "PRESSURE_ONLY" if max(momentum, structural) >= float(cfg.get("pressure_display_threshold", 0.20)) else "NONE"
            event_score = 0.0
        return {
            "momentum_score": float(momentum),
            "structural_score": float(structural),
            "structural_qualified": structural_qualified,
            "momentum_confirmed": bool(momentum_confirmed),
            "structural_confirmed": bool(structural_confirmed),
            "pressure_score": float(max(momentum, structural)),
            "event_score": float(event_score),
            "event_type": event_type,
            "event_active": bool(momentum_confirmed or structural_confirmed),
            "qualification_reason": structural_info["qualification_reason"],
            "structural_high_level": bool(structural_info.get("high_level", False)),
            "structural_persistent": bool(structural_info.get("persistent", False)),
            "structural_inventory_stress": bool(structural_info.get("inventory_stress", False)),
            "structural_spread_stress": bool(structural_info.get("spread_stress", False)),
            "structural_breadth_confirmed": bool(structural_info.get("breadth_confirmed", False)),
        }

    def _oil_event_score(self, row: pd.Series) -> float:
        if not bool(self.config.get("event_overlay", {}).get("oil", {}).get("enabled", True)):
            return 0.0
        return self._oil_event_snapshot(row)["event_score"]

    def _apply_event_overlays(self, base_weights: Dict[str, float], row: pd.Series, regime_id: int) -> Dict[str, Any]:
        w = self._normalize_weights(base_weights)
        oil_snapshot = self._oil_event_snapshot(row)
        oil_score = oil_snapshot["event_score"]
        oil_pressure = oil_snapshot["pressure_score"]
        unknown_score = self._unknown_event_score(row)
        overlay_cfg = self.config.get("event_overlay", {})
        oil_cfg = overlay_cfg.get("oil", {})
        unknown_cfg = overlay_cfg.get("unknown_event", {})
        unknown_activation = float(unknown_cfg.get("activation_score", 0.25))
        w.update({
            "oil_pressure_score": oil_pressure,
            "oil_event_score": oil_score,
            "oil_momentum_score": oil_snapshot["momentum_score"],
            "oil_structural_score": oil_snapshot["structural_score"],
            "oil_structural_qualified": oil_snapshot["structural_qualified"],
            "oil_momentum_confirmed": oil_snapshot["momentum_confirmed"],
            "oil_event_type": oil_snapshot["event_type"],
            "oil_event_active": oil_snapshot["event_active"],
            "oil_qualification_reason": oil_snapshot["qualification_reason"],
            "oil_structural_high_level": bool(oil_snapshot.get("structural_high_level", False)),
            "oil_structural_persistent": bool(oil_snapshot.get("structural_persistent", False)),
            "oil_structural_inventory_stress": bool(oil_snapshot.get("structural_inventory_stress", False)),
            "oil_structural_spread_stress": bool(oil_snapshot.get("structural_spread_stress", False)),
            "oil_structural_breadth_confirmed": bool(oil_snapshot.get("structural_breadth_confirmed", False)),
            "commodity_event_active": False,
            "commodity_event_reason": "No confirmed commodity event overlay active.",
            "unknown_event_score": unknown_score,
            "unknown_event_active": unknown_score >= unknown_activation,
            "unknown_guard_active": unknown_score >= unknown_activation,
        })
        if regime_id == 2:
            w["commodity_event_reason"] = "Systemic liquidity shock: commodity overlay disabled."
            return self._normalize_weights(w)
        unknown_guard_active = regime_id == 0 and unknown_score >= unknown_activation
        if unknown_guard_active:
            target_cash = float(unknown_cfg.get("minimum_cash", 50.0))
            if w["cash"] < target_cash:
                needed = target_cash - w["cash"]
                take_eq = min(needed, w["equity"]); w["equity"] -= take_eq; w["cash"] += take_eq; needed -= take_eq
                if needed > 0:
                    take_crp = min(needed, w["crypto"]); w["crypto"] -= take_crp; w["cash"] += take_crp
            w = self._normalize_weights(w)
        activation_score = float(oil_cfg.get("activation_score", 0.20))
        max_map = oil_cfg.get("max_commodity_by_regime", {"0": 35.0, "1": 45.0, "2": 0.0, "3": 20.0, "4": 15.0, "5": 25.0})
        max_commodity = float(max_map.get(str(regime_id), max_map.get("0", 35.0)))
        if oil_snapshot["event_active"] and oil_score >= activation_score and max_commodity > w["commodity"]:
            target = w["commodity"] + oil_score * (max_commodity - w["commodity"])
            delta = max(0.0, target - w["commodity"])
            eq_floor = float(oil_cfg.get("equity_floor_by_regime", {}).get(str(regime_id), 0.0))
            cash_floor = float(oil_cfg.get("cash_floor_by_regime", {}).get(str(regime_id), 0.0))
            if unknown_guard_active:
                cash_floor = max(cash_floor, float(unknown_cfg.get("minimum_cash", 50.0)))
                max_commodity = min(max_commodity, float(unknown_cfg.get("maximum_commodity", 20.0)))
            eq_available = max(0.0, w["equity"] - eq_floor)
            cash_available = max(0.0, w["cash"] - cash_floor)
            eq_share = float(np.clip(oil_cfg.get("funding_from_equity", 0.70), 0.0, 1.0))
            from_eq = min(delta * eq_share, eq_available)
            from_cash = min(delta - from_eq, cash_available)
            actual_delta = from_eq + from_cash
            w["equity"] -= from_eq; w["cash"] -= from_cash; w["commodity"] += actual_delta
            w["commodity_event_active"] = actual_delta > 1e-9
            w["commodity_event_reason"] = f"Confirmed oil event active ({oil_snapshot['event_type']}): score={oil_score:.2f}, commodity allocation={w['commodity']:.2f}%."
        elif oil_snapshot["event_type"] == "PRESSURE_ONLY":
            w["commodity_event_reason"] = f"Oil structural/momentum pressure monitored (pressure={oil_pressure:.2f}); confirmed-event gate not met."

        # Final canonicalization: event state must be self-consistent at the
        # persistence boundary regardless of any upstream normalization.
        if oil_snapshot["momentum_confirmed"] and oil_snapshot["structural_confirmed"]:
            canonical_type = "COMBINED"
            canonical_score = max(oil_snapshot["momentum_score"], oil_snapshot["structural_score"])
        elif oil_snapshot["momentum_confirmed"]:
            canonical_type = "MOMENTUM"
            canonical_score = oil_snapshot["momentum_score"]
        elif oil_snapshot["structural_confirmed"]:
            canonical_type = "STRUCTURAL"
            canonical_score = oil_snapshot["structural_score"]
        elif oil_pressure >= float(oil_cfg.get("pressure_display_threshold", 0.20)):
            canonical_type = "PRESSURE_ONLY"
            canonical_score = 0.0
        else:
            canonical_type = "NONE"
            canonical_score = 0.0
        w["oil_pressure_score"] = float(oil_pressure)
        w["oil_event_score"] = float(canonical_score)
        w["oil_event_type"] = canonical_type
        w["oil_event_active"] = bool(canonical_type in {"MOMENTUM", "STRUCTURAL", "COMBINED"})
        w["oil_structural_qualified"] = bool(oil_snapshot["structural_qualified"])
        w["oil_momentum_confirmed"] = bool(oil_snapshot["momentum_confirmed"])
        return self._normalize_weights(w)

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, Any]:
        keys = ("cash", "gold", "bond", "equity", "commodity", "crypto")
        clean = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
        total = sum(clean.values())
        if total <= 1e-12:
            clean = {"cash": 100.0, "gold": 0.0, "bond": 0.0, "equity": 0.0, "commodity": 0.0, "crypto": 0.0}
        elif abs(total - 100.0) > 1e-9:
            clean = {k: v * 100.0 / total for k, v in clean.items()}
        # Preserve non-weight metadata across normalization.
        for k, v in weights.items():
            if k not in keys:
                clean[k] = v
        return clean

    def get_portfolio_weights(self, regime_id: int, subtype: str = "", row: Optional[pd.Series] = None) -> Dict[str, Any]:
        weights_map = {
            0: {"cash": 35.0, "gold": 20.0, "bond": 20.0, "equity": 15.0, "commodity": 5.0, "crypto": 5.0},
        }
        # Configuration is the single source of truth for named regime allocations.
        for regime in self.config.get("regimes", []):
            try:
                rid = int(regime.get("id"))
            except (TypeError, ValueError):
                continue
            allocation = regime.get("default_allocation")
            if isinstance(allocation, dict):
                weights_map[rid] = allocation
        base = self._normalize_weights(weights_map.get(int(regime_id), weights_map[0]))
        if row is None:
            base.update(oil_pressure_score=0.0, oil_event_score=0.0, oil_momentum_score=0.0, oil_structural_score=0.0, oil_structural_qualified=False, oil_momentum_confirmed=False, oil_event_type="NONE", oil_event_active=False, oil_qualification_reason="No live row supplied; regime weights only.", commodity_event_active=False, commodity_event_reason="No live row supplied; regime weights only.", unknown_event_score=0.0, unknown_event_active=False, unknown_guard_active=False)
            return base
        if bool(self.config.get("research_strategies", {}).get("live_use_adaptive_opportunity", False)):
            base = self._adaptive_opportunity_weights(base, row, int(regime_id))
        return self._apply_event_overlays(base, row, int(regime_id))

    def _adaptive_opportunity_weights(self, base: Dict[str, Any], row: pd.Series, regime_id: int) -> Dict[str, Any]:
        """Bounded return-seeking allocation overlay; uses only prior-known row features."""
        cfg = self.config.get("research_strategies", {})
        risk_map = cfg.get("production_risk_budget_by_regime", cfg.get("risk_budget_by_regime", {}))
        risk = float(risk_map.get(str(regime_id), 0.60))
        sp20 = self._safe_float(row.get("spx_ret_20d"), 0.0)
        sp100 = self._safe_float(row.get("spx_ret_100d"), 0.0)
        btc60 = self._safe_float(row.get("btc_ret_60d"), 0.0)
        vix_pct = self._safe_float(row.get("vix_percentile_252"), 50.0)
        trend = cfg.get("production_trend_adjustments", cfg.get("trend_adjustments", {}))
        if sp20 > 0 and sp100 > 0:
            risk += float(trend.get("positive_20d_and_100d", 0.12))
        elif sp20 < 0 and sp100 < 0:
            risk += float(trend.get("negative_20d_and_100d", -0.15))
        if vix_pct > 80:
            risk += float(trend.get("vix_pct_above_80", -0.18))
        elif vix_pct < 30:
            risk += float(trend.get("vix_pct_below_30", 0.05))
        btc_cfg = cfg.get("production_btc_adjustments", cfg.get("btc_adjustments", {}))
        btc_enabled = bool(cfg.get("production_enable_btc", True))
        if btc_enabled and btc60 > 0:
            risk += float(btc_cfg.get("positive_60d", 0.06))
        elif btc_enabled and btc60 < 0:
            risk += float(btc_cfg.get("negative_60d", -0.04))
        risk = float(np.clip(risk, 0.08, 0.88))
        sleeve = cfg.get("production_risk_sleeve_weights", cfg.get("risk_sleeve_weights", {"equity":0.62,"gold":0.14,"commodity":0.10,"bond":0.14,"crypto":0.10}))
        eq_s = float(sleeve.get("equity",0.62)); gold_s=float(sleeve.get("gold",0.14)); oil_s=float(sleeve.get("commodity",0.10)); bond_s=float(sleeve.get("bond",0.14)); btc_s=float(sleeve.get("crypto",0.10)) if btc_enabled and btc60 > 0 and risk > 0.45 else 0.0
        bond_s = max(0.0, bond_s - btc_s)
        eq = risk * eq_s; gold = risk * gold_s; oil = risk * oil_s; bond = risk * bond_s; btc = risk * btc_s
        cash = max(0.0, 1.0 - (eq + gold + oil + bond + btc))
        # Preserve the regime's defensive floor even when the opportunity sleeve is active.
        min_cash = float(cfg.get("production_min_cash_by_regime", {}).get(str(regime_id), 0.0))
        if cash < min_cash:
            needed = min_cash - cash
            donor = ["equity", "commodity", "crypto", "gold"]
            vals = {"equity":eq, "commodity":oil, "crypto":btc, "gold":gold}
            for key in donor:
                take = min(needed, vals[key]); vals[key] -= take; needed -= take
                if needed <= 1e-9: break
            eq, oil, btc, gold = vals["equity"], vals["commodity"], vals["crypto"], vals["gold"]
            cash = min_cash
        base.update({"cash": cash*100.0, "gold": gold*100.0, "bond": bond*100.0, "equity": eq*100.0, "commodity": oil*100.0, "crypto": btc*100.0, "strategy_mode": "ADAPTIVE_OPPORTUNITY_BTC" if btc_enabled else "ADAPTIVE_OPPORTUNITY"})
        return self._normalize_weights(base)

    def get_event_snapshot(self, row: pd.Series) -> Dict[str, Any]:
        oil = self._oil_event_snapshot(row)
        unknown_score = self._unknown_event_score(row)
        return {
            "oil_pressure_score": oil["pressure_score"],
            "oil_event_score": oil["event_score"],
            "oil_momentum_score": oil["momentum_score"],
            "oil_structural_score": oil["structural_score"],
            "oil_structural_qualified": oil["structural_qualified"],
            "oil_momentum_confirmed": oil["momentum_confirmed"],
            "oil_event_type": oil["event_type"],
            "oil_event_active": oil["event_active"],
            "oil_qualification_reason": oil["qualification_reason"],
            "oil_structural_high_level": bool(oil.get("structural_high_level", False)),
            "oil_structural_persistent": bool(oil.get("structural_persistent", False)),
            "oil_structural_inventory_stress": bool(oil.get("structural_inventory_stress", False)),
            "oil_structural_spread_stress": bool(oil.get("structural_spread_stress", False)),
            "oil_structural_breadth_confirmed": bool(oil.get("structural_breadth_confirmed", False)),
            "oil_level_percentile_252": self._safe_float(row.get("oil_level_percentile_252"), np.nan),
            "oil_level_percentile_756": self._safe_float(row.get("oil_level_percentile_756"), np.nan),
            "oil_high_level_persistence_60d": self._safe_float(row.get("oil_high_level_persistence_60d"), np.nan),
            "energy_breadth_20d": self._safe_float(row.get("energy_breadth_20d"), np.nan),
            "crude_inventory_draw_z": self._safe_float(row.get("crude_inventory_draw_z"), np.nan),
            "commodity_breadth_20d": self._safe_float(row.get("commodity_breadth_20d"), np.nan),
            "oil_event_quality_60d": self._safe_float(row.get("oil_event_quality_60d"), 0.5),
            "tips_level_z": self._safe_float(row.get("tips_level_z"), 0.0),
            "tips_level_percentile_252": self._safe_float(row.get("tips_level_percentile_252"), 50.0),
            "unknown_event_score": unknown_score,
            "unknown_event_active": unknown_score >= float(self.config.get("event_overlay", {}).get("unknown_event", {}).get("activation_score", 0.25)),
        }

