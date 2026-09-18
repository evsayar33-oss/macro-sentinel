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
        if custom_thresholds:
            th.update(custom_thresholds)

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
        r3_daily_shock = tips_chg_z > th["r3_tips_z"]
        r3_persistent = tips_lvl >= high_real_floor or tips_level_z > th.get("r3_tips_level_z", 1.0)
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
        n = len(out)
        defaults = {
            "raw_candidate_id": 0,
            "raw_candidate_name": "REJIMSIZ_GECIS",
            "confirmed_regime_id": 0,
            "confirmed_regime_name": "REJIMSIZ_GECIS",
            "regime_type": "TRANSITION",
            "regime_subtype": "Dengeli / Nötr Piyasa",
            "regime_severity_source": "NONE",
            "hysteresis_days_left": 0,
            "conflict_note": "",
            "regime_cash_weight": 35.0,
            "regime_gold_weight": 20.0,
            "regime_bond_weight": 20.0,
            "regime_eq_weight": 15.0,
            "regime_commodity_weight": 5.0,
            "regime_crypto_weight": 5.0,
            "oil_event_score": 0.0,
            "oil_momentum_score": 0.0,
            "oil_structural_score": 0.0,
            "oil_event_type": "NONE",
            "oil_event_active": False,
            "commodity_event_active": False,
            "commodity_event_reason": "",
            "unknown_event_score": 0.0,
            "unknown_event_active": False,
            "unknown_guard_active": False,
        }
        for col, val in defaults.items():
            out[col] = val

        current_id = 0
        current_name = "REJIMSIZ_GECIS"
        current_type = "TRANSITION"
        current_subtype = "Dengeli / Nötr Piyasa"
        current_severity_source = "NONE"
        hysteresis = 0

        for i in range(n):
            row = out.iloc[i]
            evals = self.evaluate_regimes_for_row(row, custom_thresholds=custom_thresholds)
            decision = self.resolve_conflicts(evals, row)
            candidate_id = int(decision["candidate_id"])

            out.at[out.index[i], "raw_candidate_id"] = candidate_id
            out.at[out.index[i], "raw_candidate_name"] = decision["candidate_name"]
            out.at[out.index[i], "conflict_note"] = decision["conflict_note"]

            hard_shock = self._hard_event_override(row)
            if candidate_id != 0:
                current_id = candidate_id
                current_name = decision["candidate_name"]
                current_type = decision["candidate_type"]
                current_subtype = decision["candidate_subtype"]
                hysteresis = self.hysteresis_days
            elif hard_shock:
                # A very extreme sensor is allowed to clear a stale regime state;
                # it does not invent a named regime.
                current_id = 0
                current_name = "REJIMSIZ_GECIS"
                current_type = "TRANSITION"
                current_subtype = "Extreme Unclassified Event"
                current_severity_source = "UNKNOWN_EVENT"
                hysteresis = 0
            elif hysteresis > 0:
                hysteresis -= 1
            else:
                current_id = 0
                current_name = "REJIMSIZ_GECIS"
                current_type = "TRANSITION"
                current_subtype = "Dengeli / Nötr Piyasa"
                current_severity_source = "NONE"

            out.at[out.index[i], "confirmed_regime_id"] = current_id
            out.at[out.index[i], "confirmed_regime_name"] = current_name
            out.at[out.index[i], "regime_type"] = current_type
            out.at[out.index[i], "regime_subtype"] = current_subtype
            out.at[out.index[i], "regime_severity_source"] = current_severity_source
            out.at[out.index[i], "hysteresis_days_left"] = hysteresis

            weights = self.get_portfolio_weights(current_id, current_subtype, row=row)
            for source, target in (
                ("cash", "regime_cash_weight"),
                ("gold", "regime_gold_weight"),
                ("bond", "regime_bond_weight"),
                ("equity", "regime_eq_weight"),
                ("commodity", "regime_commodity_weight"),
                ("crypto", "regime_crypto_weight"),
            ):
                out.at[out.index[i], target] = float(weights[source])
            out.at[out.index[i], "oil_event_score"] = float(weights.get("oil_event_score", 0.0))
            out.at[out.index[i], "oil_momentum_score"] = float(weights.get("oil_momentum_score", 0.0))
            out.at[out.index[i], "oil_structural_score"] = float(weights.get("oil_structural_score", 0.0))
            out.at[out.index[i], "oil_event_type"] = str(weights.get("oil_event_type", "NONE"))
            out.at[out.index[i], "commodity_event_active"] = bool(weights.get("commodity_event_active", False))
            out.at[out.index[i], "commodity_event_reason"] = str(weights.get("commodity_event_reason", ""))
            out.at[out.index[i], "unknown_event_score"] = float(weights.get("unknown_event_score", 0.0))
            out.at[out.index[i], "unknown_event_active"] = bool(weights.get("unknown_event_active", False))
            out.at[out.index[i], "unknown_guard_active"] = bool(weights.get("unknown_guard_active", False))

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
        move_pct = self._safe_float(row.get("oil_abs_5d_percentile"), 0.0)

        if shock_z <= base_z and move_pct < percentile_gate:
            return 0.0

        intensity = np.clip((max(shock_z, base_z) - base_z) / (saturation_z - base_z), 0.0, 1.0)
        percentile_boost = 0.10 if move_pct >= percentile_gate else 0.0
        breadth = self._safe_float(row.get("commodity_breadth_20d"), 0.0)
        breadth_boost = 0.10 if breadth >= 0.50 else 0.0
        quality = float(np.clip(self._safe_float(row.get("oil_event_quality_60d"), 0.5), 0.5, 1.0))
        score = float(np.clip(intensity + percentile_boost + breadth_boost, 0.0, 1.0))
        return float(np.clip(score * (0.80 + 0.20 * quality), 0.0, 1.0))

    def _oil_structural_score(self, row: pd.Series) -> float:
        cfg = self.config.get("event_overlay", {}).get("oil", {}).get("structural", {})
        if not bool(cfg.get("enabled", True)):
            return 0.0

        level_pct = self._safe_float(row.get("oil_level_percentile_756"), np.nan)
        if not np.isfinite(level_pct):
            level_pct = self._safe_float(row.get("oil_level_percentile_252"), np.nan)
        persistence = self._safe_float(row.get("oil_high_level_persistence_60d"), np.nan)
        breadth = self._safe_float(row.get("energy_breadth_20d"), np.nan)
        inventory_draw = self._safe_float(row.get("crude_inventory_draw_z"), np.nan)
        spread_pct = self._safe_float(row.get("brent_wti_spread_percentile_252"), np.nan)
        brent_pct = self._safe_float(row.get("brent_level_percentile_252"), np.nan)

        components = []
        weights = []

        if np.isfinite(level_pct):
            soft = float(cfg.get("level_soft_percentile", 80.0))
            hard = max(soft + 1.0, float(cfg.get("level_hard_percentile", 95.0)))
            level_score = np.clip((level_pct - soft) / (hard - soft), 0.0, 1.0)
            components.append(float(level_score)); weights.append(float(cfg.get("level_weight", 0.35)))

        if np.isfinite(brent_pct):
            soft = float(cfg.get("brent_soft_percentile", 80.0))
            hard = max(soft + 1.0, float(cfg.get("brent_hard_percentile", 95.0)))
            brent_score = np.clip((brent_pct - soft) / (hard - soft), 0.0, 1.0)
            components.append(float(brent_score)); weights.append(float(cfg.get("brent_weight", 0.15)))

        if np.isfinite(persistence):
            gate = float(cfg.get("persistence_gate", 0.40))
            persistence_score = np.clip((persistence - gate) / max(1.0 - gate, 1e-6), 0.0, 1.0)
            components.append(float(persistence_score)); weights.append(float(cfg.get("persistence_weight", 0.25)))

        if np.isfinite(breadth):
            gate = float(cfg.get("breadth_gate", 0.60))
            breadth_score = np.clip((breadth - gate) / max(1.0 - gate, 1e-6), 0.0, 1.0)
            components.append(float(breadth_score)); weights.append(float(cfg.get("breadth_weight", 0.10)))

        if np.isfinite(inventory_draw):
            inventory_score = np.clip(inventory_draw / 2.0, 0.0, 1.0)
            components.append(float(inventory_score)); weights.append(float(cfg.get("inventory_weight", 0.10)))

        if np.isfinite(spread_pct):
            spread_score = np.clip((spread_pct - float(cfg.get("spread_soft_percentile", 70.0))) / 25.0, 0.0, 1.0)
            components.append(float(spread_score)); weights.append(float(cfg.get("spread_weight", 0.05)))

        if not components:
            return 0.0

        score = float(np.average(components, weights=weights))

        # Qualification rule: persistent high price is sufficient only when the
        # market is also broad / supply-stressed; this prevents a permanently
        # expensive but orderly oil market from becoming a "shock" every day.
        high_level = np.isfinite(level_pct) and level_pct >= float(cfg.get("qualification_level_percentile", 85.0))
        persistent = np.isfinite(persistence) and persistence >= float(cfg.get("qualification_persistence", 0.40))
        broad = np.isfinite(breadth) and breadth >= float(cfg.get("qualification_breadth", 0.60))
        inventory_stress = np.isfinite(inventory_draw) and inventory_draw >= float(cfg.get("qualification_inventory_draw_z", 1.0))
        spread_stress = np.isfinite(spread_pct) and spread_pct >= float(cfg.get("qualification_spread_percentile", 85.0))

        qualified = (
            (high_level and persistent)
            or (high_level and (inventory_stress or spread_stress))
            or (level_pct >= 90.0 and inventory_stress)
        )
        return float(score if qualified else score * float(cfg.get("unqualified_multiplier", 0.35)))

    def _oil_event_snapshot(self, row: pd.Series) -> Dict[str, Any]:
        momentum = self._oil_momentum_score(row)
        structural = self._oil_structural_score(row)
        combined = max(momentum, structural)
        if combined <= 0.0:
            event_type = "NONE"
        elif momentum >= 0.20 and structural >= 0.20:
            event_type = "COMBINED"
        elif structural > momentum:
            event_type = "STRUCTURAL"
        else:
            event_type = "MOMENTUM"
        return {
            "momentum_score": float(momentum),
            "structural_score": float(structural),
            "score": float(combined),
            "event_type": event_type,
        }

    def _oil_event_score(self, row: pd.Series) -> float:
        if not bool(self.config.get("event_overlay", {}).get("oil", {}).get("enabled", True)):
            return 0.0
        return self._oil_event_snapshot(row)["score"]

    def _apply_event_overlays(self, base_weights: Dict[str, float], row: pd.Series, regime_id: int) -> Dict[str, Any]:
        w = self._normalize_weights(base_weights)
        oil_snapshot = self._oil_event_snapshot(row)
        oil_score = oil_snapshot["score"]
        unknown_score = self._unknown_event_score(row)
        overlay_cfg = self.config.get("event_overlay", {})
        oil_cfg = overlay_cfg.get("oil", {})
        unknown_cfg = overlay_cfg.get("unknown_event", {})

        w["oil_event_score"] = oil_score
        w["oil_momentum_score"] = oil_snapshot["momentum_score"]
        w["oil_structural_score"] = oil_snapshot["structural_score"]
        w["oil_event_type"] = oil_snapshot["event_type"]
        w["oil_event_active"] = oil_score >= float(oil_cfg.get("activation_score", 0.20))
        w["commodity_event_active"] = False
        w["commodity_event_reason"] = "No commodity event overlay active."
        unknown_activation = float(unknown_cfg.get("activation_score", 0.25))
        w["unknown_event_score"] = unknown_score
        w["unknown_event_active"] = unknown_score >= unknown_activation
        w["unknown_guard_active"] = unknown_score >= unknown_activation

        # Systemic liquidity shock is an explicit hard risk-off state.
        if regime_id == 2:
            w["commodity_event_reason"] = "Systemic liquidity shock: commodity overlay disabled."
            w = self._normalize_weights(w)
            w.update(oil_event_score=oil_score, oil_momentum_score=oil_snapshot["momentum_score"], oil_structural_score=oil_snapshot["structural_score"], oil_event_type=oil_snapshot["event_type"], oil_event_active=oil_score >= float(oil_cfg.get("activation_score", 0.20)), commodity_event_active=False, unknown_event_score=unknown_score, unknown_event_active=unknown_score >= unknown_activation, unknown_guard_active=unknown_score >= unknown_activation)
            return w

        # Unknown severe anomaly: do not invent direction; reduce exposure and
        # preserve a meaningful cash buffer. This is a safety circuit, not a regime.
        if unknown_score > float(unknown_cfg.get("activation_score", 0.25)) and regime_id == 0:
            target_cash = float(unknown_cfg.get("minimum_cash", 50.0))
            if w["cash"] < target_cash:
                needed = target_cash - w["cash"]
                take_eq = min(needed, w["equity"])
                w["equity"] -= take_eq
                w["cash"] += take_eq
                needed -= take_eq
                if needed > 0:
                    take_crp = min(needed, w["crypto"])
                    w["crypto"] -= take_crp
                    w["cash"] += take_crp
            w = self._normalize_weights(w)

        activation_score = float(oil_cfg.get("activation_score", 0.20))
        max_map = oil_cfg.get("max_commodity_by_regime", {"0": 35.0, "1": 45.0, "2": 0.0, "3": 20.0, "4": 15.0, "5": 25.0})
        max_commodity = float(max_map.get(str(regime_id), max_map.get("0", 35.0)))

        unknown_guard_active = (
            regime_id == 0
            and unknown_score > float(unknown_cfg.get("activation_score", 0.25))
        )

        if oil_score >= activation_score and max_commodity > w["commodity"]:
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

            w["equity"] -= from_eq
            w["cash"] -= from_cash
            w["commodity"] += actual_delta
            w["commodity_event_active"] = actual_delta > 1e-9
            w["commodity_event_reason"] = (
                f"Independent oil event active ({oil_snapshot['event_type']}): score={oil_score:.2f}, "
                f"commodity allocation={w['commodity']:.2f}%."
            )

        w = self._normalize_weights(w)
        w["oil_event_score"] = oil_score
        w["oil_momentum_score"] = oil_snapshot["momentum_score"]
        w["oil_structural_score"] = oil_snapshot["structural_score"]
        w["oil_event_type"] = oil_snapshot["event_type"]
        w["oil_event_active"] = oil_score >= activation_score
        w["unknown_event_score"] = unknown_score
        w["unknown_event_active"] = unknown_score >= unknown_activation
        w["unknown_guard_active"] = unknown_score >= unknown_activation
        return w

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
            base.update(oil_event_score=0.0, oil_momentum_score=0.0, oil_structural_score=0.0, oil_event_type="NONE", oil_event_active=False, commodity_event_active=False, commodity_event_reason="No live row supplied; regime weights only.", unknown_event_score=0.0, unknown_event_active=False, unknown_guard_active=False)
            return base
        return self._apply_event_overlays(base, row, int(regime_id))

    def get_event_snapshot(self, row: pd.Series) -> Dict[str, Any]:
        oil = self._oil_event_snapshot(row)
        unknown_score = self._unknown_event_score(row)
        return {
            "oil_event_score": oil["score"],
            "oil_momentum_score": oil["momentum_score"],
            "oil_structural_score": oil["structural_score"],
            "oil_event_type": oil["event_type"],
            "oil_event_active": oil["score"] >= float(self.config.get("event_overlay", {}).get("oil", {}).get("activation_score", 0.20)),
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
