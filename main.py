"""
Macro Sentinel v2.0 — point-in-time autonomous daily macro engine.

Runtime contract
----------------
1. Collect real FRED + Yahoo data.
2. Never synthesize market data for a live decision.
3. Validate coverage / freshness before changing allocation.
4. Compute all indicators point-in-time; no look-ahead correlations.
5. Classify regime + independent event layer.
6. Persist the decision and a machine-readable health state.
7. On insufficient data, HOLD_LAST_VALID instead of inventing a signal.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

from regime_engine import MacroRegimeEngine

FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"
IST_TZ = pytz.timezone("Europe/Istanbul")


class UltimateSentinelEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.regime_engine = MacroRegimeEngine()
        data_cfg = self.regime_engine.config.get("data_quality", {})
        self.min_history = int(data_cfg.get("minimum_history_days", 126))
        self.market_stale_days = int(data_cfg.get("market_max_stale_days", 3))
        self.fred_stale_days = int(data_cfg.get("fred_max_stale_days", 7))

    @staticmethod
    def _as_numeric_series(values, index=None):
        s = pd.to_numeric(pd.Series(values, index=index), errors="coerce").dropna()
        return s.sort_index()

    @staticmethod
    def _extract_close(raw_df: pd.DataFrame, tickers) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()
        if isinstance(raw_df.columns, pd.MultiIndex):
            level0 = list(raw_df.columns.get_level_values(0))
            if "Close" in level0:
                close = raw_df["Close"].copy()
            elif "Adj Close" in level0:
                close = raw_df["Adj Close"].copy()
            else:
                return pd.DataFrame()
        else:
            close = raw_df.copy()
            if "Close" in close.columns:
                close = close[["Close"]]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        for ticker in tickers:
            if ticker not in close.columns:
                close[ticker] = np.nan
        return close[tickers]

    def fetch_fred(self, series_id: str, limit: int = 500) -> Tuple[pd.Series, Dict[str, object]]:
        if not self.api_key:
            return pd.Series(dtype=float), {"ok": False, "reason": "FRED_API_KEY missing", "last_date": None, "count": 0}
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
                "units": "lin",
            }
            response = requests.get(self.base_url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])
            if not observations:
                return pd.Series(dtype=float), {"ok": False, "reason": "no observations", "last_date": None, "count": 0}
            obs = pd.DataFrame(observations)[["date", "value"]]
            obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
            obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
            obs = obs.dropna().sort_values("date")
            series = obs.set_index("date")["value"]
            last_date = series.index[-1].date().isoformat() if not series.empty else None
            return series, {"ok": not series.empty, "reason": "ok" if not series.empty else "empty", "last_date": last_date, "count": int(series.shape[0])}
        except Exception as exc:
            return pd.Series(dtype=float), {"ok": False, "reason": str(exc)[:180], "last_date": None, "count": 0}

    @staticmethod
    def _freshness(last_date, max_age_days: int) -> bool:
        if last_date is None:
            return False
        try:
            d = pd.Timestamp(last_date).date()
            age = (datetime.now(timezone.utc).date() - d).days
            return age <= max_age_days
        except Exception:
            return False

    @staticmethod
    def _safe_reindex(series: pd.Series, index: pd.Index, default: float = np.nan) -> pd.Series:
        if series is None or series.empty:
            return pd.Series(default, index=index, dtype=float)
        s = pd.to_numeric(series, errors="coerce").sort_index()
        # Forward-fill only. Back-filling a historical series would introduce future data.
        return s.reindex(index).ffill().fillna(default)

    def _load_history(self) -> pd.DataFrame:
        if not os.path.exists(HISTORY_FILE):
            return pd.DataFrame()
        try:
            df = pd.read_csv(HISTORY_FILE)
            if df.empty:
                return df
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _valid_previous_allocation(history: pd.DataFrame):
        if history.empty:
            return None
        required = [
            "cash_weight", "gold_weight", "bond_weight", "eq_weight", "commodity_weight", "crypto_weight"
        ]
        for c in required:
            if c not in history.columns:
                return None
        total = history[required].astype(float).sum(axis=1)
        valid = history.loc[np.isfinite(total) & (abs(total - 100.0) < 0.25)]
        if valid.empty:
            return None
        row = valid.iloc[-1]
        return {
            "cash": float(row["cash_weight"]),
            "gold": float(row["gold_weight"]),
            "bond": float(row["bond_weight"]),
            "equity": float(row["eq_weight"]),
            "commodity": float(row["commodity_weight"]),
            "crypto": float(row["crypto_weight"]),
            "date": str(row.get("date", "")),
        }

    @staticmethod
    def _allocation_payload(weights: Dict[str, float]) -> Dict[str, float]:
        return {
            "cash": float(weights.get("cash", 0.0)),
            "gold": float(weights.get("gold", 0.0)),
            "bond": float(weights.get("bond", 0.0)),
            "equity": float(weights.get("equity", 0.0)),
            "commodity": float(weights.get("commodity", 0.0)),
            "crypto": float(weights.get("crypto", 0.0)),
        }

    def _data_health(self, fred_meta: Dict[str, Dict[str, object]], y_data: pd.DataFrame, today_index) -> Dict[str, object]:
        core_yahoo = ["ES=F", "CL=F", "GC=F", "^VIX"]
        secondary_yahoo = ["HG=F", "SI=F", "USDJPY=X", "EURUSD=X", "SOXX", "BTC-USD", "BDRY", "^VIX3M"]
        core_fred = ["spread", "tips", "vix", "dxy", "dgs2", "dgs10", "t10yie"]

        issues = []
        warnings = []
        available_core = 0

        for ticker in core_yahoo:
            ok = ticker in y_data.columns and y_data[ticker].notna().sum() >= self.min_history and pd.notna(y_data[ticker].iloc[-1])
            if ok:
                available_core += 1
                last = y_data[ticker].dropna().index[-1]
                if (today_index.date() - pd.Timestamp(last).date()).days > self.market_stale_days:
                    issues.append(f"stale market series: {ticker}")
            else:
                issues.append(f"missing/short core market series: {ticker}")

        for ticker in secondary_yahoo:
            ok = ticker in y_data.columns and y_data[ticker].notna().sum() >= 30 and pd.notna(y_data[ticker].iloc[-1])
            if not ok:
                warnings.append(f"missing secondary market series: {ticker}")

        for key in core_fred:
            meta = fred_meta.get(key, {})
            if not meta.get("ok"):
                issues.append(f"FRED unavailable: {key}")
            elif not self._freshness(meta.get("last_date"), self.fred_stale_days):
                issues.append(f"stale FRED series: {key}")
            elif int(meta.get("count", 0)) < 30:
                warnings.append(f"short FRED history: {key}")

        if available_core < 3 or issues:
            status = "HALT"
        elif warnings:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "core_market_available": available_core,
            "core_market_required": len(core_yahoo),
            "issues": issues,
            "warnings": warnings,
        }

    def _build_macro_frame(self, y_data: pd.DataFrame, raw: Dict[str, pd.Series]) -> pd.DataFrame:
        index = y_data.index
        p = pd.DataFrame(index=index)
        p["oil"] = y_data["CL=F"]
        p["freight"] = y_data.get("BDRY", pd.Series(index=index, dtype=float))
        p["hy_oas"] = self._safe_reindex(raw["spread"], index)
        p["ig_oas"] = self._safe_reindex(raw["ig_spread"], index)
        p["spx"] = y_data["ES=F"]
        p["ust10y"] = self._safe_reindex(raw["dgs10"], index)
        p["broad_dollar"] = self._safe_reindex(raw["dxy"], index)
        p["usdjpy"] = y_data.get("USDJPY=X", pd.Series(index=index, dtype=float))
        p["vix"] = self._safe_reindex(raw["vix"], index)
        p["btc"] = y_data.get("BTC-USD", pd.Series(index=index, dtype=float))
        p["tips10y"] = self._safe_reindex(raw["tips"], index)
        p["t10yie"] = self._safe_reindex(raw["t10yie"], index)
        p["dgs2"] = self._safe_reindex(raw["dgs2"], index)
        p["dgs10"] = self._safe_reindex(raw["dgs10"], index)
        p["ndl"] = self._safe_reindex(raw["fed"], index) - self._safe_reindex(raw["tga"], index) - (self._safe_reindex(raw["rrp"], index) * 1000.0)
        p["gold"] = y_data["GC=F"]
        p["copper"] = y_data.get("HG=F", pd.Series(index=index, dtype=float))
        p["silver"] = y_data.get("SI=F", pd.Series(index=index, dtype=float))
        return p

    def run(self) -> Dict[str, object]:
        now = datetime.now(IST_TZ)
        history = self._load_history()

        fred_ids = {
            "fed": "WALCL",
            "ecb": "ECBASSETSW",
            "boj": "JPNASSETS",
            "rrp": "RRPONTSYD",
            "tga": "WTREGEN",
            "spread": "BAMLH0A0HYM2",
            "tips": "DFII10",
            "pmi": "NAPM",
            "vix": "VIXCLS",
            "yc": "T10Y2Y",
            "t10yie": "T10YIE",
            "dxy": "DTWEXBGS",
            "ig_spread": "BAMLC0A0CM",
            "dgs2": "DGS2",
            "dgs10": "DGS10",
        }
        raw = {}
        fred_meta = {}
        for key, sid in fred_ids.items():
            raw[key], fred_meta[key] = self.fetch_fred(sid)

        tickers = [
            "HG=F", "SI=F", "GC=F", "ES=F", "EURUSD=X", "USDJPY=X", "JPYUSD=X",
            "^VIX", "^VIX3M", "SOXX", "CL=F", "TLT", "BTC-USD", "BDRY"
        ]
        y_data = pd.DataFrame()
        yahoo_error = ""
        try:
            downloaded = yf.download(
                tickers=tickers,
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            y_data = self._extract_close(downloaded, tickers).ffill(limit=5)
        except Exception as exc:
            yahoo_error = str(exc)[:180]

        if y_data.empty:
            health = {"status": "HALT", "issues": [f"Yahoo unavailable: {yahoo_error or 'empty response'}"], "warnings": [], "core_market_available": 0, "core_market_required": 4}
        else:
            health = self._data_health(fred_meta, y_data, now)
            if yahoo_error:
                health["warnings"].append(f"Yahoo warning: {yahoo_error}")

        prev = self._valid_previous_allocation(history)

        # Fail-closed: do not calculate a new portfolio from guessed prices.
        if health["status"] == "HALT":
            if prev:
                weights = {k: prev[k] for k in ("cash", "gold", "bond", "equity", "commodity", "crypto")}
                allocation_source = "LAST_VALID"
                previous_date = prev.get("date", "")
            else:
                weights = {"cash": 100.0, "gold": 0.0, "bond": 0.0, "equity": 0.0, "commodity": 0.0, "crypto": 0.0}
                allocation_source = "SAFE_CASH_FIRST_RUN"
                previous_date = ""

            return self._make_result(
                now=now,
                health=health,
                api_status="Offline" if not y_data.size else "Degraded",
                weights=weights,
                source=allocation_source,
                previous_date=previous_date,
                history=history,
            )

        # Build indicators. From this point onward all data are real and point-in-time.
        macro_df = self._build_macro_frame(y_data, raw)
        prepared = self.regime_engine.prepare_indicators(macro_df)

        # Growth proxy selection uses current historical sample only; allocation is
        # interpreted as an end-of-day decision for the next session.
        spx_ret = y_data["ES=F"].pct_change()
        growth_proxies = {
            "Bakir/Altin": y_data["HG=F"] / y_data["GC=F"],
            "Gumus/Altin": y_data["SI=F"] / y_data["GC=F"],
            "YariIletken/Altin": y_data["SOXX"] / y_data["GC=F"],
        }
        best_corr = -np.inf
        active_growth_name = "Bakir/Altin"
        active_growth_series = growth_proxies[active_growth_name]
        for name, series in growth_proxies.items():
            pair = pd.concat([series.pct_change(), spx_ret], axis=1).dropna().tail(90)
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 30 else np.nan
            if pd.notna(corr) and corr > best_corr:
                best_corr = float(corr)
                active_growth_name = name
                active_growth_series = series

        # No sign inversion of oil. Oil is an independent event sensor.
        oil_ret = y_data["CL=F"].pct_change()
        oil_trend_display = float(prepared["oil_trend_strength"].iloc[-1]) if pd.notna(prepared["oil_trend_strength"].iloc[-1]) else 0.0

        def z_series(series):
            s = pd.to_numeric(series, errors="coerce")
            mean = s.mean()
            std = s.std()
            return (s - mean) / (std + 1e-9)

        fed_s = self._safe_reindex(raw["fed"], y_data.index)
        tga_s = self._safe_reindex(raw["tga"], y_data.index)
        rrp_s = self._safe_reindex(raw["rrp"], y_data.index)
        ndl_s = fed_s - tga_s - (rrp_s * 1000.0)
        tips_s = self._safe_reindex(raw["tips"], y_data.index)
        yc_s = self._safe_reindex(raw["yc"], y_data.index)
        spread_s = self._safe_reindex(raw["spread"], y_data.index)

        factors = pd.DataFrame({
            "liq_now": ndl_s,
            "liq_fwd": ndl_s.pct_change(60),
            "growth_now": active_growth_series,
            "cycle_fwd": yc_s,
            "stress_now": spread_s,
            "rates_fwd": tips_s.diff(60),
        }).ffill().ewm(span=10).mean()

        # IMPORTANT: no shift(-1). These correlations use only observations known
        # by the end of the current bar; portfolio sizing applies after this point.
        corr_matrix = pd.concat([factors.tail(90), spx_ret.tail(90).rename("spx")], axis=1).corr()
        corrs = corr_matrix["spx"].drop("spx").abs().fillna(0.0).values
        weights = np.exp(corrs - np.max(corrs))
        weights = weights / max(weights.sum(), 1e-9)

        cms = (
            z_series(factors["liq_now"]).iloc[-1] * weights[0]
            + z_series(factors["liq_fwd"]).iloc[-1] * weights[1]
            + z_series(factors["growth_now"]).iloc[-1] * weights[2]
            + z_series(factors["cycle_fwd"]).iloc[-1] * weights[3]
            - z_series(factors["stress_now"]).iloc[-1] * weights[4]
            - z_series(factors["rates_fwd"]).iloc[-1] * weights[5]
        )

        vix_term = (
            float(y_data["^VIX"].iloc[-1] / y_data["^VIX3M"].iloc[-1])
            if "^VIX3M" in y_data.columns
            and pd.notna(y_data["^VIX3M"].iloc[-1])
            and y_data["^VIX3M"].iloc[-1] > 0
            else np.nan
        )

        calibrated_th = self.regime_engine.config.get("calibrated_thresholds", None)
        classified = self.regime_engine.run_time_series(prepared, custom_thresholds=calibrated_th)
        latest = classified.iloc[-1]

        regime_id = int(latest["confirmed_regime_id"])
        regime_name = str(latest["confirmed_regime_name"])
        regime_type = str(latest["regime_type"])
        regime_subtype = str(latest["regime_subtype"])
        h_days = int(latest["hysteresis_days_left"])
        conflict_note = str(latest["conflict_note"])

        if regime_id == 0:
            regime_status = "REJİMSİZ GEÇİŞ / EVENT MONITOR"
        elif h_days > 0:
            regime_status = f"Teyit Edildi — Histerezis {h_days} gün"
        else:
            regime_status = "Teyit Edildi"

        weights_live = self.regime_engine.get_portfolio_weights(regime_id, regime_subtype, row=latest)
        emergency = regime_id == 2 or float(latest.get("vix", np.nan)) > 35.0
        if emergency:
            weights_live = {
                "cash": 90.0, "gold": 0.0, "bond": 10.0,
                "equity": 0.0, "commodity": 0.0, "crypto": 0.0,
                "oil_event_score": float(weights_live.get("oil_event_score", 0.0)),
                "commodity_event_active": False,
                "commodity_event_reason": "Emergency liquidity override active.",
                "unknown_event_score": float(weights_live.get("unknown_event_score", 0.0)),
                "unknown_event_active": bool(weights_live.get("unknown_event_active", False)),
            }
        else:
            weights_live = self.regime_engine._normalize_weights(weights_live)

        ml_confidence = int(np.clip(70.0 + float(np.nan_to_num(cms)) * 12.0, 20.0, 95.0))
        pmi_val = float(raw["pmi"].iloc[-1]) if not raw["pmi"].empty else 50.0
        pmi_z = (pmi_val - 50.0) / 3.0
        yc_val = float(raw["yc"].iloc[-1]) if not raw["yc"].empty else float(latest.get("dgs10", 4.0) - latest.get("dgs2", 4.0))

        return self._make_result(
            now=now,
            health=health,
            api_status="Online" if health["status"] == "HEALTHY" else "Degraded",
            weights=weights_live,
            source="LIVE_ENGINE",
            previous_date="",
            history=history,
            extra={
                "emergency": bool(emergency),
                "cms": round(float(np.nan_to_num(cms)), 4),
                "ndl": round(float(ndl_s.iloc[-1]), 0),
                "active_growth_name": active_growth_name,
                "copper_gold": round(float(y_data["HG=F"].iloc[-1] / y_data["GC=F"].iloc[-1]), 4),
                "vix": round(float(y_data["^VIX"].iloc[-1]), 2),
                "real_rate": round(float(tips_s.iloc[-1]), 2),
                "pmi_z": round(float(pmi_z), 2),
                "yield_curve": round(float(yc_val), 2),
                "w_str": ",".join([f"{w:.4f}" for w in weights]),
                "vix_term": round(float(vix_term), 3),
                "oil_trend": round(oil_trend_display, 3),
                "oil_ret_5d_z": round(float(latest.get("oil_ret_5d_z", 0.0)), 2),
                "oil_ret_20d_z": round(float(latest.get("oil_ret_20d_z", 0.0)), 2),
                "oil_abs_5d_percentile": round(float(latest.get("oil_abs_5d_percentile", 50.0)), 1),
                "oil_event_score": round(float(weights_live.get("oil_event_score", 0.0)), 3),
                "commodity_event_active": bool(weights_live.get("commodity_event_active", False)),
                "commodity_event_reason": str(weights_live.get("commodity_event_reason", "")),
                "unknown_event_score": round(float(weights_live.get("unknown_event_score", 0.0)), 3),
                "unknown_event_active": bool(weights_live.get("unknown_event_active", False)),
                "oil_event_quality_60d": round(float(latest.get("oil_event_quality_60d", 0.5)), 3),
                "commodity_breadth_20d": round(float(latest.get("commodity_breadth_20d", 0.0)), 3),
                "ml_confidence": ml_confidence,
                "regime_id": regime_id,
                "regime_name": regime_name,
                "regime_type": regime_type,
                "regime_subtype": regime_subtype,
                "regime_status": regime_status,
                "hysteresis_days_left": h_days,
                "conflict_note": conflict_note,
                "oil_event_active": bool(weights_live.get("commodity_event_active", False)),
                "oil_ret_20d_z": round(float(latest.get("oil_ret_20d_z", 0.0)), 2),
                "freight_lvl_z": round(float(latest.get("freight_lvl_z", 0.0)), 2),
                "hy_oas_z": round(float(latest.get("hy_oas_z", 0.0)), 2),
                "spx_bond_corr": round(float(latest.get("spx_bond_corr_60d", 0.0)), 2),
                "broad_dollar_5d_z": round(float(latest.get("broad_dollar_5d_z", 0.0)), 2),
                "broad_dollar_z": round(float(latest.get("broad_dollar_z", 0.0)), 2),
                "usdjpy_1d_z": round(float(latest.get("usdjpy_1d_z", 0.0)), 2),
                "vix_z": round(float(latest.get("vix_z", 0.0)), 2),
                "vix_percentile_252": round(float(latest.get("vix_percentile_252", 50.0)), 1),
                "tips_1d_z": round(float(latest.get("tips_1d_z", 0.0)), 2),
                "t10yie_z": round(float(latest.get("t10yie_z", 0.0)), 2),
                "ndl_z": round(float(latest.get("ndl_z", 0.0)), 2),
                "data_health_status": str(health["status"]),
                "data_health_issues": " | ".join(health["issues"]),
                "data_health_warnings": " | ".join(health["warnings"]),
            },
        )

    def _make_result(self, now, health, api_status, weights, source, previous_date, history, extra=None):
        w = self.regime_engine._normalize_weights(weights)
        result = {
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "api_status": api_status,
            "decision_status": "LIVE" if source == "LIVE_ENGINE" else "HOLD_LAST_VALID",
            "allocation_source": source,
            "previous_valid_date": previous_date,
            "emergency": False,
            "cms": 0.0,
            "ndl": 0.0,
            "g3_liq": 0.0,
            "active_growth_name": "N/A",
            "copper_gold": 0.0,
            "vix": 0.0,
            "real_rate": 0.0,
            "pmi_z": 0.0,
            "yield_curve": 0.0,
            "w_str": "",
            "vix_term": np.nan,
            "oil_trend": 0.0,
            "oil_ret_5d_z": 0.0,
            "oil_ret_20d_z": 0.0,
            "oil_abs_5d_percentile": 0.0,
            "oil_event_score": float(weights.get("oil_event_score", 0.0)),
            "commodity_event_active": bool(weights.get("commodity_event_active", False)),
            "commodity_event_reason": str(weights.get("commodity_event_reason", "")),
            "unknown_event_score": float(weights.get("unknown_event_score", 0.0)),
            "unknown_event_active": bool(weights.get("unknown_event_active", False)),
            "oil_event_quality_60d": 0.5,
            "commodity_breadth_20d": 0.0,
            "ml_confidence": 0,
            "regime_id": 0,
            "regime_name": "DATA_HALT" if source != "LIVE_ENGINE" else "REJIMSIZ_GECIS",
            "regime_type": "DATA_FAILURE" if source != "LIVE_ENGINE" else "TRANSITION",
            "regime_subtype": "Last valid allocation retained." if source != "LIVE_ENGINE" else "Dengeli / Nötr Piyasa",
            "regime_status": "DATA HALT / LAST VALID ALLOCATION" if source != "LIVE_ENGINE" else "REJİMSİZ GEÇİŞ / EVENT MONITOR",
            "hysteresis_days_left": 0,
            "conflict_note": "; ".join(health.get("issues", [])),
            "freight_lvl_z": 0.0,
            "hy_oas_z": 0.0,
            "spx_bond_corr": 0.0,
            "broad_dollar_5d_z": 0.0,
            "broad_dollar_z": 0.0,
            "usdjpy_1d_z": 0.0,
            "vix_z": 0.0,
            "vix_percentile_252": 50.0,
            "tips_1d_z": 0.0,
            "t10yie_z": 0.0,
            "ndl_z": 0.0,
            "data_health_status": str(health.get("status", "HALT")),
            "data_health_issues": " | ".join(health.get("issues", [])),
            "data_health_warnings": " | ".join(health.get("warnings", [])),
            "cash_weight": round(w["cash"], 4),
            "gold_weight": round(w["gold"], 4),
            "bond_weight": round(w["bond"], 4),
            "eq_weight": round(w["equity"], 4),
            "commodity_weight": round(w["commodity"], 4),
            "crypto_weight": round(w["crypto"], 4),
        }
        if extra:
            result.update(extra)
        result["allocation_total"] = round(sum(result[k] for k in ["cash_weight", "gold_weight", "bond_weight", "eq_weight", "commodity_weight", "crypto_weight"]), 4)
        return result


def append_history(result: Dict[str, object]):
    new = pd.DataFrame([result])
    if os.path.exists(HISTORY_FILE):
        try:
            old = pd.read_csv(HISTORY_FILE)
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    for col in new.columns:
        if col not in old.columns:
            old[col] = np.nan
    for col in old.columns:
        if col not in new.columns:
            new[col] = np.nan

    combined = pd.concat([old, new[old.columns]], ignore_index=True)
    # Avoid duplicate run records when GitHub Actions is manually retried.
    if "date" in combined.columns:
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined.to_csv(HISTORY_FILE, index=False)


if __name__ == "__main__":
    engine = UltimateSentinelEngine(FRED_API_KEY)
    result = engine.run()
    append_history(result)
    print(json.dumps({
        "status": result["decision_status"],
        "data_health": result["data_health_status"],
        "regime": result["regime_name"],
        "allocation_total": result["allocation_total"],
        "oil_event_score": result.get("oil_event_score", 0.0),
        "commodity_weight": result["commodity_weight"],
    }, ensure_ascii=False))
