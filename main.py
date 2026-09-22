"""
Macro Sentinel V3.2 — point-in-time autonomous macro engine with persistent risk-appetite state.

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
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

from regime_engine import MacroRegimeEngine
from adaptive_strategy_layer import AdaptiveStrategyLayer
from risk_appetite_engine import RiskAppetiteEngine

FRED_API_KEY = os.getenv("FRED_API_KEY")
EIA_API_KEY = os.getenv("EIA_API_KEY")
HISTORY_FILE = "cms_history.csv"
IST_TZ = pytz.timezone("Europe/Istanbul")


class UltimateSentinelEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.regime_engine = MacroRegimeEngine()
        self.strategy_layer = AdaptiveStrategyLayer(self.regime_engine.config)
        self.risk_appetite_engine = RiskAppetiteEngine(self.regime_engine.config)
        data_cfg = self.regime_engine.config.get("data_quality", {})
        self.min_history = int(data_cfg.get("minimum_history_days", 126))
        self.market_stale_days = int(data_cfg.get("market_max_stale_days", 3))
        self.fred_stale_days = int(data_cfg.get("fred_max_stale_days", 7))
        self.eia_stale_days = int(data_cfg.get("eia_max_stale_days", 14))

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
            return self.fetch_fred_public(series_id, limit=max(limit, 500))
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
            series, meta = self.fetch_fred_public(series_id, limit=max(limit, 500))
            if meta.get("ok"):
                meta["primary_error"] = str(exc)[:180]
                return series, meta
            return pd.Series(dtype=float), {"ok": False, "reason": str(exc)[:180], "last_date": None, "count": 0, "source": "FRED_API"}

    def fetch_fred_public(self, series_id: str, limit: int = 1000) -> Tuple[pd.Series, Dict[str, object]]:
        """Official FRED CSV endpoint fallback that does not require an API key."""
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            response = requests.get(url, timeout=20, headers={"User-Agent": "Macro-Sentinel/2.3"})
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
            if raw.empty or raw.shape[1] < 2:
                return pd.Series(dtype=float), {"ok": False, "reason": "empty FRED CSV", "last_date": None, "count": 0, "source": "FRED_GRAPH_CSV"}
            date_col = raw.columns[0]
            value_col = series_id if series_id in raw.columns else raw.columns[1]
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
            raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
            raw = raw.dropna(subset=[date_col, value_col]).sort_values(date_col).tail(limit)
            series = raw.set_index(date_col)[value_col]
            last_date = series.index[-1].date().isoformat() if not series.empty else None
            return series, {"ok": not series.empty, "reason": "ok" if not series.empty else "empty", "last_date": last_date, "count": int(series.shape[0]), "source": "FRED_GRAPH_CSV"}
        except Exception as exc:
            return pd.Series(dtype=float), {"ok": False, "reason": str(exc)[:180], "last_date": None, "count": 0, "source": "FRED_GRAPH_CSV"}

    def fetch_eia_series(self, series_id: str, length: int = 260) -> Tuple[pd.Series, Dict[str, object]]:
        """Optional direct EIA v2 fallback for weekly petroleum supply data."""
        if not EIA_API_KEY:
            return pd.Series(dtype=float), {"ok": False, "reason": "EIA_API_KEY missing", "last_date": None, "count": 0}
        try:
            url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
            params = {
                "api_key": EIA_API_KEY,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": series_id,
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": length,
            }
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("response", {}).get("data", [])
            if not data:
                return pd.Series(dtype=float), {"ok": False, "reason": "no observations", "last_date": None, "count": 0}
            obs = pd.DataFrame(data)
            if "period" not in obs.columns or "value" not in obs.columns:
                return pd.Series(dtype=float), {"ok": False, "reason": "unexpected EIA schema", "last_date": None, "count": 0}
            obs["date"] = pd.to_datetime(obs["period"], errors="coerce")
            obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
            obs = obs.dropna(subset=["date", "value"]).sort_values("date")
            series = obs.set_index("date")["value"]
            last_date = series.index[-1].date().isoformat() if not series.empty else None
            return series, {"ok": not series.empty, "reason": "ok" if not series.empty else "empty", "last_date": last_date, "count": int(series.shape[0])}
        except Exception as exc:
            return pd.Series(dtype=float), {"ok": False, "reason": str(exc)[:180], "last_date": None, "count": 0}

    @staticmethod
    def _merge_fallback_column(frame: pd.DataFrame, column: str, fallback: pd.Series, fallback_name: str, sources: Dict[str, str]) -> None:
        """Fill missing primary observations from fallback without future extrapolation."""
        if fallback is None or fallback.empty:
            return
        if column not in frame.columns:
            frame[column] = np.nan
        primary = pd.to_numeric(frame[column], errors="coerce")
        fallback = pd.to_numeric(fallback, errors="coerce").dropna().sort_index()
        if fallback.empty:
            return
        aligned = fallback.reindex(frame.index, method="ffill")
        # Never propagate a fallback beyond the fallback source's own last observation.
        aligned.loc[aligned.index > fallback.index[-1]] = np.nan
        fill_mask = primary.isna() & aligned.notna()
        if fill_mask.any():
            frame.loc[fill_mask, column] = aligned.loc[fill_mask]
        existing_source = sources.get(column)
        if frame[column].notna().any():
            if existing_source and fill_mask.any():
                sources[column] = f"{existing_source}+{fallback_name}"
            elif not existing_source:
                sources[column] = fallback_name

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
        state = {
            "cash": float(row["cash_weight"]),
            "gold": float(row["gold_weight"]),
            "bond": float(row["bond_weight"]),
            "equity": float(row["eq_weight"]),
            "commodity": float(row["commodity_weight"]),
            "crypto": float(row["crypto_weight"]),
            "date": str(row.get("date", "")),
        }
        # Preserve the last trusted state for HOLD_LAST_VALID / market-closed
        # displays instead of replacing it with a synthetic blank regime.
        carry_fields = [
            "regime_id", "regime_name", "regime_type", "regime_subtype",
            "regime_status", "regime_severity_source", "hysteresis_days_left",
            "conflict_note", "oil_event_score", "oil_momentum_score",
            "oil_structural_score", "oil_event_type", "oil_event_active",
            "commodity_event_active", "commodity_event_reason",
            "unknown_event_score", "unknown_event_active", "unknown_guard_active"
        ]
        for field in carry_fields:
            if field in row.index and pd.notna(row.get(field)):
                state[field] = row.get(field)
        return state

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

    def _data_health(self, fred_meta: Dict[str, Dict[str, object]], y_data: pd.DataFrame, today_index, source_map: Optional[Dict[str, str]] = None, eia_meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        source_map = source_map or {}
        eia_meta = eia_meta or {}
        core_yahoo = ["ES=F", "CL=F", "GC=F", "^VIX"]
        secondary_yahoo = ["HG=F", "SI=F", "USDJPY=X", "EURUSD=X", "SOXX", "BTC-USD", "BDRY", "^VIX3M"]
        energy_complements = ["BZ=F", "HO=F", "RB=F", "NG=F"]
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

        energy_ok = sum(1 for ticker in energy_complements if ticker in y_data.columns and y_data[ticker].notna().sum() >= 30 and pd.notna(y_data[ticker].iloc[-1]))
        if energy_ok == 0:
            warnings.append("no secondary energy-complex series available; structural oil score uses WTI/Brent only")
        elif energy_ok < 2:
            warnings.append(f"limited energy-complex coverage: {energy_ok}/{len(energy_complements)} secondary series")

        source_notes = [f"{k}={v}" for k, v in sorted(source_map.items())]
        for key in core_fred:
            meta = fred_meta.get(key, {})
            if meta.get("ok"):
                source_notes.append(f"FRED:{key}={meta.get('source', 'FRED_API')}")

        for key in core_fred:
            meta = fred_meta.get(key, {})
            if not meta.get("ok"):
                issues.append(f"FRED unavailable: {key}")
            elif not self._freshness(meta.get("last_date"), self.fred_stale_days):
                issues.append(f"stale FRED series: {key}")
            elif int(meta.get("count", 0)) < 30:
                warnings.append(f"short FRED history: {key}")

        # Build an explicit sensor freshness matrix. This is informational for
        # secondary sensors and blocking only for the configured core sensors.
        freshness = {}
        for ticker in core_yahoo:
            if ticker in y_data.columns:
                series = y_data[ticker].dropna()
                last = series.index[-1] if not series.empty else None
                age = (today_index.date() - pd.Timestamp(last).date()).days if last is not None else np.inf
                freshness[ticker] = {"age_days": float(age), "ok": bool(age <= self.market_stale_days)}
            else:
                freshness[ticker] = {"age_days": np.inf, "ok": False}
        for ticker in secondary_yahoo + energy_complements:
            if ticker in y_data.columns:
                series = y_data[ticker].dropna()
                last = series.index[-1] if not series.empty else None
                age = (today_index.date() - pd.Timestamp(last).date()).days if last is not None else np.inf
                freshness[ticker] = {"age_days": float(age), "ok": bool(age <= max(self.market_stale_days, 7))}

        fred_freshness = {}
        for key in core_fred:
            meta = fred_meta.get(key, {})
            ok = bool(meta.get("ok")) and self._freshness(meta.get("last_date"), self.fred_stale_days)
            fred_freshness[f"FRED:{key}"] = {"age_days": None, "ok": ok, "last_date": meta.get("last_date")}

        fresh_ok = sum(1 for item in freshness.values() if item.get("ok")) + sum(1 for item in fred_freshness.values() if item.get("ok"))
        fresh_total = len(freshness) + len(fred_freshness)
        freshness_summary = " | ".join(
            f"{k}={v.get('last_date') if v.get('last_date') else (f"{v.get('age_days'):.1f}d" if np.isfinite(v.get('age_days', np.inf)) else 'missing')}"
            for k, v in list(freshness.items()) + list(fred_freshness.items())
        )

        # Core data and optional event-data coverage are reported separately.
        if available_core < len(core_yahoo) or issues:
            status = "HALT"
        else:
            status = "HEALTHY"

        if not eia_meta.get("ok"):
            warnings.append(f"EIA inventory unavailable: {eia_meta.get('reason', 'unknown')}")
        elif not self._freshness(eia_meta.get("last_date"), self.eia_stale_days):
            warnings.append(f"stale EIA inventory series: {eia_meta.get('last_date')}")

        event_coverage = "FULL" if not warnings else "PARTIAL"
        return {
            "status": status,
            "event_coverage_status": event_coverage,
            "core_market_available": available_core,
            "core_market_required": len(core_yahoo),
            "sensor_fresh_ok": int(fresh_ok),
            "sensor_fresh_total": int(fresh_total),
            "sensor_freshness": freshness,
            "fred_freshness": fred_freshness,
            "sensor_freshness_summary": freshness_summary,
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "source_notes": list(dict.fromkeys(source_notes)),
        }

    def _build_macro_frame(self, y_data: pd.DataFrame, raw: Dict[str, pd.Series]) -> pd.DataFrame:
        index = y_data.index
        p = pd.DataFrame(index=index)
        p["oil"] = y_data["CL=F"]
        p["brent"] = y_data.get("BZ=F", pd.Series(index=index, dtype=float))
        p["heating_oil"] = y_data.get("HO=F", pd.Series(index=index, dtype=float))
        p["gasoline"] = y_data.get("RB=F", pd.Series(index=index, dtype=float))
        p["natgas"] = y_data.get("NG=F", pd.Series(index=index, dtype=float))
        p["crude_stocks"] = self._safe_reindex(raw.get("eia_crude_stocks", pd.Series(dtype=float)), index)
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

        # V3.2 runs daily so 24/7 assets such as crypto and risk-appetite proxies
        # can refresh on weekends too. Closed-market allocation is still held.
        weekend_monitor = bool(
            now.weekday() >= 5
            and os.getenv("ALLOW_WEEKEND_RUN", "0").lower() not in {"1", "true", "yes"}
        )
        weekend_prev = self._valid_previous_allocation(history) if weekend_monitor else None

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
            "vix3m": "VXVCLS",
            "wti_spot": "DCOILWTICO",
            "brent_spot": "DCOILBRENTEU",
            "spx_proxy": "SP500",
            "gold_proxy": "GOLDAMGBD228NLBM",
        }
        raw = {}
        fred_meta = {}
        for key, sid in fred_ids.items():
            raw[key], fred_meta[key] = self.fetch_fred(sid)

        eia_crude_stocks, eia_meta = self.fetch_eia_series("WCESTUS1", length=260)
        raw["eia_crude_stocks"] = eia_crude_stocks
        fred_meta["eia_crude_stocks"] = eia_meta

        tickers = [
            "HG=F", "SI=F", "GC=F", "ES=F", "EURUSD=X", "USDJPY=X", "JPYUSD=X",
            "^VIX", "^VIX3M", "SOXX", "CL=F", "BZ=F", "HO=F", "RB=F", "NG=F", "TLT", "BTC-USD", "BDRY"
        ]
        y_data = pd.DataFrame()
        yahoo_error = ""
        try:
            downloaded = yf.download(
                tickers=tickers,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            y_data = self._extract_close(downloaded, tickers).ffill(limit=5)
        except Exception as exc:
            yahoo_error = str(exc)[:180]

        source_map: Dict[str, str] = {}
        for ticker in tickers:
            if ticker in y_data.columns and y_data[ticker].notna().any():
                source_map[ticker] = f"Yahoo:{ticker}"

        # If Yahoo is completely unavailable, construct a frame from official FRED
        # series first. This prevents the system from halting merely because the
        # market-data vendor is unavailable. Futures/index proxies are explicitly
        # labeled as FRED fallbacks in provenance.
        if y_data.empty:
            fallback_series = [
                raw.get("wti_spot"), raw.get("brent_spot"), raw.get("vix"), raw.get("vix3m"),
                raw.get("spx_proxy"), raw.get("gold_proxy"),
            ]
            indexes = [s.dropna().index for s in fallback_series if s is not None and not s.empty]
            if indexes:
                union = indexes[0]
                for idx in indexes[1:]:
                    union = union.union(idx)
                y_data = pd.DataFrame(index=union.sort_values())

        # Fill only missing Yahoo observations from official FRED fallbacks. No
        # fallback is allowed to extrapolate beyond its own final observation.
        self._merge_fallback_column(y_data, "^VIX3M", raw.get("vix3m"), "FRED:CBOE_VXVCLS", source_map)
        self._merge_fallback_column(y_data, "CL=F", raw.get("wti_spot"), "FRED:EIA_WTI_SPOT", source_map)
        self._merge_fallback_column(y_data, "BZ=F", raw.get("brent_spot"), "FRED:EIA_BRENT_SPOT", source_map)
        self._merge_fallback_column(y_data, "^VIX", raw.get("vix"), "FRED:CBOE_VIXCLS", source_map)
        self._merge_fallback_column(y_data, "ES=F", raw.get("spx_proxy"), "FRED:SP500_PROXY", source_map)
        self._merge_fallback_column(y_data, "GC=F", raw.get("gold_proxy"), "FRED:GOLD_PROXY", source_map)

        if y_data.empty:
            health = {"status": "HALT", "issues": [f"Yahoo unavailable: {yahoo_error or 'empty response'}"], "warnings": [], "core_market_available": 0, "core_market_required": 4, "source_notes": []}
        else:
            health = self._data_health(fred_meta, y_data, now, source_map=source_map, eia_meta=eia_meta)
            if yahoo_error:
                health["warnings"].append(f"Yahoo warning: {yahoo_error}")

        prev = self._valid_previous_allocation(history)

        # Fail-closed: do not calculate a new portfolio from guessed prices.
        if health["status"] == "HALT":
            if prev:
                weights = {k: prev[k] for k in ("cash", "gold", "bond", "equity", "commodity", "crypto")}
                allocation_source = "LAST_VALID"
                previous_date = prev.get("date", "")
                carry = {k: v for k, v in prev.items() if k not in {"cash", "gold", "bond", "equity", "commodity", "crypto", "date"}}
                carry["decision_note"] = "Critical data failure: last trusted state retained; no new allocation created."
            else:
                carry = {}
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
                extra=carry,
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

        # V3.2 structural risk-appetite state. It combines short and long
        # horizons and requires persistence for a long-lived state.
        risk_appetite = self.risk_appetite_engine.evaluate(prepared, history=history)
        ra_mult, ra_cash_add, ra_reason = self.risk_appetite_engine.allocation_adjustment(risk_appetite)
        risk_appetite["risk_appetite_risk_budget_multiplier"] = float(ra_mult)
        risk_appetite["risk_appetite_cash_floor_add"] = float(ra_cash_add)
        risk_appetite["risk_appetite_allocation_reason"] = str(ra_reason)
        risk_appetite["risk_appetite_asset_tilts"] = self.risk_appetite_engine.asset_tilts_for(risk_appetite)

        # Production allocation is owned by the V3.2 adaptive cross-asset layer.
        # Macro regime/event detection remains separate from the structural
        # risk-appetite state and the empirical predictive validation layer.
        strategy_result = self.strategy_layer.allocate(
            prepared_df=prepared,
            classified_df=classified,
            history=history,
            risk_appetite=risk_appetite,
        )
        weights_live = dict(strategy_result["weights"])

        # Canonical event snapshot is computed from the same point-in-time row.
        # Event metadata is never used as the allocation source of truth; the
        # adaptive strategy decides the risk budget and whether an overlay is
        # actually permitted.
        event_snapshot = self.regime_engine.get_event_snapshot(latest)
        emergency = bool(regime_id == 2 and not weekend_monitor)
        if emergency:
            # Preserve the existing macro hard-liquidity emergency contract.
            weights_live = {
                "cash": 95.0, "gold": 0.0, "bond": 5.0,
                "equity": 0.0, "commodity": 0.0, "crypto": 0.0,
            }
        elif weekend_monitor and weekend_prev:
            # Weekend monitor: refresh sensors/risk state but never create a new
            # allocation from Friday-closed markets.
            weights_live = {
                "cash": weekend_prev["cash"],
                "gold": weekend_prev["gold"],
                "bond": weekend_prev["bond"],
                "equity": weekend_prev["equity"],
                "commodity": weekend_prev["commodity"],
                "crypto": weekend_prev["crypto"],
            }

        weights_live = self.regime_engine._normalize_weights(weights_live)

        # Re-apply canonical event metadata after allocation. Strategy metadata
        # is persisted separately so pressure, confirmed event, and allocation
        # overlay cannot be conflated.
        weights_live.update({
            "oil_pressure_score": float(event_snapshot.get("oil_pressure_score", 0.0)),
            "oil_event_score": float(event_snapshot.get("oil_event_score", 0.0)),
            "oil_momentum_score": float(event_snapshot.get("oil_momentum_score", 0.0)),
            "oil_structural_score": float(event_snapshot.get("oil_structural_score", 0.0)),
            "oil_structural_qualified": bool(event_snapshot.get("oil_structural_qualified", False)),
            "oil_momentum_confirmed": bool(event_snapshot.get("oil_momentum_confirmed", False)),
            "oil_event_type": str(event_snapshot.get("oil_event_type", "NONE")),
            "oil_event_active": bool(event_snapshot.get("oil_event_active", False)),
            "oil_qualification_reason": str(event_snapshot.get("oil_qualification_reason", "")),
            "commodity_event_active": bool(strategy_result.get("oil_allocation_overlay", False)) and not weekend_monitor,
            "commodity_event_reason": (
                "Weekend monitor: current event observed; allocation held."
                if weekend_monitor and bool(strategy_result.get("oil_allocation_overlay", False))
                else str(strategy_result.get("oil_allocation_reason", "No confirmed oil allocation overlay."))
            ),
            "unknown_event_score": float(event_snapshot.get("unknown_event_score", 0.0)),
            "unknown_event_active": bool(event_snapshot.get("unknown_event_active", False)),
            "unknown_guard_active": bool(strategy_result.get("unknown_guard_active", False)),
            "risk_appetite_score": float(risk_appetite.get("risk_appetite_score", 50.0)),
            "risk_appetite_state": str(risk_appetite.get("risk_appetite_state", "NEUTRAL")),
            "risk_appetite_confidence": float(risk_appetite.get("risk_appetite_confidence", 0.0)),
            "risk_appetite_persistence_20d": float(risk_appetite.get("risk_appetite_persistence_20d", 0.0)),
            "risk_appetite_major_event_score": float(risk_appetite.get("risk_appetite_major_event_score", 0.0)),
            "risk_appetite_major_event_active": bool(risk_appetite.get("risk_appetite_major_event_active", False)),
            "risk_appetite_major_event_type": str(risk_appetite.get("risk_appetite_major_event_type", "NONE")),
            "risk_appetite_tightening_score": float(risk_appetite.get("risk_appetite_tightening_score", 0.5)),
            "risk_appetite_synchronized_stress": float(risk_appetite.get("risk_appetite_synchronized_stress", 0.0)),
        })

        ml_confidence = int(np.clip(70.0 + float(np.nan_to_num(cms)) * 12.0, 20.0, 95.0))
        pmi_val = float(raw["pmi"].iloc[-1]) if not raw["pmi"].empty else 50.0
        pmi_z = (pmi_val - 50.0) / 3.0
        yc_val = float(raw["yc"].iloc[-1]) if not raw["yc"].empty else float(latest.get("dgs10", 4.0) - latest.get("dgs2", 4.0))

        result = self._make_result(
            now=now,
            health=health,
            api_status="Online" if health["status"] == "HEALTHY" else "Degraded",
            weights=weights_live,
            source="MARKET_CLOSED_HOLD" if weekend_monitor else "LIVE_ENGINE",
            previous_date=(weekend_prev.get("date", "") if weekend_monitor and weekend_prev else ""),
            history=history,
            extra={
                "emergency": bool(emergency),
                "decision_note": (
                    "Weekend monitor: sensors and risk appetite refreshed; allocation retained from last trusted weekday state."
                    if weekend_monitor else ""
                ),
                "risk_appetite_score": round(float(risk_appetite.get("risk_appetite_score", 50.0)), 4),
                "risk_appetite_state": str(risk_appetite.get("risk_appetite_state", "NEUTRAL")),
                "risk_appetite_confidence": round(float(risk_appetite.get("risk_appetite_confidence", 0.0)), 4),
                "risk_appetite_persistence_20d": round(float(risk_appetite.get("risk_appetite_persistence_20d", 0.0)), 4),
                "risk_appetite_on_persistence_20d": round(float(risk_appetite.get("risk_appetite_on_persistence_20d", 0.0)), 4),
                "risk_appetite_off_persistence_20d": round(float(risk_appetite.get("risk_appetite_off_persistence_20d", 0.0)), 4),
                "risk_appetite_on_persistence_60d": round(float(risk_appetite.get("risk_appetite_on_persistence_60d", 0.0)), 4),
                "risk_appetite_off_persistence_60d": round(float(risk_appetite.get("risk_appetite_off_persistence_60d", 0.0)), 4),
                "risk_appetite_risk_on_evidence": round(float(risk_appetite.get("risk_appetite_risk_on_evidence", 0.0)), 4),
                "risk_appetite_risk_off_evidence": round(float(risk_appetite.get("risk_appetite_risk_off_evidence", 0.0)), 4),
                "risk_appetite_multi_horizon_score": round(float(risk_appetite.get("risk_appetite_multi_horizon_score", 50.0)), 4),
                "risk_appetite_shift_20d_z": round(float(risk_appetite.get("risk_appetite_shift_20d_z", 0.0)), 4),
                "risk_appetite_shift_60d_z": round(float(risk_appetite.get("risk_appetite_shift_60d_z", 0.0)), 4),
                "risk_appetite_major_event_score": round(float(risk_appetite.get("risk_appetite_major_event_score", 0.0)), 4),
                "risk_appetite_major_event_active": bool(risk_appetite.get("risk_appetite_major_event_active", False)),
                "risk_appetite_major_event_type": str(risk_appetite.get("risk_appetite_major_event_type", "NONE")),
                "risk_appetite_tightening_score": round(float(risk_appetite.get("risk_appetite_tightening_score", 0.5)), 4),
                "risk_appetite_liquidity_score": round(float(risk_appetite.get("risk_appetite_liquidity_score", 0.5)), 4),
                "risk_appetite_credit_score": round(float(risk_appetite.get("risk_appetite_credit_score", 0.5)), 4),
                "risk_appetite_volatility_score": round(float(risk_appetite.get("risk_appetite_volatility_score", 0.5)), 4),
                "risk_appetite_growth_risk_score": round(float(risk_appetite.get("risk_appetite_growth_risk_score", 0.5)), 4),
                "risk_appetite_synchronized_stress": round(float(risk_appetite.get("risk_appetite_synchronized_stress", 0.0)), 4),
                "risk_appetite_risk_budget_multiplier": round(float(risk_appetite.get("risk_appetite_risk_budget_multiplier", 1.0)), 4),
                "risk_appetite_cash_floor_add": round(float(risk_appetite.get("risk_appetite_cash_floor_add", 0.0)), 4),
                "risk_appetite_allocation_reason": str(risk_appetite.get("risk_appetite_allocation_reason", "")),
                "risk_appetite_available_factor_count": int(risk_appetite.get("risk_appetite_available_factor_count", 0)),
                "risk_appetite_factor_count": int(risk_appetite.get("risk_appetite_factor_count", 0)),
                "risk_appetite_source": str(risk_appetite.get("risk_appetite_source", "UNAVAILABLE")),
                "strategy_mode": str(strategy_result.get("strategy_mode", "ADAPTIVE_STRATEGY_LAYER_V3_2")),
                "strategy_risk_budget_base": round(float(strategy_result.get("risk_budget_base", 0.0)), 4),
                "strategy_risk_budget_final": round(float(strategy_result.get("risk_budget_final", 0.0)), 4),
                "strategy_deploy_risk": round(float(strategy_result.get("risk_budget_final", 0.0)), 4),
                "strategy_vol_scale": round(float(strategy_result.get("vol_scale", 1.0)), 4),
                "strategy_opportunity_score": round(float(strategy_result.get("opportunity_score", 0.0)), 4),
                "strategy_stress_score": round(float(strategy_result.get("stress_score", 0.0)), 4),
                "strategy_stress_broad": bool(strategy_result.get("stress_broad", False)),
                "strategy_stress_hard": bool(strategy_result.get("stress_hard", False)),
                "strategy_stress_negative_breadth": round(float(strategy_result.get("stress_negative_breadth", 0.0)), 4),
                "strategy_stress_median_return_20d": round(float(strategy_result.get("stress_median_return_20d", 0.0)), 4),
                "strategy_stress_avg_corr_40d": round(float(strategy_result.get("stress_avg_corr_40d", 0.0)), 4),
                "strategy_stress_vix_percentile": round(float(strategy_result.get("stress_vix_percentile", 50.0)), 2),
                "strategy_stress_ndl_z": round(float(strategy_result.get("stress_ndl_z", 0.0)), 2),
                "strategy_recovery_ready": bool(strategy_result.get("stress_recovery_ready", True)),
                "strategy_cash_floor": round(float(strategy_result.get("cash_floor", 0.0)), 4),
                "strategy_unknown_guard": bool(strategy_result.get("unknown_guard_active", False)),
                "strategy_stress_reason": str(strategy_result.get("stress_reason", "")),
                "strategy_oil_allocation_overlay": bool(strategy_result.get("oil_allocation_overlay", False)),
                "strategy_oil_allocation_reason": str(strategy_result.get("oil_allocation_reason", "")),
                "strategy_asset_score_equity": round(float(strategy_result.get("asset_scores", {}).get("equity", 0.0)), 4),
                "strategy_asset_score_gold": round(float(strategy_result.get("asset_scores", {}).get("gold", 0.0)), 4),
                "strategy_asset_score_bond": round(float(strategy_result.get("asset_scores", {}).get("bond", 0.0)), 4),
                "strategy_asset_score_commodity": round(float(strategy_result.get("asset_scores", {}).get("commodity", 0.0)), 4),
                "strategy_asset_score_crypto": round(float(strategy_result.get("asset_scores", {}).get("crypto", 0.0)), 4),
                "strategy_available_asset_count": int(strategy_result.get("available_asset_count", 0)),
                "strategy_decision_reason": str(strategy_result.get("decision_reason", "")),
                "cms": round(float(np.nan_to_num(cms)), 4),
                "ndl": round(float(ndl_s.iloc[-1]), 0),
                "active_growth_name": active_growth_name,
                "copper_gold": round(float(y_data["HG=F"].iloc[-1] / y_data["GC=F"].iloc[-1]), 4),
                "vix": round(float(y_data["^VIX"].iloc[-1]), 2),
                "oil_price": round(float(y_data["CL=F"].iloc[-1]), 4),
                "brent_price": round(float(y_data["BZ=F"].iloc[-1]), 4) if pd.notna(y_data["BZ=F"].iloc[-1]) else np.nan,
                "spx_price": round(float(y_data["ES=F"].iloc[-1]), 4),
                "gold_price": round(float(y_data["GC=F"].iloc[-1]), 4),
                "copper_price": round(float(y_data["HG=F"].iloc[-1]), 4) if pd.notna(y_data["HG=F"].iloc[-1]) else np.nan,
                "silver_price": round(float(y_data["SI=F"].iloc[-1]), 4) if pd.notna(y_data["SI=F"].iloc[-1]) else np.nan,
                "btc_price": round(float(y_data["BTC-USD"].iloc[-1]), 4) if "BTC-USD" in y_data.columns and pd.notna(y_data["BTC-USD"].iloc[-1]) else np.nan,
                "ust10y_yield": round(float(raw["dgs10"].iloc[-1]), 4) if not raw["dgs10"].empty and pd.notna(raw["dgs10"].iloc[-1]) else np.nan,
                "real_rate": round(float(tips_s.iloc[-1]), 2),
                "pmi_z": round(float(pmi_z), 2),
                "yield_curve": round(float(yc_val), 2),
                "w_str": ",".join([f"{w:.4f}" for w in weights]),
                "vix_term": round(float(vix_term), 3),
                "oil_trend": round(oil_trend_display, 3),
                "oil_ret_5d_z": round(float(latest.get("oil_ret_5d_z", 0.0)), 2),
                "oil_ret_20d_z": round(float(latest.get("oil_ret_20d_z", 0.0)), 2),
                "oil_abs_5d_percentile": round(float(latest.get("oil_abs_5d_percentile", 50.0)), 1),
                "oil_level_percentile_252": round(float(latest.get("oil_level_percentile_252", 0.0)), 1),
                "oil_level_percentile_756": round(float(latest.get("oil_level_percentile_756", 0.0)), 1),
                "oil_level_z_252": round(float(latest.get("oil_level_z_252", 0.0)), 2),
                "oil_level_z_756": round(float(latest.get("oil_level_z_756", 0.0)), 2),
                "oil_high_level_persistence_60d": round(float(latest.get("oil_high_level_persistence_60d", 0.0)), 3),
                "brent_level_percentile_252": round(float(latest.get("brent_level_percentile_252", 0.0)), 1),
                "brent_wti_spread": round(float(latest.get("brent_wti_spread", 0.0)), 2),
                "brent_wti_spread_percentile_252": round(float(latest.get("brent_wti_spread_percentile_252", 0.0)), 1),
                "energy_breadth_20d": round(float(latest.get("energy_breadth_20d", 0.0)), 3),
                "crude_inventory_draw_z": round(float(latest.get("crude_inventory_draw_z", 0.0)), 2),
                "oil_pressure_score": round(float(weights_live.get("oil_pressure_score", 0.0)), 3),
                "oil_event_score": round(float(weights_live.get("oil_event_score", 0.0)), 3),
                "oil_momentum_score": round(float(weights_live.get("oil_momentum_score", 0.0)), 3),
                "oil_structural_score": round(float(weights_live.get("oil_structural_score", 0.0)), 3),
                "oil_structural_qualified": bool(weights_live.get("oil_structural_qualified", False)),
                "oil_momentum_confirmed": bool(weights_live.get("oil_momentum_confirmed", False)),
                "oil_event_type": str(weights_live.get("oil_event_type", "NONE")),
                "oil_event_active": bool(weights_live.get("oil_event_active", False)),
                "oil_qualification_reason": str(event_snapshot.get("oil_qualification_reason", "")),
                "commodity_event_active": bool(weights_live.get("commodity_event_active", False)),
                "commodity_event_reason": str(weights_live.get("commodity_event_reason", "")),
                "unknown_event_score": round(float(weights_live.get("unknown_event_score", 0.0)), 3),
                "unknown_event_active": bool(weights_live.get("unknown_event_active", False)),
                "unknown_guard_active": bool(weights_live.get("unknown_guard_active", False)),
                "oil_event_quality_60d": round(float(latest.get("oil_event_quality_60d", 0.5)), 3),
                "commodity_breadth_20d": round(float(latest.get("commodity_breadth_20d", 0.0)), 3),
                "ml_confidence": ml_confidence,
                "regime_id": regime_id,
                "regime_name": regime_name,
                "regime_type": regime_type,
                "regime_subtype": regime_subtype,
                "regime_severity_source": str(latest.get("regime_severity_source", "NONE")),
                "regime_status": regime_status,
                "hysteresis_days_left": h_days,
                "conflict_note": conflict_note,
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
                "tips_level_z": round(float(latest.get("tips_level_z", 0.0)), 2),
                "tips_level_percentile_252": round(float(latest.get("tips_level_percentile_252", 50.0)), 1),
                "t10yie_z": round(float(latest.get("t10yie_z", 0.0)), 2),
                "ndl_z": round(float(latest.get("ndl_z", 0.0)), 2),
                "data_health_status": str(health["status"]),
                "event_coverage_status": str(health.get("event_coverage_status", "UNKNOWN")),
                "data_health_issues": " | ".join(health["issues"]),
                "data_health_warnings": " | ".join(health["warnings"]),
                "data_source_notes": " | ".join(health.get("source_notes", [])),
                "sensor_fresh_ok": int(health.get("sensor_fresh_ok", 0)),
                "sensor_fresh_total": int(health.get("sensor_fresh_total", 0)),
                "sensor_freshness_summary": str(health.get("sensor_freshness_summary", "")),
                "oil_price_source": source_map.get("CL=F", "UNAVAILABLE"),
                "brent_price_source": source_map.get("BZ=F", "UNAVAILABLE"),
                "vix3m_source": source_map.get("^VIX3M", "UNAVAILABLE"),
                "eia_inventory_source": "EIA_API:WCESTUS1" if not eia_crude_stocks.empty else "unavailable",
            },
        )
        audit_cfg = self.regime_engine.config.get("self_learning", {})
        audit = _daily_audit(
            history,
            result,
            min_events=int(audit_cfg.get("minimum_completed_event_observations", 10)),
        )
        result.update({
            "daily_audit_status": audit["status"],
            "audit_completed_oil_events_5d": audit["completed_events"],
            "audit_oil_positive_rate_5d": audit["positive_rate_5d"],
            "audit_completed_structural_events_5d": audit["structural_events"],
            "audit_structural_positive_rate_5d": audit["structural_positive_rate_5d"],
            "audit_completed_momentum_events_5d": audit["momentum_events"],
            "audit_momentum_positive_rate_5d": audit["momentum_positive_rate_5d"],
        })
        return result

    def _make_result(self, now, health, api_status, weights, source, previous_date, history, extra=None):
        w = self.regime_engine._normalize_weights(weights)
        result = {
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "api_status": api_status,
            "decision_status": (
                "LIVE" if source == "LIVE_ENGINE"
                else "MARKET_CLOSED_HOLD" if source == "MARKET_CLOSED_HOLD"
                else "HOLD_LAST_VALID"
            ),
            "allocation_source": source,
            "previous_valid_date": previous_date,
            "emergency": False,
            "strategy_mode": "ADAPTIVE_STRATEGY_LAYER_V3",
            "strategy_risk_budget_base": 0.0,
            "strategy_risk_budget_final": 0.0,
            "strategy_deploy_risk": 0.0,
            "strategy_vol_scale": 1.0,
            "strategy_opportunity_score": 0.0,
            "strategy_stress_score": 0.0,
            "strategy_stress_broad": False,
            "strategy_stress_hard": False,
            "strategy_stress_negative_breadth": 0.0,
            "strategy_stress_median_return_20d": 0.0,
            "strategy_stress_avg_corr_40d": 0.0,
            "strategy_stress_vix_percentile": 50.0,
            "strategy_stress_ndl_z": 0.0,
            "strategy_recovery_ready": True,
            "strategy_cash_floor": 0.0,
            "strategy_unknown_guard": False,
            "strategy_stress_reason": "",
            "strategy_oil_allocation_overlay": False,
            "strategy_oil_allocation_reason": "",
            "strategy_asset_score_equity": 0.0,
            "strategy_asset_score_gold": 0.0,
            "strategy_asset_score_bond": 0.0,
            "strategy_asset_score_commodity": 0.0,
            "strategy_asset_score_crypto": 0.0,
            "strategy_available_asset_count": 0,
            "strategy_decision_reason": "",
            "risk_appetite_score": float(weights.get("risk_appetite_score", 50.0)),
            "risk_appetite_state": str(weights.get("risk_appetite_state", "NEUTRAL")),
            "risk_appetite_confidence": float(weights.get("risk_appetite_confidence", 0.0)),
            "risk_appetite_persistence_20d": float(weights.get("risk_appetite_persistence_20d", 0.0)),
            "risk_appetite_major_event_score": float(weights.get("risk_appetite_major_event_score", 0.0)),
            "risk_appetite_major_event_active": bool(weights.get("risk_appetite_major_event_active", False)),
            "risk_appetite_major_event_type": str(weights.get("risk_appetite_major_event_type", "NONE")),
            "risk_appetite_tightening_score": float(weights.get("risk_appetite_tightening_score", 0.5)),
            "risk_appetite_synchronized_stress": float(weights.get("risk_appetite_synchronized_stress", 0.0)),
            "risk_appetite_risk_budget_multiplier": float(weights.get("risk_appetite_risk_budget_multiplier", 1.0)),
            "risk_appetite_cash_floor_add": float(weights.get("risk_appetite_cash_floor_add", 0.0)),
            "risk_appetite_allocation_reason": str(weights.get("risk_appetite_allocation_reason", "")),
            "risk_appetite_on_persistence_20d": float(weights.get("risk_appetite_on_persistence_20d", 0.0)),
            "risk_appetite_off_persistence_20d": float(weights.get("risk_appetite_off_persistence_20d", 0.0)),
            "risk_appetite_on_persistence_60d": float(weights.get("risk_appetite_on_persistence_60d", 0.0)),
            "risk_appetite_off_persistence_60d": float(weights.get("risk_appetite_off_persistence_60d", 0.0)),
            "risk_appetite_multi_horizon_score": float(weights.get("risk_appetite_multi_horizon_score", 50.0)),
            "risk_appetite_shift_20d_z": float(weights.get("risk_appetite_shift_20d_z", 0.0)),
            "risk_appetite_shift_60d_z": float(weights.get("risk_appetite_shift_60d_z", 0.0)),
            "risk_appetite_available_factor_count": int(weights.get("risk_appetite_available_factor_count", 0)),
            "risk_appetite_factor_count": int(weights.get("risk_appetite_factor_count", 10)),
            "risk_appetite_source": str(weights.get("risk_appetite_source", "UNAVAILABLE")),
            "sensor_fresh_ok": int(weights.get("sensor_fresh_ok", health.get("sensor_fresh_ok", 0))),
            "sensor_fresh_total": int(weights.get("sensor_fresh_total", health.get("sensor_fresh_total", 0))),
            "sensor_freshness_summary": str(weights.get("sensor_freshness_summary", health.get("sensor_freshness_summary", ""))),
            "cms": 0.0,
            "ndl": 0.0,
            "g3_liq": np.nan,
            "active_growth_name": "N/A",
            "oil_price": np.nan,
            "brent_price": np.nan,
            "spx_price": np.nan,
            "gold_price": np.nan,
            "copper_price": np.nan,
            "silver_price": np.nan,
            "btc_price": np.nan,
            "ust10y_yield": np.nan,
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
            "oil_level_percentile_252": 0.0,
            "oil_level_percentile_756": 0.0,
            "oil_level_z_252": 0.0,
            "oil_level_z_756": 0.0,
            "oil_high_level_persistence_60d": 0.0,
            "brent_level_percentile_252": 0.0,
            "brent_wti_spread": 0.0,
            "brent_wti_spread_percentile_252": 0.0,
            "energy_breadth_20d": 0.0,
            "crude_inventory_draw_z": 0.0,
            "oil_pressure_score": float(weights.get("oil_pressure_score", 0.0)),
            "oil_event_score": float(weights.get("oil_event_score", 0.0)),
            "oil_momentum_score": float(weights.get("oil_momentum_score", 0.0)),
            "oil_structural_score": float(weights.get("oil_structural_score", 0.0)),
            "oil_structural_qualified": bool(weights.get("oil_structural_qualified", False)),
            "oil_momentum_confirmed": bool(weights.get("oil_momentum_confirmed", False)),
            "oil_event_type": str(weights.get("oil_event_type", "NONE")),
            "oil_event_active": bool(weights.get("oil_event_active", False)),
            "oil_qualification_reason": str(weights.get("oil_qualification_reason", "")),
            "commodity_event_active": bool(weights.get("commodity_event_active", False)),
            "commodity_event_reason": str(weights.get("commodity_event_reason", "")),
            "unknown_event_score": float(weights.get("unknown_event_score", 0.0)),
            "unknown_event_active": bool(weights.get("unknown_event_active", False)),
            "unknown_guard_active": bool(weights.get("unknown_guard_active", False)),
            "strategy_mode": str(weights.get("strategy_mode", "STATIC_REGIME")),
            "oil_event_quality_60d": 0.5,
            "commodity_breadth_20d": 0.0,
            "ml_confidence": 0,
            "regime_id": 0,
            "regime_name": (
                "MARKET_CLOSED" if source == "MARKET_CLOSED_HOLD"
                else "DATA_HALT" if source != "LIVE_ENGINE"
                else "REJIMSIZ_GECIS"
            ),
            "regime_type": (
                "MARKET_CLOSED" if source == "MARKET_CLOSED_HOLD"
                else "DATA_FAILURE" if source != "LIVE_ENGINE"
                else "TRANSITION"
            ),
            "regime_subtype": (
                "Market closed; previous valid allocation retained." if source == "MARKET_CLOSED_HOLD"
                else "Last valid allocation retained." if source != "LIVE_ENGINE"
                else "Dengeli / Nötr Piyasa"
            ),
            "regime_severity_source": "NONE",
            "regime_status": (
                "MARKET CLOSED / PREVIOUS VALID ALLOCATION" if source == "MARKET_CLOSED_HOLD"
                else "DATA HALT / LAST VALID ALLOCATION" if source != "LIVE_ENGINE"
                else "REJİMSİZ GEÇİŞ / EVENT MONITOR"
            ),
            "hysteresis_days_left": 0,
            "pending_regime_id": 0,
            "pending_regime_count": 0,
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
            "tips_level_z": 0.0,
            "tips_level_percentile_252": 50.0,
            "t10yie_z": 0.0,
            "ndl_z": 0.0,
            "data_health_status": str(health.get("status", "HALT")),
            "event_coverage_status": str(health.get("event_coverage_status", "UNKNOWN")),
            "data_health_issues": " | ".join(health.get("issues", [])),
            "data_health_warnings": " | ".join(health.get("warnings", [])),
            "data_source_notes": " | ".join(health.get("source_notes", [])),
            "oil_price_source": "unavailable",
            "brent_price_source": "unavailable",
            "vix3m_source": "unavailable",
            "eia_inventory_source": "unavailable",
            "daily_audit_status": "INSUFFICIENT_HISTORY",
            "audit_completed_oil_events_5d": 0,
            "audit_oil_positive_rate_5d": np.nan,
            "audit_completed_structural_events_5d": 0,
            "audit_structural_positive_rate_5d": np.nan,
            "audit_completed_momentum_events_5d": 0,
            "audit_momentum_positive_rate_5d": np.nan,
            "predictive_status": str(weights.get("predictive_status", "WARMUP")),
            "predictive_confidence": float(weights.get("predictive_confidence", 0.0)),
            "predictive_matured_observations": int(weights.get("predictive_matured_observations", 0)),
            "predictive_risk_bin_observations": int(weights.get("predictive_risk_bin_observations", 0)),
            "predictive_opportunity_bin_observations": int(weights.get("predictive_opportunity_bin_observations", 0)),
            "predictive_risk_probability_5d": float(weights.get("predictive_risk_probability_5d", 0.0)),
            "predictive_expected_median_return_5d": float(weights.get("predictive_expected_median_return_5d", 0.0)),
            "predictive_expected_loss_5d": float(weights.get("predictive_expected_loss_5d", 0.0)),
            "predictive_opportunity_probability_20d": float(weights.get("predictive_opportunity_probability_20d", 0.0)),
            "predictive_expected_opportunity_return_20d": float(weights.get("predictive_expected_opportunity_return_20d", 0.0)),
            "predictive_deterioration_score": float(weights.get("predictive_deterioration_score", 0.0)),
            "predictive_adjustment_reason": str(weights.get("predictive_adjustment_reason", "Predictive layer not yet calibrated.")),
            "predictive_reason": str(weights.get("predictive_reason", "")),
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
        result.setdefault("daily_audit_status", "INSUFFICIENT_HISTORY")
        result.setdefault("audit_completed_oil_events_5d", 0)
        result.setdefault("audit_oil_positive_rate_5d", np.nan)
        result.setdefault("audit_completed_structural_events_5d", 0)
        result.setdefault("audit_structural_positive_rate_5d", np.nan)
        result.setdefault("audit_completed_momentum_events_5d", 0)
        result.setdefault("audit_momentum_positive_rate_5d", np.nan)
        return result



def _daily_audit(history: pd.DataFrame, current_result: Dict[str, object], min_events: int = 10) -> Dict[str, object]:
    """Post-outcome audit only; never feeds future data back into today's decision."""
    current = pd.DataFrame([current_result])
    hist = history.copy() if history is not None else pd.DataFrame()

    required = ["oil_price", "oil_event_score", "oil_momentum_score", "oil_structural_score", "oil_event_active", "oil_event_type"]
    if not hist.empty:
        hist = hist.reindex(columns=required)
        current = current.reindex(columns=required)
        combined = pd.concat([hist, current], ignore_index=True)
    else:
        combined = current.reindex(columns=required).copy()

    if not all(c in combined.columns for c in required):
        return {"status": "UNAVAILABLE", "events": 0, "completed_events": 0, "positive_rate_5d": np.nan, "structural_positive_rate_5d": np.nan, "momentum_positive_rate_5d": np.nan}

    prices = pd.to_numeric(combined["oil_price"], errors="coerce")
    fwd5 = prices.shift(-5) / prices - 1.0
    completed = fwd5.notna()

    def audit(mask):
        m = mask & completed
        n = int(m.sum())
        if n == 0:
            return 0, np.nan
        return n, float((fwd5[m] > 0).mean())

    event_mask = combined["oil_event_active"].astype(str).str.lower().isin(["1", "true", "yes", "on"])
    event_types = combined["oil_event_type"].astype(str).str.upper()
    structural_mask = event_mask & event_types.isin(["STRUCTURAL", "COMBINED"])
    momentum_mask = event_mask & event_types.isin(["MOMENTUM", "COMBINED"])
    event_n, event_rate = audit(event_mask)
    structural_n, structural_rate = audit(structural_mask)
    momentum_n, momentum_rate = audit(momentum_mask)
    status = "INSUFFICIENT_HISTORY" if event_n < min_events else "TRACKING"
    return {
        "status": status,
        "events": event_n,
        "completed_events": event_n,
        "positive_rate_5d": event_rate,
        "structural_events": structural_n,
        "structural_positive_rate_5d": structural_rate,
        "momentum_events": momentum_n,
        "momentum_positive_rate_5d": momentum_rate,
    }


def _canonicalize_result_event_state(result: Dict[str, object], activation_score: float = 0.20) -> Dict[str, object]:
    """Build one authoritative oil-event state from persisted numeric evidence.

    Event flags/types are never trusted on their own. Structural qualification is
    recomputed from the saved percentile/persistence/supply metrics so history
    cannot contain impossible states such as STRUCTURAL + qualified=False.
    """
    out = dict(result)

    def flag(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "yes", "on"}:
            return True
        if text in {"false", "no", "off", "nan", "none", ""}:
            return False
        try:
            return bool(float(text))
        except (TypeError, ValueError):
            return False

    def num(name, default=np.nan):
        try:
            value = float(out.get(name, default))
            return value if np.isfinite(value) else float(default)
        except (TypeError, ValueError):
            return float(default)

    engine = MacroRegimeEngine()
    oil_cfg = engine.config.get("event_overlay", {}).get("oil", {})
    structural_cfg = oil_cfg.get("structural", {})
    threshold = float(activation_score)

    momentum = float(np.clip(num("oil_momentum_score", 0.0), 0.0, 1.0))
    pressure = num("oil_pressure_score", np.nan)
    structural = float(np.clip(num("oil_structural_score", 0.0), 0.0, 1.0))
    if not np.isfinite(pressure):
        pressure = max(momentum, structural)
    pressure = float(np.clip(max(pressure, momentum, structural), 0.0, 1.0))

    level_pct = num("oil_level_percentile_756", np.nan)
    if not np.isfinite(level_pct):
        level_pct = num("oil_level_percentile_252", np.nan)
    persistence = num("oil_high_level_persistence_60d", np.nan)
    breadth = num("energy_breadth_20d", np.nan)
    inventory = num("crude_inventory_draw_z", np.nan)
    spread_pct = num("brent_wti_spread_percentile_252", np.nan)

    high_level = np.isfinite(level_pct) and level_pct >= float(structural_cfg.get("qualification_level_percentile", 85.0))
    persistent = np.isfinite(persistence) and persistence >= float(structural_cfg.get("qualification_persistence", 0.40))
    inventory_stress = np.isfinite(inventory) and inventory >= float(structural_cfg.get("qualification_inventory_draw_z", 1.0))
    spread_stress = np.isfinite(spread_pct) and spread_pct >= float(structural_cfg.get("qualification_spread_percentile", 85.0))
    breadth_confirmed = np.isfinite(breadth) and breadth >= float(structural_cfg.get("qualification_breadth", 0.60))

    # Keep the engine's confirmed-event policy: high level + persistence, or
    # high level + supply confirmation, or very high level + inventory stress.
    structural_qualified = bool(
        (high_level and persistent)
        or (high_level and (inventory_stress or spread_stress))
        or (np.isfinite(level_pct) and level_pct >= 90.0 and inventory_stress)
    )
    momentum_ok = momentum >= float(oil_cfg.get("momentum_activation_score", threshold))
    structural_ok = structural_qualified and structural >= float(oil_cfg.get("structural_activation_score", threshold))

    if momentum_ok and structural_ok:
        event_type, event_score = "COMBINED", max(momentum, structural)
    elif momentum_ok:
        event_type, event_score = "MOMENTUM", momentum
    elif structural_ok:
        event_type, event_score = "STRUCTURAL", structural
    elif pressure >= float(oil_cfg.get("pressure_display_threshold", 0.20)):
        event_type, event_score = "PRESSURE_ONLY", 0.0
    else:
        event_type, event_score = "NONE", 0.0

    if structural_qualified:
        reason = "Structural qualification confirmed by high price level plus persistence/supply confirmation."
    elif np.isfinite(level_pct) and level_pct >= float(oil_cfg.get("structural", {}).get("pressure_level_percentile", 75.0)):
        reason = "Structural pressure present, but confirmed-event qualification is not met."
    else:
        reason = "Structural pressure below the confirmed-event qualification zone."

    overlay_active = flag(out.get("commodity_event_active", False))
    if not (event_type in {"MOMENTUM", "STRUCTURAL", "COMBINED"}):
        overlay_active = False
        overlay_reason = "No confirmed oil event; commodity event overlay must remain disabled."
    else:
        overlay_reason = str(out.get("commodity_event_reason", "Confirmed oil event overlay state."))

    out.update({
        "oil_pressure_score": round(pressure, 6),
        "oil_event_score": round(float(np.clip(event_score, 0.0, 1.0)), 6),
        "oil_event_type": event_type,
        "oil_event_active": bool(event_type in {"MOMENTUM", "STRUCTURAL", "COMBINED"}),
        "oil_structural_qualified": bool(structural_qualified),
        "oil_momentum_confirmed": bool(momentum_ok),
        "oil_structural_high_level": bool(high_level),
        "oil_structural_persistent": bool(persistent),
        "oil_structural_inventory_stress": bool(inventory_stress),
        "oil_structural_spread_stress": bool(spread_stress),
        "oil_structural_breadth_confirmed": bool(breadth_confirmed),
        "oil_qualification_reason": reason,
        "commodity_event_active": bool(overlay_active),
        "commodity_event_reason": overlay_reason,
    })
    return out


def append_history(result: Dict[str, object]):
    """Persist a canonical row without dtype-unsafe scalar mutation."""
    engine_cfg = MacroRegimeEngine().config
    activation_score = float(
        engine_cfg.get("event_overlay", {})
        .get("oil", {})
        .get("activation_score", 0.20)
    )
    canonical_result = _canonicalize_result_event_state(result, activation_score)
    new_row = pd.DataFrame([canonical_result])

    if os.path.exists(HISTORY_FILE):
        try:
            old = pd.read_csv(HISTORY_FILE)
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if not old.empty and "date" in old.columns and "date" in new_row.columns:
        old = old.loc[old["date"].astype(str) != str(new_row.iloc[0]["date"])].copy()

    all_columns = list(dict.fromkeys(list(old.columns) + list(new_row.columns)))
    combined = pd.concat(
        [old.reindex(columns=all_columns), new_row.reindex(columns=all_columns)],
        ignore_index=True,
        sort=False,
    )

    # Rebuild the final row instead of assigning booleans/scalars into pre-typed
    # float columns. This is safe on current pandas 2.x and future strict dtypes.
    final_dict = _canonicalize_result_event_state(combined.iloc[-1].to_dict(), activation_score)
    rows = combined.iloc[:-1].to_dict(orient="records")
    rows.append(final_dict)
    combined = pd.DataFrame(rows, columns=all_columns)

    combined.to_csv(HISTORY_FILE, index=False)

    persisted = pd.read_csv(HISTORY_FILE)
    if persisted.empty:
        raise RuntimeError("HISTORY_PERSISTENCE_FAILURE: cms_history.csv is empty after write")

    persisted_last = _canonicalize_result_event_state(persisted.iloc[-1].to_dict(), activation_score)
    check_fields = (
        "oil_event_type", "oil_event_active", "oil_event_score",
        "oil_pressure_score", "oil_structural_qualified",
        "oil_momentum_confirmed", "commodity_event_active"
    )
    for field in check_fields:
        saved = persisted.iloc[-1].get(field)
        expected = persisted_last.get(field)
        if field in {"oil_event_type"}:
            ok = str(saved).upper() == str(expected).upper()
        elif field in {"oil_event_active", "oil_structural_qualified", "oil_momentum_confirmed", "commodity_event_active"}:
            ok = flag_value(saved) == flag_value(expected)
        else:
            try:
                ok = abs(float(saved) - float(expected)) <= 0.01
            except (TypeError, ValueError):
                ok = False
        if not ok:
            raise RuntimeError(
                f"HISTORY_EVENT_STATE_PERSISTENCE_FAILURE: {field} persisted={saved!r} canonical={expected!r}"
            )


def flag_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off", "nan", "none", ""}:
        return False
    try:
        return bool(float(text))
    except (TypeError, ValueError):
        return False


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
