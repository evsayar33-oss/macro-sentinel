"""
Macro Sentinel V3.2 — Multi-Horizon Risk Appetite / Tightening State Engine.

The engine does not claim to observe the private intent of people, banks, or
states. It estimates market-wide risk appetite using observable proxies:
credit, volatility, dollar, liquidity, real rates, growth-sensitive assets,
crypto, metals and cross-asset breadth.

Point-in-time contract:
    - no look-ahead data are used;
    - trailing statistics use current/past observations only;
    - long-lived states require persistence, not a one-day spike;
    - major events require multi-factor evidence or a large multi-horizon shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskAppetiteState:
    score: float
    state: str
    confidence: float
    persistence_20d: float
    persistence_60d: float
    risk_on_evidence: float
    risk_off_evidence: float
    multi_horizon_score: float
    shift_20d_z: float
    shift_60d_z: float
    major_event_score: float
    major_event_active: bool
    major_event_type: str
    tightening_score: float
    liquidity_score: float
    credit_score: float
    volatility_score: float
    growth_risk_score: float
    synchronized_stress: float
    available_factor_count: int
    factor_count: int
    reason: str


class RiskAppetiteEngine:
    """Bounded market risk-appetite state estimator.

    Scores are normalized to [0, 100]. Higher values mean broader risk-taking
    conditions; lower values mean stronger capital-preservation pressure.
    """

    DEFAULT_HORIZONS = (5, 20, 60, 120, 252, 756)
    FACTORS = (
        "equity",
        "crypto",
        "industrial_metals",
        "credit",
        "volatility",
        "dollar",
        "real_rates",
        "liquidity",
        "carry",
        "breadth",
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        root = config or {}
        self.cfg = root.get("risk_appetite", root.get("risk_appetite_v3_2", {}))
        self.enabled = bool(self.cfg.get("enabled", True))
        raw_h = self.cfg.get("horizons", list(self.DEFAULT_HORIZONS))
        self.horizons = tuple(sorted({max(1, int(x)) for x in raw_h})) or self.DEFAULT_HORIZONS
        default_w = {5: 0.08, 20: 0.20, 60: 0.27, 120: 0.22, 252: 0.15, 756: 0.08}
        configured_w = self.cfg.get("horizon_weights", {})
        weights = {h: float(configured_w.get(str(h), configured_w.get(h, default_w.get(h, 0.1)))) for h in self.horizons}
        total = sum(max(0.0, v) for v in weights.values()) or 1.0
        self.horizon_weights = {h: max(0.0, weights[h]) / total for h in self.horizons}

        self.risk_on_threshold = float(self.cfg.get("risk_on_threshold", 60.0))
        self.risk_off_threshold = float(self.cfg.get("risk_off_threshold", 40.0))
        self.persistent_threshold = float(self.cfg.get("persistent_persistence", 0.65))
        self.crisis_threshold = float(self.cfg.get("crisis_threshold", 22.0))
        self.major_event_threshold = float(self.cfg.get("major_event_threshold", 0.68))
        self.extreme_z = float(self.cfg.get("extreme_factor_z", 2.0))
        self.major_change_z = float(self.cfg.get("major_change_z", 2.0))
        self.min_factor_coverage = int(self.cfg.get("minimum_factor_coverage", 5))
        self.min_long_horizon_points = int(self.cfg.get("minimum_long_horizon_points", 120))
        self.persistence_fast = int(self.cfg.get("persistence_window_fast", 20))
        self.persistence_slow = int(self.cfg.get("persistence_window_slow", 60))
        self.max_risk_on_boost = float(self.cfg.get("max_risk_on_boost", 0.12))
        self.max_risk_off_reduction = float(self.cfg.get("max_risk_off_reduction", 0.45))

        self.asset_tilts = self.cfg.get("asset_tilts", {
            "RISK_ON_PERSISTENT": {"equity": 1.12, "crypto": 1.20, "commodity": 1.05, "gold": 0.92, "bond": 0.92},
            "RISK_ON": {"equity": 1.06, "crypto": 1.10, "commodity": 1.03, "gold": 0.97, "bond": 0.97},
            "NEUTRAL": {"equity": 1.00, "crypto": 1.00, "commodity": 1.00, "gold": 1.00, "bond": 1.00},
            "RISK_OFF": {"equity": 0.82, "crypto": 0.62, "commodity": 0.86, "gold": 1.10, "bond": 1.10},
            "RISK_OFF_PERSISTENT": {"equity": 0.68, "crypto": 0.45, "commodity": 0.78, "gold": 1.16, "bond": 1.16},
            "CRISIS": {"equity": 0.45, "crypto": 0.25, "commodity": 0.60, "gold": 1.18, "bond": 1.18},
        })

    @staticmethod
    def _num(value: Any, default: float = np.nan) -> float:
        try:
            x = float(value)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _robust_z(series: pd.Series, window: int = 252, min_periods: int = 40) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        prior = s.shift(1)
        mean = prior.rolling(window, min_periods=min_periods).mean()
        std = prior.rolling(window, min_periods=min_periods).std()
        return (s - mean) / (std.replace(0.0, np.nan) + 1e-9)

    @staticmethod
    def _trailing_percentile(series: pd.Series, window: int = 252, min_periods: int = 40) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")

        def pct(x: np.ndarray) -> float:
            if len(x) < 2 or not np.isfinite(x[-1]):
                return np.nan
            hist = x[:-1]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 2:
                return np.nan
            return float(np.mean(hist <= x[-1]) * 100.0)

        return s.rolling(window + 1, min_periods=min_periods + 1).apply(pct, raw=True)

    @staticmethod
    def _clip01(x: Any, default: float = 0.5) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        return float(np.clip(v if np.isfinite(v) else default, 0.0, 1.0))

    @staticmethod
    def _finite_mean(values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan

    def _series(self, df: pd.DataFrame, name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    def _factor_scores(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        idx = df.index
        factor = pd.DataFrame(index=idx, columns=self.FACTORS, dtype=float)
        latest_meta: Dict[str, float] = {}

        # Growth-sensitive assets: positive returns are pro-risk.
        eq = self._series(df, "spx")
        btc = self._series(df, "btc")
        copper = self._series(df, "copper")
        silver = self._series(df, "silver")
        gold = self._series(df, "gold")
        industrial = pd.concat({"copper": copper.pct_change(20), "silver": silver.pct_change(20)}, axis=1).mean(axis=1, skipna=True)

        def trend_score(s: pd.Series) -> pd.Series:
            pieces = []
            for h in (5, 20, 60, 120):
                r = s.pct_change(h)
                vol = s.pct_change().rolling(min(60, max(10, h)), min_periods=min(20, max(5, h // 2))).std()
                pieces.append(np.tanh(r / (vol * np.sqrt(max(h, 1)) + 1e-9)))
            return pd.concat(pieces, axis=1).mean(axis=1, skipna=True)

        factor["equity"] = trend_score(eq)
        factor["crypto"] = trend_score(btc)
        factor["industrial_metals"] = np.tanh(industrial / 0.12)

        hy = self._series(df, "hy_oas")
        hy_z = self._robust_z(hy, 252, 40)
        factor["credit"] = -np.tanh(hy_z / 2.0)

        vix = self._series(df, "vix")
        vix_z = self._robust_z(vix, 252, 40)
        factor["volatility"] = -np.tanh(vix_z / 2.0)

        dxy = self._series(df, "broad_dollar")
        dxy_z = self._robust_z(dxy, 252, 40)
        factor["dollar"] = -np.tanh(dxy_z / 2.0)

        tips = self._series(df, "tips10y")
        tips_z = self._robust_z(tips, 252, 40)
        factor["real_rates"] = -np.tanh(tips_z / 2.0)

        ndl = self._series(df, "ndl")
        ndl_z = self._robust_z(ndl, 252, 40)
        factor["liquidity"] = np.tanh(ndl_z / 2.0)

        usdjpy = self._series(df, "usdjpy")
        carry_r = usdjpy.pct_change(20)
        carry_vol = usdjpy.pct_change().rolling(60, min_periods=20).std()
        factor["carry"] = np.tanh(carry_r / (carry_vol * np.sqrt(20.0) + 1e-9))

        breadth = pd.concat({
            "equity": eq.pct_change(20) > 0,
            "crypto": btc.pct_change(20) > 0,
            "copper": copper.pct_change(20) > 0,
            "silver": silver.pct_change(20) > 0,
        }, axis=1)
        factor["breadth"] = breadth.astype(float).mean(axis=1, skipna=True) * 2.0 - 1.0

        for col in factor.columns:
            latest = factor[col].dropna()
            latest_meta[col] = float(latest.iloc[-1]) if not latest.empty else np.nan
        return factor, latest_meta

    def _multi_horizon(self, factor: pd.DataFrame) -> pd.Series:
        pieces = []
        for h in self.horizons:
            rolled = factor.rolling(h, min_periods=max(5, min(h, h // 2))).mean()
            score = rolled.mean(axis=1, skipna=True)
            pieces.append(score.rename(str(h)))
        panel = pd.concat(pieces, axis=1)
        out = pd.Series(np.nan, index=factor.index, dtype=float)
        for h, weight in self.horizon_weights.items():
            s = panel[str(h)]
            if out.isna().all():
                out = s * weight
            else:
                out = out.fillna(0.0) + s.fillna(0.0) * weight
        weights_available = pd.DataFrame({str(h): panel[str(h)].notna().astype(float) * w for h, w in self.horizon_weights.items()}, index=factor.index).sum(axis=1)
        return out / weights_available.replace(0.0, np.nan)

    def _shift_z(self, score: pd.Series, horizon: int) -> pd.Series:
        change = score - score.shift(horizon)
        prior = change.shift(1)
        mean = prior.rolling(252, min_periods=40).mean()
        std = prior.rolling(252, min_periods=40).std()
        return (change - mean) / (std.replace(0.0, np.nan) + 1e-9)

    def _major_event(self, df: pd.DataFrame, factor: pd.DataFrame, appetite_raw: pd.Series) -> Tuple[float, bool, str]:
        """Detect unusually large multi-horizon shifts without look-ahead.

        ``factor`` contains bounded appetite scores in [-1, 1], so it cannot be
        compared directly with a Z-score threshold. Recompute a point-in-time
        shock panel from the underlying observables and use that for the extreme
        factor breadth component.
        """
        shift20 = self._shift_z(appetite_raw, 20)
        shift60 = self._shift_z(appetite_raw, 60)
        latest20 = self._num(shift20.iloc[-1], 0.0)
        latest60 = self._num(shift60.iloc[-1], 0.0)

        eq = self._series(df, "spx")
        btc = self._series(df, "btc")
        copper = self._series(df, "copper")
        silver = self._series(df, "silver")
        hy = self._series(df, "hy_oas")
        vix = self._series(df, "vix")
        dxy = self._series(df, "broad_dollar")
        tips = self._series(df, "tips10y")
        ndl = self._series(df, "ndl")
        usdjpy = self._series(df, "usdjpy")

        shock_series = []

        def add_z(series: pd.Series, window: int, min_periods: int, kind: str = "level") -> None:
            if series.isna().all():
                return
            if kind == "return":
                base = series.pct_change(window)
            else:
                base = series
            z = self._robust_z(base, 252, max(20, min_periods))
            shock_series.append(z.iloc[-1])

        add_z(eq, 5, 40, "return")
        add_z(eq, 20, 40, "return")
        add_z(btc, 5, 40, "return")
        add_z(btc, 20, 40, "return")
        add_z(copper, 20, 40, "return")
        add_z(silver, 20, 40, "return")
        add_z(hy, 252, 40, "level")
        add_z(vix, 252, 40, "level")
        add_z(dxy, 252, 40, "level")
        add_z(tips, 252, 40, "level")
        add_z(ndl, 252, 40, "level")
        add_z(usdjpy, 20, 40, "return")

        clean = np.asarray([x for x in shock_series if np.isfinite(x)], dtype=float)
        extreme_breadth = float(np.mean(np.abs(clean) >= self.extreme_z)) if clean.size else 0.0

        # The configured ``major_change_z`` is now the actual threshold for the
        # multi-horizon shift component; larger changes saturate at 1.
        change_unit = max(abs(self.major_change_z), 1.0)
        change_score = float(np.clip(max(abs(latest20), abs(latest60)) / change_unit, 0.0, 1.0))
        score = float(np.clip(0.55 * change_score + 0.45 * extreme_breadth, 0.0, 1.0))

        delta20 = self._num(appetite_raw.iloc[-1] - appetite_raw.iloc[-21], 0.0) if len(appetite_raw) > 21 else 0.0
        if score >= self.major_event_threshold and delta20 <= -0.08:
            event_type = "RISK_OFF_SHIFT"
        elif score >= self.major_event_threshold and delta20 >= 0.08:
            event_type = "RISK_ON_SHIFT"
        elif score >= self.major_event_threshold:
            event_type = "MULTI_FACTOR_SHOCK"
        else:
            event_type = "NONE"
        return score, bool(score >= self.major_event_threshold), event_type

    def evaluate(self, prepared_df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if not self.enabled or prepared_df is None or prepared_df.empty:
            return self._empty("Risk appetite engine disabled or no prepared data.")

        df = prepared_df.copy().sort_index()
        factor, _ = self._factor_scores(df)
        horizon = self._multi_horizon(factor)
        appetite = (horizon + 1.0) * 50.0
        persistence_on20 = (appetite >= self.risk_on_threshold).rolling(self.persistence_fast, min_periods=max(5, self.persistence_fast // 2)).mean()
        persistence_off20 = (appetite <= self.risk_off_threshold).rolling(self.persistence_fast, min_periods=max(5, self.persistence_fast // 2)).mean()
        persistence_on60 = (appetite >= self.risk_on_threshold).rolling(self.persistence_slow, min_periods=max(10, self.persistence_slow // 2)).mean()
        persistence_off60 = (appetite <= self.risk_off_threshold).rolling(self.persistence_slow, min_periods=max(10, self.persistence_slow // 2)).mean()
        on_persistence = self._finite_mean([persistence_on20.iloc[-1], persistence_on60.iloc[-1]])
        off_persistence = self._finite_mean([persistence_off20.iloc[-1], persistence_off60.iloc[-1]])

        latest_factor = factor.iloc[-1]
        finite = latest_factor.dropna()
        available = int(len(finite))
        factor_mean = float(finite.mean()) if available else 0.0
        coverage = available / max(len(self.FACTORS), 1)
        base_score = float(np.clip(50.0 + 50.0 * factor_mean, 0.0, 100.0))
        multi_horizon_score = float(np.clip(self._num(appetite.iloc[-1], 50.0), 0.0, 100.0))

        # Synchronised downside pressure is treated as a risk amplifier.
        returns = pd.DataFrame({
            "equity": self._series(df, "spx").pct_change(20),
            "gold": self._series(df, "gold").pct_change(20),
            "commodity": self._series(df, "oil").pct_change(20),
            "crypto": self._series(df, "btc").pct_change(20),
        })
        last_r = returns.iloc[-1].dropna()
        negative_breadth = float((last_r < 0).mean()) if len(last_r) else 0.0
        median_return = float(last_r.median()) if len(last_r) else 0.0
        synchronized = float(np.clip(0.65 * negative_breadth + 0.35 * np.clip((-median_return) / 0.12, 0.0, 1.0), 0.0, 1.0))

        shift20 = self._shift_z(appetite, 20)
        shift60 = self._shift_z(appetite, 60)
        shift20_now = float(np.clip(self._num(shift20.iloc[-1], 0.0), -6.0, 6.0))
        shift60_now = float(np.clip(self._num(shift60.iloc[-1], 0.0), -6.0, 6.0))
        major_score, major_active, major_type = self._major_event(df, factor, appetite / 100.0)

        # Tightening is intentionally multi-factor: real rates + credit + dollar
        # + liquidity + volatility + synchronised asset pressure.
        def lastf(name: str) -> float:
            return float(np.clip((self._num(latest_factor.get(name), 0.0) + 1.0) / 2.0, 0.0, 1.0))

        credit_risk = 1.0 - lastf("credit")
        vol_risk = 1.0 - lastf("volatility")
        dollar_risk = 1.0 - lastf("dollar")
        rate_risk = 1.0 - lastf("real_rates")
        liq_risk = 1.0 - lastf("liquidity")
        tightening = float(np.clip(
            0.20 * rate_risk + 0.20 * credit_risk + 0.16 * dollar_risk +
            0.16 * liq_risk + 0.14 * vol_risk + 0.14 * synchronized,
            0.0, 1.0,
        ))

        score = float(np.clip(
            0.75 * base_score + 0.25 * (50.0 * (1.0 - tightening)),
            0.0, 100.0,
        ))
        if synchronized >= 0.72:
            score = min(score, 34.0)
        if tightening >= 0.82:
            score = min(score, 22.0)

        if score <= self.crisis_threshold or (tightening >= 0.82 and synchronized >= 0.70):
            state = "CRISIS"
        elif score <= self.risk_off_threshold and off_persistence >= self.persistent_threshold:
            state = "RISK_OFF_PERSISTENT"
        elif score <= self.risk_off_threshold:
            state = "RISK_OFF"
        elif score >= self.risk_on_threshold and on_persistence >= self.persistent_threshold:
            state = "RISK_ON_PERSISTENT"
        elif score >= self.risk_on_threshold:
            state = "RISK_ON"
        else:
            state = "NEUTRAL"

        if major_active and major_type == "RISK_OFF_SHIFT" and state == "NEUTRAL":
            state = "RISK_OFF"
        if major_active and major_type == "RISK_ON_SHIFT" and state == "NEUTRAL" and on_persistence >= 0.35:
            state = "RISK_ON"

        confidence = float(np.clip(0.35 * coverage + 0.25 * min(len(df) / 252.0, 1.0) + 0.20 * min(len(df) / self.min_long_horizon_points, 1.0) + 0.20 * (0.5 + 0.5 * max(on_persistence if np.isfinite(on_persistence) else 0.0, off_persistence if np.isfinite(off_persistence) else 0.0)), 0.0, 1.0))

        # Component scores are displayed as risk appetite sub-scores [0,1].
        def subscore(name: str) -> float:
            return lastf(name)

        reason_parts = [
            f"multi-horizon={score:.1f}/100",
            f"on_persist20/60={persistence_on20.iloc[-1] if np.isfinite(persistence_on20.iloc[-1]) else 0:.2f}/{persistence_on60.iloc[-1] if np.isfinite(persistence_on60.iloc[-1]) else 0:.2f}",
            f"off_persist20/60={persistence_off20.iloc[-1] if np.isfinite(persistence_off20.iloc[-1]) else 0:.2f}/{persistence_off60.iloc[-1] if np.isfinite(persistence_off60.iloc[-1]) else 0:.2f}",
            f"tightening={tightening:.2f}",
            f"sync_stress={synchronized:.2f}",
        ]
        if major_active:
            reason_parts.append(f"major_event={major_type}:{major_score:.2f}")

        return {
            "risk_appetite_score": round(score, 4),
            "risk_appetite_state": state,
            "risk_appetite_confidence": round(confidence, 4),
            "risk_appetite_persistence_20d": round(float(np.clip(on_persistence if state.startswith("RISK_ON") else off_persistence if state.startswith("RISK_OFF") else max(on_persistence, off_persistence), 0.0, 1.0)) if np.isfinite(on_persistence) or np.isfinite(off_persistence) else 0.0, 4),
            "risk_appetite_on_persistence_20d": round(float(self._num(persistence_on20.iloc[-1], 0.0)), 4),
            "risk_appetite_off_persistence_20d": round(float(self._num(persistence_off20.iloc[-1], 0.0)), 4),
            "risk_appetite_on_persistence_60d": round(float(self._num(persistence_on60.iloc[-1], 0.0)), 4),
            "risk_appetite_off_persistence_60d": round(float(self._num(persistence_off60.iloc[-1], 0.0)), 4),
            "risk_appetite_risk_on_evidence": round(float(np.clip((factor[factor.columns].clip(-1,1).gt(0).mean(axis=1)).iloc[-1], 0.0, 1.0)), 4),
            "risk_appetite_risk_off_evidence": round(float(np.clip((factor[factor.columns].clip(-1,1).lt(0).mean(axis=1)).iloc[-1], 0.0, 1.0)), 4),
            "risk_appetite_multi_horizon_score": round(multi_horizon_score, 4),
            "risk_appetite_shift_20d_z": round(shift20_now, 4),
            "risk_appetite_shift_60d_z": round(shift60_now, 4),
            "risk_appetite_major_event_score": round(major_score, 4),
            "risk_appetite_major_event_active": bool(major_active),
            "risk_appetite_major_event_type": major_type,
            "risk_appetite_tightening_score": round(tightening, 4),
            "risk_appetite_liquidity_score": round(subscore("liquidity"), 4),
            "risk_appetite_credit_score": round(subscore("credit"), 4),
            "risk_appetite_volatility_score": round(subscore("volatility"), 4),
            "risk_appetite_growth_risk_score": round(float(np.clip(0.5 * (subscore("equity") + subscore("industrial_metals")), 0.0, 1.0)), 4),
            "risk_appetite_synchronized_stress": round(synchronized, 4),
            "risk_appetite_available_factor_count": available,
            "risk_appetite_factor_count": len(self.FACTORS),
            "risk_appetite_reason": "Risk appetite state: " + state + " | " + "; ".join(reason_parts),
            "risk_appetite_source": "MARKET_PROXY_ENSEMBLE_V3_2",
        }

    def allocation_adjustment(self, state: Dict[str, Any]) -> Tuple[float, float, str]:
        """Return (risk-budget multiplier, cash-floor add, reason)."""
        score = self._num(state.get("risk_appetite_score"), 50.0)
        conf = self._clip01(state.get("risk_appetite_confidence"), 0.0)
        major = bool(state.get("risk_appetite_major_event_active", False))
        appetite_state = str(state.get("risk_appetite_state", "NEUTRAL"))
        sync = self._clip01(state.get("risk_appetite_synchronized_stress"), 0.0)

        if appetite_state == "CRISIS":
            return 0.30, 0.25, "Risk-appetite CRISIS: aggressive risk-budget compression and defensive cash floor."
        if appetite_state == "RISK_OFF_PERSISTENT":
            reduction = 0.22 + 0.18 * conf + 0.10 * max(0.0, (50.0 - score) / 50.0)
            return max(0.55 - self.max_risk_off_reduction, 1.0 - reduction), 0.15, "Persistent risk-off state: risk budget compressed; capital preservation increased."
        if appetite_state == "RISK_OFF":
            reduction = 0.12 + 0.10 * conf
            return max(0.65, 1.0 - min(reduction, self.max_risk_off_reduction)), 0.08, "Risk-off state: moderate risk-budget compression."
        if appetite_state == "RISK_ON_PERSISTENT":
            boost = min(self.max_risk_on_boost, 0.04 + 0.08 * conf + 0.04 * max(0.0, (score - 60.0) / 40.0))
            return 1.0 + boost, 0.0, "Persistent risk-on state: bounded increase in risk budget; growth-sensitive assets tilted up."
        if appetite_state == "RISK_ON":
            boost = min(self.max_risk_on_boost * 0.65, 0.025 + 0.05 * conf)
            return 1.0 + boost, 0.0, "Risk-on state: bounded increase in risk budget."
        if major and sync >= 0.65:
            return 0.80, 0.08, "Major multi-factor shift with synchronized stress: temporary defensive compression."
        return 1.0, 0.0, "Neutral risk-appetite state: no structural risk-budget adjustment."

    def asset_tilts_for(self, state: Dict[str, Any]) -> Dict[str, float]:
        name = str(state.get("risk_appetite_state", "NEUTRAL"))
        raw = self.asset_tilts.get(name, self.asset_tilts.get("NEUTRAL", {}))
        return {k: float(np.clip(self._num(v, 1.0), 0.50, 1.30)) for k, v in raw.items()}

    def _empty(self, reason: str) -> Dict[str, Any]:
        return {
            "risk_appetite_score": 50.0,
            "risk_appetite_state": "NEUTRAL",
            "risk_appetite_confidence": 0.0,
            "risk_appetite_persistence_20d": 0.0,
            "risk_appetite_on_persistence_20d": 0.0,
            "risk_appetite_off_persistence_20d": 0.0,
            "risk_appetite_on_persistence_60d": 0.0,
            "risk_appetite_off_persistence_60d": 0.0,
            "risk_appetite_risk_on_evidence": 0.0,
            "risk_appetite_risk_off_evidence": 0.0,
            "risk_appetite_multi_horizon_score": 50.0,
            "risk_appetite_shift_20d_z": 0.0,
            "risk_appetite_shift_60d_z": 0.0,
            "risk_appetite_major_event_score": 0.0,
            "risk_appetite_major_event_active": False,
            "risk_appetite_major_event_type": "NONE",
            "risk_appetite_tightening_score": 0.5,
            "risk_appetite_liquidity_score": 0.5,
            "risk_appetite_credit_score": 0.5,
            "risk_appetite_volatility_score": 0.5,
            "risk_appetite_growth_risk_score": 0.5,
            "risk_appetite_synchronized_stress": 0.0,
            "risk_appetite_available_factor_count": 0,
            "risk_appetite_factor_count": len(self.FACTORS),
            "risk_appetite_reason": reason,
            "risk_appetite_source": "UNAVAILABLE",
        }
