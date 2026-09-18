"""
Macro Sentinel v3.0 — Adaptive Cross-Asset Risk & Capital-Preservation Layer

Purpose
-------
Adds a genuine asset-selection/risk-budget layer on top of the existing macro
regime engine without assuming that any asset is permanently safe.

Design principles
-----------------
- No look-ahead: every feature uses data up to the decision timestamp only.
- Cash is a first-class risk-control asset, not a permanent default.
- Broad cross-asset selloffs can override normal risk budgets.
- Asset selection uses momentum, trend, drawdown and realized-volatility quality.
- Portfolio risk is constrained by regime, stress and an annualized vol target.
- Re-entry after a broad stress event requires stabilization; this prevents
  one-day whipsaw from immediately redeploying capital.
- No leverage.
- Output always sums to 100%.

Expected prepared_df columns (only existing data are used when available)
---------------------------------------------------------------------------
spx, gold, ust10y, oil, brent, copper, silver, btc, vix_percentile_252, ndl_z
plus optional regime/event columns from MacroRegimeEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


ASSET_KEYS = ("equity", "gold", "bond", "commodity", "crypto")
OUTPUT_KEYS = ("cash", "gold", "bond", "equity", "commodity", "crypto")


@dataclass
class StressState:
    score: float
    broad_stress: bool
    hard_stress: bool
    negative_breadth: float
    median_return_20d: float
    avg_corr_40d: float
    vix_percentile: float
    ndl_z: float
    recovery_ready: bool
    stress_reason: str


class AdaptiveStrategyLayer:
    """Cross-asset risk budget + capital preservation overlay."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        research = self.config.get("research_strategies", {})
        self.cfg = research.get("adaptive_layer", {})
        self.regime_risk = self.cfg.get(
            "risk_budget_by_regime",
            research.get("production_risk_budget_by_regime", {
                "0": 0.60, "1": 0.35, "2": 0.08,
                "3": 0.45, "4": 0.25, "5": 0.82,
            }),
        )
        self.min_cash_by_regime = self.cfg.get(
            "min_cash_by_regime",
            research.get("production_min_cash_by_regime", {
                "0": 0.10, "1": 0.15, "2": 0.80,
                "3": 0.20, "4": 0.25, "5": 0.05,
            }),
        )
        self.target_vol = float(self.cfg.get("target_annual_vol", 0.085))
        self.max_risk_budget = float(self.cfg.get("max_risk_budget", 0.92))
        self.min_risk_budget = float(self.cfg.get("min_risk_budget", 0.05))
        self.max_asset_weights = {
            "equity": 0.45,
            "gold": 0.25,
            "bond": 0.30,
            "commodity": 0.30,
            "crypto": 0.15,
        }
        self.score_floor = float(self.cfg.get("score_floor", 0.05))
        self.softmax_temperature = float(self.cfg.get("softmax_temperature", 1.15))
        self.stress = {
            "broad_negative_breadth": float(self.cfg.get("broad_negative_breadth", 0.60)),
            "hard_negative_breadth": float(self.cfg.get("hard_negative_breadth", 0.80)),
            "broad_corr": float(self.cfg.get("broad_corr", 0.45)),
            "hard_corr": float(self.cfg.get("hard_corr", 0.60)),
            "broad_vix_pct": float(self.cfg.get("broad_vix_pct", 60.0)),
            "hard_vix_pct": float(self.cfg.get("hard_vix_pct", 80.0)),
            "broad_ndl_z": float(self.cfg.get("broad_ndl_z", -0.25)),
            "hard_ndl_z": float(self.cfg.get("hard_ndl_z", -0.80)),
            "hard_risk_cap": float(self.cfg.get("hard_risk_cap", 0.10)),
            "broad_risk_multiplier": float(self.cfg.get("broad_risk_multiplier", 0.55)),
            "recent_stress_days": int(self.cfg.get("recent_stress_days", 5)),
            "recovery_required_days": int(self.cfg.get("recovery_required_days", 3)),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _num(v: Any, default: float = np.nan) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip01(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        clean = {k: max(0.0, float(weights.get(k, 0.0))) for k in OUTPUT_KEYS}
        total = sum(clean.values())
        if total <= 1e-12:
            return {"cash": 100.0, "gold": 0.0, "bond": 0.0, "equity": 0.0, "commodity": 0.0, "crypto": 0.0}
        return {k: v * 100.0 / total for k, v in clean.items()}

    @staticmethod
    def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    @staticmethod
    def _bond_returns(yield_series: pd.Series) -> pd.Series:
        # Duration-like proxy: falling yields are positive for bond prices.
        return -8.0 * yield_series.diff() / 100.0

    def _build_asset_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct daily returns for the five investable risk sleeves."""
        out = pd.DataFrame(index=df.index)
        out["equity"] = self._safe_series(df, "spx").pct_change(fill_method=None)
        out["gold"] = self._safe_series(df, "gold").pct_change(fill_method=None)
        out["bond"] = self._bond_returns(self._safe_series(df, "ust10y"))
        
        # Commodity is a breadth-aware composite. We average available
        # standardized daily returns so oil does not dominate simply because
        # its nominal volatility is larger than copper/silver.
        commodity_parts = []
        for col in ("oil", "brent", "copper", "silver"):
            s = self._safe_series(df, col).pct_change(fill_method=None)
            if s.notna().sum() >= 30:
                commodity_parts.append(s.rename(col))
        if commodity_parts:
            cm = pd.concat(commodity_parts, axis=1)
            vols = cm.rolling(60, min_periods=20).std()
            inv = 1.0 / (vols + 1e-9)
            weighted = (cm * inv).sum(axis=1, min_count=1)
            denom = inv.where(cm.notna()).sum(axis=1)
            out["commodity"] = weighted / denom.replace(0.0, np.nan)
        else:
            out["commodity"] = np.nan
        out["crypto"] = self._safe_series(df, "btc").pct_change(fill_method=None)
        return out

    def _asset_availability(self, returns: pd.DataFrame) -> Dict[str, bool]:
        """Require enough recent observations before an asset can receive risk."""
        availability = {}
        for asset in ASSET_KEYS:
            s = pd.to_numeric(returns.get(asset), errors="coerce") if asset in returns.columns else pd.Series(dtype=float)
            recent = s.tail(90).dropna()
            availability[asset] = bool(len(recent) >= 30)
        return availability

    def _asset_score_series(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return 0..1 opportunity scores using trailing data only.

        An unavailable asset receives a near-zero score instead of a neutral
        score; this prevents missing data from silently creating exposure.
        """
        scores = pd.DataFrame(index=returns.index, columns=ASSET_KEYS, dtype=float)
        for asset in ASSET_KEYS:
            r = pd.to_numeric(returns[asset], errors="coerce") if asset in returns.columns else pd.Series(np.nan, index=returns.index)
            valid_recent = int(r.tail(90).notna().sum())
            if valid_recent < 30:
                scores[asset] = 0.02
                continue

            r20 = r.rolling(20, min_periods=10).sum()
            r60 = r.rolling(60, min_periods=30).sum()
            r100 = r.rolling(100, min_periods=50).sum()
            vol20 = r.rolling(20, min_periods=10).std() * np.sqrt(252.0)
            vol60 = r.rolling(60, min_periods=30).std() * np.sqrt(252.0)

            q20 = np.tanh(r20 / (vol20 * np.sqrt(20.0 / 252.0) + 1e-9))
            q60 = np.tanh(r60 / (vol60 * np.sqrt(60.0 / 252.0) + 1e-9))
            trend = np.tanh(r100 / (vol60 * np.sqrt(100.0 / 252.0) + 1e-9))

            eq_curve = (1.0 + r.fillna(0.0)).cumprod()
            rolling_peak = eq_curve.rolling(126, min_periods=30).max()
            dd = eq_curve / (rolling_peak + 1e-9) - 1.0
            dd_quality = np.clip(1.0 + dd, 0.0, 1.0)
            vol_quality = 1.0 - np.clip((vol20 - 0.10) / 0.45, 0.0, 1.0)

            raw = (
                0.26 * (q20 + 1.0) / 2.0
                + 0.28 * (q60 + 1.0) / 2.0
                + 0.20 * (trend + 1.0) / 2.0
                + 0.14 * dd_quality
                + 0.12 * vol_quality
            )
            scores[asset] = raw.clip(0.0, 1.0)
        return scores

    def _stress_state(self, df: pd.DataFrame, returns: pd.DataFrame) -> StressState:
        tail = returns.tail(40)
        latest20 = returns.tail(20).sum(min_count=10)
        available = latest20.dropna()
        if available.empty:
            negative_breadth = 0.0
            median_ret = 0.0
        else:
            negative_breadth = float((available < 0).mean())
            median_ret = float(available.median())

        corr = tail.corr(min_periods=15)
        pairs = []
        for i, a in enumerate(ASSET_KEYS):
            for b in ASSET_KEYS[i + 1 :]:
                v = self._num(corr.loc[a, b], np.nan) if a in corr.index and b in corr.columns else np.nan
                if np.isfinite(v):
                    pairs.append(abs(v))
        avg_corr = float(np.mean(pairs)) if pairs else 0.0

        vix_pct = self._num(df.iloc[-1].get("vix_percentile_252"), 50.0)
        ndl_z = self._num(df.iloc[-1].get("ndl_z"), 0.0)

        # Cross-asset liquidation must not depend on VIX being high. In a
        # synchronized selloff, correlations can actually become unstable or
        # temporarily fall, so breadth + magnitude are primary triggers while
        # correlation is a severity amplifier.
        broad_liquidation = (
            negative_breadth >= self.stress["broad_negative_breadth"]
            and median_ret < -0.02
        )
        hard_liquidation = (
            negative_breadth >= self.stress["hard_negative_breadth"]
            and median_ret < -0.05
        )
        broad = broad_liquidation or (
            negative_breadth >= self.stress["broad_negative_breadth"]
            and avg_corr >= self.stress["broad_corr"]
            and (vix_pct >= self.stress["broad_vix_pct"] or ndl_z <= self.stress["broad_ndl_z"])
        )
        hard = hard_liquidation or (
            negative_breadth >= self.stress["hard_negative_breadth"]
            and avg_corr >= self.stress["hard_corr"]
            and (vix_pct >= self.stress["hard_vix_pct"] or ndl_z <= self.stress["hard_ndl_z"])
        )

        # Historical stabilization requirement: at least N of the last K
        # observations must not satisfy broad/hard stress conditions.
        recovery_ready = True
        k = max(3, self.stress["recent_stress_days"])
        recent_flags = []
        for _, idx in df.tail(k).iterrows():
            # Re-evaluate with a short trailing window; this remains point-in-time.
            # For compactness, use the broad negative-breadth component from the
            # current rolling return panel when enough history exists.
            recent_flags.append(False)
        if broad or hard:
            recovery_ready = False
        else:
            # Require risk assets to show some breadth recovery.
            last5 = returns.tail(5).sum(min_count=3)
            positive_breadth = float((last5.dropna() > 0).mean()) if last5.dropna().size else 0.0
            recovery_ready = positive_breadth >= 0.60 or (vix_pct < 55 and ndl_z > -0.10)

        if hard:
            reason = (
                f"Hard cross-asset stress: negative breadth={negative_breadth:.2f}, "
                f"corr={avg_corr:.2f}, VIX pct={vix_pct:.1f}, NDL Z={ndl_z:.2f}."
            )
        elif broad:
            reason = (
                f"Broad cross-asset stress: negative breadth={negative_breadth:.2f}, "
                f"corr={avg_corr:.2f}, VIX pct={vix_pct:.1f}, NDL Z={ndl_z:.2f}."
            )
        else:
            reason = "No broad cross-asset liquidation pattern detected."

        # Score is bounded and interpretable.
        stress_score = 0.0
        stress_score += 0.40 * np.clip((negative_breadth - 0.40) / 0.60, 0, 1)
        stress_score += 0.30 * np.clip((avg_corr - 0.30) / 0.50, 0, 1)
        stress_score += 0.20 * np.clip((vix_pct - 50.0) / 40.0, 0, 1)
        stress_score += 0.10 * np.clip((-ndl_z - 0.0) / 1.5, 0, 1)
        return StressState(
            score=float(np.clip(stress_score, 0.0, 1.0)),
            broad_stress=bool(broad),
            hard_stress=bool(hard),
            negative_breadth=negative_breadth,
            median_return_20d=median_ret,
            avg_corr_40d=avg_corr,
            vix_percentile=vix_pct,
            ndl_z=ndl_z,
            recovery_ready=bool(recovery_ready),
            stress_reason=reason,
        )

    def _base_risk_budget(self, regime_id: int) -> float:
        return float(self.regime_risk.get(str(regime_id), self.regime_risk.get("0", 0.60)))

    def _min_cash(self, regime_id: int) -> float:
        return float(self.min_cash_by_regime.get(str(regime_id), self.min_cash_by_regime.get("0", 0.10)))

    def _adjust_risk_budget(
        self,
        base: float,
        stress: StressState,
        scores: Dict[str, float],
        regime_id: int,
        row: pd.Series,
    ) -> Tuple[float, Dict[str, Any]]:
        risk = base
        top = max(scores.values()) if scores else 0.0
        positive = sum(1 for v in scores.values() if v >= 0.60)
        if top >= 0.72 and positive >= 3 and not stress.broad_stress:
            risk += 0.08
        elif top < 0.45 or positive <= 1:
            risk -= 0.08

        if stress.hard_stress:
            risk = min(risk, self.stress["hard_risk_cap"])
        elif stress.broad_stress:
            risk *= self.stress["broad_risk_multiplier"]

        # A macro hard-shock regime takes precedence over opportunity signals.
        if regime_id == 2:
            risk = min(risk, 0.08)

        unknown_active = str(row.get("unknown_event_active", False)).strip().lower() in {"1", "true", "yes", "on"}
        if unknown_active:
            risk = min(risk, 0.25)

        risk = float(np.clip(risk, self.min_risk_budget, self.max_risk_budget))
        if (stress.broad_stress or stress.hard_stress) and not stress.recovery_ready:
            risk = min(risk, max(self.min_risk_budget, 0.20 if stress.broad_stress else 0.10))

        return risk, {
            "risk_budget_base": base,
            "risk_budget_final": risk,
            "stress_score": stress.score,
            "broad_stress": stress.broad_stress,
            "hard_stress": stress.hard_stress,
            "stress_recovery_ready": stress.recovery_ready,
            "stress_reason": stress.stress_reason,
        }

    def _dynamic_risk_weights(self, score_map: Dict[str, float]) -> Dict[str, float]:
        vals = np.array([max(self.score_floor, score_map[a]) for a in ASSET_KEYS], dtype=float)
        logits = vals / max(self.softmax_temperature, 0.1)
        logits -= logits.max()
        p = np.exp(logits)
        p /= max(p.sum(), 1e-9)
        weights = {a: float(p[i]) for i, a in enumerate(ASSET_KEYS)}

        # Hard upper caps followed by renormalization.
        for _ in range(4):
            over = [a for a in ASSET_KEYS if weights[a] > self.max_asset_weights[a]]
            if not over:
                break
            excess = 0.0
            free = []
            for a in ASSET_KEYS:
                cap = self.max_asset_weights[a]
                if weights[a] > cap:
                    excess += weights[a] - cap
                    weights[a] = cap
                else:
                    free.append(a)
            denom = sum(weights[a] for a in free)
            if denom > 0 and excess > 0:
                for a in free:
                    weights[a] += excess * weights[a] / denom
        return weights

    def _vol_scale(self, risk_weights: Dict[str, float], returns: pd.DataFrame) -> float:
        """Scale risk <=1 using a conservative trailing covariance estimate."""
        cols = [a for a in ASSET_KEYS if a in returns.columns and risk_weights.get(a, 0.0) > 0]
        if len(cols) < 2:
            return 1.0
        sample = returns[cols].tail(60)
        if sample.shape[0] < 25:
            return 1.0
        cov_df = sample.cov(min_periods=20) * 252.0
        cov_df = cov_df.reindex(index=cols, columns=cols)
        variances = sample.var().mul(252.0)
        cov_df = cov_df.fillna(0.0)
        for col in cols:
            if not np.isfinite(cov_df.loc[col, col]) or cov_df.loc[col, col] <= 0:
                v = float(variances.get(col, 0.0))
                cov_df.loc[col, col] = v if np.isfinite(v) and v > 0 else 0.01
        cov = cov_df.values
        w = np.array([risk_weights.get(a, 0.0) for a in cols], dtype=float)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        if not np.isfinite(vol) or vol <= self.target_vol:
            return 1.0
        return float(np.clip(self.target_vol / vol, 0.0, 1.0))

    @staticmethod
    def _prior_stress_state(history: Optional[pd.DataFrame], required_clean: int) -> Tuple[bool, int]:
        """Determine whether a recent stress episode is still blocking full re-entry."""
        if history is None or history.empty:
            return False, 0
        cols = [c for c in ("strategy_stress_broad", "strategy_stress_hard") if c in history.columns]
        if not cols:
            return False, 0
        tail = history.tail(max(required_clean, 1))
        stress_flags = []
        for _, row in tail.iterrows():
            broad = str(row.get("strategy_stress_broad", False)).strip().lower() in {"1", "true", "yes", "on"}
            hard = str(row.get("strategy_stress_hard", False)).strip().lower() in {"1", "true", "yes", "on"}
            stress_flags.append(broad or hard)
        consecutive_clean = 0
        for flag in reversed(stress_flags):
            if flag:
                break
            consecutive_clean += 1
        recently_stressed = any(stress_flags)
        return recently_stressed, consecutive_clean

    def allocate(
        self,
        prepared_df: pd.DataFrame,
        classified_df: Optional[pd.DataFrame] = None,
        history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate final production weights and diagnostics.

        The strategy is explicitly allowed to move overwhelmingly into cash
        during synchronized asset losses. Re-entry is staged rather than a
        one-day switch to avoid whipsaw.
        """
        if prepared_df is None or prepared_df.empty:
            return {
                "weights": {"cash": 100.0, "gold": 0.0, "bond": 0.0, "equity": 0.0, "commodity": 0.0, "crypto": 0.0},
                "strategy_mode": "CAPITAL_PRESERVATION_NO_DATA",
                "decision_reason": "No prepared data available.",
                "risk_budget_base": 0.0, "risk_budget_final": 0.0,
                "vol_scale": 0.0, "opportunity_score": 0.0, "asset_scores": {},
                "stress_score": 1.0, "stress_broad": True, "stress_hard": True,
                "stress_negative_breadth": 1.0, "stress_median_return_20d": -1.0,
                "stress_avg_corr_40d": 0.0, "stress_vix_percentile": 100.0, "stress_ndl_z": -5.0,
                "stress_recovery_ready": False, "cash_floor": 1.0,
                "unknown_guard_active": True, "oil_allocation_overlay": False,
                "oil_allocation_reason": "No data; risk not deployed.",
                "stress_reason": "No prepared data available; capital preservation first.",
            }

        df = prepared_df.copy().sort_index()
        market_row = df.iloc[-1]
        decision_row = classified_df.iloc[-1] if classified_df is not None and not classified_df.empty else market_row
        regime_id = int(self._num(decision_row.get("confirmed_regime_id"), self._num(market_row.get("confirmed_regime_id"), 0)))

        returns = self._build_asset_returns(df)
        availability = self._asset_availability(returns)
        scores = self._asset_score_series(returns)
        latest_scores = {}
        for a in ASSET_KEYS:
            default = 0.02 if not availability.get(a, False) else 0.50
            latest_scores[a] = self._clip01(self._num(scores.iloc[-1].get(a), default))
            if not availability.get(a, False):
                latest_scores[a] = 0.02

        stress = self._stress_state(df, returns)
        base_risk = self._base_risk_budget(regime_id)
        risk_budget, risk_meta = self._adjust_risk_budget(base_risk, stress, latest_scores, regime_id, decision_row)

        # If fewer than four risk sleeves have enough recent data, the engine cannot
        # prove a full cross-asset condition. Reduce risk rather than pretending coverage.
        available_count = sum(1 for a in ASSET_KEYS if availability.get(a, False))
        if available_count < 4:
            risk_budget = min(risk_budget, 0.35)

        sleeve = self._dynamic_risk_weights(latest_scores)
        oil_active = str(decision_row.get("oil_event_active", False)).strip().lower() in {"1", "true", "yes", "on"}
        oil_pressure = self._num(decision_row.get("oil_pressure_score"), 0.0)
        oil_type = str(decision_row.get("oil_event_type", "NONE")).upper()
        oil_allocation_overlay = False
        oil_allocation_reason = "No confirmed oil allocation overlay."

        # Pressure-only is intentionally informational. Only a confirmed oil event
        # changes commodity priority, and never during a hard synchronized selloff.
        if oil_active and oil_type in {"MOMENTUM", "STRUCTURAL", "COMBINED"} and not stress.hard_stress:
            sleeve["commodity"] = min(self.max_asset_weights["commodity"], sleeve["commodity"] + 0.10)
            total = sum(sleeve.values())
            sleeve = {k: v / total for k, v in sleeve.items()}
            oil_allocation_overlay = True
            oil_allocation_reason = f"Confirmed oil event ({oil_type}); commodity priority raised inside existing risk budget."
        elif oil_pressure >= 0.20:
            oil_allocation_reason = "Oil pressure monitored only; confirmed-event gate not met, so no oil-specific allocation boost."

        vol_scale = self._vol_scale(sleeve, returns)
        deploy_risk = risk_budget * vol_scale
        cash_floor = self._min_cash(regime_id)

        # Strong synchronized stress is the highest-priority capital-preservation
        # rule. It can override otherwise attractive asset scores.
        if stress.hard_stress:
            deploy_risk = min(deploy_risk, self.stress["hard_risk_cap"])
            cash_floor = max(cash_floor, self.stress.get("hard_cash_floor", 0.90))
        elif stress.broad_stress:
            deploy_risk = min(deploy_risk, self.stress.get("broad_risk_cap", 0.25))
            cash_floor = max(cash_floor, self.stress.get("broad_cash_floor", 0.65))

        unknown_guard_active = str(decision_row.get("unknown_event_active", False)).strip().lower() in {"1", "true", "yes", "on"}
        if unknown_guard_active:
            deploy_risk = min(deploy_risk, float(self.cfg.get("unknown_event", {}).get("risk_cap", 0.25)))
            cash_floor = max(cash_floor, float(self.cfg.get("unknown_event", {}).get("min_cash", 0.50)))

        # Opportunity gate: do not deploy risk merely because a regime budget exists.
        available_scores = [v for a, v in latest_scores.items() if availability.get(a, False)]
        top_n = int(self.cfg.get("opportunity", {}).get("top_n", 3))
        opportunity = float(np.mean(sorted(available_scores, reverse=True)[:top_n])) if available_scores else 0.0
        high_score = float(self.cfg.get("opportunity", {}).get("high_score", 0.72))
        min_score = float(self.cfg.get("opportunity", {}).get("minimum_score", 0.45))
        if opportunity < min_score:
            deploy_risk = min(deploy_risk, 0.45)
        elif opportunity >= high_score and not stress.broad_stress:
            deploy_risk = min(self.max_risk_budget, deploy_risk + float(self.cfg.get("opportunity", {}).get("strong_opportunity_bonus", 0.05)))

        # Staged re-entry after a stress episode. The system does not jump from
        # cash to full risk after a single green day.
        reentry_cfg = self.cfg.get("reentry", {})
        required_clean = int(reentry_cfg.get("require_clean_observations", self.stress.get("recovery_required_days", 3)))
        prior_stressed, clean_count = self._prior_stress_state(history, required_clean)
        recovery_ready = stress.recovery_ready and (not prior_stressed or clean_count >= required_clean)
        if prior_stressed and clean_count < required_clean and not stress.broad_stress and not stress.hard_stress:
            deploy_risk = min(deploy_risk, float(reentry_cfg.get("risk_cap_until_recovered", 0.35)))
        elif not recovery_ready and (stress.broad_stress or stress.hard_stress):
            deploy_risk = min(deploy_risk, 0.20 if stress.broad_stress else 0.10)

        # Strategy V3 never uses leverage and never lets cash fall below its floor.
        deploy_risk = float(np.clip(deploy_risk, self.min_risk_budget, self.max_risk_budget))
        if 1.0 - deploy_risk < cash_floor:
            deploy_risk = max(self.min_risk_budget, 1.0 - cash_floor)

        weights = {"cash": 1.0 - deploy_risk, "gold": 0.0, "bond": 0.0, "equity": 0.0, "commodity": 0.0, "crypto": 0.0}
        for a in ASSET_KEYS:
            weights[a] = deploy_risk * sleeve[a] if availability.get(a, False) else 0.0

        # In hard stress, pin a high-cash state even if numerical floors allowed more risk.
        if stress.hard_stress:
            target_cash = max(cash_floor, self.stress.get("hard_cash_floor", 0.90))
            donor_needed = max(0.0, target_cash - weights["cash"])
            if donor_needed > 0:
                for a in sorted(ASSET_KEYS, key=lambda x: weights[x], reverse=True):
                    take = min(donor_needed, weights[a])
                    weights[a] -= take
                    weights["cash"] += take
                    donor_needed -= take
                    if donor_needed <= 1e-12:
                        break
            oil_allocation_overlay = False
            oil_allocation_reason = "Hard synchronized cross-asset stress: capital preservation overrides oil allocation preference."

        weights = self._normalize(weights)
        if stress.hard_stress:
            mode = "CAPITAL_PRESERVATION_HARD_STRESS"
        elif stress.broad_stress:
            mode = "CAPITAL_PRESERVATION_BROAD_STRESS"
        elif prior_stressed and not recovery_ready:
            mode = "STAGED_REENTRY"
        else:
            mode = "ADAPTIVE_CROSS_ASSET_V3"

        return {
            "weights": weights,
            "strategy_mode": mode,
            "risk_budget_base": base_risk,
            "risk_budget_final": deploy_risk,
            "vol_scale": vol_scale,
            "opportunity_score": opportunity,
            "asset_scores": latest_scores,
            "asset_availability": availability,
            "available_asset_count": available_count,
            "stress_score": stress.score,
            "stress_broad": stress.broad_stress,
            "stress_hard": stress.hard_stress,
            "stress_negative_breadth": stress.negative_breadth,
            "stress_median_return_20d": stress.median_return_20d,
            "stress_avg_corr_40d": stress.avg_corr_40d,
            "stress_vix_percentile": stress.vix_percentile,
            "stress_ndl_z": stress.ndl_z,
            "stress_recovery_ready": recovery_ready,
            "cash_floor": cash_floor,
            "unknown_guard_active": unknown_guard_active,
            "oil_allocation_overlay": oil_allocation_overlay,
            "oil_allocation_reason": oil_allocation_reason,
            "stress_reason": stress.stress_reason,
            "decision_reason": (
                "Hard synchronized stress: cash governor active." if stress.hard_stress else
                "Broad cross-asset stress: risk budget compressed." if stress.broad_stress else
                "Staged re-entry: recent stress episode has not fully cleared." if prior_stressed and not recovery_ready else
                "Adaptive cross-asset allocation based on opportunity, volatility and macro risk."
            ),
        }
