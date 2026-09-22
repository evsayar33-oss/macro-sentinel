"""
Macro Sentinel V3.2 — Adaptive Cross-Asset Strategy + Persistent Risk Appetite + Predictive Validation.

Drop-in replacement for the existing adaptive_strategy_layer.py.
Adds a bounded empirical predictive layer without changing the point-in-time
contract: only matured historical outcomes may affect current risk sizing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from predictive_risk_engine import PredictiveRiskEngine

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
    """Cross-asset risk budget, capital preservation and predictive layer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        research = self.config.get("research_strategies", {})
        self.cfg = research.get("adaptive_layer", {})
        self.regime_risk = self.cfg.get("risk_budget_by_regime", research.get("production_risk_budget_by_regime", {
            "0": 0.60, "1": 0.35, "2": 0.08, "3": 0.45, "4": 0.25, "5": 0.82,
        }))
        self.min_cash_by_regime = self.cfg.get("min_cash_by_regime", research.get("production_min_cash_by_regime", {
            "0": 0.10, "1": 0.15, "2": 0.80, "3": 0.20, "4": 0.25, "5": 0.05,
        }))
        self.target_vol = float(self.cfg.get("target_annual_vol", 0.085))
        self.max_risk_budget = float(self.cfg.get("max_risk_budget", 0.92))
        self.min_risk_budget = float(self.cfg.get("min_risk_budget", 0.05))
        self.max_asset_weights = self.cfg.get("max_asset_weights", {
            "equity": 0.45, "gold": 0.25, "bond": 0.30, "commodity": 0.30, "crypto": 0.15,
        })
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
            "recovery_required_days": int(self.cfg.get("recovery_required_days", 3)),
            "hard_cash_floor": float(self.cfg.get("hard_cash_floor", 0.90)),
            "broad_cash_floor": float(self.cfg.get("broad_cash_floor", 0.65)),
            "broad_risk_cap": float(self.cfg.get("broad_risk_cap", 0.25)),
        }
        self.predictor = PredictiveRiskEngine(self.config)

    @staticmethod
    def _num(v: Any, default: float = np.nan) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _truth(v: Any) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        s = str(v).strip().lower()
        if s in {"true", "yes", "on"}:
            return True
        if s in {"false", "no", "off", "", "nan", "none"}:
            return False
        try:
            return float(v) != 0.0
        except (TypeError, ValueError):
            return False

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
        return -8.0 * yield_series.diff() / 100.0

    def _build_asset_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["equity"] = self._safe_series(df, "spx").pct_change(fill_method=None)
        out["gold"] = self._safe_series(df, "gold").pct_change(fill_method=None)
        out["bond"] = self._bond_returns(self._safe_series(df, "ust10y"))
        parts = []
        for col in ("oil", "brent", "copper", "silver"):
            s = self._safe_series(df, col).pct_change(fill_method=None)
            if s.notna().sum() >= 30:
                parts.append(s.rename(col))
        if parts:
            cm = pd.concat(parts, axis=1)
            vol = cm.rolling(60, min_periods=20).std()
            inv = 1.0 / (vol + 1e-9)
            denom = inv.where(cm.notna()).sum(axis=1)
            out["commodity"] = (cm * inv).sum(axis=1, min_count=1) / denom.replace(0.0, np.nan)
        else:
            out["commodity"] = np.nan
        out["crypto"] = self._safe_series(df, "btc").pct_change(fill_method=None)
        return out

    def _availability(self, returns: pd.DataFrame) -> Dict[str, bool]:
        return {a: int(pd.to_numeric(returns[a], errors="coerce").tail(90).notna().sum()) >= 30 for a in ASSET_KEYS}

    def _asset_scores(self, returns: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(index=returns.index, columns=ASSET_KEYS, dtype=float)
        for a in ASSET_KEYS:
            r = pd.to_numeric(returns[a], errors="coerce")
            if int(r.tail(90).notna().sum()) < 30:
                scores[a] = 0.02
                continue
            r20 = r.rolling(20, min_periods=10).sum()
            r60 = r.rolling(60, min_periods=30).sum()
            r100 = r.rolling(100, min_periods=50).sum()
            v20 = r.rolling(20, min_periods=10).std() * np.sqrt(252.0)
            v60 = r.rolling(60, min_periods=30).std() * np.sqrt(252.0)
            q20 = np.tanh(r20 / (v20 * np.sqrt(20.0 / 252.0) + 1e-9))
            q60 = np.tanh(r60 / (v60 * np.sqrt(60.0 / 252.0) + 1e-9))
            tr = np.tanh(r100 / (v60 * np.sqrt(100.0 / 252.0) + 1e-9))
            curve = (1.0 + r.fillna(0.0)).cumprod()
            peak = curve.rolling(126, min_periods=30).max()
            ddq = np.clip(curve / (peak + 1e-9), 0.0, 1.0)
            volq = 1.0 - np.clip((v20 - 0.10) / 0.45, 0.0, 1.0)
            scores[a] = (0.26 * (q20 + 1) / 2 + 0.28 * (q60 + 1) / 2 + 0.20 * (tr + 1) / 2 + 0.14 * ddq + 0.12 * volq).clip(0, 1)
        return scores

    def _stress(self, df: pd.DataFrame, returns: pd.DataFrame) -> StressState:
        r20 = returns.tail(20).sum(min_count=10)
        avail = r20.dropna()
        neg = float((avail < 0).mean()) if len(avail) else 0.0
        median = float(avail.median()) if len(avail) else 0.0
        corr = returns.tail(40).corr(min_periods=15)
        pairs = []
        for i, a in enumerate(ASSET_KEYS):
            for b in ASSET_KEYS[i + 1:]:
                if a in corr.index and b in corr.columns and np.isfinite(corr.loc[a, b]):
                    pairs.append(abs(float(corr.loc[a, b])))
        avg_corr = float(np.mean(pairs)) if pairs else 0.0
        row = df.iloc[-1]
        vix = self._num(row.get("vix_percentile_252"), 50.0)
        ndl = self._num(row.get("ndl_z"), 0.0)
        broad = (neg >= self.stress["broad_negative_breadth"] and median < -0.02) or (neg >= self.stress["broad_negative_breadth"] and avg_corr >= self.stress["broad_corr"] and (vix >= self.stress["broad_vix_pct"] or ndl <= self.stress["broad_ndl_z"]))
        hard = (neg >= self.stress["hard_negative_breadth"] and median < -0.05) or (neg >= self.stress["hard_negative_breadth"] and avg_corr >= self.stress["hard_corr"] and (vix >= self.stress["hard_vix_pct"] or ndl <= self.stress["hard_ndl_z"]))
        last5 = returns.tail(5).sum(min_count=3).dropna()
        pos5 = float((last5 > 0).mean()) if len(last5) else 0.0
        recovery = (not broad and not hard and (pos5 >= 0.60 or (vix < 55 and ndl > -0.10)))
        score = 0.40 * np.clip((neg - 0.40) / 0.60, 0, 1) + 0.30 * np.clip((avg_corr - 0.30) / 0.50, 0, 1) + 0.20 * np.clip((vix - 50) / 40, 0, 1) + 0.10 * np.clip(-ndl / 1.5, 0, 1)
        reason = f"Hard cross-asset stress: breadth={neg:.2f}, corr={avg_corr:.2f}, VIXpct={vix:.1f}, NDLZ={ndl:.2f}." if hard else f"Broad cross-asset stress: breadth={neg:.2f}, corr={avg_corr:.2f}, VIXpct={vix:.1f}, NDLZ={ndl:.2f}." if broad else "No broad cross-asset liquidation pattern detected."
        return StressState(float(np.clip(score, 0, 1)), bool(broad), bool(hard), neg, median, avg_corr, vix, ndl, recovery, reason)

    def _dynamic_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        vals = np.array([max(self.score_floor, scores[a]) for a in ASSET_KEYS], dtype=float)
        logits = vals / max(self.softmax_temperature, 0.1); logits -= logits.max(); p = np.exp(logits); p /= max(p.sum(), 1e-9)
        w = {a: float(p[i]) for i, a in enumerate(ASSET_KEYS)}
        for _ in range(4):
            excess = 0.0; free = []
            for a in ASSET_KEYS:
                cap = float(self.max_asset_weights.get(a, 1.0))
                if w[a] > cap:
                    excess += w[a] - cap; w[a] = cap
                else:
                    free.append(a)
            if excess <= 0 or not free: break
            denom = sum(w[a] for a in free)
            for a in free: w[a] += excess * w[a] / max(denom, 1e-9)
        return w

    def _vol_scale(self, sleeve: Dict[str, float], returns: pd.DataFrame) -> float:
        cols = [a for a in ASSET_KEYS if sleeve.get(a, 0) > 0 and a in returns.columns]
        sample = returns[cols].tail(60)
        if len(cols) < 2 or len(sample) < 25: return 1.0
        cov = sample.cov(min_periods=20).fillna(0.0) * 252.0
        w = np.array([sleeve[a] for a in cols], dtype=float)
        vol = float(np.sqrt(max(w @ cov.values @ w, 0.0)))
        if not np.isfinite(vol) or vol <= self.target_vol: return 1.0
        return float(np.clip(self.target_vol / vol, 0.0, 1.0))

    @staticmethod
    def _prior_stress(history: Optional[pd.DataFrame], required: int):
        if history is None or history.empty or "strategy_stress_broad" not in history.columns:
            return False, required
        h = history.tail(max(required, 1)); flags=[]
        for _, r in h.iterrows(): flags.append(bool(str(r.get("strategy_stress_broad", False)).lower() in {"1","true","yes","on"}) or bool(str(r.get("strategy_stress_hard", False)).lower() in {"1","true","yes","on"}))
        clean=0
        for f in reversed(flags):
            if f: break
            clean += 1
        return any(flags), clean

    def allocate(
        self,
        prepared_df: pd.DataFrame,
        classified_df: Optional[pd.DataFrame] = None,
        history: Optional[pd.DataFrame] = None,
        risk_appetite: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if prepared_df is None or prepared_df.empty:
            return {"weights": {"cash":100.0,"gold":0.0,"bond":0.0,"equity":0.0,"commodity":0.0,"crypto":0.0}, "strategy_mode":"CAPITAL_PRESERVATION_NO_DATA", "risk_budget_base":0.0, "risk_budget_final":0.0, "vol_scale":0.0, "opportunity_score":0.0, "asset_scores":{}, "available_asset_count":0, "stress_score":1.0,"stress_broad":True,"stress_hard":True,"stress_negative_breadth":1.0,"stress_median_return_20d":-1.0,"stress_avg_corr_40d":0.0,"stress_vix_percentile":100.0,"stress_ndl_z":-5.0,"stress_recovery_ready":False,"cash_floor":1.0,"unknown_guard_active":True,"oil_allocation_overlay":False,"oil_allocation_reason":"No data; risk not deployed.","stress_reason":"No prepared data available; capital preservation first.","decision_reason":"No prepared data available; capital preservation first."}

        df = prepared_df.copy().sort_index(); market = df.iloc[-1]
        decision = classified_df.iloc[-1] if classified_df is not None and not classified_df.empty else market
        rid = int(self._num(decision.get("confirmed_regime_id"), 0))
        returns = self._build_asset_returns(df); avail = self._availability(returns); score_df = self._asset_scores(returns)
        scores = {a: float(np.clip(self._num(score_df.iloc[-1].get(a), 0.02), 0, 1)) if avail[a] else 0.02 for a in ASSET_KEYS}
        stress = self._stress(df, returns)
        base = float(self.regime_risk.get(str(rid), self.regime_risk.get("0",0.60)))
        risk = base
        top=max(scores.values()) if scores else 0.0; positive=sum(v>=0.60 for v in scores.values())
        if top>=0.72 and positive>=3 and not stress.broad_stress: risk += 0.08
        elif top<0.45 or positive<=1: risk -= 0.08

        # V3.2 persistent risk-appetite state. Hard capital-preservation guards
        # below remain authoritative and can override this bounded adjustment.
        ra = risk_appetite or {}
        ra_state = str(ra.get("risk_appetite_state", "NEUTRAL"))
        ra_score = self._num(ra.get("risk_appetite_score"), 50.0)
        ra_conf = float(np.clip(self._num(ra.get("risk_appetite_confidence"), 0.0), 0.0, 1.0))
        ra_adj = float(np.clip(self._num(ra.get("risk_appetite_risk_budget_multiplier"), 1.0), 0.30, 1.12))
        ra_cash_add = float(np.clip(self._num(ra.get("risk_appetite_cash_floor_add"), 0.0), 0.0, 0.25))
        ra_reason = str(ra.get("risk_appetite_allocation_reason", "Risk-appetite state unavailable; no structural adjustment."))

        # Backward-compatible fallback when allocate() is called without a
        # pre-computed risk-appetite allocation adjustment.
        if "risk_appetite_risk_budget_multiplier" not in ra:
            if ra_state == "CRISIS":
                ra_adj, ra_cash_add, ra_reason = 0.30, 0.25, "Risk-appetite CRISIS: structural capital-preservation governor."
            elif ra_state == "RISK_OFF_PERSISTENT":
                ra_adj, ra_cash_add, ra_reason = 0.60, 0.15, "Persistent risk-off state: structural risk budget compression."
            elif ra_state == "RISK_OFF":
                ra_adj, ra_cash_add, ra_reason = 0.82, 0.08, "Risk-off state: bounded risk-budget compression."
            elif ra_state == "RISK_ON_PERSISTENT":
                boost = min(0.12, 0.04 + 0.08 * ra_conf + 0.04 * max(0.0, (ra_score - 60.0) / 40.0))
                ra_adj, ra_cash_add, ra_reason = 1.0 + boost, 0.0, "Persistent risk-on state: bounded risk-budget increase."
            elif ra_state == "RISK_ON":
                ra_adj, ra_cash_add, ra_reason = 1.03, 0.0, "Risk-on state: bounded risk-budget increase."

        risk *= ra_adj
        if stress.hard_stress: risk=min(risk,self.stress["hard_risk_cap"])
        elif stress.broad_stress: risk*=self.stress["broad_risk_multiplier"]
        if rid==2: risk=min(risk,0.08)
        if self._truth(decision.get("unknown_event_active",False)): risk=min(risk,0.25)

        available_scores=[scores[a] for a in ASSET_KEYS if avail[a]]
        topn=int(self.cfg.get("opportunity",{}).get("top_n",3)); opportunity=float(np.mean(sorted(available_scores,reverse=True)[:topn])) if available_scores else 0.0
        oppcfg=self.cfg.get("opportunity",{}); minopp=float(oppcfg.get("minimum_score",0.45)); high=float(oppcfg.get("high_score",0.72))
        if opportunity<minopp: risk=min(risk,0.45)
        elif opportunity>=high and not stress.broad_stress: risk=min(self.max_risk_budget,risk+float(oppcfg.get("strong_opportunity_bonus",0.05)))

        sleeve=self._dynamic_weights(scores)

        # Risk-appetite state changes the relative composition inside the deployed
        # risk sleeve. This is not leverage; the total risk budget remains capped.
        tilt_values = ra.get("risk_appetite_asset_tilts", {})
        if isinstance(tilt_values, dict):
            for asset in ASSET_KEYS:
                sleeve[asset] *= float(np.clip(self._num(tilt_values.get(asset), 1.0), 0.50, 1.30))
            tilt_total = sum(sleeve.values())
            if tilt_total > 0:
                sleeve = {k: v / tilt_total for k, v in sleeve.items()}

        predictive = self.predictor.evaluate(history, current_risk_score=max(stress.score,0.0), current_opportunity_score=opportunity, current_asset_scores=scores)
        det = predictive.deterioration_score
        risk, predictive_reason = self.predictor.adjust_risk(risk, predictive, stress.broad_stress, stress.hard_stress)

        # Asset-specific empirical edge adjusts the sleeve only after the
        # predictive layer is statistically warmed up. Each multiplier is bounded
        # to avoid overreacting to noisy samples.
        if predictive.status == "CALIBRATED":
            for asset in ASSET_KEYS:
                n = int(predictive.asset_bin_observations.get(asset, 0))
                if n >= self.predictor.min_bin and asset in predictive.asset_positive_probabilities_20d:
                    pp = float(predictive.asset_positive_probabilities_20d[asset])
                    er = float(predictive.asset_expected_returns_20d.get(asset, 0.0))
                    edge = np.clip(0.50 + 0.55 * (pp - 0.50) + 0.25 * np.tanh(er / 0.05), 0.60, 1.40)
                    sleeve[asset] *= float(edge)
            zsum = sum(sleeve.values())
            if zsum > 0:
                sleeve = {k: v / zsum for k, v in sleeve.items()}

        # Immediate deterioration guard is point-in-time and independent of learned history.
        if det >= self.predictor.deterioration_hard and not stress.hard_stress: risk=min(risk,0.18)
        elif det >= self.predictor.deterioration_soft and not stress.broad_stress: risk=min(risk, max(0.25,risk*0.70))

        available_count=sum(avail.values());
        if available_count < 4: risk=min(risk,0.35)

        oil_active=self._truth(decision.get("oil_event_active",False)); oil_type=str(decision.get("oil_event_type","NONE")).upper(); oil_pressure=self._num(decision.get("oil_pressure_score"),0.0)
        oil_overlay=False; oil_reason="No confirmed oil allocation overlay."
        if oil_active and oil_type in {"MOMENTUM","STRUCTURAL","COMBINED"} and not stress.hard_stress:
            sleeve["commodity"]=min(float(self.max_asset_weights.get("commodity",0.30)),sleeve["commodity"]+0.10); total=sum(sleeve.values()); sleeve={k:v/total for k,v in sleeve.items()}; oil_overlay=True; oil_reason=f"Confirmed oil event ({oil_type}); commodity priority raised inside existing risk budget."
        elif oil_pressure>=0.20: oil_reason="Oil pressure monitored only; confirmed-event gate not met, so no oil-specific allocation boost."

        vol_scale=self._vol_scale(sleeve,returns); deploy=risk*vol_scale; cash_floor=float(self.min_cash_by_regime.get(str(rid),self.min_cash_by_regime.get("0",0.10)))
        cash_floor = float(np.clip(cash_floor + ra_cash_add, 0.0, 0.95))
        if stress.hard_stress: deploy=min(deploy,self.stress["hard_risk_cap"]); cash_floor=max(cash_floor,self.stress["hard_cash_floor"])
        elif stress.broad_stress: deploy=min(deploy,self.stress["broad_risk_cap"]); cash_floor=max(cash_floor,self.stress["broad_cash_floor"])
        if self._truth(decision.get("unknown_event_active",False)): deploy=min(deploy,0.25); cash_floor=max(cash_floor,0.50)

        req=int(self.cfg.get("reentry",{}).get("require_clean_observations",3)); prior_stressed,clean=self._prior_stress(history,req); recovery=stress.recovery_ready and (not prior_stressed or clean>=req)
        if prior_stressed and clean<req and not stress.broad_stress and not stress.hard_stress: deploy=min(deploy,float(self.cfg.get("reentry",{}).get("risk_cap_until_recovered",0.35)))
        elif not recovery and (stress.broad_stress or stress.hard_stress): deploy=min(deploy,0.20 if stress.broad_stress else 0.10)
        deploy=float(np.clip(deploy,self.min_risk_budget,self.max_risk_budget))
        if 1.0-deploy<cash_floor: deploy=max(self.min_risk_budget,1.0-cash_floor)

        weights={"cash":1.0-deploy,"gold":0.0,"bond":0.0,"equity":0.0,"commodity":0.0,"crypto":0.0}
        for a in ASSET_KEYS: weights[a]=deploy*sleeve[a] if avail[a] else 0.0
        if stress.hard_stress:
            target=max(cash_floor,self.stress["hard_cash_floor"]); need=max(0.0,target-weights["cash"])
            for a in sorted(ASSET_KEYS,key=lambda x:weights[x],reverse=True):
                take=min(need,weights[a]); weights[a]-=take; weights["cash"]+=take; need-=take
                if need<=1e-12: break
            oil_overlay=False; oil_reason="Hard synchronized cross-asset stress: capital preservation overrides oil allocation preference."
        weights=self._normalize(weights)
        mode="CAPITAL_PRESERVATION_HARD_STRESS" if stress.hard_stress else "CAPITAL_PRESERVATION_BROAD_STRESS" if stress.broad_stress else "STAGED_REENTRY" if prior_stressed and not recovery else "ADAPTIVE_CROSS_ASSET_V3_2"
        return {
            "weights":weights,"strategy_mode":mode,"risk_budget_base":base,"risk_budget_final":deploy,"vol_scale":vol_scale,"opportunity_score":opportunity,"asset_scores":scores,"asset_availability":avail,"available_asset_count":available_count,
            "stress_score":stress.score,"stress_broad":stress.broad_stress,"stress_hard":stress.hard_stress,"stress_negative_breadth":stress.negative_breadth,"stress_median_return_20d":stress.median_return_20d,"stress_avg_corr_40d":stress.avg_corr_40d,"stress_vix_percentile":stress.vix_percentile,"stress_ndl_z":stress.ndl_z,"stress_recovery_ready":recovery,"cash_floor":cash_floor,
            "unknown_guard_active":self._truth(decision.get("unknown_event_active",False)),"oil_allocation_overlay":oil_overlay,"oil_allocation_reason":oil_reason,"stress_reason":stress.stress_reason,
            "risk_appetite_score":float(np.clip(self._num(ra.get("risk_appetite_score"), 50.0), 0.0, 100.0)),
            "risk_appetite_state":ra_state,
            "risk_appetite_confidence":ra_conf,
            "risk_appetite_persistence_20d":float(np.clip(self._num(ra.get("risk_appetite_persistence_20d"), 0.0), 0.0, 1.0)),
            "risk_appetite_major_event_score":float(np.clip(self._num(ra.get("risk_appetite_major_event_score"), 0.0), 0.0, 1.0)),
            "risk_appetite_major_event_active":bool(ra.get("risk_appetite_major_event_active", False)),
            "risk_appetite_major_event_type":str(ra.get("risk_appetite_major_event_type", "NONE")),
            "risk_appetite_tightening_score":float(np.clip(self._num(ra.get("risk_appetite_tightening_score"), 0.5), 0.0, 1.0)),
            "risk_appetite_synchronized_stress":float(np.clip(self._num(ra.get("risk_appetite_synchronized_stress"), 0.0), 0.0, 1.0)),
            "risk_appetite_risk_budget_multiplier":float(ra_adj),
            "risk_appetite_cash_floor_add":float(ra_cash_add),
            "risk_appetite_allocation_reason":ra_reason,
            "risk_appetite_asset_tilts":dict(tilt_values) if isinstance(tilt_values, dict) else {},
            "predictive_status":predictive.status,"predictive_confidence":predictive.confidence,"predictive_matured_observations":predictive.matured_observations,"predictive_risk_bin_observations":predictive.risk_bin_observations,"predictive_opportunity_bin_observations":predictive.opportunity_bin_observations,"predictive_risk_probability_5d":predictive.risk_probability_5d,"predictive_expected_median_return_5d":predictive.expected_median_return_5d,"predictive_expected_loss_5d":predictive.expected_loss_5d,"predictive_opportunity_probability_20d":predictive.opportunity_probability_20d,"predictive_expected_opportunity_return_20d":predictive.expected_opportunity_return_20d,"predictive_deterioration_score":predictive.deterioration_score,"predictive_adjustment_reason":predictive_reason,"predictive_reason":predictive.reason,
            "decision_reason":("Hard synchronized stress: cash governor active." if stress.hard_stress else "Broad cross-asset stress: risk budget compressed." if stress.broad_stress else "Persistent risk-off state: capital preservation prioritized." if ra_state in {"RISK_OFF_PERSISTENT", "CRISIS"} else "Persistent risk-on state: bounded growth-sensitive exposure enabled." if ra_state == "RISK_ON_PERSISTENT" else "Staged re-entry: recent stress episode has not fully cleared." if prior_stressed and not recovery else "Adaptive cross-asset allocation with V3.2 risk-appetite and predictive validation."),
        }
