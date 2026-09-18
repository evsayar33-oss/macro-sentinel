"""
Macro Sentinel V3.1 — Predictive Risk / Opportunity Validation Engine.

Point-in-time rule: only matured historical observations are allowed to inform
current adaptive decisions. A signal at t can only use outcomes that are fully
known before the current decision timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictiveEstimate:
    status: str
    confidence: float
    matured_observations: int
    risk_bin_observations: int
    opportunity_bin_observations: int
    risk_probability_5d: float
    expected_median_return_5d: float
    expected_loss_5d: float
    opportunity_probability_20d: float
    expected_opportunity_return_20d: float
    deterioration_score: float
    reason: str
    asset_expected_returns_20d: Dict[str, float]
    asset_positive_probabilities_20d: Dict[str, float]
    asset_bin_observations: Dict[str, int]


class PredictiveRiskEngine:
    """Empirical, bounded, point-in-time prediction layer.

    It does not forecast prices directly. It estimates conditional probabilities
    from previously matured Macro Sentinel states and their later outcomes.
    """

    ASSETS = ("equity", "gold", "bond", "commodity", "crypto")
    PRICE_COLUMNS = {
        "equity": "spx_price",
        "gold": "gold_price",
        "commodity": "oil_price",
        "crypto": "btc_price",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        root = config or {}
        pcfg = root.get("predictive_validation", {})
        self.enabled = bool(pcfg.get("enabled", True))
        self.lookahead_5d = int(pcfg.get("risk_horizon_days", 5))
        self.lookahead_20d = int(pcfg.get("opportunity_horizon_days", 20))
        self.min_matured = int(pcfg.get("minimum_matured_observations", 60))
        self.min_bin = int(pcfg.get("minimum_bin_observations", 20))
        self.risk_threshold = float(pcfg.get("risk_return_threshold_5d", -0.02))
        self.opportunity_threshold = float(pcfg.get("opportunity_return_threshold_20d", 0.04))
        self.prior_strength = float(pcfg.get("bayesian_prior_strength", 12.0))
        self.prior_risk = float(pcfg.get("prior_risk_probability_5d", 0.20))
        self.prior_opportunity = float(pcfg.get("prior_opportunity_probability_20d", 0.50))
        self.score_edges = tuple(float(x) for x in pcfg.get("score_bins", [0.0, 0.35, 0.55, 0.75, 1.01]))
        self.max_risk_reduction = float(pcfg.get("max_risk_reduction", 0.25))
        self.max_risk_increase = float(pcfg.get("max_risk_increase", 0.08))
        self.risk_reduce_at = float(pcfg.get("risk_reduce_probability", 0.60))
        self.opportunity_increase_at = float(pcfg.get("opportunity_increase_probability", 0.62))
        self.deterioration_soft = float(pcfg.get("deterioration_soft", 0.55))
        self.deterioration_hard = float(pcfg.get("deterioration_hard", 0.80))

    @staticmethod
    def _num(v: Any, default: float = np.nan) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool(v: Any) -> bool:
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

    def _bin_mask(self, series: pd.Series, value: float) -> pd.Series:
        edges = self.score_edges
        v = float(np.clip(value, 0.0, 1.0))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo <= v < hi or (i == len(edges) - 2 and v <= hi):
                return (series >= lo) & (series < hi if i < len(edges) - 2 else series <= hi)
        return pd.Series(False, index=series.index)

    @staticmethod
    def _price_return(series: pd.Series, horizon: int) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        return s.shift(-horizon) / s - 1.0

    @staticmethod
    def _bond_forward_return(yield_series: pd.Series, horizon: int) -> pd.Series:
        s = pd.to_numeric(yield_series, errors="coerce")
        return -8.0 * (s.shift(-horizon) - s) / 100.0

    def _forward_outcomes(self, history: pd.DataFrame) -> pd.DataFrame:
        """Build matured signal outcomes from persisted prices only."""
        if history is None or history.empty:
            return pd.DataFrame(index=pd.Index([], dtype=int))
        h = history.copy()
        out = pd.DataFrame(index=h.index)

        returns5 = {}
        returns20 = {}
        for asset, col in self.PRICE_COLUMNS.items():
            if col in h.columns:
                returns5[asset] = self._price_return(h[col], self.lookahead_5d)
                returns20[asset] = self._price_return(h[col], self.lookahead_20d)

        if "ust10y_yield" in h.columns:
            returns5["bond"] = self._bond_forward_return(h["ust10y_yield"], self.lookahead_5d)
            returns20["bond"] = self._bond_forward_return(h["ust10y_yield"], self.lookahead_20d)

        r5 = pd.DataFrame(returns5, index=h.index)
        r20 = pd.DataFrame(returns20, index=h.index)
        out["median5"] = r5.median(axis=1, skipna=True)
        out["negative_breadth5"] = (r5.lt(0).sum(axis=1) / r5.notna().sum(axis=1).replace(0, np.nan))
        out["risk_event5"] = (out["median5"] <= self.risk_threshold) | (out["negative_breadth5"] >= 0.60)
        out["top20"] = r20.max(axis=1, skipna=True)
        out["opportunity20"] = out["top20"] >= self.opportunity_threshold
        out["usable5"] = r5.notna().sum(axis=1) >= 3
        out["usable20"] = r20.notna().sum(axis=1) >= 3
        return out

    def _matured_mask(self, history: pd.DataFrame, outcomes: pd.DataFrame, current_timestamp: Optional[Any]) -> pd.Series:
        usable = outcomes["usable5"] & outcomes["usable20"]
        # Use row position, not calendar-day subtraction. This is robust to
        # weekends/holidays because history is a trading-day observation stream.
        cutoff = max(self.lookahead_20d, self.lookahead_5d)
        eligible = pd.Series(False, index=history.index)
        if len(history) > cutoff:
            eligible.iloc[:-cutoff] = True
        return usable & eligible

    def deterioration_score(self, history: Optional[pd.DataFrame]) -> float:
        """Immediate trend-break score based only on data through the current row."""
        if history is None or len(history) < 30:
            return 0.0
        h = history.copy()
        series = []
        for col in ("spx_price", "gold_price", "oil_price", "btc_price"):
            if col in h.columns:
                s = pd.to_numeric(h[col], errors="coerce")
                r20 = s.pct_change(20)
                r60 = s.pct_change(60)
                r100 = s.pct_change(100)
                v20 = s.pct_change().rolling(20).std()
                v60 = s.pct_change().rolling(60).std()
                series.append(pd.DataFrame({
                    "break": ((r20 < 0) & ((r60 > 0) | (r100 > 0))).astype(float),
                    "vol": np.clip((v20 / (v60 + 1e-9) - 1.0) / 1.5, 0.0, 1.0),
                    "dd": np.clip(-(s / s.rolling(126, min_periods=30).max() - 1.0) / 0.20, 0.0, 1.0),
                }, index=h.index))
        if not series:
            return 0.0
        panel = pd.concat(series, axis=1)
        latest = panel.iloc[-1].dropna()
        if latest.empty:
            return 0.0
        # Equal contribution across asset diagnostics, still capped and interpretable.
        score = 0.45 * float(panel.filter(like="break").iloc[-1].mean())
        score += 0.30 * float(panel.filter(like="vol").iloc[-1].mean())
        score += 0.25 * float(panel.filter(like="dd").iloc[-1].mean())
        return float(np.clip(score, 0.0, 1.0))

    def _smoothed_probability(self, successes: int, total: int, prior: float) -> float:
        if total <= 0:
            return float(prior)
        return float((successes + self.prior_strength * prior) / (total + self.prior_strength))

    def evaluate(
        self,
        history: Optional[pd.DataFrame],
        current_risk_score: float,
        current_opportunity_score: float,
        current_timestamp: Optional[Any] = None,
        current_asset_scores: Optional[Dict[str, float]] = None,
    ) -> PredictiveEstimate:
        if not self.enabled:
            return PredictiveEstimate("DISABLED", 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, self.deterioration_score(history), "Predictive validation disabled.", {}, {}, {})

        history = history.copy() if history is not None else pd.DataFrame()
        outcomes = self._forward_outcomes(history)
        if outcomes.empty:
            det = self.deterioration_score(history)
            return PredictiveEstimate("WARMUP", 0.0, 0, 0, 0, self.prior_risk, 0.0, 0.0, self.prior_opportunity, 0.0, det, "No historical outcomes available yet.", {}, {}, {})

        matured = self._matured_mask(history, outcomes, current_timestamp)
        matured_n = int(matured.sum())
        if "strategy_stress_score" not in history.columns:
            risk_series = pd.Series(0.5, index=history.index)
        else:
            risk_series = pd.to_numeric(history["strategy_stress_score"], errors="coerce").fillna(0.5).clip(0, 1)
        if "strategy_opportunity_score" not in history.columns:
            opp_series = pd.Series(0.5, index=history.index)
        else:
            opp_series = pd.to_numeric(history["strategy_opportunity_score"], errors="coerce").fillna(0.5).clip(0, 1)

        risk_bin = matured & self._bin_mask(risk_series, current_risk_score)
        opp_bin = matured & self._bin_mask(opp_series, current_opportunity_score)
        risk_n = int(risk_bin.sum())
        opp_n = int(opp_bin.sum())

        if risk_n > 0:
            risk_successes = int(outcomes.loc[risk_bin, "risk_event5"].fillna(False).sum())
            risk_prob = self._smoothed_probability(risk_successes, risk_n, self.prior_risk)
            median5 = float(pd.to_numeric(outcomes.loc[risk_bin, "median5"], errors="coerce").mean())
            expected_loss = float(max(0.0, -median5))
        else:
            risk_prob, median5, expected_loss = self.prior_risk, 0.0, 0.0

        if opp_n > 0:
            opp_successes = int(outcomes.loc[opp_bin, "opportunity20"].fillna(False).sum())
            opp_prob = self._smoothed_probability(opp_successes, opp_n, self.prior_opportunity)
            opp_return = float(pd.to_numeric(outcomes.loc[opp_bin, "top20"], errors="coerce").mean())
        else:
            opp_prob, opp_return = self.prior_opportunity, 0.0

        # Asset-specific forward-edge calibration. This is what converts a
        # generic opportunity score into an adaptive selector: a strong score
        # is rewarded only when that score historically led to positive 20-day
        # outcomes for the same asset.
        asset_expected: Dict[str, float] = {}
        asset_positive: Dict[str, float] = {}
        asset_n: Dict[str, int] = {}
        current_asset_scores = current_asset_scores or {}
        score_cols = {
            "equity": "strategy_asset_score_equity",
            "gold": "strategy_asset_score_gold",
            "bond": "strategy_asset_score_bond",
            "commodity": "strategy_asset_score_commodity",
            "crypto": "strategy_asset_score_crypto",
        }
        price_cols = {
            "equity": "spx_price", "gold": "gold_price", "commodity": "oil_price", "crypto": "btc_price",
        }
        for asset, score_col in score_cols.items():
            if score_col not in history.columns:
                continue
            series = pd.to_numeric(history[score_col], errors="coerce").clip(0, 1)
            cur = self._num(current_asset_scores.get(asset), 0.5)
            mask = matured & self._bin_mask(series, cur)
            if asset == "bond" and "ust10y_yield" in history.columns:
                fwd = self._bond_forward_return(history["ust10y_yield"], self.lookahead_20d)
            elif asset in price_cols and price_cols[asset] in history.columns:
                fwd = self._price_return(history[price_cols[asset]], self.lookahead_20d)
            else:
                continue
            mask = mask & fwd.notna()
            n = int(mask.sum())
            asset_n[asset] = n
            if n:
                er = float(pd.to_numeric(fwd[mask], errors="coerce").mean())
                pos = int((pd.to_numeric(fwd[mask], errors="coerce") > 0).sum())
                pp = self._smoothed_probability(pos, n, self.prior_opportunity)
                asset_expected[asset] = er
                asset_positive[asset] = pp

        det = self.deterioration_score(history)
        ready = matured_n >= self.min_matured and risk_n >= self.min_bin and opp_n >= self.min_bin
        confidence = float(np.clip(
            min(matured_n / max(self.min_matured, 1), 1.0)
            * min(risk_n / max(self.min_bin, 1), 1.0)
            * min(opp_n / max(self.min_bin, 1), 1.0),
            0.0, 1.0,
        ))
        if not ready:
            status = "WARMUP"
            reason = f"Predictive calibration warming up: matured={matured_n}, risk_bin={risk_n}, opportunity_bin={opp_n}."
        else:
            status = "CALIBRATED"
            reason = "Current risk/opportunity state evaluated against matured historical outcomes only."

        return PredictiveEstimate(
            status=status,
            confidence=confidence,
            matured_observations=matured_n,
            risk_bin_observations=risk_n,
            opportunity_bin_observations=opp_n,
            risk_probability_5d=float(np.clip(risk_prob, 0.0, 1.0)),
            expected_median_return_5d=median5,
            expected_loss_5d=expected_loss,
            opportunity_probability_20d=float(np.clip(opp_prob, 0.0, 1.0)),
            expected_opportunity_return_20d=opp_return,
            deterioration_score=det,
            reason=reason,
            asset_expected_returns_20d=asset_expected,
            asset_positive_probabilities_20d=asset_positive,
            asset_bin_observations=asset_n,
        )

    def adjust_risk(self, risk: float, estimate: PredictiveEstimate, stress_broad: bool, stress_hard: bool) -> Tuple[float, str]:
        """Bounded adaptive adjustment. Never overrides hard capital-preservation guards."""
        if estimate.status != "CALIBRATED" or estimate.confidence < 0.35:
            return risk, "Predictive layer in warm-up; no learned adjustment applied."
        original = risk
        if not stress_hard and estimate.risk_probability_5d >= self.risk_reduce_at:
            severity = np.clip((estimate.risk_probability_5d - self.risk_reduce_at) / max(1.0 - self.risk_reduce_at, 1e-9), 0.0, 1.0)
            risk *= 1.0 - float(severity) * self.max_risk_reduction
        if not stress_broad and not stress_hard and estimate.opportunity_probability_20d >= self.opportunity_increase_at:
            strength = np.clip((estimate.opportunity_probability_20d - self.opportunity_increase_at) / max(1.0 - self.opportunity_increase_at, 1e-9), 0.0, 1.0)
            risk += float(strength) * self.max_risk_increase
        if estimate.deterioration_score >= self.deterioration_hard:
            risk = min(risk, 0.18)
        elif estimate.deterioration_score >= self.deterioration_soft:
            risk = min(risk, max(0.25, risk * 0.70))
        if abs(risk - original) < 1e-9:
            return risk, "Predictive calibration active; no additional risk-budget adjustment required."
        return float(risk), (
            f"Predictive adjustment: risk {original:.3f}->{risk:.3f}; "
            f"riskProb5={estimate.risk_probability_5d:.2f}, oppProb20={estimate.opportunity_probability_20d:.2f}, "
            f"deterioration={estimate.deterioration_score:.2f}, confidence={estimate.confidence:.2f}."
        )
