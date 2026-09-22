"""
Macro Sentinel V3.2 — Predictive Risk / Opportunity Validation Engine.

Point-in-time rule: only matured historical observations are allowed to inform
current adaptive decisions. A signal at t can only use outcomes that are fully
known before the current decision timestamp.

V3.1 daily-observation rule
---------------------------
The raw history file may contain several runs on the same calendar day. The
predictive layer therefore canonicalizes history to ONE observation per day
before calculating 5-trading-day / 20-trading-day outcomes. This prevents
intraday reruns from masquerading as extra days and avoids overstating sample
size or shortening the forward horizon.
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
    from previously matured *daily* Macro Sentinel states and their later
    outcomes.
    """

    ASSETS = ("equity", "gold", "bond", "commodity", "crypto")
    PRICE_COLUMNS = {
        "equity": "spx_price",
        "gold": "gold_price",
        "commodity": "oil_price",
        "crypto": "btc_price",
    }

    # These columns represent the outcome-bearing market state. When a day has
    # multiple raw rows, the latest non-null value in the day is retained.
    DAILY_VALUE_COLUMNS = tuple(PRICE_COLUMNS.values()) + ("ust10y_yield",)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        root = config or {}
        pcfg = root.get("predictive_validation", {})
        self.enabled = bool(pcfg.get("enabled", True))
        self.lookahead_5d = max(1, int(pcfg.get("risk_horizon_days", 5)))
        self.lookahead_20d = max(1, int(pcfg.get("opportunity_horizon_days", 20)))
        self.min_matured = max(1, int(pcfg.get("minimum_matured_observations", 60)))
        self.min_bin = max(1, int(pcfg.get("minimum_bin_observations", 20)))
        self.risk_threshold = float(pcfg.get("risk_return_threshold_5d", -0.02))
        self.opportunity_threshold = float(pcfg.get("opportunity_return_threshold_20d", 0.04))
        self.prior_strength = max(0.0, float(pcfg.get("bayesian_prior_strength", 12.0)))
        self.prior_risk = float(np.clip(pcfg.get("prior_risk_probability_5d", 0.20), 0.0, 1.0))
        self.prior_opportunity = float(np.clip(pcfg.get("prior_opportunity_probability_20d", 0.50), 0.0, 1.0))
        edges = tuple(float(x) for x in pcfg.get("score_bins", [0.0, 0.35, 0.55, 0.75, 1.01]))
        self.score_edges = edges if len(edges) >= 2 else (0.0, 1.01)
        self.max_risk_reduction = float(np.clip(pcfg.get("max_risk_reduction", 0.25), 0.0, 1.0))
        self.max_risk_increase = float(np.clip(pcfg.get("max_risk_increase", 0.08), 0.0, 1.0))
        self.risk_reduce_at = float(np.clip(pcfg.get("risk_reduce_probability", 0.60), 0.0, 1.0))
        self.opportunity_increase_at = float(np.clip(pcfg.get("opportunity_increase_probability", 0.62), 0.0, 1.0))
        self.deterioration_soft = float(np.clip(pcfg.get("deterioration_soft", 0.55), 0.0, 1.0))
        self.deterioration_hard = float(np.clip(pcfg.get("deterioration_hard", 0.80), 0.0, 1.0))

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

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            # History timestamps are naive Istanbul-local strings. If an
            # offset-aware timestamp arrives, compare using its instant by
            # converting to a naive UTC representation.
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return ts
        except Exception:
            return None

    def _dailyize_history(self, history: Optional[pd.DataFrame], current_timestamp: Optional[Any] = None) -> pd.DataFrame:
        """Return one canonical observation per calendar day, point-in-time.

        The latest row of each day supplies state/signal fields. Outcome-bearing
        price/yield fields use the latest non-null value seen that day so a
        late degraded rerun cannot erase a valid market observation.
        """
        if history is None or history.empty:
            return pd.DataFrame()

        h = history.copy()
        if "date" not in h.columns:
            # Without an observation date there is no safe daily grouping rule.
            # Keep a copy for graceful warm-up, but do not pretend rows are days.
            return pd.DataFrame()

        h["__ts"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.dropna(subset=["__ts"]).sort_values("__ts").reset_index(drop=True)
        if h.empty:
            return pd.DataFrame()

        cutoff = self._parse_timestamp(current_timestamp)
        if cutoff is not None:
            h = h.loc[h["__ts"] <= cutoff].copy()
        if h.empty:
            return pd.DataFrame()

        h["__day"] = h["__ts"].dt.normalize()
        rows = []
        for _, group in h.groupby("__day", sort=True):
            row = group.iloc[-1].copy()
            for col in self.DAILY_VALUE_COLUMNS:
                if col not in group.columns:
                    continue
                valid = group[col].dropna()
                if not valid.empty:
                    row[col] = valid.iloc[-1]
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("__ts").reset_index(drop=True)
        return out.drop(columns=["__ts", "__day"], errors="ignore")

    def _bin_mask(self, series: pd.Series, value: float) -> pd.Series:
        edges = self.score_edges
        v = float(np.clip(value, 0.0, 1.0))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            upper_ok = series <= hi if i == len(edges) - 2 else series < hi
            if lo <= v <= hi if i == len(edges) - 2 else lo <= v < hi:
                return (series >= lo) & upper_ok
        return pd.Series(False, index=series.index)

    @staticmethod
    def _price_return(series: pd.Series, horizon: int) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        base = s.replace(0.0, np.nan)
        return base.shift(-horizon) / base - 1.0

    @staticmethod
    def _bond_forward_return(yield_series: pd.Series, horizon: int) -> pd.Series:
        s = pd.to_numeric(yield_series, errors="coerce")
        return -8.0 * (s.shift(-horizon) - s) / 100.0

    def _forward_outcomes(self, history: pd.DataFrame) -> pd.DataFrame:
        """Build 5/20 trading-day forward outcomes from canonical daily rows."""
        if history is None or history.empty:
            return pd.DataFrame(index=pd.Index([], dtype=int))

        h = history.copy().reset_index(drop=True)
        out = pd.DataFrame(index=h.index)
        returns5: Dict[str, pd.Series] = {}
        returns20: Dict[str, pd.Series] = {}

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
        valid5 = r5.notna().sum(axis=1)
        valid20 = r20.notna().sum(axis=1)
        out["negative_breadth5"] = r5.lt(0).sum(axis=1) / valid5.replace(0, np.nan)
        out["risk_event5"] = (out["median5"] <= self.risk_threshold) | (out["negative_breadth5"] >= 0.60)
        out["top20"] = r20.max(axis=1, skipna=True)
        out["opportunity20"] = out["top20"] >= self.opportunity_threshold
        out["usable5"] = valid5 >= 3
        out["usable20"] = valid20 >= 3
        return out

    def _matured_mask(self, history: pd.DataFrame, outcomes: pd.DataFrame, current_timestamp: Optional[Any]) -> pd.Series:
        """Select only observations with a fully known 20-day future.

        Since history has already been canonicalized to one row per day, row
        offsets now correspond to trading observations rather than intraday runs.
        """
        usable = outcomes["usable5"] & outcomes["usable20"]
        cutoff = max(self.lookahead_20d, self.lookahead_5d)
        eligible = pd.Series(False, index=history.index)
        if len(history) > cutoff:
            eligible.iloc[:-cutoff] = True

        # Defensive point-in-time check in case the caller supplied a mixed
        # history that extends beyond current_timestamp.
        if current_timestamp is not None and "date" in history.columns:
            cutoff_ts = self._parse_timestamp(current_timestamp)
            if cutoff_ts is not None:
                ts = pd.to_datetime(history["date"], errors="coerce")
                eligible &= ts <= cutoff_ts
        return usable & eligible

    def deterioration_score(self, history: Optional[pd.DataFrame], current_timestamp: Optional[Any] = None) -> float:
        """Immediate deterioration score from daily, point-in-time data only."""
        h = self._dailyize_history(history, current_timestamp=current_timestamp)
        if len(h) < 30:
            return 0.0

        series = []
        for col in ("spx_price", "gold_price", "oil_price", "btc_price"):
            if col not in h.columns:
                continue

            s = pd.to_numeric(h[col], errors="coerce")
            r1 = s.pct_change()
            r20 = s.pct_change(20)
            r60 = s.pct_change(60)
            r100 = s.pct_change(100)
            v20 = r1.rolling(20, min_periods=10).std()
            v60 = r1.rolling(60, min_periods=20).std()
            rolling_peak = s.rolling(126, min_periods=30).max()

            break_signal = ((r20 < 0) & ((r60 > 0) | (r100 > 0))).astype(float)
            vol_ratio = v20 / (v60 + 1e-9)
            vol_signal = np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0)
            dd_signal = np.clip(-(s / (rolling_peak + 1e-9) - 1.0) / 0.20, 0.0, 1.0)

            series.append(pd.DataFrame({
                "break": break_signal,
                "vol": vol_signal,
                "dd": dd_signal,
            }, index=h.index))

        if not series:
            return 0.0

        panel = pd.concat(series, axis=1)
        latest = panel.iloc[-1]

        def finite_mean(values: pd.Series) -> float:
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            return float(finite.mean()) if finite.size else 0.0

        break_mean = finite_mean(latest.filter(like="break"))
        vol_mean = finite_mean(latest.filter(like="vol"))
        dd_mean = finite_mean(latest.filter(like="dd"))
        score = 0.45 * break_mean + 0.30 * vol_mean + 0.25 * dd_mean
        return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0

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
            return PredictiveEstimate(
                "DISABLED", 0.0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                self.deterioration_score(history, current_timestamp),
                "Predictive validation disabled.", {}, {}, {},
            )

        history = self._dailyize_history(history, current_timestamp=current_timestamp)
        if history.empty:
            det = 0.0
            return PredictiveEstimate(
                "WARMUP", 0.0, 0, 0, 0,
                self.prior_risk, 0.0, 0.0,
                self.prior_opportunity, 0.0,
                det,
                "No dated historical observations available yet.",
                {}, {}, {},
            )

        outcomes = self._forward_outcomes(history)
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
            median_values = pd.to_numeric(outcomes.loc[risk_bin, "median5"], errors="coerce").dropna()
            median5 = float(median_values.mean()) if not median_values.empty else 0.0
            expected_loss = float(max(0.0, -median5))
        else:
            risk_prob, median5, expected_loss = self.prior_risk, 0.0, 0.0

        if opp_n > 0:
            opp_successes = int(outcomes.loc[opp_bin, "opportunity20"].fillna(False).sum())
            opp_prob = self._smoothed_probability(opp_successes, opp_n, self.prior_opportunity)
            opp_values = pd.to_numeric(outcomes.loc[opp_bin, "top20"], errors="coerce").dropna()
            opp_return = float(opp_values.mean()) if not opp_values.empty else 0.0
        else:
            opp_prob, opp_return = self.prior_opportunity, 0.0

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
            "equity": "spx_price",
            "gold": "gold_price",
            "commodity": "oil_price",
            "crypto": "btc_price",
        }

        for asset, score_col in score_cols.items():
            if score_col not in history.columns:
                continue
            score_series = pd.to_numeric(history[score_col], errors="coerce").clip(0, 1)
            cur = self._num(current_asset_scores.get(asset), 0.5)
            mask = matured & self._bin_mask(score_series, cur)

            if asset == "bond" and "ust10y_yield" in history.columns:
                fwd = self._bond_forward_return(history["ust10y_yield"], self.lookahead_20d)
            elif asset in price_cols and price_cols[asset] in history.columns:
                fwd = self._price_return(history[price_cols[asset]], self.lookahead_20d)
            else:
                continue

            mask &= fwd.notna()
            n = int(mask.sum())
            asset_n[asset] = n
            if n:
                valid = pd.to_numeric(fwd[mask], errors="coerce").dropna()
                if valid.empty:
                    continue
                er = float(valid.mean())
                pos = int((valid > 0).sum())
                pp = self._smoothed_probability(pos, n, self.prior_opportunity)
                asset_expected[asset] = er
                asset_positive[asset] = pp

        det = self.deterioration_score(history, current_timestamp=current_timestamp)
        ready = matured_n >= self.min_matured and risk_n >= self.min_bin and opp_n >= self.min_bin
        confidence = float(np.clip(
            min(matured_n / self.min_matured, 1.0)
            * min(risk_n / self.min_bin, 1.0)
            * min(opp_n / self.min_bin, 1.0),
            0.0,
            1.0,
        ))

        if not ready:
            status = "WARMUP"
            reason = (
                f"Predictive calibration warming up: daily_rows={len(history)}, "
                f"matured={matured_n}, risk_bin={risk_n}, opportunity_bin={opp_n}."
            )
        else:
            status = "CALIBRATED"
            reason = (
                "Current risk/opportunity state evaluated against matured historical "
                "daily outcomes only (5/20 trading-day horizons)."
            )

        return PredictiveEstimate(
            status=status,
            confidence=confidence,
            matured_observations=matured_n,
            risk_bin_observations=risk_n,
            opportunity_bin_observations=opp_n,
            risk_probability_5d=float(np.clip(risk_prob, 0.0, 1.0)),
            expected_median_return_5d=float(median5),
            expected_loss_5d=float(max(0.0, expected_loss)),
            opportunity_probability_20d=float(np.clip(opp_prob, 0.0, 1.0)),
            expected_opportunity_return_20d=float(opp_return),
            deterioration_score=float(np.clip(det, 0.0, 1.0)),
            reason=reason,
            asset_expected_returns_20d=asset_expected,
            asset_positive_probabilities_20d=asset_positive,
            asset_bin_observations=asset_n,
        )

    def adjust_risk(
        self,
        risk: float,
        estimate: PredictiveEstimate,
        stress_broad: bool,
        stress_hard: bool,
    ) -> Tuple[float, str]:
        """Bounded adaptive adjustment. Hard guards remain authoritative."""
        risk = float(risk)
        if estimate.status != "CALIBRATED" or estimate.confidence < 0.35:
            return risk, "Predictive layer in warm-up; no learned adjustment applied."

        original = risk
        if not stress_hard and estimate.risk_probability_5d >= self.risk_reduce_at:
            severity = np.clip(
                (estimate.risk_probability_5d - self.risk_reduce_at)
                / max(1.0 - self.risk_reduce_at, 1e-9),
                0.0,
                1.0,
            )
            risk *= 1.0 - float(severity) * self.max_risk_reduction

        if not stress_broad and not stress_hard and estimate.opportunity_probability_20d >= self.opportunity_increase_at:
            strength = np.clip(
                (estimate.opportunity_probability_20d - self.opportunity_increase_at)
                / max(1.0 - self.opportunity_increase_at, 1e-9),
                0.0,
                1.0,
            )
            risk += float(strength) * self.max_risk_increase

        if estimate.deterioration_score >= self.deterioration_hard:
            risk = min(risk, 0.18)
        elif estimate.deterioration_score >= self.deterioration_soft:
            risk = min(risk, max(0.25, risk * 0.70))

        risk = max(0.0, float(risk))
        if abs(risk - original) < 1e-9:
            return risk, "Predictive calibration active; no additional risk-budget adjustment required."
        return risk, (
            f"Predictive adjustment: risk {original:.3f}->{risk:.3f}; "
            f"riskProb5={estimate.risk_probability_5d:.2f}, "
            f"oppProb20={estimate.opportunity_probability_20d:.2f}, "
            f"deterioration={estimate.deterioration_score:.2f}, "
            f"confidence={estimate.confidence:.2f}."
        )
