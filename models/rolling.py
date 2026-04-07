from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from models.decomposition import SpreadComponents, decompose
from data.stream import Trade


class Regime(str, Enum):
    DEEP_STABLE    = "deep / stable"
    THIN_ONE_SIDED = "thin / one-sided"
    FRAGMENTED     = "fragmented"
    UNKNOWN        = "unknown"

REGIME_COLORS = {
    Regime.DEEP_STABLE:    "green",
    Regime.THIN_ONE_SIDED: "red",
    Regime.FRAGMENTED:     "yellow",
    Regime.UNKNOWN:        "white",
}


@dataclass
class RollingEstimate:
    components: SpreadComponents
    regime: Regime
    alpha_zscore: float
    lambda_zscore: float
    window_seconds: float


class RollingDecomposer:
    def __init__(self, trade_window=500, history_len=120, regime_lookback=20):
        self._trades = deque(maxlen=trade_window)
        self._history = deque(maxlen=history_len)
        self._regime_lookback = regime_lookback

    def push(self, trade: Trade) -> Optional[RollingEstimate]:
        self._trades.append(trade)
        comp = decompose(list(self._trades))
        if comp is None or not comp.is_valid():
            return None
        self._history.append(comp)
        return RollingEstimate(
            components=comp,
            regime=self._classify(comp),
            alpha_zscore=self._zscore("adverse_selection_share"),
            lambda_zscore=self._zscore("kyle_lambda"),
            window_seconds=self._window_seconds(),
        )

    def _zscore(self, attr: str) -> float:
        vals = [getattr(c, attr) for c in list(self._history)[-self._regime_lookback:]]
        if len(vals) < 5:
            return 0.0
        arr = np.array(vals, dtype=float)
        std = arr.std()
        if std < 1e-10:
            return 0.0
        return float((arr[-1] - arr.mean()) / std)

    def _classify(self, comp: SpreadComponents) -> Regime:
        if len(self._history) < 5:
            return Regime.UNKNOWN
        hist_es    = np.array([c.effective_spread for c in self._history])
        hist_alpha = np.array([c.adverse_selection_share for c in self._history])
        spread_pct = (comp.effective_spread - hist_es.mean()) / (hist_es.std() + 1e-10)
        alpha_pct  = (comp.adverse_selection_share - hist_alpha.mean()) / (hist_alpha.std() + 1e-10)
        if spread_pct < 0.5 and alpha_pct < 0.5:
            return Regime.DEEP_STABLE
        elif alpha_pct > 0.5 or (comp.kyle_lambda > 0 and spread_pct > 1.0):
            return Regime.THIN_ONE_SIDED
        else:
            return Regime.FRAGMENTED

    def _window_seconds(self) -> float:
        trades = list(self._trades)
        if len(trades) < 2:
            return 0.0
        return trades[-1].timestamp - trades[0].timestamp

    @property
    def history(self):
        return list(self._history)

    @property
    def n_trades_buffered(self):
        return len(self._trades)
