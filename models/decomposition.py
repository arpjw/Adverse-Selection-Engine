from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from data.stream import Trade


@dataclass
class SpreadComponents:
    effective_spread: float
    adverse_selection: float
    inventory: float
    order_processing: float
    adverse_selection_share: float
    inventory_share: float
    order_processing_share: float
    kyle_lambda: float
    r_squared: float
    n_trades: int
    mid_price: float

    def is_valid(self) -> bool:
        return (
            math.isfinite(self.effective_spread)
            and math.isfinite(self.adverse_selection)
            and self.effective_spread > 0
            and 0.0 <= self.adverse_selection_share <= 1.0
        )


def _roll_spread(prices: np.ndarray) -> float:
    dp = np.diff(prices)
    if len(dp) < 4:
        return float("nan")
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov >= 0:
        return float("nan")
    return 2.0 * math.sqrt(-cov)


def _kyle_lambda(prices: np.ndarray, volumes: np.ndarray, sides: np.ndarray):
    dp = np.diff(prices)
    signed_vol = volumes[:-1] * sides[:-1]
    if len(dp) < 10:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(signed_vol)), signed_vol])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, dp, rcond=None)
        lam = coeffs[1]
        y_hat = X @ coeffs
        ss_res = np.sum((dp - y_hat) ** 2)
        ss_tot = np.sum((dp - dp.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(lam), float(r2)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


def _huang_stoll(prices, sides, roll_spread, window=5):
    if roll_spread <= 0 or not math.isfinite(roll_spread):
        return float("nan"), float("nan"), float("nan")
    dp = np.diff(prices)
    half_spread = roll_spread / 2.0
    if len(dp) < window + 2:
        return float("nan"), float("nan"), float("nan")
    y = dp[1:]
    d_curr = sides[1:len(dp)]
    d_prev = sides[:len(dp)-1]
    X = np.column_stack([d_curr, d_prev])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeff_curr, coeff_prev = coeffs
        alpha_plus_beta = coeff_curr / half_spread
        beta = -coeff_prev / half_spread
        alpha = alpha_plus_beta - beta
        alpha = float(np.clip(alpha, 0.0, 1.0))
        beta  = float(np.clip(beta,  0.0, 1.0 - alpha))
        gamma = max(0.0, 1.0 - alpha - beta)
        return alpha, beta, gamma
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")


def decompose(trades, min_trades=50):
    if len(trades) < min_trades:
        return None
    prices  = np.array([t.price    for t in trades], dtype=np.float64)
    volumes = np.array([t.quantity for t in trades], dtype=np.float64)
    sides   = np.array([t.side     for t in trades], dtype=np.float64)
    mid_price = float(np.median(prices))
    roll_s    = _roll_spread(prices)
    lam, r2   = _kyle_lambda(prices, volumes, sides)
    alpha, beta, gamma = _huang_stoll(prices, sides, roll_s)
    if not math.isfinite(roll_s) or roll_s <= 0:
        roll_s = float(np.mean(np.abs(np.diff(prices))) * 2)
    if not all(math.isfinite(x) for x in [alpha, beta, gamma]):
        alpha, beta, gamma = 0.33, 0.33, 0.34
    return SpreadComponents(
        effective_spread=roll_s,
        adverse_selection=alpha * roll_s,
        inventory=beta * roll_s,
        order_processing=gamma * roll_s,
        adverse_selection_share=alpha,
        inventory_share=beta,
        order_processing_share=gamma,
        kyle_lambda=lam if math.isfinite(lam) else 0.0,
        r_squared=r2 if math.isfinite(r2) else 0.0,
        n_trades=len(trades),
        mid_price=mid_price,
    )
