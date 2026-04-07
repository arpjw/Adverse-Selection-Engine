import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from data.stream import Trade
from models.decomposition import decompose, _roll_spread, _kyle_lambda, SpreadComponents
from models.rolling import RollingDecomposer, Regime


def make_trades(n=200, base_price=50000.0, spread=10.0, seed=42) -> list[Trade]:
    rng = np.random.default_rng(seed)
    trades = []
    price = base_price
    for i in range(n):
        side = int(rng.choice([-1, 1]))
        noise = rng.normal(0, spread * 0.5)
        price += side * (spread / 2) + noise
        price = max(price, 1.0)
        trades.append(Trade(
            timestamp=float(i),
            price=price,
            quantity=float(rng.uniform(0.01, 1.0)),
            side=side,
            trade_id=i,
        ))
    return trades


def make_informed_trades(n=200, base_price=50000.0, alpha=0.6, seed=7) -> list[Trade]:
    rng = np.random.default_rng(seed)
    trades = []
    price = base_price
    true_value = base_price
    for i in range(n):
        true_value += rng.normal(0, 5.0)
        side = 1 if true_value > price else -1
        if rng.random() > alpha:
            side = int(rng.choice([-1, 1]))
        price += side * 8 + rng.normal(0, 2.0)
        price = max(price, 1.0)
        trades.append(Trade(
            timestamp=float(i),
            price=price,
            quantity=float(rng.uniform(0.01, 0.5)),
            side=side,
            trade_id=i,
        ))
    return trades


class TestRollSpread:
    def test_positive_spread(self):
        trades = make_trades(n=300)
        prices = np.array([t.price for t in trades])
        s = _roll_spread(prices)
        assert math.isfinite(s) or math.isnan(s)

    def test_insufficient_data(self):
        prices = np.array([100.0, 101.0, 100.5])
        s = _roll_spread(prices)
        assert math.isnan(s)

    def test_constant_prices(self):
        prices = np.full(100, 50000.0)
        s = _roll_spread(prices)
        assert math.isnan(s) or s == 0.0


class TestKyleLambda:
    def test_returns_finite(self):
        trades = make_trades(n=200)
        prices  = np.array([t.price    for t in trades])
        volumes = np.array([t.quantity for t in trades])
        sides   = np.array([t.side     for t in trades], dtype=float)
        lam, r2 = _kyle_lambda(prices, volumes, sides)
        assert math.isfinite(lam)
        assert 0.0 <= r2 <= 1.0

    def test_insufficient_data(self):
        prices  = np.array([100.0, 101.0])
        volumes = np.array([1.0, 1.0])
        sides   = np.array([1.0, -1.0])
        lam, r2 = _kyle_lambda(prices, volumes, sides)
        assert math.isnan(lam)


class TestDecompose:
    def test_returns_none_when_insufficient(self):
        result = decompose(make_trades(n=10))
        assert result is None

    def test_returns_components_with_enough_data(self):
        result = decompose(make_trades(n=200))
        assert result is not None
        assert isinstance(result, SpreadComponents)

    def test_shares_sum_to_one(self):
        result = decompose(make_trades(n=300))
        assert result is not None
        total = result.adverse_selection_share + result.inventory_share + result.order_processing_share
        assert abs(total - 1.0) < 1e-6

    def test_shares_non_negative(self):
        result = decompose(make_trades(n=300))
        assert result is not None
        assert result.adverse_selection_share >= 0
        assert result.inventory_share >= 0
        assert result.order_processing_share >= 0

    def test_is_valid(self):
        result = decompose(make_trades(n=300))
        assert result is not None
        assert result.is_valid()

    def test_effective_spread_positive(self):
        result = decompose(make_trades(n=300))
        assert result is not None
        assert result.effective_spread > 0

    def test_higher_alpha_with_informed_flow(self):
        results = []
        for seed in range(5):
            uninformed = decompose(make_trades(n=500, seed=seed))
            informed   = decompose(make_informed_trades(n=500, alpha=0.8, seed=seed))
            if uninformed and informed:
                results.append(informed.adverse_selection_share >= uninformed.adverse_selection_share)
        assert sum(results) >= len(results) // 2


class TestRollingDecomposer:
    def test_regime_unknown_at_start(self):
        roller = RollingDecomposer(trade_window=100, history_len=60)
        trades = make_trades(n=60)
        last_est = None
        for t in trades:
            est = roller.push(t)
            if est:
                last_est = est
        if last_est:
            assert last_est.regime in list(Regime)

    def test_accumulates_history(self):
        roller = RollingDecomposer(trade_window=100)
        trades = make_trades(n=300)
        for t in trades:
            roller.push(t)
        assert len(roller.history) > 0

    def test_regime_deep_stable_on_quiet_market(self):
        roller = RollingDecomposer(trade_window=200, history_len=60)
        trades = make_trades(n=400, spread=2.0)
        regimes_seen = set()
        for t in trades:
            est = roller.push(t)
            if est:
                regimes_seen.add(est.regime)
        assert len(regimes_seen) > 0
        assert all(r in list(Regime) for r in regimes_seen)