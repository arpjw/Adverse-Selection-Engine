import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional
import websockets

@dataclass
class Trade:
    timestamp: float
    price: float
    quantity: float
    side: int
    trade_id: int

class BinanceTradeStream:
    WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "BTCUSDT", maxlen: int = 2000):
        self.symbol = symbol.lower()
        self.trades: deque[Trade] = deque(maxlen=maxlen)
        self._running = False
        self._callbacks = []
        self._last_price: Optional[float] = None
        self._reconnect_delay = 1.0

    def on_trade(self, fn):
        self._callbacks.append(fn)
        return fn

    def _parse(self, msg: dict) -> Optional[Trade]:
        try:
            price = float(msg["p"])
            qty   = float(msg["q"])
            side  = -1 if msg["m"] else +1
            t = Trade(
                timestamp=msg["T"] / 1000.0,
                price=price,
                quantity=qty,
                side=side,
                trade_id=int(msg["t"]),
            )
            self._last_price = price
            return t
        except (KeyError, ValueError):
            return None

    async def _connect(self):
        url = f"{self.WS_BASE}/{self.symbol}@trade"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._reconnect_delay = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        trade = self._parse(msg)
                        if trade:
                            self.trades.append(trade)
                            for cb in self._callbacks:
                                await cb(trade) if asyncio.iscoroutinefunction(cb) else cb(trade)
            except (websockets.ConnectionClosed, OSError):
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    def start(self, loop: asyncio.AbstractEventLoop):
        self._running = True
        return loop.create_task(self._connect())

    def stop(self):
        self._running = False

    @property
    def last_price(self) -> Optional[float]:
        return self._last_price

    def snapshot(self) -> list[Trade]:
        return list(self.trades)