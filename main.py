"""
Adverse Selection Spread Decomposition Engine

Usage:
    python main.py                        # BTC/USDT, live terminal dashboard
    python main.py --symbol ETHUSDT       # Different pair
    python main.py --symbol SOLUSDT --window 300  # Smaller trade window
    python main.py --export               # Also write estimates to CSV
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

from rich.live import Live

sys.path.insert(0, str(Path(__file__).parent))

from data.stream import BinanceTradeStream
from models.rolling import RollingDecomposer
from viz.dashboard import Dashboard


def parse_args():
    p = argparse.ArgumentParser(description="Adverse Selection Spread Decomposition Engine")
    p.add_argument("--symbol",     default="BTCUSDT",  help="Binance pair (default: BTCUSDT)")
    p.add_argument("--window",     type=int, default=500, help="Trade window size (default: 500)")
    p.add_argument("--history",    type=int, default=120, help="History buffer length (default: 120)")
    p.add_argument("--refresh",    type=float, default=1.0, help="Dashboard refresh seconds (default: 1.0)")
    p.add_argument("--export",     action="store_true", help="Export estimates to CSV")
    p.add_argument("--export-dir", default="exports", help="Export directory (default: exports)")
    return p.parse_args()


def make_csv_writer(symbol: str, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(export_dir, f"{symbol}_{ts}.csv")
    f = open(path, "w", newline="")
    fields = [
        "timestamp", "mid_price", "effective_spread",
        "adverse_selection", "inventory", "order_processing",
        "adverse_selection_share", "inventory_share", "order_processing_share",
        "kyle_lambda", "r_squared", "regime", "n_trades",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w, path


async def main():
    args = parse_args()
    symbol = args.symbol.upper()

    stream    = BinanceTradeStream(symbol=symbol, maxlen=args.window * 2)
    roller    = RollingDecomposer(trade_window=args.window, history_len=args.history)
    dashboard = Dashboard(symbol=symbol)

    csv_file = csv_writer = csv_path = None
    if args.export:
        csv_file, csv_writer, csv_path = make_csv_writer(symbol, args.export_dir)
        print(f"Exporting to: {csv_path}")

    estimate_count = 0

    def on_trade(trade):
        nonlocal estimate_count
        est = roller.push(trade)
        if est is None:
            return
        dashboard.update(est)
        estimate_count += 1
        if csv_writer and estimate_count % 10 == 0:
            c = est.components
            csv_writer.writerow({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mid_price": round(c.mid_price, 4),
                "effective_spread": round(c.effective_spread, 8),
                "adverse_selection": round(c.adverse_selection, 8),
                "inventory": round(c.inventory, 8),
                "order_processing": round(c.order_processing, 8),
                "adverse_selection_share": round(c.adverse_selection_share, 4),
                "inventory_share": round(c.inventory_share, 4),
                "order_processing_share": round(c.order_processing_share, 4),
                "kyle_lambda": round(c.kyle_lambda, 10),
                "r_squared": round(c.r_squared, 6),
                "regime": est.regime.value,
                "n_trades": c.n_trades,
            })

    stream.on_trade(on_trade)
    loop = asyncio.get_event_loop()

    ws_task = stream.start(loop)

    try:
        with Live(
            dashboard.render(),
            console=dashboard.console,
            refresh_per_second=1.0 / args.refresh,
            screen=True,
        ) as live:
            while True:
                await asyncio.sleep(args.refresh)
                live.update(dashboard.render())
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        ws_task.cancel()
        if csv_file:
            csv_file.close()
            print(f"\nExported to: {csv_path}")
        print("\nShutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())