from __future__ import annotations
import math
import time
from collections import deque
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from models.rolling import RollingEstimate, Regime, REGIME_COLORS
from models.decomposition import SpreadComponents

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_WIDTH = 40

REGIME_STYLE = {
    Regime.DEEP_STABLE:    "bold green",
    Regime.THIN_ONE_SIDED: "bold red",
    Regime.FRAGMENTED:     "bold yellow",
    Regime.UNKNOWN:        "dim white",
}


def _sparkline(values, width=30):
    if len(values) < 2:
        return "─" * width
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    rng = hi - lo or 1e-9
    return "".join(SPARK_CHARS[round((v - lo) / rng * (len(SPARK_CHARS) - 1))] for v in tail)


def _stacked_bar(alpha, beta, gamma, width=BAR_WIDTH):
    a_w = round(alpha * width)
    b_w = round(beta  * width)
    g_w = max(0, width - a_w - b_w)
    t = Text()
    t.append("█" * a_w, style="bold red")
    t.append("█" * b_w, style="bold yellow")
    t.append("█" * g_w, style="bold blue")
    return t


def _bps(value, mid):
    if mid <= 0 or not math.isfinite(value):
        return "—"
    return f"{value / mid * 10_000:.2f}"


class Dashboard:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.console = Console()
        self._last: Optional[RollingEstimate] = None
        self._alpha_history = deque(maxlen=60)
        self._trade_count = 0
        self._start_time = time.time()

    def update(self, est: RollingEstimate):
        self._last = est
        self._alpha_history.append(est.components.adverse_selection_share)
        self._trade_count += 1

    def render(self) -> Panel:
        if self._last is None:
            return Panel(
                Text("Waiting for data…", style="dim"),
                title=f"[bold]Adverse Selection Engine[/bold]  ·  {self.symbol}",
                border_style="bright_black",
            )

        est = self._last
        comp = est.components
        elapsed = time.time() - self._start_time

        header = Table.grid(expand=True, padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="center")
        header.add_column(justify="center")
        header.add_column(justify="right")
        header.add_row(
            Text(f"${comp.mid_price:,.2f}", style="bold white"),
            Text(f"{self._trade_count:,} trades", style="dim"),
            Text(f"{est.window_seconds:.0f}s window", style="dim"),
            Text(f"↑ {elapsed:.0f}s uptime", style="dim"),
        )

        bar = _stacked_bar(comp.adverse_selection_share, comp.inventory_share, comp.order_processing_share)
        legend = Text()
        legend.append("  ██ ", style="bold red")
        legend.append("Adverse sel  ", style="dim")
        legend.append("██ ", style="bold yellow")
        legend.append("Inventory  ", style="dim")
        legend.append("██ ", style="bold blue")
        legend.append("Order proc", style="dim")

        tbl = Table(box=box.SIMPLE, expand=True, show_header=True, padding=(0, 1))
        tbl.add_column("Component",   style="dim",     width=22)
        tbl.add_column("Half-spread", justify="right", width=14)
        tbl.add_column("Basis pts",   justify="right", width=12)
        tbl.add_column("Share",       justify="right", width=10)
        tbl.add_column("Z-score",     justify="right", width=10)

        mid = comp.mid_price
        tbl.add_row(Text("Adverse selection",     style="bold red"),   f"${comp.adverse_selection:.6f}", _bps(comp.adverse_selection, mid),  f"{comp.adverse_selection_share:.1%}", f"{est.alpha_zscore:+.2f}σ")
        tbl.add_row(Text("Inventory cost",        style="bold yellow"),f"${comp.inventory:.6f}",         _bps(comp.inventory, mid),           f"{comp.inventory_share:.1%}",         "—")
        tbl.add_row(Text("Order processing",      style="bold blue"),  f"${comp.order_processing:.6f}",  _bps(comp.order_processing, mid),    f"{comp.order_processing_share:.1%}",  "—")
        tbl.add_row(Text("Effective spread (Roll)",style="bold white"), f"${comp.effective_spread:.6f}", _bps(comp.effective_spread, mid),    "100%",                                "—")

        stats = Table.grid(expand=True, padding=(0, 3))
        stats.add_column(justify="left")
        stats.add_column(justify="left")
        stats.add_column(justify="right")
        stats.add_row(
            Text(f"Kyle λ  {comp.kyle_lambda:.2e}", style="cyan"),
            Text(f"R²  {comp.r_squared:.3f}", style="dim"),
            Text(f"Regime: {est.regime.value}", style=REGIME_STYLE.get(est.regime, "white")),
        )

        spark_vals = list(self._alpha_history)
        spark_line = Text()
        spark_line.append("α share  ", style="dim")
        spark_line.append(_sparkline(spark_vals, width=50), style="bold red")
        if spark_vals:
            spark_line.append(f"  {spark_vals[-1]:.1%}", style="dim")

        return Panel(
            Group(header, Text(""), bar, legend, Text(""), tbl, stats, Text(""), spark_line),
            title=f"[bold]Adverse Selection Engine[/bold]  ·  [cyan]{self.symbol}[/cyan]",
            subtitle=f"[dim]n={comp.n_trades}  Roll (1984) · Huang-Stoll (1997) · Kyle (1985)[/dim]",
            border_style="bright_black",
            padding=(1, 2),
        )
