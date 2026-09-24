"""Terminal output for runkit, built on `rich`.

Two shared Consoles: `err` (stderr) for runkit's own chatter -- the start banner,
warnings, status -- so it never lands on stdout where an experiment may be writing
data; and `out` (stdout) for output the user explicitly asked to see, i.e. a
command whose whole point is the text it prints.

`run_started` is the panel runkit prints itself. The rest is a small
vocabulary of line helpers (`line`, `title`, `ok`, `warn`, `done`, ...) for ad-hoc
output, all left-padded by `PADDING_LEFT` so they line up.
"""
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table

err = Console(stderr=True)
out = Console()

PADDING_LEFT = 2


def silence() -> None:
    """Suppress all UI output. Call once at startup for a machine-readable mode."""
    global err, out
    err = Console(stderr=True, quiet=True)
    out = Console(quiet=True)


# ── line vocabulary (stderr) ─────────────────────────────────────────────────

def line(text: str = "", *, pad: int = PADDING_LEFT, highlight: bool = False) -> None:
    """One left-padded line (the base the helpers below build on)."""
    err.print(Padding(text, (0, pad), expand=False), highlight=highlight)


def title(text: str) -> None:
    line(f"⏵⏵ [bold]{text}[/bold]")


def info(text: str) -> None:
    line(text, highlight=True)


def detail(key: str, value) -> None:
    line(f"[dim]{key}:[/dim]  {value}")


def item(text: str) -> None:
    line(f"[dim]→[/dim]  {text}")


def ok(text: str) -> None:
    line(f"[green]✓[/green] {text}")


def fail(text: str) -> None:
    line(f"[red]✗[/red] {text}")


def warn(text: str) -> None:
    line(f"[yellow]⚠[/yellow] {text}")


def done(msg: str, hint: str = "") -> None:
    """Success block at the end of a command, with vertical breathing room."""
    body = f"[bold green]✓ {msg}[/bold green]"
    if hint:
        body += f"\n[dim]{hint}[/dim]"
    err.print(Padding(body, (1, PADDING_LEFT), expand=False))


def status(msg: str) -> Live:
    """A transient spinner; use as `with ui.status(...):` around slow work."""
    return Live(
        Padding(Spinner("dots", text=msg, style="magenta"), (0, PADDING_LEFT),
                expand=False),
        console=err, transient=True, refresh_per_second=12.5,
    )


# ── runkit's own panels ──────────────────────────────────────────────────────

def _kv(rows):
    """A right-aligned label / value grid for a handful of short fields."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="cyan", no_wrap=True)
    grid.add_column(overflow="fold")
    for label, value in rows:
        grid.add_row(label, str(value))
    return grid


def _cfg_block(cfg):
    """Render a plain (already-serialized) cfg dict as a yaml syntax block."""
    text = yaml.safe_dump(cfg, sort_keys=False).strip() or "{}"
    return Syntax(text, "yaml", background_color="default")


def run_started(*, name, run_id, run_dir, config_path, cfg):
    """Print the start banner: which run, where it writes, with what config.

    `cfg` is a plain dict (the resolved config, also frozen at `config_path`).
    """
    body = Group(
        _kv([("id", run_id), ("dir", run_dir), ("config", config_path)]),
        "", _cfg_block(cfg),
    )
    err.print(Panel(body, title=f"▶ runkit · {name}", title_align="left",
                    border_style="green", expand=False))
