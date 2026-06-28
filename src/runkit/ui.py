"""Terminal output for runkit, built on `rich`.

Two shared Consoles: `err` (stderr) for runkit's own chatter -- the start banner,
warnings -- so it never lands on stdout where an experiment may be writing data;
and `out` (stdout) for output the user explicitly asked to see (`--dry-run`).

Helpers just render; they hold no state beyond the consoles. Anything domain-ish
(serializing a cfg into a plain dict) happens before it reaches here.
"""
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

err = Console(stderr=True)
out = Console()


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


def run_started(*, name, run_id, out_dir, config_path, cfg):
    """Print the start banner: which run, where it writes, with what config.

    `cfg` is a plain dict (the resolved config, also frozen at `config_path`).
    """
    paths = _kv([("id", run_id), ("out", out_dir), ("config", config_path)])
    body = Group(paths, "", _cfg_block(cfg))
    err.print(Panel(body, title=f"▶ runkit · {name}", title_align="left",
                    border_style="green", expand=False))


def dry_run(*, config_file, cfg, flags):
    """Print the resolved plan for `--dry-run` (to stdout -- it's the deliverable)."""
    body = Group(
        _kv([("config file", config_file), ("flags", flags or "{}")]),
        "", _cfg_block(cfg),
    )
    out.print(Panel(body, title="runkit · dry-run", title_align="left",
                    border_style="yellow", expand=False))


def warn(message):
    """A styled, non-fatal warning on stderr."""
    err.print(f"[yellow]⚠  runkit:[/] {message}")
