"""Terminal output for runkit, built on `rich`.

Two shared Consoles: `err` (stderr) for runkit's own chatter -- the start banner,
warnings, status -- so it never lands on stdout where an experiment may be writing
data; and `out` (stdout) for output the user explicitly asked to see, i.e. a
command whose whole point is the text it prints.

`run_started` / `run_finished` bracket a run; `opened` heads an eval / viz.
The rest is a small vocabulary of line helpers (`line`, `title`, `ok`, `warn`,
`done`, ...) for ad-hoc output, all left-padded by `PADDING_LEFT` so they line up.

Everything written to `err` first flushes stdout. When stdout is not a terminal
(piped, a log file) python buffers it while stderr goes out at once, so without
the flush a body's prints can land after runkit's closing line in the log.
Paths are shown short (`short_path`): relative to the current folder when below
it, else under `~`, else absolute.
"""
import os
import pathlib
import sys

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

err = Console(stderr=True)
out = Console()

PADDING_LEFT = 2


def silence() -> None:
    """Suppress all UI output. Call once at startup for a machine-readable mode."""
    global err, out
    err = Console(stderr=True, quiet=True)
    out = Console(quiet=True)


# ── line vocabulary (stderr) ─────────────────────────────────────────────────

def _flush_stdout():
    """Let what the experiment printed so far out first, so the log keeps the
    order things happened in."""
    try:
        sys.stdout.flush()
    except Exception:                                        # noqa: BLE001
        pass


def short_path(p):
    """A path as short as it can be shown unambiguously: relative to the current
    folder when it is below it, else `~/...` when under home, else absolute."""
    p = pathlib.Path(p)
    try:
        rel = os.path.relpath(p)
        if not rel.startswith(".."):
            return rel
    except ValueError:                       # another drive (windows)
        pass
    try:
        return "~/" + str(p.relative_to(pathlib.Path.home()))
    except ValueError:
        return str(p)


def line(text: str = "", *, pad: int = PADDING_LEFT, highlight: bool = False) -> None:
    """One left-padded line (the base the helpers below build on)."""
    _flush_stdout()
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
        grid.add_row(label, value if not isinstance(value, str) else escape(value))
    return grid


def _cfg_block(cfg):
    """Render a plain (already-serialized) cfg dict as a yaml syntax block."""
    text = yaml.safe_dump(cfg, sort_keys=False).strip() or "{}"
    return Syntax(text, "yaml", background_color="default")


def opened(*, name, verb, run_dir):
    """One line when eval/viz opens an existing run: which, and for what."""
    line(f"[bold green]▶ runkit · {name} · {verb}[/bold green]  "
         f"[dim]{escape(short_path(run_dir))}[/dim]")


def _launch_text(launch):
    """meta.yaml's `launch` as one short line: how the process was started."""
    parts = []
    if launch.get("project"):
        parts.append("uv")
    if launch.get("extras"):
        parts.append("extras " + ", ".join(launch["extras"]))
    if launch.get("vars"):
        parts.append("vars " + ", ".join(launch["vars"]))
    return " · ".join(parts)


def run_started(*, name, run_id, run_dir, tag, changes, n_fields, launch=None):
    """Print the start banner: which run, where it writes, how it was launched,
    what it changes -- a title line, then a label / value block, no frame.

    Only the config fields that differ from the dataclass defaults are shown
    (`changes`, dotted keys) -- a large config would otherwise fill the screen,
    and the whole of it is in `{run_dir}/config.yaml` anyway.
    """
    rest = n_fields - len(changes)
    if not changes:
        config = Text(f"all {n_fields} fields at their defaults", style="dim")
    else:
        note = f"{rest} more at their defaults · " if rest else ""
        config = Group(_cfg_block(changes), f"[dim]{note}the full resolved config is "
                                            f"in the run dir's config.yaml[/dim]")
    rows = [("id", run_id), *([("tag", tag)] if tag else []),
            ("dir", short_path(run_dir)),
            *([("launch", _launch_text(launch))] if launch else []),
            ("config", config)]
    _flush_stdout()
    err.print(Padding(Group(f"[bold green]▶ runkit · {escape(name)}[/bold green]",
                            Padding(_kv(rows), (0, 0, 0, 2))),
                      (0, PADDING_LEFT), expand=False))


def _duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def checkpoint_saved(*, path, elapsed_s, info, progress=None, total=None, summary=None):
    """Lines when a checkpoint is complete: its folder (relative to the run dir),
    how far into the run, the progress and its info; then, per metrics stream
    the run recorded to since the last checkpoint (`run` first), those rows --
    means, and counters by their last value. Each shortened if long."""
    text = f"[cyan]◆[/cyan] checkpoint  {escape(path)}  [dim]at {_duration(elapsed_s)}"
    if progress is not None or total:
        text += "  " + escape(_progress_text(progress or 0, total))
    if info:
        text += "  " + escape(_clip("  ".join(f"{k}={_num(v)}" for k, v in info.items())))
    line(text + "[/dim]")
    summary = summary or {}
    if "rows" in summary:                    # one stream's summary, as older records have
        summary = {"run": summary}
    width = max((len(s) for s in summary), default=0)
    for stream, s in summary.items():
        means = "  ".join(f"{k} {_num(v)}" for k, v in s["mean"].items())
        lasts = "  ".join(f"{k} {_num(v)}" for k, v in s["last"].items())
        rows = s["rows"]
        body = "  ·  ".join(p for p in (means, lasts) if p)
        line(f"[dim]  {escape(stream.ljust(width))}  {rows} row{'s' if rows != 1 else ''} "
             f"since the last: {escape(_clip(body, 100))}[/dim]")


def _progress_text(progress, total):
    if not total:
        return f"progress {_num(progress)}"
    return f"{_num(progress)} / {_num(total)}  {100 * progress / total:.0f}%"


def _num(v):
    """A number, short: 2.66M, 12.3k, 0.0213, 260."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return str(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.3g}M"
    if a >= 1e4:
        return f"{v / 1e3:.3g}k"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.3g}"


def _clip(text, n=80):
    return text if len(text) <= n else text[:n - 3] + "..."


def run_finished(*, run_id, status, duration_s, error, run_dir):
    """One line at the end of a run: how it went, how long, where it is."""
    dur, where = _duration(duration_s), escape(short_path(run_dir))
    if status == "ok":
        ok(f"{run_id}  [green]ok[/green] in {dur}  [dim]→ {where}[/dim]")
    elif status == "interrupted":
        warn(f"{run_id}  [yellow]interrupted[/yellow] after {dur}  [dim]→ {where}[/dim]")
    else:
        fail(f"{run_id}  [red]failed[/red] after {dur}  ({escape(str(error))})  "
             f"[dim]→ {where}/traceback.txt[/dim]")
