"""Metrics: named values recorded over time, one JSON Lines file per stream.

    ctx.record(it=it, steps=n, ep_return=ret)   # the run  -> metrics/run.jsonl
    ctx.record("eval", ep_return=ret)           # in eval  -> metrics/eval.jsonl

Each call appends one line; runkit adds `_time` and `_elapsed_s` (since the run
started, or -- for a context opened from disk, as in eval -- since it was
opened). Keys starting with `_` are runkit's: added automatically, and refused
from `ctx.record`, so `time` and any other plain name stay free for yours. JSON Lines because calls may carry different keys, an append is
crash-safe (a killed run loses at most its last line), and it reads back as
rows (`load_metrics`), as columns (`compile_metrics`), or with
`pandas.read_json(path, lines=True)`.
"""
import datetime
import json
import pathlib
import time

import numpy as np

from .utils import plain

FOLDER = "metrics"
RESERVED_PREFIX = "_"          # keys starting with it are runkit's


def _check_stream(stream):
    if (not isinstance(stream, str) or not stream or stream.startswith(".")
            or "/" in stream or "\\" in stream):
        raise ValueError(f"bad metrics stream {stream!r}: a plain file name, "
                         f"not starting with '.'")


def append(ctx, stream, values):
    """Append one row to `{ctx.dir}/metrics/<stream>.jsonl`."""
    _check_stream(stream)
    clash = sorted(k for k in values if k.startswith(RESERVED_PREFIX))
    if clash:
        raise ValueError(f"ctx.record: {clash} -- keys starting with "
                         f"'{RESERVED_PREFIX}' are runkit's; use other names")
    t0 = ctx._live.t0 if ctx._live is not None else ctx._opened
    row = {"_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
           "_elapsed_s": round(time.monotonic() - t0, 3) if t0 is not None else None,
           **plain(values)}
    folder = ctx.dir / FOLDER
    folder.mkdir(exist_ok=True)
    with open(folder / f"{stream}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    if ctx._live is not None:               # counted for checkpoint.yaml
        ctx._live.records[stream] = ctx._live.records.get(stream, 0) + 1
        if stream == "run":                 # summarized at the next checkpoint
            _add_to_window(ctx._live.window, row)


def load_metrics(run_dir, stream="run"):
    """The rows of one stream, in order, as dicts. A line that does not parse
    (the last one of a killed run, say) is skipped. [] if there is none."""
    path = pathlib.Path(run_dir) / FOLDER / f"{stream}.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def compile_metrics(run_dir, stream="run", *, x=None, start=None, end=None, x_end=None):
    """One stream as columns: `{key: array}`, one entry per line, in order.

        m = compile_metrics(run_dir)
        m["loss"]                                    # a column
        m["loss"][-1000:]                            # its last 1000 entries
        m["loss"][m["steps"] > 3e6]                  # a condition on another column
        plt.plot(m["_line"], m["loss"])              # against the line number
        plt.plot(m["_elapsed_s"], m["ep_return"])    # against time
        pandas.DataFrame(m)                          # if you have pandas

    Every column is as long as the stream, so they all line up: the same slice
    or mask selects the same lines in each. A line without a key leaves a gap,
    so a sparse key (an eval every tenth iteration) stays aligned. Numeric
    columns -- numbers, and nulls -- are float arrays with NaN for gaps
    (matplotlib leaves gaps at NaN; `ok = ~np.isnan(y)` drops them); any other
    column (strings, booleans, lists, like `_time`) is an object array with
    None for gaps -- an array too, so slices and masks work on every column.
    `_line` is added here, not stored: each line's number in the file, so it
    stays right through any slicing. Keys come in the order they first appear,
    `_line` first. {} if the stream has no lines.

    Slicing covers row ranges and conditions. `start` / `end` do what it
    cannot: an inclusive window on the x axis -- `_line`, or the column `x` --
    where each bound is a value, negative counting back from the end of the axis
    (`x_end`, default its largest value): `start=-1000` is the last 1000 lines,
    or with `x="steps"` the last 1000 steps. Or a bound is a checkpoint name (a
    name wins over a number that reads the same, as `"000003"`): the rows after /
    before that checkpoint completed, from its `checkpoint.yaml`.
    """
    columns = _columns(load_metrics(run_dir, stream))
    return _select(run_dir, stream, columns, x=x, start=start, end=end, x_end=x_end)


def _columns(lines):
    keys = list(dict.fromkeys(k for row in lines for k in row))
    columns = {"_line": np.arange(len(lines), dtype=float)} if lines else {}
    for k in keys:
        values = [row.get(k) for row in lines]
        if all(v is None or _is_number(v) for v in values):
            columns[k] = np.array([np.nan if v is None else float(v) for v in values])
        else:
            col = np.empty(len(values), dtype=object)
            for i, v in enumerate(values):     # element-wise: a value may itself be a list
                col[i] = v
            columns[k] = col
    return columns


def _select(run_dir, stream, columns, *, rows=None, x=None, start=None, end=None, x_end=None):
    """The same lines of every column: a window (start / end), then a slice."""
    if not columns:
        return columns
    if start is not None or end is not None:
        keep = _window(run_dir, stream, columns, x, start, end, x_end)
        columns = {k: c[keep] for k, c in columns.items()}
    if rows is not None:
        rows = parse_rows(rows)
        columns = {k: c[rows] for k, c in columns.items()}
    return columns


def parse_rows(rows):
    """"-1000:" / "200:800" / slice -> slice. Python slice semantics."""
    if isinstance(rows, slice):
        return rows
    parts = str(rows).split(":")
    if len(parts) != 2:
        raise ValueError(f"rows: expected a slice like '-1000:' or '200:800', got {rows!r}")
    try:
        a, b = (int(p) if p.strip() else None for p in parts)
    except ValueError:
        raise ValueError(f"rows: expected integers, got {rows!r}") from None
    return slice(a, b)


def _window(run_dir, stream, columns, x, start, end, x_end):
    """Boolean mask of the lines inside [start, end] on the x axis."""
    line = columns["_line"]
    axis = _numeric(columns, x or "_line", stream)
    axis_end = x_end if x_end is not None else np.nanmax(axis)
    keep = np.ones(len(line), dtype=bool)
    for bound, is_start in ((start, True), (end, False)):
        if bound is None:
            continue
        rows_at = _checkpoint_rows(run_dir, stream, bound)
        if rows_at is not None:              # a checkpoint: a boundary between lines
            keep &= (line >= rows_at) if is_start else (line < rows_at)
            continue
        v = _as_number(bound)
        if v < 0:
            v = axis_end + v
        with np.errstate(invalid="ignore"):
            keep &= (axis >= v) if is_start else (axis <= v)
    return keep


def _checkpoint_rows(run_dir, stream, name):
    """The row count of `stream` when checkpoint `name` completed, or None if
    there is no checkpoint of that name."""
    from .checkpoints import load_checkpoints
    for c in load_checkpoints(run_dir):
        if c.name == str(name):
            if stream not in c.metrics:
                raise ValueError(
                    f"checkpoint {c.name!r} has no row count for stream {stream!r} (written "
                    f"by another process, or before it existed); use a value of an x key")
            return c.metrics[stream]
    return None


def _as_number(v):
    if _is_number(v):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"start/end: {v!r} is neither a number nor a checkpoint name") from None


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def streams(run_dir):
    """The metrics streams a run has, by name."""
    folder = pathlib.Path(run_dir) / FOLDER
    return sorted(f.stem for f in folder.glob("*.jsonl")) if folder.is_dir() else []


def summarize_metrics(run_dir, stream="run", *, rows=None, x=None, start=None, end=None):
    """Per key of one stream: how many lines have it, and its last / min / max /
    mean over the values it has (numeric keys only; None otherwise). What `runkit metrics` prints. `x`,
    `start`, `end` as for `compile_metrics`; then `rows`, a slice of the lines
    (`"-1000:"`)."""
    m = _select(run_dir, stream, _columns(load_metrics(run_dir, stream)),
                rows=rows, x=x, start=start, end=end)
    out = []
    for k, col in m.items():
        if k == "_line":                     # the index itself: nothing to summarize
            continue
        if col.dtype == float:
            vals = col[~np.isnan(col)]
            out.append({"key": k, "rows": len(vals),
                        "last": vals[-1].item() if len(vals) else None,
                        "min": vals.min().item() if len(vals) else None,
                        "mean": vals.mean().item() if len(vals) else None,
                        "max": vals.max().item() if len(vals) else None})
        else:
            present = [v for v in col if v is not None]
            out.append({"key": k, "rows": len(present), "last": present[-1] if present else None,
                        "min": None, "mean": None, "max": None})
    return out


def _series(spec):
    """"key" -> ("run", "key");  "eval:key" -> ("eval", "key")."""
    stream, sep, key = spec.partition(":")    # the first ':': keys may contain one
    return (stream, key) if sep else ("run", spec)


def plot_metrics(run_dir, ys, x=None, out=None, *, rows=None, start=None, end=None):
    """Plot metrics of a run to a PNG; returns its path.

    `ys`: keys to plot, each `key` (the `run` stream) or `stream:key`, drawn
    on one axis. Empty (or a lone `stream:`): every numeric key of that stream
    (default `run`), one subplot each -- the quick overview of a run.
    `x`: a key to plot against; each series uses the `x` of its own stream, so
    streams need not line up -- with `x="steps"`, a training and an eval return
    land on one axis. Without `x`, the line number, which only means the same
    thing within one stream, so series from several streams need an `x`.
    `out`: the file; default `{run_dir}/metrics/<ys>_vs_<x>[_<range>].png`, so
    the same plot redrawn overwrites itself.
    `rows`, `start`, `end`: which lines, as for `compile_metrics`; a negative
    `start` / `end` counts back from the largest x across all the series, so
    several streams share one window.
    """
    import matplotlib                       # imported here: a run never loads it
    matplotlib.use("Agg")                   # no display needed (clusters, ssh)
    import matplotlib.pyplot as plt
    run_dir = pathlib.Path(run_dir)
    if isinstance(ys, str):
        ys = [ys]
    ys = list(ys or [])
    if not ys or (len(ys) == 1 and ys[0].endswith(":")):
        stream = ys[0][:-1] if ys else "run"
        return _plot_all(plt, run_dir, stream or "run", x, out, rows, start, end)
    series = [_series(s) for s in ys]
    if any(key.endswith("/") for _, key in series):      # `loss/`: every `loss/...` key
        series, ys = _expand_groups(run_dir, series)
    if x is None and len({stream for stream, _ in series}) > 1:
        raise ValueError("plot: series from several streams need an x key they share, "
                         "e.g. --x steps (x='steps'), or --x _elapsed_s -- line numbers "
                         "of different streams do not line up")

    if rows is not None and len({stream for stream, _ in series}) > 1:
        from . import ui
        ui.warn("rows are counted per stream: the same rows of two streams cover "
                "different stretches -- a window on a shared x (start/end) lines them up")
    full = {}
    for stream in dict.fromkeys(s for s, _ in series):
        full[stream] = _columns(load_metrics(run_dir, stream))
        if not full[stream]:
            raise ValueError(f"plot: no stream {stream!r} in {run_dir / FOLDER} "
                             f"(streams: {', '.join(streams(run_dir)) or 'none'})")
    # a negative start / end counts back from the end of the axis over all series
    x_end = max(np.nanmax(_numeric(c, x or "_line", s)) for s, c in full.items())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for spec, (stream, key) in zip(ys, series):
        cols = _select(run_dir, stream, full[stream], rows=rows, x=x, start=start,
                       end=end, x_end=x_end)
        _numeric(full[stream], key, stream)             # a missing key is an error, not a gap
        if not cols:
            continue
        yv, xv = cols[key], cols[x or "_line"]
        ok = ~np.isnan(yv) & ~np.isnan(xv)
        ax.plot(xv[ok], yv[ok], marker="o", ms=_marker_size(ok.sum()),
                label=spec if stream != "run" or len(series) > 1 else key)
    ax.set_xlabel(x or "line (record call)")
    keys = {key for _, key in series}
    ax.set_ylabel(keys.pop() if len(keys) == 1 else "value")   # one key, maybe from several streams
    if len(ys) > 1:
        ax.legend()
    ax.set_title(run_dir.resolve().name, fontsize=9)
    ax.grid(alpha=0.3)
    name = "__".join(s.replace(":", "-") for s in ys)
    return _save(plt, fig, out, run_dir, name, x, rows, start, end)


def _expand_groups(run_dir, series):
    """(stream, "loss/") -> (stream, "loss/train"), (stream, "loss/eval"), ..."""
    out = []
    for stream, key in series:
        if not key.endswith("/"):
            out.append((stream, key))
            continue
        have = [k for k in _columns(load_metrics(run_dir, stream)) if k.startswith(key)]
        if not have:
            raise ValueError(f"plot: no keys {key}... in stream {stream!r}")
        out += [(stream, k) for k in have]
    return out, [k if s == "run" else f"{s}:{k}" for s, k in out]


def _plot_all(plt, run_dir, stream, x, out, rows, start, end):
    """Every numeric key of `stream` in its own subplot, against x or the line."""
    full = _columns(load_metrics(run_dir, stream))
    if not full:
        raise ValueError(f"plot: no stream {stream!r} in {run_dir / FOLDER} "
                         f"(streams: {', '.join(streams(run_dir)) or 'none'})")
    axis_key = x or "_line"
    _numeric(full, axis_key, stream)
    keys = [k for k, c in full.items()
            if not k.startswith("_") and k != x and c.dtype == float]
    if not keys:
        raise ValueError(f"plot: stream {stream!r} has no numeric keys to plot")
    cols = _select(run_dir, stream, full, rows=rows, x=x, start=start, end=end)
    groups = _groups(keys)
    ncols = min(3, len(groups))
    nrows = -(-len(groups) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.7 * nrows),
                             sharex=True, squeeze=False)
    for i, (title, members) in enumerate(groups.items()):
        ax = axes.flat[i]
        for key in members:
            if cols:
                yv, xv = cols[key], cols[axis_key]
                ok = ~np.isnan(yv) & ~np.isnan(xv)
                ax.plot(xv[ok], yv[ok], marker="o", ms=_marker_size(ok.sum()), lw=1,
                        label=key.rsplit("/", 1)[-1] if "/" in key else key)
        if len(members) > 1:
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    n = len(groups)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:                                   # x label on the bottom row
        ax.set_xlabel(x or "line (record call)", fontsize=8)
    for i in range(n, nrows * ncols):                     # a hidden panel: label the one above
        axes.flat[i - ncols].set_xlabel(x or "line (record call)", fontsize=8)
        axes.flat[i - ncols].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle(f"{run_dir.resolve().name} · {FOLDER}/{stream}.jsonl", fontsize=9)
    return _save(plt, fig, out, run_dir, "all" if stream == "run" else f"{stream}-all",
                 x, rows, start, end)


def _groups(keys):
    """Keys by their prefix before the last `/`, in order: `loss/train` and
    `loss/eval` share the subplot `loss`; a key without `/` is its own (and
    joins a group of its name: `loss` with `loss/eval`)."""
    groups = {}
    for k in keys:
        groups.setdefault(k.rsplit("/", 1)[0] if "/" in k else k, []).append(k)
    return groups


def _marker_size(n):
    """Every point gets a marker -- each is a record call -- smaller as they
    crowd, so a long series stays a readable line."""
    return 3.5 if n <= 50 else 2.0 if n <= 500 else 1.0


def _save(plt, fig, out, run_dir, name, x, rows, start, end):
    """Save to `out`, or to `{run_dir}/metrics/<name>_vs_<x>[_<range>].png`."""
    if out is None:
        name += f"_vs_{x or 'line'}"
        if rows is not None:
            name += f"_rows{rows if isinstance(rows, str) else f'{rows.start}:{rows.stop}'}"
        if start is not None:
            name += f"_start{start}"
        if end is not None:
            name += f"_end{end}"
        out = run_dir / FOLDER / f"{_safe(name)}.png"
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _numeric(cols, key, stream):
    if key not in cols:
        raise ValueError(f"plot: no key {key!r} in stream {stream!r} "
                         f"(keys: {', '.join(cols)})")
    if cols[key].dtype != float:              # an object column: strings, booleans, lists
        raise ValueError(f"plot: key {key!r} in stream {stream!r} is not numeric")
    return cols[key]


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)[:150]


# -- the summary a checkpoint gives of the rows since the one before -----------

def _add_to_window(window, row):
    """Running per-key state for the rows since the last checkpoint: a sum and
    count over the key's numeric values (None / NaN / missing are not counted),
    its last value, and whether it is a counter so far -- integers, strictly
    increasing (`it`, `steps`). A float metric is never a counter, however it
    moves: a return that improves every row is still averaged."""
    window["_rows"] = window.get("_rows", 0) + 1
    for k, v in row.items():
        if k.startswith("_") or not _is_number(v) or v != v:      # runkit's keys; None; NaN
            continue
        s = window.get(k)
        if s is None:
            window[k] = {"sum": float(v), "n": 1, "last": v, "counter": isinstance(v, int)}
        else:
            s["counter"] = s["counter"] and isinstance(v, int) and v > s["last"]
            s["sum"] += v
            s["n"] += 1
            s["last"] = v


def summarize_window(window):
    """{"rows", "mean", "last"} of a window: counters (`it`, `steps`) by their
    last value, the rest by their mean over the values they had. None if nothing
    was recorded."""
    rows = window.get("_rows", 0)
    if not rows:
        return None
    keys = {k: s for k, s in window.items() if k != "_rows"}
    return {"rows": rows,
            "mean": {k: s["sum"] / s["n"] for k, s in keys.items() if not s["counter"]},
            "last": {k: s["last"] for k, s in keys.items() if s["counter"]}}
