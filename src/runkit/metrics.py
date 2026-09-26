"""Metrics: named values recorded over time, one JSON Lines file per stream.

    ctx.record(it=it, steps=n, ep_return=ret)   # the run  -> metrics/run.jsonl
    ctx.record("eval", ep_return=ret)           # in eval  -> metrics/eval.jsonl

Each call appends one line; runkit adds `time` and `elapsed_s` (since the run
started, or -- for a context opened from disk, as in eval -- since it was
opened). JSON Lines because calls may carry different keys, an append is
crash-safe (a killed run loses at most its last line), and it reads back with
`load_metrics` or `pandas.read_json(path, lines=True)`.
"""
import datetime
import json
import pathlib
import time

from .utils import plain

FOLDER = "metrics"
_ADDED = ("time", "elapsed_s")


def _check_stream(stream):
    if (not isinstance(stream, str) or not stream or stream.startswith(".")
            or "/" in stream or "\\" in stream):
        raise ValueError(f"bad metrics stream {stream!r}: a plain file name, "
                         f"not starting with '.'")


def append(ctx, stream, values):
    """Append one row to `{ctx.dir}/metrics/<stream>.jsonl`."""
    _check_stream(stream)
    clash = sorted(set(values) & set(_ADDED))
    if clash:
        raise ValueError(f"ctx.record: {clash} are added by runkit; use other names")
    t0 = ctx._live.t0 if ctx._live is not None else ctx._opened
    row = {"time": datetime.datetime.now().isoformat(timespec="milliseconds"),
           "elapsed_s": round(time.monotonic() - t0, 3) if t0 is not None else None,
           **plain(values)}
    folder = ctx.dir / FOLDER
    folder.mkdir(exist_ok=True)
    with open(folder / f"{stream}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


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
