"""Following a run as it goes: `runkit metrics RUN_DIR [KEY ...] --follow`.

Prints a stream's last rows as a table, then each new row as it is appended,
the run's checkpoints as they complete, and runkit's closing line when the run
ends -- the same events its own terminal shows, from any other terminal or
machine that sees the run dir. It polls: the stream file from where it last
read (only whole lines; a line still being written waits for the next poll),
and `status.yaml` for checkpoints and the end. Ctrl-C stops following, never
the run.
"""
import json
import os
import socket
import time

import yaml

from . import ui
from .metrics import FOLDER

POLL_S = 1.0          # how often to look for new rows
HEADER_EVERY = 40     # reprint the header every so many rows, so it stays in view


def follow(run_dir, stream="run", keys=None, *, first=None, poll=POLL_S, sleep=time.sleep):
    """Print rows of `stream` as they come, until the run ends. Returns the
    run's final status (`ok`, `failed`, `interrupted`), or `gone` if it is left
    at `running` with no process behind it on this host.

    `keys`: the columns (default: every key but runkit's `_` ones, as they
    appear). `first`: which of the rows already there to print first, a slice
    (default: the last 10). `sleep` is for tests.
    """
    path = run_dir / FOLDER / f"{stream}.jsonl"
    table = _Table(keys)
    rows, offset = _read_from(path, 0)
    count = len(rows)                       # rows of the stream so far, printed or not
    shown = rows[first if first is not None else slice(-10, None)]
    table.size(shown)                       # one header for the rows shown first
    for row in shown:
        table.row(row)
    seen = _checkpoint_seen(run_dir)
    while True:
        status = _status(run_dir)
        # a new checkpoint goes where it happened: after the row count it recorded
        pending = None
        now = _checkpoint_seen(run_dir)
        if now != seen and now is not None:
            seen = now
            pending = _checkpoint_record(run_dir, status)
        rows, offset = _read_from(path, offset)
        at = (pending or {}).get("metrics", {}).get(stream, count)
        if pending is not None and at <= count:
            _print_checkpoint(status, pending)
            pending = None
        for row in rows:
            table.row(row)
            count += 1
            if pending is not None and count >= at:
                _print_checkpoint(status, pending)
                pending = None
        if pending is not None:
            _print_checkpoint(status, pending)
        if status.get("status") != "running":
            _print_end(run_dir, status)
            return status.get("status")
        if _gone(status):
            ui.warn(f"{ui.short_path(run_dir)} is still `running`, but its process "
                    f"({status.get('pid')}) is gone -- killed hard, or out of memory")
            return "gone"
        sleep(poll)


class _Table:
    """Rows as aligned columns: `_elapsed_s` as the first, then the keys.

    Keys in a group (`reward/lin`, `reward/yaw`) sit side by side under one
    header naming the group, each column headed by its short name -- so a
    column is as wide as `lin` and its values, not as `reward/lin`. Nothing is
    cut: the group and the short name make the key. A column widens (and the
    header comes again) when a value needs more room.

    `keys` may name groups (`reward/`): the group's keys become columns as they
    appear in the rows, so a term first recorded later still shows up. No
    keys: every key but runkit's `_` ones, as they appear.
    """

    MIN_WIDTH = 7          # fits most compact values (-0.0133), so widening is rare

    def __init__(self, keys):
        self.wanted = list(keys) if keys else None
        self.keys = [k for k in keys if not k.endswith("/")] if keys else []
        self.widths = {}
        self.since_header = None
        self.warned = False

    def _wants(self, key):
        if self.wanted is None:
            return not key.startswith("_")
        return key in self.wanted or any(key.startswith(g) for g in self.wanted
                                         if g.endswith("/"))

    @staticmethod
    def _split(key):
        """`reward/lin` -> ("reward", "lin"); `it` -> ("", "it")."""
        group, sep, leaf = key.rpartition("/")
        return (group, leaf) if sep else ("", key)

    def _order(self):
        """The keys with each group's columns side by side, groups in the order
        they first appeared."""
        groups = {}
        for k in self.keys:
            groups.setdefault(self._split(k)[0], []).append(k)
        return [k for members in groups.values() for k in members]

    def size(self, rows):
        """Take in rows without printing them: their keys and widths, so the
        rows shown first share one header."""
        for row in rows:
            self._take(row)

    def _take(self, row):
        """Columns and widths for a row; its cells, in column order."""
        new = [k for k in row if k not in self.keys and self._wants(k)]
        if new:
            self.keys += new
            self.since_header = None              # a new column: the header again
        order = self._order()
        elapsed = row.get("_elapsed_s")
        # runkit's `_elapsed_s`, under its own name, shown as a duration (9m 22s)
        cells = {"_elapsed_s": ui._duration(elapsed) if isinstance(elapsed, (int, float)) else ""}
        cells.update({k: "" if row.get(k) is None else ui._num(row[k]) for k in order})
        for k, text in cells.items():             # a value that needs more room widens it
            need = max(len(text), len(self._split(k)[1]), self.MIN_WIDTH)
            if need > self.widths.get(k, 0):
                self.widths[k] = need
                self.since_header = None
        return order, cells

    def row(self, row):
        order, cells = self._take(row)
        if self.since_header is None or self.since_header >= HEADER_EVERY:
            self._print_header(order)
            self.since_header = 0
        self._print_line([cells[k].rjust(self.widths[k]) for k in ["_elapsed_s", *order]])
        self.since_header += 1

    def _print_header(self, order):
        cols = ["_elapsed_s", *order]
        # widen a group's last column so the group name fits over its columns
        spans = {}
        for k in order:
            spans.setdefault(self._split(k)[0], []).append(k)
        for group, members in spans.items():
            span = sum(self.widths[k] for k in members) + 2 * (len(members) - 1)
            if group and len(group) > span:
                self.widths[members[-1]] += len(group) - span
        groups = [""] + [self._split(k)[0] for k in order]
        if any(groups):
            parts, i = [], 0
            while i < len(cols):
                j = i
                while j + 1 < len(cols) and groups[j + 1] == groups[i]:
                    j += 1
                span = sum(self.widths[c] for c in cols[i:j + 1]) + 2 * (j - i)
                parts.append(groups[i].ljust(span))
                i = j + 1
            self._print_line(parts, header=True)
        self._print_line([self._split(c)[1].rjust(self.widths[c]) for c in cols], header=True)

    def _print_line(self, parts, header=False):
        text = "  ".join(parts).rstrip()
        if header and len(text) > ui.out.width and not self.warned:
            self.warned = True               # never cut: the terminal wraps; say how to narrow
            ui.line(f"[dim]{len(self.keys)} columns are wider than the terminal; name "
                    f"keys to narrow it, e.g. `runkit metrics follow RUN loss reward/`[/dim]")
        ui.out.print(f"[bold]{text}[/bold]" if header else text, markup=header,
                     highlight=False, soft_wrap=True, crop=False)


def _read_from(path, offset):
    """Whole lines of `path` from byte `offset` on, parsed; and the offset after
    the last whole line (a line still being written is left for next time)."""
    if not path.is_file():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows, offset + end


def _status(run_dir):
    try:
        return yaml.safe_load((run_dir / "status.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _checkpoint_seen(run_dir):
    """(path, index) of the latest complete checkpoint, or None: a repeated
    name (`current`) is a new checkpoint when its index changes."""
    path = _status(run_dir).get("checkpoint")
    if not path:
        return None
    try:
        rec = yaml.safe_load((run_dir / path / "checkpoint.yaml").read_text())
        return path, rec.get("index")
    except (OSError, yaml.YAMLError, AttributeError):
        return path, None


def _checkpoint_record(run_dir, status):
    try:
        return yaml.safe_load((run_dir / status["checkpoint"] / "checkpoint.yaml").read_text())
    except (OSError, yaml.YAMLError, TypeError, KeyError):
        return None


def _print_checkpoint(status, rec):
    if not rec:
        return
    ui.checkpoint_saved(path=status.get("checkpoint"), elapsed_s=rec.get("elapsed_s") or 0,
                        info=rec.get("info"),
                        progress=rec.get("progress"), total=rec.get("total"),
                        summary=rec.get("summary") or {})


def _print_end(run_dir, status):
    try:
        run_id = yaml.safe_load((run_dir / "run_context.yaml").read_text())["id"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        run_id = run_dir.name
    ui.run_finished(run_id=run_id, status=status.get("status"),
                    duration_s=status.get("duration_s") or 0, error=status.get("error"),
                    run_dir=run_dir)


def _gone(status):
    """Is a `running` run's process gone? Only answerable on its own host."""
    pid, host = status.get("pid"), status.get("host")
    if not pid or host != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:                  # alive, someone else's
        return False
    return False
