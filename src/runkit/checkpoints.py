"""Checkpoints: named, recorded snapshots a live run saves along the way.

    with ctx.checkpoint() as ckpt:              # {run dir}/checkpoints/000003/
        model.save(ckpt.state / "model.zip")

    with ctx.checkpoint("best") as ckpt:        # {run dir}/checkpoints/best/
        model.save(ckpt.state / "model.zip")
        ckpt.info["ep_return"] = ret            # saved with it, in checkpoint.yaml

A checkpoint is laid out like a run dir: runkit's own file at the top, the rest
in named folders --

    checkpoints/best/
      checkpoint.yaml    runkit's: index, name, time, info, row counts, summary
      state/             the run body's (`ckpt.state`): what continuing needs
      eval/              the eval's (`ckpt.eval`), made when one opens it

runkit owns the folder structure -- names, `checkpoint.yaml`, the `latest` link
-- and the body owns `state/`, as it owns `ctx.out`. A branch reads `state/`
and nothing else. A checkpoint from before `state/` has its files at the top:
its `state` is the checkpoint folder itself -- as for a body that still saves
into `ckpt.dir`, whose empty `state/` is dropped when the block exits.

A checkpoint is complete once its `checkpoint.yaml` exists; that file is written
last, and only when the block exits cleanly. The block writes into a hidden
staging folder that is swapped into place at the end, so a failed or killed save
never replaces an earlier checkpoint of the same name, and leaves at most a
dot-folder that everything ignores. Order is runkit's `index`, never the name.
"""
import dataclasses
import datetime
import os
import pathlib
import shutil
import time

import yaml

from .metrics import summarize_window
from .utils import plain, point_latest

FOLDER = "checkpoints"
RECORD = "checkpoint.yaml"
STATE = "state"
EVAL = "eval"
_RESERVED = ("latest",)


@dataclasses.dataclass
class Checkpoint:
    name: str                 # the folder name; the counter ("000003") when none is given
    dir: pathlib.Path         # {run dir}/checkpoints/<name>/ (a staging folder during the block)
    index: int                # runkit's counter: the order of checkpoints, whatever their names
    info: dict = dataclasses.field(default_factory=dict)   # yours; saved to checkpoint.yaml
    metrics: dict = dataclasses.field(default_factory=dict)  # lines per metrics stream when it completed
    summary: dict = dataclasses.field(default_factory=dict)  # per stream: the rows since the last one
    run: pathlib.Path | None = None                         # the run dir it belongs to
    time: str | None = None           # when it completed (ISO, to the second)
    elapsed_s: float | None = None    # how far into the run, in seconds
    progress: object = None           # ctx.progress when it completed ...
    total: object = None              # ... and its total

    @property
    def state(self) -> pathlib.Path:
        """`{dir}/state` -- what the run body saves, and a branch continues from
        (as `ctx.out` is `{run dir}/out`). A checkpoint from before `state/`:
        the checkpoint folder itself."""
        state = self.dir / STATE
        return state if state.is_dir() else self.dir

    @property
    def eval(self) -> pathlib.Path:
        """`{dir}/eval` -- what an eval of this checkpoint writes. Replaced with
        the checkpoint: saving one of the same name again empties it."""
        return self.dir / EVAL

    def is_empty(self):
        """Nothing saved in `state` -- the body wrote no files."""
        if (self.dir / STATE).is_dir():
            return not any((self.dir / STATE).iterdir())
        return not any(p.name not in (RECORD, EVAL) for p in self.dir.iterdir())


class _Saving:
    """The context manager `RunContext.checkpoint` returns."""

    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __enter__(self):
        ctx = self.ctx
        index = ctx._live.checkpoints + 1
        name = self.name if self.name is not None else f"{index:06d}"
        _check_name(name)
        folder = ctx.dir / FOLDER
        folder.mkdir(exist_ok=True)
        self.final = folder / name
        staging = folder / f".{name}.staging-{index}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        (staging / STATE).mkdir()
        self.ckpt = Checkpoint(name=name, dir=staging, index=index, run=ctx.dir)
        return self.ckpt

    def __exit__(self, exc_type, exc, tb):
        ckpt, staging = self.ckpt, self.ckpt.dir
        if exc_type is not None:              # not complete: nothing recorded, nothing replaced
            shutil.rmtree(staging, ignore_errors=True)
            return False
        ctx = self.ctx
        record = {
            "index": ckpt.index,
            "name": ckpt.name,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.monotonic() - ctx._live.t0, 3),
            "run": ctx.id,
            "info": plain(ckpt.info),
            # rows each metrics stream had when this checkpoint completed: an exact
            # boundary for "the metrics between two checkpoints"
            "metrics": dict(ctx._live.records),
            # where the run was, and how it went since the last checkpoint
            "progress": ctx._live.progress,
            "total": ctx._live.total,
            # per stream the run recorded to since the last checkpoint (`run` first)
            "summary": {s: w for s, w in sorted(
                ((s, summarize_window(w)) for s, w in ctx._live.windows.items()),
                key=lambda sw: (sw[0] != "run", sw[0])) if w},
        }
        state = staging / STATE
        if not any(state.iterdir()):
            # nothing in state/: the body saved at the top (`ckpt.dir`, as before
            # state/ existed) or nothing at all; the folder is then the state
            state.rmdir()
        (staging / RECORD).write_text(yaml.safe_dump(record, sort_keys=False))
        _swap_in(staging, self.final)
        ctx._live.checkpoints = ckpt.index
        ctx._live.windows = {}                # the next checkpoint summarizes from here
        ckpt.metrics = dict(ctx._live.records)
        ckpt.summary = record["summary"]
        ckpt.time, ckpt.elapsed_s = record["time"], record["elapsed_s"]
        ckpt.progress, ckpt.total = record["progress"], record["total"]
        ckpt.dir = self.final
        point_latest(self.final)
        # status.yaml names it at once (not throttled like progress): what a
        # resume or an eval of an unfinished run would read
        ctx._live.checkpoint = f"{FOLDER}/{ckpt.name}"
        ctx._write_running()
        try:                                  # say so; best-effort, never costs the checkpoint
            from . import ui
            ui.checkpoint_saved(path=ctx._live.checkpoint, elapsed_s=record["elapsed_s"],
                                info=record["info"], progress=record["progress"],
                                total=record["total"], summary=record["summary"])
            if ctx._live.table is not None:           # the run's own --follow table
                ctx._live.table.interrupt()
        except Exception:                                    # noqa: BLE001
            pass
        return False


def _check_name(name):
    if (not isinstance(name, str) or not name or name in _RESERVED
            or name.startswith(".") or "/" in name or os.sep in name):
        raise ValueError(f"bad checkpoint name {name!r}: a plain folder name, "
                         f"not starting with '.', and not {', '.join(_RESERVED)}")


def _swap_in(staging, final):
    """Put `staging` where `final` is. An old checkpoint of that name is moved
    aside first and removed only after the new one is in place."""
    old = None
    if final.exists():
        old = final.with_name(f".{final.name}.old-{os.getpid()}")
        os.replace(final, old)
    os.replace(staging, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def load_checkpoints(run_dir):
    """The complete checkpoints of a run, oldest first (by `index`).

    A folder counts only if its `checkpoint.yaml` is there; staging leftovers,
    the `latest` link and anything unreadable are skipped.
    """
    folder = pathlib.Path(run_dir) / FOLDER
    if not folder.is_dir():
        return []
    found = []
    for d in folder.iterdir():
        if d.name.startswith(".") or d.is_symlink() or not (d / RECORD).is_file():
            continue
        try:
            rec = yaml.safe_load((d / RECORD).read_text())
            found.append(Checkpoint(name=rec["name"], dir=d.resolve(), index=rec["index"],
                                    info=rec.get("info") or {},
                                    metrics=rec.get("metrics") or {},
                                    summary=rec.get("summary") or {},
                                    run=pathlib.Path(run_dir).resolve(),
                                    time=rec.get("time"), elapsed_s=rec.get("elapsed_s"),
                                    progress=rec.get("progress"), total=rec.get("total")))
        except Exception:                                    # noqa: BLE001
            continue
    return sorted(found, key=lambda c: c.index)
