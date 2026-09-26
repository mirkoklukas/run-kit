"""Checkpoints: `with ctx.checkpoint(name=None) as ckpt:` in a live run."""
import numpy as np
import pytest
import yaml

from dataclasses import dataclass

from runkit import Experiment, RunContext, load_run


@dataclass
class Cfg:
    n: int = 3
    fail_at: int = -1          # raise inside the checkpoint block of this save


def _exp(body):
    exp = Experiment("ck")
    exp.run(body)
    return exp


def _go(exp, tmp_path, *args):
    return exp.main([*args, f"--root={tmp_path}"])


def _record(ckpt_dir):
    return yaml.safe_load((ckpt_dir / "checkpoint.yaml").read_text())


def test_counter_names_record_and_latest(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        for i in range(cfg.n):
            with ctx.checkpoint() as ckpt:
                (ckpt.dir / "w.txt").write_text(str(i))
                ckpt.info["i"] = i
        return [c.name for c in ctx.checkpoints()]

    r = _go(_exp(body), tmp_path)
    assert r.retval == ["000001", "000002", "000003"]
    folder = r.context.dir / "checkpoints"
    rec = _record(folder / "000003")
    assert rec["index"] == 3 and rec["name"] == "000003" and rec["run"] == r.context.id
    assert rec["info"] == {"i": 2} and rec["elapsed_s"] >= 0 and rec["time"]
    assert (folder / "latest").resolve() == (folder / "000003").resolve()
    assert (folder / "000003" / "w.txt").read_text() == "2"
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]    # no leftovers


def test_a_repeated_name_replaces_and_takes_a_new_index(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint("best") as ckpt:
            (ckpt.dir / "w.txt").write_text("first")
        with ctx.checkpoint() as ckpt:                        # 000002
            pass
        with ctx.checkpoint("best") as ckpt:
            (ckpt.dir / "new.txt").write_text("second")
        return [(c.name, c.index) for c in ctx.checkpoints()]

    r = _go(_exp(body), tmp_path)
    assert r.retval == [("000002", 2), ("best", 3)]          # ordered by index, not name
    best = r.context.dir / "checkpoints" / "best"
    assert sorted(p.name for p in best.iterdir()) == ["checkpoint.yaml", "new.txt"]
    assert (r.context.dir / "checkpoints" / "latest").resolve() == best.resolve()


def test_a_failed_save_records_nothing_and_keeps_the_old_one(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint("best") as ckpt:
            (ckpt.dir / "w.txt").write_text("good")
        try:
            with ctx.checkpoint("best") as ckpt:
                (ckpt.dir / "w.txt").write_text("half")
                raise OSError("disk full")
        except OSError:
            pass
        return ctx.checkpoints()

    r = _go(_exp(body), tmp_path)
    assert [c.name for c in r.retval] == ["best"] and r.retval[0].index == 1
    folder = r.context.dir / "checkpoints"
    assert (folder / "best" / "w.txt").read_text() == "good"
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]


def test_ckpt_dir_is_the_final_folder_after_the_block(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint("x") as ckpt:
            during = ckpt.dir
        return during, ckpt.dir

    during, after = _go(_exp(body), tmp_path).retval
    assert during.name.startswith(".x.staging")                # hidden while being written
    assert after.name == "x" and (after / "checkpoint.yaml").is_file()


def test_numpy_info_is_saved_as_plain_values(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint() as ckpt:
            ckpt.info.update(ret=np.float32(1.5), arr=np.arange(2), obj=object())

    r = _go(_exp(body), tmp_path)
    info = _record(r.context.dir / "checkpoints" / "000001")["info"]
    assert info["ret"] == 1.5 and info["arr"] == [0, 1] and info["obj"].startswith("<object")


@pytest.mark.parametrize("bad", ["latest", ".hidden", "a/b", ""])
def test_bad_names_are_refused(tmp_path, bad):
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint(bad):
            pass

    with pytest.raises(ValueError, match="bad checkpoint name"):
        _go(_exp(body), tmp_path)


def test_only_the_live_run_can_checkpoint(tmp_path):
    exp = Experiment("ck")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        assert ctx.live
        with ctx.checkpoint():
            pass

    @exp.viz
    def show(cfg: Cfg, ctx: RunContext):
        assert not ctx.live
        found = ctx.checkpoints()                             # reading is fine
        with ctx.checkpoint():                                # writing is not
            pass
        return found

    r = _go(exp, tmp_path)
    assert not r.context.live                                 # finished: no longer live
    with pytest.raises(RuntimeError, match="only the live run"):
        r.context.checkpoint()
    with pytest.raises(RuntimeError, match="only the live run"):
        _go(exp, tmp_path, "viz")
    assert [c.name for c in load_run(r.context.dir, Cfg).context.checkpoints()] == ["000001"]


def test_a_killed_save_is_ignored(tmp_path):
    """A staging folder left by a hard kill, and a folder without a record,
    are not checkpoints."""
    def body(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint():
            pass
        (ctx.dir / "checkpoints" / ".000002.staging-2").mkdir()
        (ctx.dir / "checkpoints" / "manual").mkdir()          # no checkpoint.yaml
        return [c.name for c in ctx.checkpoints()]

    assert _go(_exp(body), tmp_path).retval == ["000001"]


def test_status_names_the_latest_checkpoint_by_its_path(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        seen = [yaml.safe_load((ctx.dir / "status.yaml").read_text())["checkpoint"]]
        with ctx.checkpoint():
            pass
        seen.append(yaml.safe_load((ctx.dir / "status.yaml").read_text())["checkpoint"])
        with ctx.checkpoint("best"):
            pass
        if cfg.fail_at == 0:
            raise ValueError("after checkpointing")
        return seen

    r = _go(_exp(body), tmp_path)
    assert r.retval == [None, "checkpoints/000001"]           # written at once, mid-run
    status = yaml.safe_load((r.context.dir / "status.yaml").read_text())
    assert status["checkpoint"] == "checkpoints/best"         # kept in the final write
    assert (r.context.dir / status["checkpoint"] / "checkpoint.yaml").is_file()

    with pytest.raises(ValueError):                           # a failed run says what to resume from
        _go(_exp(body), tmp_path, "fail_at=0")
    statuses = [yaml.safe_load((d / "status.yaml").read_text())
                for d in (tmp_path / "ck").glob("2*")]
    st = next(s for s in statuses if s["status"] == "failed")
    assert st["checkpoint"] == "checkpoints/best"
