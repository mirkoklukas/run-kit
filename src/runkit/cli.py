"""The `runkit` console script.

Usage:
    runkit <verb> [runkit options] <experiment> [args ...]
    runkit root [FOLDER]
    runkit metrics [RUN_DIR] [KEY ...] [--follow] [--rows A:B] [--start V] [--end V] [--x KEY]
    runkit plot [RUN_DIR] [KEY ...] [--x KEY] [--rows A:B] [--start V] [--end V] [--out FILE]

verbs: run, eval, viz (as registered on the experiment), root, latest

`<experiment>` is a file (`experiment.py`) or a dotted module
(`lab.rl_env.test_policy`). `args` are what `python experiment.py <verb> ...`
takes after the verb. The slot between the verb and the experiment is for
runkit's own options (none yet).

`root` and `latest` print a path, for `cd "$(runkit latest experiment.py)"`:
the experiment's runs folder ({root}/{name}) and its latest run dir. Without an
experiment, `runkit root [FOLDER]` prints the root itself, resolved from FOLDER
(default: the current one): the nearest `experiment.toml`'s `[env] root`, else
`./runs`.

`metrics` and `plot` read a run dir's metrics without importing its experiment
(RUN_DIR defaults to the current folder; `runs/<name>/latest` works).
`metrics` prints every stream's keys with their rows, last, min and max (or,
given KEYs, just those); with
`--follow` (`-f`), its rows as they are written, the run's checkpoints, and its
end. KEY picks keys, and a stream, as for plot (`eval:` for all of eval). `plot`
draws keys to a PNG in the run's metrics/ folder: KEY is `key` (the `run`
stream) or `stream:key`; `--x KEY` plots against a key of each series' own
stream, else against the line number. No KEY (or a lone `stream:`): every
numeric key of the stream, one subplot each.

Both select lines with `--rows A:B` (a python slice of line numbers, negatives
from the end: `--rows -1000:`) and `--start V` / `--end V` (an inclusive window
on the x axis: a value, negative counting back from the end -- `--start -1000` is
the last 1000 lines, or with `--x steps` the last 1000 steps -- or a checkpoint
name, for the lines between checkpoints).
"""
import importlib
import importlib.util
import pathlib
import sys

from . import ui
from .autocli import PATH_VERBS, VERBS, dispatch
from .exp import SCRIPT_MODULE, Experiment
from .settings import resolve_root

HELP = __doc__[__doc__.index("Usage:"):].strip()


def app():
    """Console-script entry point. Returns None: whatever the verb returned (a
    `Run`, say) must not become the process's exit status."""
    main()


def main(argv=None):
    """`runkit <verb> [options] <experiment> ...` -> the verb's return value."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"runkit — lightweight, reproducible experiment runs.\n\n{HELP}")
        return
    verb, rest = argv[0], argv[1:]
    if verb in ("metrics", "plot"):          # read a run dir; no experiment import
        return (metrics_cmd if verb == "metrics" else plot_cmd)(rest)
    if verb not in VERBS + PATH_VERBS:
        hint = ""
        if verb.endswith(".py") or "." in verb:      # the old order: runkit exp.py viz
            v = rest[0] if rest and rest[0] in VERBS + PATH_VERBS else "run"
            hint = f" -- the verb comes first: runkit {v} {verb} ..."
        sys.exit(f"unknown verb {verb!r}{hint}\n\n{HELP}")

    # runkit's own options sit between the verb and the experiment
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt in ("-h", "--help"):
            print(HELP)
            return
        sys.exit(f"unknown runkit option {opt!r} for {verb} "
                 f"(experiment arguments go after the experiment)")

    if verb == "root" and (not rest or pathlib.Path(rest[0]).is_dir()):
        return root_cmd(rest)
    if not rest:
        sys.exit(f"usage: runkit {verb} <experiment> ...")
    target, args = rest[0], rest[1:]
    from .launch import prepare
    prepare(target, argv)                   # experiment.toml's extras / vars, before the import
    return dispatch(_find_experiment(_import(target), target), [verb, *args])


def root_cmd(argv):
    """`runkit root [FOLDER]`: print the root resolved from FOLDER.

    Printing is all it can do: a process cannot change its parent shell's
    directory. Only the path goes to stdout, so `cd "$(runkit root)"` is clean.
    """
    if len(argv) > 1:
        sys.exit("usage: runkit root [FOLDER | <experiment>]")
    start = pathlib.Path(argv[0]) if argv else pathlib.Path.cwd()
    try:
        root = resolve_root(start).resolve()
    except ValueError as e:
        sys.exit(str(e))
    if not root.is_dir():
        ui.warn(f"{root} does not exist yet (no runs made there)")
    print(root)
    return root


def _run_dir_and_rest(argv, usage):
    """Split off a leading RUN_DIR (an existing folder), else the current one."""
    if any(a in ("-h", "--help") for a in argv):
        print(usage)
        sys.exit(0)
    if argv and pathlib.Path(argv[0]).is_dir():
        run_dir, rest = pathlib.Path(argv[0]), argv[1:]
    else:
        run_dir, rest = pathlib.Path.cwd(), list(argv)
    if not (run_dir / "run_context.yaml").is_file():
        sys.exit(f"{run_dir} is not a run dir (no run_context.yaml); pass one, "
                 f"e.g. runs/<name>/latest\n\n{usage}")
    return run_dir, rest


def _options(argv, allowed, usage):
    """Split `--name value` / `--name=value` options (from `allowed`) off argv.
    A value may start with '-' (`--start -1000`)."""
    opts, positional, it = {}, [], iter(argv)
    for a in it:
        if a.startswith("--") and a not in ("--help",):
            name, eq, value = a[2:].partition("=")
            if name not in allowed:
                sys.exit(f"unknown option {a!r}\n\n{usage}")
            opts[name] = value if eq else next(it, None)
            if opts[name] is None:
                sys.exit(f"--{name} needs a value\n\n{usage}")
        else:
            positional.append(a)
    return opts, positional


def _selection(opts):
    """--rows / --start / --end / --x -> keyword arguments for runkit.metrics
    (named as in python: `from` is a keyword there)."""
    return {"rows": opts.get("rows"), "start": opts.get("start"), "end": opts.get("end"),
            "x": opts.get("x")}


def _stream_and_keys(specs, usage):
    """`key` / `stream:key` / `stream:` -> (stream, keys); all from one stream."""
    from .metrics import _series
    stream, keys = None, []
    for spec in specs:
        s, key = _series(spec)
        if stream is not None and s != stream:
            sys.exit(f"keys from one stream at a time (got {stream!r} and {s!r})\n\n{usage}")
        stream = s
        if key:
            keys.append(key)
    return stream or "run", keys


def metrics_cmd(argv):
    """`runkit metrics [RUN_DIR] [KEY ...] [selection] [--follow]`: a stream's
    keys as a table, or its rows as they come."""
    from .metrics import FOLDER, parse_rows, streams, summarize_metrics
    usage = ("usage: runkit metrics [RUN_DIR] [KEY ...] [--follow] "
             "[--rows A:B] [--start V] [--end V] [--x KEY]")
    following = any(a in ("-f", "--follow") for a in argv)
    argv = [a for a in argv if a not in ("-f", "--follow")]
    opts, argv = _options(argv, ("rows", "start", "end", "x"), usage)
    run_dir, specs = _run_dir_and_rest(argv, usage)
    stream, keys = _stream_and_keys(specs, usage)
    have = streams(run_dir)

    if following:
        from .follow import follow
        if opts.get("start") or opts.get("end"):
            sys.exit(f"--follow takes --rows (which rows to show first), not --start/--end"
                     f"\n\n{usage}")
        try:
            first = parse_rows(opts["rows"]) if opts.get("rows") else None
        except ValueError as e:
            sys.exit(str(e))
        ui.line(f"[dim]following {FOLDER}/{stream}.jsonl of "
                f"{ui.short_path(run_dir)} -- Ctrl-C stops following, not the run[/dim]")
        try:
            return follow(run_dir, stream, keys or None, first=first)
        except KeyboardInterrupt:
            return None

    if not specs:                           # no key, no stream: all of them
        return _metrics_overview(run_dir, have, opts)
    if stream not in have:
        sys.exit(f"no metrics stream {stream!r} in {run_dir / FOLDER} "
                 f"(streams: {', '.join(have) or 'none'})")
    try:
        rows = summarize_metrics(run_dir, stream, **_selection(opts))
    except ValueError as e:
        sys.exit(str(e))
    if keys:
        missing = [k for k in keys if k not in {r["key"] for r in rows}]
        if missing:
            sys.exit(f"no key(s) {missing} in stream {stream!r} "
                     f"(keys: {', '.join(r['key'] for r in rows)})")
        rows = [r for r in rows if r["key"] in keys]
    others = [s for s in have if s != stream]
    ui.out.print(f"[bold]{run_dir.resolve().name}[/bold]")
    _print_stream(stream, rows, note=f"also: {', '.join(others)}" if others else None)
    return rows


def _metrics_overview(run_dir, have, opts):
    """Every stream of a run, each with its keys: what `runkit metrics RUN_DIR` shows."""
    from .metrics import FOLDER, summarize_metrics
    ui.out.print(f"[bold]{run_dir.resolve().name}[/bold]")
    if not have:
        ui.out.print("[dim]no metrics yet (nothing recorded with ctx.record)[/dim]")
        return {}
    overview, failed = {}, {}
    for stream in sorted(have, key=lambda s: (s != "run", s)):   # the run's own stream first
        try:
            overview[stream] = summarize_metrics(run_dir, stream, **_selection(opts))
        except ValueError as e:              # e.g. a checkpoint that did not count this stream
            failed[stream] = e
    if failed and not overview:              # a bad selection: nothing to show at all
        sys.exit(str(next(iter(failed.values()))))
    for stream in sorted(have, key=lambda s: (s != "run", s)):
        if stream in overview:
            _print_stream(stream, overview[stream])
        else:
            ui.out.print(f"\n{FOLDER}/{stream}.jsonl  [dim]{failed[stream]}[/dim]")
    return overview


def _print_stream(stream, rows, note=None):
    """One stream's keys as a table under a `metrics/<stream>.jsonl  N rows` line."""
    from rich.padding import Padding
    from rich.table import Table
    from .metrics import FOLDER
    n = max((r["rows"] for r in rows), default=0)
    ui.out.print(f"\n{FOLDER}/{stream}.jsonl  {n} rows"
                 + (f"  [dim]({note})[/dim]" if note else ""))
    table = Table(box=None, pad_edge=False, header_style="bold")
    for col, justify in (("key", "left"), ("rows", "right"), ("last", "right"),
                         ("min", "right"), ("max", "right")):
        table.add_column(col, justify=justify)
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
            return str(int(v))                       # 60000, not 6e+04
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    for r in rows:
        table.add_row(r["key"], str(r["rows"]), fmt(r["last"]), fmt(r["min"]), fmt(r["max"]))
    ui.out.print(Padding(table, (0, 0, 0, 2)))


def plot_cmd(argv):
    """`runkit plot [RUN_DIR] KEY [KEY ...] [--x KEY] [selection] [--out FILE]`: to a PNG."""
    from .metrics import plot_metrics
    usage = ("usage: runkit plot [RUN_DIR] [KEY ...] [--x KEY] "
             "[--rows A:B] [--start V] [--end V] [--out FILE]")
    opts, positional = _options(argv, ("x", "out", "rows", "start", "end"), usage)
    run_dir, ys = _run_dir_and_rest(positional, usage)
    sel = _selection(opts)
    try:
        out = plot_metrics(run_dir, ys, x=sel["x"], out=opts.get("out"), rows=sel["rows"],
                           start=sel["start"], end=sel["end"])
    except ValueError as e:
        sys.exit(str(e))
    print(out)
    return out


def _import(target):
    """Import a file-path or dotted-module target the way `python` would run it."""
    path = pathlib.Path(target)
    if target.endswith(".py") or path.is_file():
        if not path.is_file():
            sys.exit(f"experiment file not found: {target}")
        # like `python file.py`: the file's own folder is importable
        sys.path.insert(0, str(path.resolve().parent))
        spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[SCRIPT_MODULE] = mod
        spec.loader.exec_module(mod)
        return mod
    # like `python -m pkg.mod`: importable from the current directory
    sys.path.insert(0, str(pathlib.Path.cwd()))
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        sys.exit(f"cannot import {target!r} (not a file, and not a module: {e})")


def _find_experiment(mod, target):
    """The one `Experiment` in `mod` -- an `Experiment(...)` object, or the one
    behind an `@experiment`-decorated function. One experiment per file."""
    found = {}
    for value in vars(mod).values():
        exp = value if isinstance(value, Experiment) else getattr(
            value, "_runkit_experiment", None)
        if isinstance(exp, Experiment):
            found[id(exp)] = exp
    if not found:
        sys.exit(f"{target}: no experiment found (an `Experiment(...)` or an "
                 f"@experiment function at module level)")
    if len(found) > 1:
        names = sorted(e.name for e in found.values())
        sys.exit(f"{target}: {len(found)} experiments ({', '.join(names)}); "
                 f"one experiment per file")
    return next(iter(found.values()))
