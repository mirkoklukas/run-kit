"""argv parsing and dataclass instantiation.

Two namespaces, strictly disjoint:
  - bare `key=value`  -> cfg overrides    (the experiment definition)
  - `--flag[=value]`  -> ctx overrides    (how this attempt is run)

Pure functions; no IO, no globals. Tested in isolation.
"""
import copy
import dataclasses
import inspect
import secrets
import types
import typing


def split_argv(tokens):
    """Split tokens into (cfg_overrides, ctx_flags, positionals).

    cfg_overrides: list of "key=value" strings (the cfg layer).
    ctx_flags: dict of {flag_name: value} parsed from --flag / --flag=value /
               --flag value pairs. Bare --flag (and any `-x` short flag) becomes
               True, so a stray `-x` is rejected as a flag rather than
               mistaken for a positional.
    positionals: bare tokens with no leading dash and no '=' (e.g. a config
                 file path). Callers decide what they mean.
    """
    cfg, flags, positionals = [], {}, []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            body = t[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                flags[k.replace("-", "_")] = _coerce(v)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--") \
                    and "=" not in tokens[i + 1]:
                flags[body.replace("-", "_")] = _coerce(tokens[i + 1])
                i += 1
            else:
                flags[body.replace("-", "_")] = True
        elif len(t) > 1 and t[0] == "-" and not t[1].isdigit():
            body = t[1:]
            flags[body.replace("-", "_")] = True
        elif "=" in t:
            cfg.append(t)
        else:
            positionals.append(t)
        i += 1
    return cfg, flags, positionals


def deep_merge(base, over):
    """Recursively merge `over` into `base`; `over` wins. Returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(tokens):
    """Parse ["a=1", "b.c=hi"] -> {"a": 1, "b": {"c": "hi"}}.

    Dotted keys become nested dicts. Values are type-coerced.
    """
    out = {}
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"override {t!r} must be key=value")
        key, raw = t.split("=", 1)
        _set_dotted(out, key.split("."), _coerce(raw))
    return out


# `key+=v`, `key-=v`, `key*=v`, `key/=v`: change a number instead of setting it
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b}
_SYMBOL = {"+": "+", "-": "−", "*": "×", "/": "÷"}


def split_ops(tokens):
    """Split `key=value` tokens into (plain ones, operations): `env.w*=2` is the
    operation ("env.w", "*", "2"). Keys never end in `+ - * /`, so the character
    before the first `=` says which it is; `x=-5` stays a plain negative value."""
    sets, ops = [], []
    for t in tokens:
        key, sep, raw = t.partition("=")
        if sep and len(key) > 1 and key[-1] in OPS:
            ops.append((key[:-1], key[-1], raw))
        else:
            sets.append(t)
    return sets, ops


def apply_ops(cfg, ops):
    """Apply `(key, op, value)` operations to a built config, left to right, each
    to the value the key has at that point. -> (the new config, {key: how its
    value came about, e.g. "15 × 2"}).

    Numbers only: a key whose value is not a number (a string, a bool, None) is
    an error, as is a value that is not one. An `int` field stays an int -- a
    result that is not whole is an error rather than being cut."""
    derived = {}
    for key, op, raw in ops:
        value = _coerce(raw)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key}{op}={raw}: {raw!r} is not a number")
        path = key.split(".")
        current, is_int = _leaf(cfg, path, key)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError(f"{key}{op}={raw}: {key} is {current!r}, not a number")
        if op == "/" and value == 0:
            raise ValueError(f"{key}{op}={raw}: division by zero")
        new = OPS[op](current, value)
        if is_int:
            if not float(new).is_integer():
                raise ValueError(f"{key}{op}={raw}: {key} is an int, and "
                                 f"{_show(current)} {_SYMBOL[op]} {_show(value)} = {new:g}")
            new = int(new)
        nested = {}
        _set_dotted(nested, path, new)
        cfg = build_cfg(type(cfg), nested, base=cfg)
        before = derived.get(key, _show(current))
        if op in "*/" and any(f" {c} " in before for c in "+−"):
            before = f"({before})"                       # left to right, as written
        derived[key] = f"{before} {_SYMBOL[op]} {raw}"
    return cfg, derived


def _leaf(cfg, path, key):
    """The value at a dotted path of a config (through nested configs and dict
    fields), and whether it is an `int` -- by the field's annotation where there
    is one (`w: float = 5` is a float), else by the value (inside a dict)."""
    obj, hint = cfg, None
    for i, part in enumerate(path):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            hints = typing.get_type_hints(type(obj))
            if part not in hints:
                raise ValueError(f"{key}: unknown field {'.'.join(path[:i + 1])!r}")
            hint, obj = _unwrap_optional(hints[part]), getattr(obj, part)
        elif isinstance(obj, dict):
            if part not in obj:
                raise ValueError(f"{key}: no {'.'.join(path[:i + 1])!r} to change")
            hint, obj = None, obj[part]
        else:
            raise ValueError(f"{key}: {'.'.join(path[:i])!r} is {obj!r}, it has no fields")
    is_int = hint is int if hint is not None else isinstance(obj, int)
    return obj, is_int


def _show(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _set_dotted(d, path, value):
    for p in path[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ValueError(f"override path collides with scalar at {p!r}")
    d[path[-1]] = value


def _coerce(s):
    """Cheap YAML-ish scalar coercion. No external dep."""
    if not isinstance(s, str):
        return s
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def random_seed(bits=31):
    """A dataclass field whose default is a fresh random seed:

        @dataclass
        class Config:
            seed: int = random_seed()

    The seed is drawn when the config is built -- before runkit freezes it -- so
    `config.yaml` records the number actually used and the run can be repeated
    (`seed=<that number>`). Overriding (`seed=7`) skips the draw, and a config
    thawed from `config.yaml` (eval, viz, `load_run`) keeps its recorded seed.

    31 bits fits a signed 32-bit int, which every library takes as a seed.
    Drawn from the OS (`secrets`), so it does not depend on any global random
    state. The field is tagged `metadata={"runkit": "seed"}`.
    """
    return dataclasses.field(default_factory=lambda: secrets.randbits(bits),
                             metadata={"runkit": "seed"})


def annotated_cfg(fn):
    """The annotation on `fn`'s `cfg` parameter, or None if it has none.

    Follows `functools.wraps` to the body, and `eval_str=True` so a string
    annotation (`from __future__ import annotations`) resolves to the class.
    """
    params = inspect.signature(fn, eval_str=True).parameters
    if "cfg" not in params or params["cfg"].annotation is inspect.Parameter.empty:
        return None
    return params["cfg"].annotation


def build_cfg(cls, overrides, base=None):
    """Instantiate dataclass `cls` with `overrides` applied to defaults.

    Supports nested dataclasses: a dict value overrides fields *of the nested
    config the field would otherwise hold* -- its own default (`default_factory()`
    or `default`), or, below the top, the parent's value -- so every field the
    overrides do not mention keeps that value, not the nested class's defaults.
    A `dict` field works the same way: the override is deep-merged into the dict
    the field would otherwise hold, so `sched.env.w.start=2e6` changes one leaf.
    A subclass default stays that subclass. Only a field with no default is
    built fresh from its annotated class.

    `base`: an instance to override instead of `cls`'s defaults (used for the
    nesting; the result is `dataclasses.replace(base, ...)`).
    """
    if base is not None:
        cls = type(base)
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    field_types = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in overrides.items():
        if k not in field_types:
            raise ValueError(
                f"unknown field {k!r} for {cls.__name__}; "
                f"known: {sorted(field_types)}")
        t = field_types[k]
        if isinstance(v, dict) and dataclasses.is_dataclass(t):
            current = getattr(base, k) if base is not None else _field_default(fields[k])
            if _is_instance(current):
                kwargs[k] = build_cfg(type(current), v, base=current)
            else:
                kwargs[k] = build_cfg(t, v)
        elif isinstance(v, dict) and _is_dict_type(t):
            # a dict field works like a nested config: the override is merged
            # into the dict the field would otherwise hold, not put in its place
            current = getattr(base, k) if base is not None else _field_default(fields[k])
            kwargs[k] = deep_merge(copy.deepcopy(current), v) if isinstance(current, dict) else v
        else:
            kwargs[k] = _cast(t, v)
    if base is not None:
        return dataclasses.replace(base, **kwargs)
    return cls(**kwargs)


def _is_dict_type(t):
    """`dict`, `dict[str, X]`, `typing.Dict[...]`, or an optional one."""
    t = _unwrap_optional(t)
    return t is dict or typing.get_origin(t) is dict


def _is_instance(x):
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _field_default(f):
    """The value a dataclass field gets when it is not passed, or None."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def _unwrap_optional(t):
    """`Optional[X]` / `X | None` -> X; other unions and plain types pass through."""
    if typing.get_origin(t) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _cast(t, v):
    """Coerce `v` to the field's annotated scalar type `t`, when it's safe to.

    The point of failure this fixes: YAML 1.1 only reads a float when the
    mantissa has a dot, so `1e-4` loads as the *string* "1e-4". A `float`
    annotation lets us recover the intended value. Only the scalar leaf types are
    touched; containers, dataclasses, and anything we can't convert pass through
    unchanged, so a genuinely wrong value still surfaces at `cls(**kwargs)`.

    Tuples too: yaml has none, so a `tuple` field comes back from `config.yaml` as
    a list, which never equals the tuple default (the banner would list it as
    changed). A list for a `tuple` / `tuple[...]` field becomes a tuple.
    """
    t = _unwrap_optional(t)
    if (t is tuple or typing.get_origin(t) is tuple) and isinstance(v, list):
        return _to_tuple(t, v)
    if v is None or not isinstance(t, type) or isinstance(v, t):
        return v
    if t is bool:
        return _coerce(v) if isinstance(v, str) else v
    if t is int:
        return _to_int(v)
    if t in (float, str):
        try:
            return t(v)
        except (TypeError, ValueError):
            return v
    return v


def _to_tuple(t, v):
    """List `v` as a tuple for annotation `t`: elements cast per `tuple[X, ...]` /
    `tuple[A, B]`; under a bare `tuple`, nested lists become tuples as well
    (`((-60, 60), ...)` round-trips)."""
    args = typing.get_args(t)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_cast(args[0], x) for x in v)
    if args and len(args) == len(v):
        return tuple(_cast(a, x) for a, x in zip(args, v))
    return tuple(_to_tuple(tuple, x) if isinstance(x, list) else x for x in v)


def _to_int(v):
    """`int(v)`, but also accept exponent strings yaml leaves as text (`1e3`).

    Such a value is read through `float` first; it's only taken as an int when it
    lands on a whole number (`1e3` -> 1000, but `1.5e0` passes through untouched).
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        return int(f) if f.is_integer() else v
