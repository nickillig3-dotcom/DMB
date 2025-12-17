"""
d001_kernel.py — Domino Builder Kernel (contracts, registry, discovery, safety gates)

This file is intentionally "content-agnostic":
- No hardcoded market data schemas (not "only OHLCV", etc.)
- No hardcoded strategy lists
- No external APIs
- Designed for "latest wins" overrides via registry + versioning, without editing older files.

How this scales with the project:
- Future files (d002_*.py, d003_*.py, ...) register components at import-time:
    import d001_kernel as k
    k.REGISTRY.register(kind="strategy", name="my_strategy", version="1.0.0", provider=MyStrategy)

- The kernel can discover and import all d*.py modules in numeric order (plugin pattern)
  without requiring a central list that must be edited.

Safety:
- Live trading is disabled by default.
- Live trading is only "allowed" when ALL gates pass:
    1) nb_local.py contains ENABLE_LIVE_TRADING = True
    2) nb_local.py contains LIVE_TRADING_CONFIRMATION = "I_UNDERSTAND_LIVE_TRADING_RISKS"
    3) Environment variable DMB_LIVE_TRADING is set to YES/TRUE/1
- This file never sends orders; there is no execution code here.

Tests:
    python d001_kernel.py --selftest
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# ---------------------------
# Exceptions
# ---------------------------

class DMBError(Exception):
    """Base error for Domino Builder."""


class RegistryError(DMBError):
    """Registry-related errors."""


class ConfigError(DMBError):
    """Local config load/validation errors."""


class SafetyError(DMBError):
    """Safety gate errors."""


# ---------------------------
# Constants / Contracts
# ---------------------------

KERNEL_VERSION = "0.1.0"

# Local-only (never committed) configuration file. Must be in .gitignore.
NB_LOCAL_FILENAME = "nb_local.py"

# Safety gates
LIVE_TRADING_ENV_VAR = "DMB_LIVE_TRADING"
LIVE_TRADING_CONFIRMATION_STRING = "I_UNDERSTAND_LIVE_TRADING_RISKS"

# Domino modules live in repo root and follow dNNN_*.py naming.
DOMINO_MODULE_RE = re.compile(r"^d(\d{3,})_.+\.py$")


# ---------------------------
# Helpers: canonicalization & stable hashing
# ---------------------------

_JSON_PRIMITIVE = (str, int, bool, type(None))


def _is_pathlike(obj: Any) -> bool:
    return isinstance(obj, (pathlib.Path,)) or hasattr(obj, "__fspath__")


def _as_path_str(obj: Any) -> str:
    try:
        return os.fspath(obj)  # type: ignore[arg-type]
    except Exception:
        return str(obj)


def canonicalize(obj: Any) -> Any:
    """
    Convert an arbitrary Python object into a deterministic, JSON-serializable structure.

    Goals:
    - Stable across dict key order
    - Preserves basic type information where it matters (tuple vs list, bytes, set, float edge cases)
    - Avoids using raw repr() for unknown objects unless it is unavoidable
      (repr() can contain memory addresses and break reproducibility).

    Supported:
    - None, bool, int, str
    - float (finite + NaN/Inf)
    - bytes/bytearray
    - list/tuple/set/frozenset
    - dict with any JSON-ish or simple keys
    - dataclasses
    - objects with to_dict(), __getstate__(), or __dict__ (best-effort)

    Raises:
    - TypeError when it cannot safely canonicalize
    """
    if isinstance(obj, _JSON_PRIMITIVE):
        return obj

    if isinstance(obj, float):
        # JSON float formatting is stable, but NaN/Inf are not standard JSON. Preserve them explicitly.
        if math.isnan(obj):
            return {"__t": "float", "v": "nan"}
        if math.isinf(obj):
            return {"__t": "float", "v": "inf" if obj > 0 else "-inf"}
        # Use repr for stable round-tripping representation
        return {"__t": "float", "repr": repr(obj)}

    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
        return {"__t": "bytes", "b64": base64.b64encode(b).decode("ascii")}

    if _is_pathlike(obj):
        return {"__t": "path", "v": _as_path_str(obj)}

    if isinstance(obj, (list, tuple)):
        items = [canonicalize(x) for x in obj]
        if isinstance(obj, list):
            return {"__t": "list", "v": items}
        return {"__t": "tuple", "v": items}

    if isinstance(obj, (set, frozenset)):
        items = [canonicalize(x) for x in obj]
        # Sort deterministically by canonical JSON string of each item
        items_sorted = sorted(items, key=lambda x: dumps_canonical(x))
        return {"__t": "set", "v": items_sorted, "frozen": isinstance(obj, frozenset)}

    if isinstance(obj, dict):
        # Dict keys may be non-strings. Represent dict as a list of key/value pairs with canonical keys.
        items: List[Tuple[str, Any]] = []
        for k, v in obj.items():
            ck = canonicalize(k)
            # key_token must be a string for deterministic sorting
            key_token = dumps_canonical(ck)
            items.append((key_token, canonicalize(v)))
        items.sort(key=lambda kv: kv[0])
        return {"__t": "dict", "items": [[k, v] for k, v in items]}

    # Dataclasses: stable conversion via asdict (still needs canonicalization)
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            cls = obj.__class__
            payload = dataclasses.asdict(obj)
            return {
                "__t": "dataclass",
                "cls": f"{cls.__module__}.{cls.__qualname__}",
                "v": canonicalize(payload),
            }
    except Exception:
        # fallthrough to other strategies
        pass

    # to_dict hook
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            payload = obj.to_dict()
            return {
                "__t": "to_dict",
                "cls": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "v": canonicalize(payload),
            }
        except Exception as e:
            raise TypeError(f"to_dict() failed for {obj.__class__.__name__}: {e}") from e

    # __getstate__ hook
    if hasattr(obj, "__getstate__") and callable(getattr(obj, "__getstate__")):
        try:
            payload = obj.__getstate__()  # type: ignore[misc]
            return {
                "__t": "getstate",
                "cls": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "v": canonicalize(payload),
            }
        except Exception as e:
            raise TypeError(f"__getstate__() failed for {obj.__class__.__name__}: {e}") from e

    # __dict__ best-effort
    if hasattr(obj, "__dict__"):
        try:
            payload = dict(obj.__dict__)
            return {
                "__t": "object",
                "cls": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "v": canonicalize(payload),
            }
        except Exception as e:
            raise TypeError(f"__dict__ canonicalization failed for {obj.__class__.__name__}: {e}") from e

    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not canonicalizable. "
        "Provide a to_dict/__getstate__ or use only JSON-serializable primitives."
    )


def dumps_canonical(obj: Any) -> str:
    """
    Deterministic JSON dump of the canonicalized representation.
    """
    canon = canonicalize(obj)
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, *, digest: str = "sha256", salt: Optional[str] = None) -> str:
    """
    Deterministic content hash of an object based on its canonical JSON representation.
    """
    payload = dumps_canonical(obj)
    if salt:
        payload = f"{salt}\n{payload}"
    h = hashlib.new(digest)
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


# ---------------------------
# Local config loader (nb_local.py)
# ---------------------------

@dataclass(frozen=True)
class LocalConfigView:
    """
    Read-only view over nb_local.py with safe defaults.

    Notes:
    - nb_local.py is optional and must not be committed.
    - This view never exposes secrets; it only provides runtime access.
    """
    module: Optional[Any]
    defaults: Mapping[str, Any]
    load_error: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        if self.module is not None and hasattr(self.module, key):
            return getattr(self.module, key)
        if key in self.defaults:
            return self.defaults[key]
        return default

    def as_dict(self) -> Dict[str, Any]:
        # Prefer module attributes where available; fall back to defaults.
        out: Dict[str, Any] = dict(self.defaults)
        if self.module is not None:
            for k in dir(self.module):
                if k.startswith("_"):
                    continue
                try:
                    out[k] = getattr(self.module, k)
                except Exception:
                    # ignore weird descriptors
                    pass
        if self.load_error:
            out["_nb_local_load_error"] = self.load_error
        return out


def repo_root() -> pathlib.Path:
    """
    Repo root is assumed to be the directory containing this file (no subfolders by contract).
    """
    return pathlib.Path(__file__).resolve().parent


def _load_module_from_path(module_name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ConfigError(f"Could not create spec for {module_name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure relative imports inside nb_local.py (if any) won't accidentally happen.
    # We keep it minimal: execute the module in isolation.
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_nb_local(root: Optional[pathlib.Path] = None) -> LocalConfigView:
    """
    Load nb_local.py if present. If missing, return defaults.
    If present but fails to load, return defaults plus load_error.
    """
    root = root or repo_root()
    defaults: Dict[str, Any] = {
        "ENABLE_LIVE_TRADING": False,
        "LIVE_TRADING_CONFIRMATION": "",
        # Resource controls (optional)
        "PARALLEL_WORKERS_CAP": 32,
        "MAX_PARALLEL_WORKERS": None,
        "RANDOM_SEED": 1337,
    }

    path = root / NB_LOCAL_FILENAME
    if not path.exists():
        return LocalConfigView(module=None, defaults=defaults, load_error=None)

    try:
        mod = _load_module_from_path("nb_local", path)
        return LocalConfigView(module=mod, defaults=defaults, load_error=None)
    except Exception as e:
        # For safety, treat config load failures as "safe defaults" and report the error.
        err = f"{type(e).__name__}: {e}"
        return LocalConfigView(module=None, defaults=defaults, load_error=err)


# ---------------------------
# Safety gates
# ---------------------------

def live_trading_allowed(cfg: Optional[Union[LocalConfigView, Mapping[str, Any]]] = None) -> Tuple[bool, str]:
    """
    Determine whether live trading is permitted.

    Returns:
        (allowed, reason)

    Gate conditions (ALL must pass):
    - ENABLE_LIVE_TRADING == True in nb_local.py
    - LIVE_TRADING_CONFIRMATION matches LIVE_TRADING_CONFIRMATION_STRING
    - Environment variable DMB_LIVE_TRADING is YES/TRUE/1
    """
    if cfg is None:
        cfg = load_nb_local()

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(cfg, LocalConfigView):
            return cfg.get(key, default)
        return cfg.get(key, default)  # type: ignore[union-attr]

    if not bool(_get("ENABLE_LIVE_TRADING", False)):
        return False, "Gate 1/3 failed: ENABLE_LIVE_TRADING is not True (nb_local.py)."

    if str(_get("LIVE_TRADING_CONFIRMATION", "")).strip() != LIVE_TRADING_CONFIRMATION_STRING:
        return False, (
            "Gate 2/3 failed: LIVE_TRADING_CONFIRMATION mismatch (nb_local.py). "
            f"Expected: {LIVE_TRADING_CONFIRMATION_STRING!r}."
        )

    env = str(os.environ.get(LIVE_TRADING_ENV_VAR, "")).strip().upper()
    if env not in {"1", "YES", "TRUE"}:
        return False, (
            f"Gate 3/3 failed: environment variable {LIVE_TRADING_ENV_VAR} not set to YES/TRUE/1."
        )

    return True, "Live trading permitted (ALL gates passed)."


# ---------------------------
# Resource budgeting
# ---------------------------

def cpu_count() -> int:
    c = os.cpu_count() or 1
    return max(1, int(c))


def recommended_max_workers(cfg: Optional[LocalConfigView] = None) -> int:
    """
    Conservative default: leave one core free, cap via nb_local.py.
    Users can override explicitly with MAX_PARALLEL_WORKERS in nb_local.py.
    """
    cfg = cfg or load_nb_local()
    explicit = cfg.get("MAX_PARALLEL_WORKERS", None)
    if explicit is not None:
        try:
            n = int(explicit)
            return max(1, n)
        except Exception:
            # Ignore invalid explicit override and fall back
            pass

    cap = cfg.get("PARALLEL_WORKERS_CAP", 32)
    try:
        cap_i = max(1, int(cap))
    except Exception:
        cap_i = 32

    suggest = max(1, cpu_count() - 1)
    return min(suggest, cap_i)


# ---------------------------
# Registry
# ---------------------------

def normalize_version(v: Any) -> Tuple[int, ...]:
    """
    Normalize version into a comparable tuple of ints.

    Accepts:
    - None -> (0,)
    - int -> (int,)
    - tuple/list of ints -> tuple(int, ...)
    - str like '1.2.3' -> (1,2,3)
    Any non-numeric parts are ignored.
    """
    if v is None:
        return (0,)
    if isinstance(v, int):
        return (int(v),)
    if isinstance(v, (tuple, list)):
        out: List[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(out) if out else (0,)
    if isinstance(v, str):
        parts = re.split(r"[^\d]+", v.strip())
        nums = [int(p) for p in parts if p.isdigit()]
        return tuple(nums) if nums else (0,)
    # Fallback: try int conversion, else 0
    try:
        return (int(v),)
    except Exception:
        return (0,)


@dataclass(frozen=True)
class RegisteredItem:
    kind: str
    name: str
    version: Tuple[int, ...]
    provider: Any
    meta: Mapping[str, Any]
    source: str
    ordinal: int

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """
        Instantiate/produce the component if provider is callable, else return as-is.
        """
        if callable(self.provider):
            return self.provider(*args, **kwargs)
        return self.provider


class Registry:
    """
    A simple, deterministic component registry.

    "Latest wins" resolution:
    - higher normalized version wins
    - if versions equal, higher ordinal wins (later registration)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: Dict[Tuple[str, str], List[RegisteredItem]] = {}
        self._ordinal = 0

    def register(
        self,
        *,
        kind: str,
        name: str,
        provider: Any,
        version: Any = None,
        meta: Optional[Mapping[str, Any]] = None,
        source: Optional[str] = None,
    ) -> RegisteredItem:
        if not kind or not isinstance(kind, str):
            raise RegistryError("kind must be a non-empty string")
        if not name or not isinstance(name, str):
            raise RegistryError("name must be a non-empty string")

        v = normalize_version(version)
        m = dict(meta or {})
        src = source or _infer_caller_source()

        with self._lock:
            self._ordinal += 1
            item = RegisteredItem(
                kind=kind,
                name=name,
                version=v,
                provider=provider,
                meta=m,
                source=src,
                ordinal=self._ordinal,
            )
            key = (kind, name)
            self._items.setdefault(key, []).append(item)
            return item

    def list(self, *, kind: Optional[str] = None) -> List[RegisteredItem]:
        with self._lock:
            items = []
            for (k, _), lst in self._items.items():
                if kind is None or k == kind:
                    items.extend(lst)
            # Stable order: kind, name, version, ordinal
            return sorted(items, key=lambda it: (it.kind, it.name, it.version, it.ordinal))

    def resolve(self, *, kind: str, name: str) -> RegisteredItem:
        key = (kind, name)
        with self._lock:
            if key not in self._items or not self._items[key]:
                raise RegistryError(f"No entry registered for kind={kind!r}, name={name!r}")
            lst = list(self._items[key])

        # Latest wins: max by (version, ordinal)
        return max(lst, key=lambda it: (it.version, it.ordinal))

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a canonicalizable snapshot of the registry (no providers included).
        Useful for hashing experiment state without serializing callables.
        """
        snap: Dict[str, Any] = {}
        for it in self.list():
            snap.setdefault(it.kind, {}).setdefault(it.name, []).append(
                {
                    "version": list(it.version),
                    "meta": dict(it.meta),
                    "source": it.source,
                    "ordinal": it.ordinal,
                    "provider_type": f"{type(it.provider).__module__}.{type(it.provider).__qualname__}",
                }
            )
        return snap


def _infer_caller_source() -> str:
    """
    Best-effort source inference for debugging.
    """
    try:
        # Walk up stack frames to find a non-kernel file
        # Using traceback.extract_stack is stable enough for debug labels.
        stack = traceback.extract_stack(limit=10)
        # Last frame is here; walk backwards to find first outside this file.
        this = pathlib.Path(__file__).resolve()
        for fr in reversed(stack[:-1]):
            p = pathlib.Path(fr.filename).resolve()
            if p != this:
                return f"{p.name}:{fr.lineno}"
    except Exception:
        pass
    return "unknown"


# Global registry instance (contract)
REGISTRY = Registry()


# ---------------------------
# Discovery / Plugin loading
# ---------------------------

@dataclass(frozen=True)
class DiscoveryResult:
    root: str
    seen_files: List[str]
    loaded_modules: List[str]
    skipped_files: List[str]
    errors: List[str]

    def ok(self) -> bool:
        return not self.errors


class _SysPathContext:
    def __init__(self, path: str) -> None:
        self._path = path
        self._added = False

    def __enter__(self) -> None:
        if self._path not in sys.path:
            sys.path.insert(0, self._path)
            self._added = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._added:
            try:
                sys.path.remove(self._path)
            except ValueError:
                pass


def _module_loaded_from_file(file_path: pathlib.Path) -> bool:
    """
    Prevent importing the same file twice (e.g. when running as __main__).
    """
    target = str(file_path.resolve())
    for mod in list(sys.modules.values()):
        try:
            mfile = getattr(mod, "__file__", None)
            if mfile and str(pathlib.Path(mfile).resolve()) == target:
                return True
        except Exception:
            continue
    return False


def discover(root: Optional[Union[str, pathlib.Path]] = None, *, strict: bool = False) -> DiscoveryResult:
    """
    Discover and import domino modules in numeric order based on filename.

    strict=False:
        - Import errors are collected but do not raise.
    strict=True:
        - First import error raises an exception.

    Returns a DiscoveryResult with loaded modules and errors.
    """
    root_path = pathlib.Path(root).resolve() if root is not None else repo_root()
    if not root_path.exists() or not root_path.is_dir():
        raise DMBError(f"Discovery root does not exist or is not a directory: {root_path}")

    files = [p for p in root_path.iterdir() if p.is_file() and DOMINO_MODULE_RE.match(p.name)]
    files_sorted = sorted(files, key=lambda p: (int(DOMINO_MODULE_RE.match(p.name).group(1)), p.name))  # type: ignore[union-attr]

    seen_files: List[str] = [p.name for p in files_sorted]
    loaded: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    with _SysPathContext(str(root_path)):
        for p in files_sorted:
            mod_name = p.stem
            if _module_loaded_from_file(p):
                skipped.append(p.name)
                continue
            try:
                importlib.import_module(mod_name)
                loaded.append(mod_name)
            except Exception as e:
                msg = f"{p.name}: {type(e).__name__}: {e}"
                errors.append(msg)
                if strict:
                    raise
                continue

    return DiscoveryResult(
        root=str(root_path),
        seen_files=seen_files,
        loaded_modules=loaded,
        skipped_files=skipped,
        errors=errors,
    )


# ---------------------------
# Guardrails / Repo hygiene checks
# ---------------------------

def assert_gitignore_protects_nb_local(root: Optional[pathlib.Path] = None) -> None:
    """
    Enforce that nb_local.py is ignored by git (must never be committed).
    Fails fast if .gitignore is missing or doesn't mention nb_local.py.
    """
    root = root or repo_root()
    gi = root / ".gitignore"
    if not gi.exists():
        raise AssertionError("Missing .gitignore in repo root. It must ignore nb_local.py.")

    txt = gi.read_text(encoding="utf-8", errors="replace")
    if "nb_local.py" not in txt:
        raise AssertionError(
            "Safety requirement failed: .gitignore does not contain 'nb_local.py'.\n"
            "Add a line to .gitignore:\n"
            "    nb_local.py\n"
            "Then rerun the selftest."
        )


# ---------------------------
# Self-test
# ---------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest() -> int:
    """
    Minimal offline deterministic tests (<30s).
    Returns process exit code.
    """
    print(f"[d001] Kernel selftest — version {KERNEL_VERSION}")
    print(f"[d001] Repo root: {repo_root()}")

    # 1) Hash determinism (dict order independent)
    h1 = stable_hash({"b": 2, "a": 1})
    h2 = stable_hash({"a": 1, "b": 2})
    _assert(h1 == h2, "stable_hash must be independent of dict key order")
    print("[d001] OK: stable_hash is deterministic for dict ordering")

    # 2) Registry 'latest wins'
    reg = Registry()
    reg.register(kind="k", name="n", version="1.0.0", provider=lambda: "v1")
    reg.register(kind="k", name="n", version="2.0.0", provider=lambda: "v2")
    best = reg.resolve(kind="k", name="n")
    _assert(best.create() == "v2", "Registry should resolve to highest version")
    # Tie-breaker: same version, later ordinal wins
    reg.register(kind="k", name="n", version="2.0.0", provider=lambda: "v2b")
    best2 = reg.resolve(kind="k", name="n")
    _assert(best2.create() == "v2b", "Registry should resolve to later registration when versions tie")
    print("[d001] OK: registry latest-wins resolution")

    # 3) Safety gate defaults
    cfg = load_nb_local()
    allowed, reason = live_trading_allowed(cfg)
    _assert(not allowed, "Live trading must be disabled by default")
    print(f"[d001] OK: live trading disabled by default ({reason})")

    # 4) Safety gate requires all 3 gates (simulate config)
    mock_cfg = {
        "ENABLE_LIVE_TRADING": True,
        "LIVE_TRADING_CONFIRMATION": LIVE_TRADING_CONFIRMATION_STRING,
    }
    # Without env var -> still false
    prev = os.environ.get(LIVE_TRADING_ENV_VAR)
    try:
        if LIVE_TRADING_ENV_VAR in os.environ:
            del os.environ[LIVE_TRADING_ENV_VAR]
        allowed2, reason2 = live_trading_allowed(mock_cfg)
        _assert(not allowed2, "Live trading must require env var gate as well")
        print(f"[d001] OK: live trading still blocked without env var ({reason2})")

        os.environ[LIVE_TRADING_ENV_VAR] = "YES"
        allowed3, reason3 = live_trading_allowed(mock_cfg)
        _assert(allowed3, "Live trading should be allowed only when all gates pass")
        print(f"[d001] OK: live trading allowed when all gates pass ({reason3})")
    finally:
        if prev is None:
            os.environ.pop(LIVE_TRADING_ENV_VAR, None)
        else:
            os.environ[LIVE_TRADING_ENV_VAR] = prev

    # 5) .gitignore must ignore nb_local.py (hard requirement)
    assert_gitignore_protects_nb_local()
    print("[d001] OK: .gitignore protects nb_local.py")

    # 6) Discovery smoke test
    dr = discover(strict=False)
    # Ensure our file is at least seen by naming rules.
    _assert("d001_kernel.py" in dr.seen_files, "Discovery must see d001_kernel.py in repo root")
    # When running as __main__, it will likely be skipped due to already loaded from path.
    _assert(
        ("d001_kernel.py" in dr.skipped_files) or ("d001_kernel" in dr.loaded_modules),
        "Discovery must either load or skip d001_kernel.py (already loaded is OK)",
    )
    if dr.errors:
        print("[d001] WARNING: discovery import errors (strict=False):")
        for e in dr.errors:
            print(f"  - {e}")
    print("[d001] OK: discovery smoke test")

    # 7) Resource budget sanity
    mw = recommended_max_workers(cfg)
    _assert(isinstance(mw, int) and mw >= 1, "recommended_max_workers must be >= 1")
    print(f"[d001] OK: recommended_max_workers={mw} (cpu_count={cpu_count()})")

    # Snapshot hash (debug visibility)
    snap_h = stable_hash({"registry": REGISTRY.snapshot(), "kernel": KERNEL_VERSION})
    print(f"[d001] Snapshot hash (debug): {snap_h}")

    print("[d001] SELFTEST PASS")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        try:
            return selftest()
        except AssertionError as e:
            print("[d001] SELFTEST FAIL:", e)
            return 2
        except Exception as e:
            print("[d001] SELFTEST ERROR:", f"{type(e).__name__}: {e}")
            return 3

    # Minimal CLI help
    print("d001_kernel.py")
    print("Usage:")
    print("  python d001_kernel.py --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
