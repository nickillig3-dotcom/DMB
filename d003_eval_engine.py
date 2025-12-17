"""
d003_eval_engine.py — Offline Evaluation Engine (Signal -> Position -> PnL) + Leakage Guards

Contracts > Contents:
- Data-agnostic dataset contract: leaf_id -> numeric time series (no OHLCV assumptions).
- Deterministic evaluator: DSL expression -> signal series -> position series -> PnL series.
- Robustness / safety gates for research (NOT trading):
    - Live trading is irrelevant here; evaluator is fully offline.
    - Target-leakage guard blocks using the target return leaf inside the strategy expression by default.
    - Optional forbidden leaf prefixes/ids block common leakage patterns (configurable).

Integration:
- Imports d001_kernel (registry, hashing, discovery, local config).
- Imports d002_strategy_dsl (StrategySpec + Expr evaluation + leaf-id extraction).
- Registers an evaluator component in d001_kernel.REGISTRY under kind="evaluator"
  so later files can override it ("latest wins") without editing existing files.

Selftest:
    python d003_eval_engine.py --selftest
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import d001_kernel as k
import d002_strategy_dsl as dsl


EVAL_ENGINE_VERSION = "0.1.0"

REG_KIND_EVALUATOR = "evaluator"
REG_EVALUATOR_NAME = "simple_signal_pnl_v1"

# Conservative epsilons for numerical stability
_EPS = 1e-12


# ---------------------------
# Dataset Contracts
# ---------------------------

class Dataset(Protocol):
    """
    Data-agnostic dataset contract.

    A dataset provides numeric time series keyed by leaf_id (string).
    No assumption is made about which leaf_ids exist or what they represent.
    """

    def length(self) -> int:
        ...

    def series_ids(self) -> Sequence[str]:
        ...

    def get_series(self, leaf_id: str) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class InMemoryDataset:
    """
    Minimal in-memory dataset implementation for offline research/tests.

    Validates that all series have equal length.
    """
    series: Mapping[str, Sequence[float]]
    meta: Mapping[str, Any] = field(default_factory=dict)
    _length: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lengths = set(len(v) for v in self.series.values())
        if not lengths:
            object.__setattr__(self, "_length", 0)
            return
        if len(lengths) != 1:
            # Keep message short but actionable
            raise ValueError(f"InMemoryDataset: series length mismatch: {sorted(lengths)}")
        object.__setattr__(self, "_length", int(next(iter(lengths))))

    def length(self) -> int:
        return self._length

    def series_ids(self) -> Sequence[str]:
        return sorted(self.series.keys())

    def get_series(self, leaf_id: str) -> Sequence[float]:
        return self.series[leaf_id]


def dataset_fingerprint(ds: Dataset, *, sample_n: int = 3) -> str:
    """
    A cheap, deterministic dataset fingerprint (not a full hash of all points).

    Intended to:
    - make evaluation artifacts reproducible and auditable
    - remain PC-friendly even when datasets become large

    Fingerprint includes:
    - length
    - leaf_ids
    - head/tail samples per series (sample_n values)
    """
    leafs = list(ds.series_ids())
    L = int(ds.length())
    sample: Dict[str, Dict[str, List[float]]] = {}
    for leaf in leafs:
        s = ds.get_series(leaf)
        head = [float(x) for x in list(s[:sample_n])]
        tail = [float(x) for x in list(s[-sample_n:])] if L >= sample_n else [float(x) for x in list(s)]
        sample[leaf] = {"head": head, "tail": tail}
    payload = {
        "t": "dataset_fingerprint_v1",
        "len": L,
        "leafs": leafs,
        "sample_n": int(sample_n),
        "sample": sample,
    }
    return k.stable_hash(payload, salt="dataset_fingerprint_v1")


# ---------------------------
# Evaluation Config + ScoreCard
# ---------------------------

@dataclass(frozen=True)
class EvalConfig:
    """
    Offline evaluation configuration.

    target_return_leaf:
        The leaf_id of the return series to score against (a "label").
    forbidden_leaf_prefixes / forbidden_leaf_ids:
        Simple leakage guards. If an expression uses such a leaf, evaluation is rejected by default.
    allow_target_leaf_in_expr:
        Defaults to False. If True, leakage guards won't reject target_return_leaf usage.
        WARNING: enabling this can allow trivial target leakage; keep False for research safety.
    transaction_cost:
        Simple turnover penalty applied on position changes (proxy, unitless).
    position_mode:
        "sign" for -1/0/+1, "raw" for continuous position (still penalized by turnover).
    signal_clip:
        Optional absolute clip for signal values before position transform (limits outliers).
    min_obs:
        Minimum number of observations required.
    min_coverage:
        Minimum fraction of finite (signal & return) points required.
    """
    target_return_leaf: str = "target.ret"
    forbidden_leaf_prefixes: Tuple[str, ...] = ("target.", "label.", "future.")
    forbidden_leaf_ids: Tuple[str, ...] = ()
    allow_target_leaf_in_expr: bool = False

    transaction_cost: float = 0.0
    position_mode: str = "sign"  # "sign" | "raw"
    signal_clip: Optional[float] = 10.0

    min_obs: int = 64
    min_coverage: float = 0.98

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "eval_config",
            "target_return_leaf": self.target_return_leaf,
            "forbidden_leaf_prefixes": list(self.forbidden_leaf_prefixes),
            "forbidden_leaf_ids": list(self.forbidden_leaf_ids),
            "allow_target_leaf_in_expr": bool(self.allow_target_leaf_in_expr),
            "transaction_cost": float(self.transaction_cost),
            "position_mode": self.position_mode,
            "signal_clip": None if self.signal_clip is None else float(self.signal_clip),
            "min_obs": int(self.min_obs),
            "min_coverage": float(self.min_coverage),
        }

    def cid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="eval_config_v1")


@dataclass(frozen=True)
class ScoreCard:
    """
    Minimal strategy evaluation result.

    Note:
    - This is NOT a profit promise; it's a deterministic scorecard for comparing candidates
      on the same dataset and config.
    """
    strategy_id: str
    ok: bool
    reason: str

    n: int
    coverage: float

    mean: float
    stdev: float
    info_ratio: float

    max_drawdown: float
    turnover: float

    score: float

    dataset_fp: str
    config_id: str

    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "scorecard",
            "v": 1,
            "strategy_id": self.strategy_id,
            "ok": bool(self.ok),
            "reason": self.reason,
            "n": int(self.n),
            "coverage": float(self.coverage),
            "mean": float(self.mean),
            "stdev": float(self.stdev),
            "info_ratio": float(self.info_ratio),
            "max_drawdown": float(self.max_drawdown),
            "turnover": float(self.turnover),
            "score": float(self.score),
            "dataset_fp": self.dataset_fp,
            "config_id": self.config_id,
            "meta": dict(self.meta),
        }

    def sid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="scorecard_v1")


# ---------------------------
# Leakage / Validity checks
# ---------------------------

def detect_leakage(expr: dsl.Expr, cfg: EvalConfig) -> Tuple[bool, str]:
    """
    Returns (is_leakage, reason).
    """
    leaf_ids = dsl.expr_leaf_ids(expr)

    if not cfg.allow_target_leaf_in_expr:
        if cfg.target_return_leaf in leaf_ids:
            return True, f"Target leakage: expression uses target_return_leaf={cfg.target_return_leaf!r}"

    for bad in cfg.forbidden_leaf_ids:
        if bad in leaf_ids:
            return True, f"Forbidden leaf id used: {bad!r}"

    for pfx in cfg.forbidden_leaf_prefixes:
        if any(lid.startswith(pfx) for lid in leaf_ids):
            # If the prefix is the target prefix and allow_target_leaf_in_expr=True, still block others.
            if cfg.allow_target_leaf_in_expr and pfx and cfg.target_return_leaf.startswith(pfx):
                # Only exempt exact target leaf; other prefixed leaves remain forbidden.
                other = [lid for lid in leaf_ids if lid.startswith(pfx) and lid != cfg.target_return_leaf]
                if other:
                    return True, f"Forbidden leaf prefix used: {pfx!r} (examples: {other[:3]!r})"
                continue
            return True, f"Forbidden leaf prefix used: {pfx!r}"

    return False, ""


def _isfinite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


# ---------------------------
# Core evaluation logic
# ---------------------------

def evaluate_strategy(
    spec: dsl.StrategySpec,
    ds: Dataset,
    *,
    cfg: EvalConfig,
) -> ScoreCard:
    """
    Evaluate one StrategySpec on one Dataset deterministically.

    Mechanics (simple, offline):
    - signal[t] = eval(expr) using features at time t
    - position[t] = sign(signal[t]) or raw(signal[t])
    - pnl[t] = position[t-1] * ret[t] - cost * abs(position[t-1] - position[t-2])
      (pnl[0] = 0, avoids lookahead because it uses the *previous* position)

    Returns a ScoreCard. If evaluation is rejected/invalid, ok=False with reason and score=-inf.
    """
    strategy_id = spec.sid()

    # Dataset checks
    L = int(ds.length())
    ds_fp = dataset_fingerprint(ds)
    cfg_id = cfg.cid()

    if L < int(cfg.min_obs):
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason=f"Not enough observations: L={L} < min_obs={cfg.min_obs}",
            n=max(0, L - 1),
            coverage=0.0,
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION},
        )

    # Leakage guard
    leak, leak_reason = detect_leakage(spec.expr, cfg)
    if leak:
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason=leak_reason,
            n=L - 1,
            coverage=0.0,
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION, "guard": "leakage"},
        )

    # Fetch target return series
    try:
        ret = ds.get_series(cfg.target_return_leaf)
    except Exception as e:
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason=f"Missing target_return_leaf={cfg.target_return_leaf!r}: {type(e).__name__}: {e}",
            n=L - 1,
            coverage=0.0,
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION},
        )

    # Leaf resolver for DSL evaluation
    def resolver(leaf_id: str, leaf_ctx: Mapping[str, Any]) -> float:
        t = int(leaf_ctx["t"])
        s = ds.get_series(leaf_id)
        return float(s[t])

    # Compute signal series
    signals: List[float] = [0.0] * L
    bad_signal = 0
    clip = None if cfg.signal_clip is None else float(cfg.signal_clip)

    for t in range(L):
        try:
            s = float(dsl.eval_expr(spec.expr, leaf_resolver=resolver, leaf_ctx={"t": t}))
        except Exception:
            s = float("nan")

        if not _isfinite(s):
            bad_signal += 1
            s = 0.0

        if clip is not None:
            if s > clip:
                s = clip
            elif s < -clip:
                s = -clip

        signals[t] = s

    # Compute position series
    positions: List[float] = [0.0] * L
    if cfg.position_mode == "sign":
        for t in range(L):
            positions[t] = float(_sign(signals[t]))
    elif cfg.position_mode == "raw":
        for t in range(L):
            positions[t] = float(signals[t])
    else:
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason=f"Unknown position_mode={cfg.position_mode!r}",
            n=L - 1,
            coverage=0.0,
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION},
        )

    # Compute pnl (shifted to avoid lookahead)
    pnl: List[float] = [0.0] * L
    turnover_sum = 0.0
    bad_ret = 0
    cost = float(cfg.transaction_cost)

    for t in range(L):
        r = float(ret[t])
        if not _isfinite(r):
            bad_ret += 1
            r = 0.0

        if t == 0:
            pnl[t] = 0.0
            continue

        prev_pos = float(positions[t - 1])
        prev_prev_pos = float(positions[t - 2]) if t >= 2 else 0.0
        turn = abs(prev_pos - prev_prev_pos)
        turnover_sum += turn

        pnl[t] = prev_pos * r - cost * turn

    # Coverage
    coverage_signal = 1.0 - (bad_signal / float(L))
    coverage_ret = 1.0 - (bad_ret / float(L))
    coverage = min(coverage_signal, coverage_ret)

    if coverage < float(cfg.min_coverage):
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason=f"Coverage too low: {coverage:.6f} < min_coverage={cfg.min_coverage}",
            n=L - 1,
            coverage=float(coverage),
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=float(turnover_sum / max(1, L - 1)),
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION, "bad_signal": bad_signal, "bad_ret": bad_ret},
        )

    # Metrics on pnl[1:] (pnl[0] is always 0 due to shift)
    series = pnl[1:]
    n = len(series)
    if n <= 0:
        return ScoreCard(
            strategy_id=strategy_id,
            ok=False,
            reason="No pnl samples after shift",
            n=0,
            coverage=float(coverage),
            mean=0.0,
            stdev=0.0,
            info_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            score=float("-inf"),
            dataset_fp=ds_fp,
            config_id=cfg_id,
            meta={"engine": EVAL_ENGINE_VERSION},
        )

    mean = math.fsum(series) / float(n)

    # population stdev (deterministic and defined for n=1)
    var = math.fsum((x - mean) * (x - mean) for x in series) / float(n)
    stdev = math.sqrt(max(0.0, var))

    info_ratio = mean / (stdev + _EPS)

    # Drawdown on cumulative sum equity curve
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in series:
        equity += float(x)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    turnover = turnover_sum / float(max(1, L - 1))

    # A simple, deterministic score (research-only):
    # - prefer higher information ratio
    # - penalize turnover (cost proxy) and drawdown (stability proxy)
    dd_norm = max_dd / (stdev * math.sqrt(float(n)) + _EPS)
    score = float(coverage) * (info_ratio - 0.10 * turnover - 0.10 * dd_norm)

    return ScoreCard(
        strategy_id=strategy_id,
        ok=True,
        reason="OK",
        n=int(n),
        coverage=float(coverage),
        mean=float(mean),
        stdev=float(stdev),
        info_ratio=float(info_ratio),
        max_drawdown=float(max_dd),
        turnover=float(turnover),
        score=float(score),
        dataset_fp=ds_fp,
        config_id=cfg_id,
        meta={
            "engine": EVAL_ENGINE_VERSION,
            "bad_signal": int(bad_signal),
            "bad_ret": int(bad_ret),
            "expr_depth": int(dsl.expr_depth(spec.expr)),
            "expr_nodes": int(dsl.expr_node_count(spec.expr)),
        },
    )


def evaluate_many(
    specs: Sequence[dsl.StrategySpec],
    ds: Dataset,
    *,
    cfg: EvalConfig,
) -> List[ScoreCard]:
    """
    Evaluate multiple StrategySpecs sequentially (deterministic, PC-friendly default).

    Note:
    - We keep this sequential to avoid Windows multiprocessing pitfalls early on.
    - Later domino files can introduce parallel evaluation behind robust gates and budgets.
    """
    out: List[ScoreCard] = []
    for s in specs:
        out.append(evaluate_strategy(s, ds, cfg=cfg))
    return out


# ---------------------------
# Evaluator component (for registry)
# ---------------------------

@dataclass(frozen=True)
class SimpleSignalPnLEvaluator:
    """
    Registry-friendly evaluator wrapper.
    """
    cfg: EvalConfig

    def evaluate(self, spec: dsl.StrategySpec, ds: Dataset) -> ScoreCard:
        return evaluate_strategy(spec, ds, cfg=self.cfg)


def default_eval_config_from_nb_local() -> EvalConfig:
    """
    Construct a default EvalConfig using nb_local.py overrides (optional).

    This is safe:
    - If nb_local.py is missing or fails to load, safe defaults are used.
    - No secrets are printed.

    Supported optional keys in nb_local.py:
      - TARGET_RETURN_LEAF (str)
      - EVAL_FORBIDDEN_LEAF_PREFIXES (tuple/list of str)
      - EVAL_FORBIDDEN_LEAF_IDS (tuple/list of str)
      - EVAL_ALLOW_TARGET_LEAF_IN_EXPR (bool)
      - EVAL_TRANSACTION_COST (float)
      - EVAL_POSITION_MODE (str)
      - EVAL_SIGNAL_CLIP (float or None)
      - EVAL_MIN_OBS (int)
      - EVAL_MIN_COVERAGE (float)
    """
    nb = k.load_nb_local()
    target_leaf = str(nb.get("TARGET_RETURN_LEAF", "target.ret"))
    prefixes = nb.get("EVAL_FORBIDDEN_LEAF_PREFIXES", ("target.", "label.", "future."))
    ids = nb.get("EVAL_FORBIDDEN_LEAF_IDS", ())
    allow_target = bool(nb.get("EVAL_ALLOW_TARGET_LEAF_IN_EXPR", False))
    tcost = float(nb.get("EVAL_TRANSACTION_COST", 0.0))
    pmode = str(nb.get("EVAL_POSITION_MODE", "sign"))
    clip = nb.get("EVAL_SIGNAL_CLIP", 10.0)
    clip_v: Optional[float]
    if clip is None:
        clip_v = None
    else:
        try:
            clip_v = float(clip)
        except Exception:
            clip_v = 10.0

    min_obs = int(nb.get("EVAL_MIN_OBS", 64))
    min_cov = float(nb.get("EVAL_MIN_COVERAGE", 0.98))

    # Normalize prefixes/ids
    if isinstance(prefixes, (list, tuple)):
        pfx = tuple(str(x) for x in prefixes)
    else:
        pfx = (str(prefixes),)

    if isinstance(ids, (list, tuple)):
        bid = tuple(str(x) for x in ids)
    else:
        bid = (str(ids),)

    return EvalConfig(
        target_return_leaf=target_leaf,
        forbidden_leaf_prefixes=pfx,
        forbidden_leaf_ids=bid,
        allow_target_leaf_in_expr=allow_target,
        transaction_cost=tcost,
        position_mode=pmode,
        signal_clip=clip_v,
        min_obs=min_obs,
        min_coverage=min_cov,
    )


def _register_evaluator_once() -> None:
    sentinel = "_dmb_d003_eval_engine_registered_v1"
    if getattr(sys, sentinel, False):
        return
    setattr(sys, sentinel, True)

    # Register a default evaluator factory. Later files can override with higher version.
    k.REGISTRY.register(
        kind=REG_KIND_EVALUATOR,
        name=REG_EVALUATOR_NAME,
        version="1.0.0",
        provider=lambda: SimpleSignalPnLEvaluator(cfg=default_eval_config_from_nb_local()),
        meta={
            "engine": EVAL_ENGINE_VERSION,
            "dsl": "expr_tree",
            "strategy_spec_version": dsl.STRATEGY_SPEC_VERSION,
            "notes": "Offline research evaluator (signal->position->pnl), leakage-guarded.",
        },
        source="d003_eval_engine.py",
    )


_register_evaluator_once()


# ---------------------------
# Selftest
# ---------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_synth_dataset(*, L: int = 512, seed: int = 123) -> InMemoryDataset:
    """
    Create a deterministic synthetic dataset where feature.a has predictive power for target.ret.
    This is ONLY for functional testing (not a profit claim).
    """
    rng = random.Random(int(seed))

    feature_a: List[float] = [rng.gauss(0.0, 1.0) for _ in range(L)]

    # target returns depend on previous feature sign, plus noise.
    target_ret: List[float] = [0.0] * L
    target_ret[0] = 0.0
    for t in range(1, L):
        drift = 0.002 * float(_sign(feature_a[t - 1]))
        noise = 0.001 * rng.gauss(0.0, 1.0)
        target_ret[t] = drift + noise

    # A second (uninformative) feature
    feature_b: List[float] = [rng.gauss(0.0, 1.0) for _ in range(L)]

    return InMemoryDataset(
        series={
            "feature.a": feature_a,
            "feature.b": feature_b,
            "target.ret": target_ret,
        },
        meta={"t": "synth", "L": L, "seed": seed},
    )


def _selftest_registry_integration() -> None:
    dr = k.discover(strict=False)
    _assert("d003_eval_engine.py" in dr.seen_files, "Discovery must see d003_eval_engine.py in repo root")

    item = k.REGISTRY.resolve(kind=REG_KIND_EVALUATOR, name=REG_EVALUATOR_NAME)
    ev = item.create()
    _assert(hasattr(ev, "evaluate") and callable(getattr(ev, "evaluate")), "Registered evaluator must have .evaluate()")


def _selftest_leakage_guard() -> None:
    ds = _make_synth_dataset(L=256, seed=1)
    cfg = EvalConfig(target_return_leaf="target.ret")

    # Strategy that directly uses the target return leaf: should be rejected.
    leak_spec = dsl.StrategySpec(expr=dsl.LeafExpr("target.ret"))
    sc = evaluate_strategy(leak_spec, ds, cfg=cfg)
    _assert(not sc.ok, "Leakage strategy must be rejected")
    _assert("leak" in sc.reason.lower(), "Rejection reason should mention leakage")


def _selftest_determinism() -> None:
    ds = _make_synth_dataset(L=384, seed=7)
    cfg = EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=64, min_coverage=0.98)

    spec = dsl.StrategySpec(expr=dsl.LeafExpr("feature.a"))

    sc1 = evaluate_strategy(spec, ds, cfg=cfg)
    sc2 = evaluate_strategy(spec, ds, cfg=cfg)

    _assert(sc1.to_dict() == sc2.to_dict(), "Evaluation must be deterministic (same config & dataset)")
    _assert(sc1.sid() == sc2.sid(), "ScoreCard IDs must match for deterministic evaluation")


def _selftest_sanity_ranking() -> None:
    ds = _make_synth_dataset(L=512, seed=42)
    cfg = EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=64, min_coverage=0.98)

    spec_signal = dsl.StrategySpec(expr=dsl.LeafExpr("feature.a"))
    spec_zero = dsl.StrategySpec(expr=dsl.ConstExpr(0.0))

    sc_signal = evaluate_strategy(spec_signal, ds, cfg=cfg)
    sc_zero = evaluate_strategy(spec_zero, ds, cfg=cfg)

    _assert(sc_signal.ok and sc_zero.ok, "Both strategies should evaluate OK on synth dataset")
    _assert(sc_signal.score > sc_zero.score, "Predictive signal strategy should outrank zero strategy on synth data")


def selftest() -> int:
    print(f"[d003] Eval engine selftest — version {EVAL_ENGINE_VERSION}")

    _selftest_registry_integration()
    print("[d003] OK: registry integration")

    _selftest_leakage_guard()
    print("[d003] OK: leakage guard")

    _selftest_determinism()
    print("[d003] OK: determinism")

    _selftest_sanity_ranking()
    print("[d003] OK: sanity ranking")

    print("[d003] SELFTEST PASS")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        try:
            return selftest()
        except AssertionError as e:
            print("[d003] SELFTEST FAIL:", e)
            return 2
        except Exception as e:
            print("[d003] SELFTEST ERROR:", f"{type(e).__name__}: {e}")
            return 3

    print("d003_eval_engine.py")
    print("Usage:")
    print("  python d003_eval_engine.py --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
