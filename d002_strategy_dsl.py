"""
d002_strategy_dsl.py — Strategy DSL (Expression Trees) + Budgeted Enumeration

Goal (Contracts > Contents):
- Provide an extensible, data-agnostic Strategy DSL represented as Expression Trees (AST).
- Provide a controlled (budgeted) enumerator that generates StrategySpecs deterministically.
- Avoid hardcoding any market-data schema (no "only OHLCV"), and avoid hardcoded strategy lists.
- Enable later extension by registry overrides ("latest wins") without editing older files.

Integration:
- Uses d001_kernel.REGISTRY to register operator specifications under kind="dsl_op".
- Operators and leaf providers are contracts: later files can register new ops, new leaf providers,
  and new evaluators/backtesters without changing this file or d001_kernel.py.

Safety:
- This module does NOT execute trades, place orders, or contact any external API.

Selftest:
    python d002_strategy_dsl.py --selftest
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import d001_kernel as k


DSL_VERSION = "0.1.0"
STRATEGY_SPEC_VERSION = 1

# Registry kinds (contracts)
REG_KIND_DSL_OP = "dsl_op"
REG_KIND_STRATEGY_ENUMERATOR = "strategy_enumerator"


# ---------------------------
# Contracts: Leaf Specs & Providers
# ---------------------------

@dataclass(frozen=True)
class LeafSpec:
    """
    A feature "leaf" contract for the DSL.

    leaf_id:
        A stable identifier string. Not constrained. Examples:
          - "price.close"
          - "funding.rate"
          - "orderbook.imbalance@1"
          - "custom:my_feature_v2"
    meta:
        Optional metadata (stream name, units, etc.). No schema is enforced here.
    """
    leaf_id: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"t": "leaf_spec", "id": self.leaf_id, "meta": dict(self.meta)}


class LeafProvider(Protocol):
    """
    Contract for producing a set of available LeafSpecs.

    This is the key mechanism that avoids hardcoding any feature list.
    Data adapters can implement this contract later (OHLCV, funding, orderbook, ...),
    without changing the DSL or the enumerator.
    """
    def iter_leaf_specs(self) -> Iterable[LeafSpec]:
        ...


@dataclass(frozen=True)
class StaticLeafProvider:
    """
    Minimal provider for tests and offline development.
    """
    leaves: Sequence[LeafSpec]

    def iter_leaf_specs(self) -> Iterable[LeafSpec]:
        return list(self.leaves)


# ---------------------------
# Contracts: Operator Specs
# ---------------------------

@dataclass(frozen=True)
class OperatorSpec:
    """
    Specification of an operator in the DSL.

    name:
        Stable operator identifier ("add", "mul", "neg", ...). Registry key.
    arity:
        Number of arguments (1 for unary, 2 for binary, etc.).
    commutative:
        If True, argument order is canonicalized to reduce duplicates in enumeration.
    meta:
        Free-form metadata (e.g., numeric stability notes, tags, constraints).
    """
    name: str
    arity: int
    commutative: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "op_spec",
            "name": self.name,
            "arity": int(self.arity),
            "commutative": bool(self.commutative),
            "meta": dict(self.meta),
        }


def _register_default_ops_once() -> None:
    """
    Register a minimal set of generic numeric operators.
    Later files can override by registering the same (kind,name) with higher version.
    """
    sentinel = "_dmb_d002_strategy_dsl_registered_ops_v1"
    if getattr(sys, sentinel, False):
        return
    setattr(sys, sentinel, True)

    default_ops: List[Tuple[OperatorSpec, str]] = [
        (OperatorSpec(name="add", arity=2, commutative=True, meta={"family": "arith"}), "1.0.0"),
        (OperatorSpec(name="sub", arity=2, commutative=False, meta={"family": "arith"}), "1.0.0"),
        (OperatorSpec(name="mul", arity=2, commutative=True, meta={"family": "arith"}), "1.0.0"),
        (OperatorSpec(name="div", arity=2, commutative=False, meta={"family": "arith", "safe": True}), "1.0.0"),
        (OperatorSpec(name="neg", arity=1, commutative=False, meta={"family": "arith"}), "1.0.0"),
    ]

    for spec, ver in default_ops:
        # provider is the OperatorSpec object (not callable) for simplicity and determinism.
        k.REGISTRY.register(
            kind=REG_KIND_DSL_OP,
            name=spec.name,
            version=ver,
            provider=spec,
            meta={"arity": spec.arity, "commutative": spec.commutative, **dict(spec.meta)},
            source="d002_strategy_dsl.py",
        )

    # Optional: register the enumerator factory by name (contract for later orchestration).
    # This doesn't enforce any single enumerator; "latest wins" can replace it.
    k.REGISTRY.register(
        kind=REG_KIND_STRATEGY_ENUMERATOR,
        name="expr_enum_v1",
        version="1.0.0",
        provider=lambda: enumerate_strategies,  # type: ignore[name-defined]
        meta={"dsl": "expr_tree", "version": STRATEGY_SPEC_VERSION},
        source="d002_strategy_dsl.py",
    )


# Register defaults on import (plugin style)
_register_default_ops_once()


def collect_latest_operator_specs(
    *,
    registry: k.Registry = k.REGISTRY,
    kind: str = REG_KIND_DSL_OP,
) -> List[OperatorSpec]:
    """
    Collect the latest OperatorSpec for each operator name from the registry.

    Supports providers that are:
    - OperatorSpec objects (recommended)
    - callables returning OperatorSpec (best-effort)
    """
    items = registry.list(kind=kind)
    best_by_name: Dict[str, k.RegisteredItem] = {}
    for it in items:
        cur = best_by_name.get(it.name)
        if cur is None or (it.version, it.ordinal) > (cur.version, cur.ordinal):
            best_by_name[it.name] = it

    specs: List[OperatorSpec] = []
    for name in sorted(best_by_name.keys()):
        it = best_by_name[name]
        prov = it.provider
        if isinstance(prov, OperatorSpec):
            specs.append(prov)
        elif callable(prov):
            # best-effort: allow factory returning OperatorSpec.
            try:
                obj = prov()
                if isinstance(obj, OperatorSpec):
                    specs.append(obj)
            except Exception:
                continue
    return specs


# ---------------------------
# DSL: Expression Tree
# ---------------------------

class Expr(Protocol):
    """
    Minimal Expression Tree contract.
    """
    def to_dict(self) -> Dict[str, Any]:
        ...

    def children(self) -> Sequence["Expr"]:
        ...


@dataclass(frozen=True)
class LeafExpr:
    leaf_id: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"t": "leaf", "id": self.leaf_id, "meta": dict(self.meta)}

    def children(self) -> Sequence[Expr]:
        return ()


@dataclass(frozen=True)
class ConstExpr:
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        # Value is intentionally "Any" but must be canonicalizable for hashing.
        return {"t": "const", "v": self.value}

    def children(self) -> Sequence[Expr]:
        return ()


@dataclass(frozen=True)
class OpExpr:
    op: str
    args: Tuple[Expr, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {"t": "op", "op": self.op, "args": [a.to_dict() for a in self.args]}

    def children(self) -> Sequence[Expr]:
        return self.args


def expr_depth(e: Expr) -> int:
    ch = list(e.children())
    if not ch:
        return 0
    return 1 + max(expr_depth(c) for c in ch)


def expr_node_count(e: Expr) -> int:
    return 1 + sum(expr_node_count(c) for c in e.children())


def expr_leaf_ids(e: Expr) -> List[str]:
    """
    Collect leaf_ids used by the expression (sorted, unique).
    """
    out: List[str] = []

    def _walk(x: Expr) -> None:
        d = x.to_dict()
        if d.get("t") == "leaf":
            out.append(str(d.get("id")))
        for c in x.children():
            _walk(c)

    _walk(e)
    return sorted(set(out))


def canonicalize_commutative_args(args: Sequence[Expr]) -> Tuple[Expr, ...]:
    """
    Canonicalize commutative operator arguments deterministically.
    """
    # Use a stable key derived from canonical JSON of the expression dict.
    # This avoids dependence on object identity / repr memory addresses.
    keyed = [(k.dumps_canonical(a.to_dict()), a) for a in args]
    keyed.sort(key=lambda kv: kv[0])
    return tuple(a for _, a in keyed)


# ---------------------------
# Strategy Spec
# ---------------------------

@dataclass(frozen=True)
class StrategySpec:
    """
    A minimal, content-agnostic strategy spec.

    For now it holds one expression. Later versions may add fields (risk filters,
    portfolio hints, etc.) by creating new files and registering new spec versions.

    IMPORTANT:
    - The *identity* of a spec is based on its canonical dict form (stable_hash).
    - Do not include unstable metadata (timestamps, random state dumps) if you want stable IDs.
    """
    expr: Expr
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "strategy_spec",
            "v": STRATEGY_SPEC_VERSION,
            "dsl": {"name": "expr_tree", "version": DSL_VERSION},
            "expr": self.expr.to_dict(),
            "meta": dict(self.meta),
        }

    def sid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="strategy_spec")


# ---------------------------
# Budgeted enumeration
# ---------------------------

@dataclass(frozen=True)
class EnumBudget:
    """
    Hard budget controls to keep enumeration PC-friendly and deterministic.

    max_strategies:
        Maximum number of StrategySpecs returned (including atoms).
    max_depth:
        Maximum expression depth.
    max_nodes:
        Maximum nodes per expression.
    max_attempts:
        Max random attempts for composites to avoid infinite loops.
    seed:
        Deterministic seed controlling random composite generation order.
    """
    max_strategies: int = 64
    max_depth: int = 3
    max_nodes: int = 15
    max_attempts: int = 5000
    seed: int = 1337


def enumerate_strategies(
    *,
    leaf_provider: LeafProvider,
    operators: Optional[Sequence[OperatorSpec]] = None,
    const_pool: Sequence[Any] = (0.0, 1.0),
    budget: EnumBudget = EnumBudget(),
) -> List[StrategySpec]:
    """
    Deterministically enumerate StrategySpecs under a strict budget.

    Design:
    - Yield atoms first (all leaves, then constants) in deterministic order.
      This guarantees that expanding the leaf set visibly expands the output.
    - Then generate composite expressions using a seeded RNG over the growing pool.

    Returns:
        A list of StrategySpec up to budget.max_strategies.
    """
    if budget.max_strategies < 1:
        return []

    if operators is None:
        operators = collect_latest_operator_specs()

    ops = [op for op in operators if isinstance(op, OperatorSpec) and op.arity >= 1]
    ops.sort(key=lambda o: (o.arity, o.name))  # stable base order

    # Leaves: deterministic order by leaf_id
    leaves = list(leaf_provider.iter_leaf_specs())
    leaves_sorted = sorted(leaves, key=lambda ls: ls.leaf_id)

    pool: List[Expr] = []
    out: List[StrategySpec] = []
    seen: set[str] = set()

    def _maybe_add(expr: Expr) -> None:
        if len(out) >= budget.max_strategies:
            return
        if expr_depth(expr) > budget.max_depth:
            return
        if expr_node_count(expr) > budget.max_nodes:
            return
        spec = StrategySpec(expr=expr)
        sid = spec.sid()
        if sid in seen:
            return
        seen.add(sid)
        out.append(spec)
        pool.append(expr)

    # 1) Atoms first
    for ls in leaves_sorted:
        _maybe_add(LeafExpr(leaf_id=ls.leaf_id, meta=ls.meta))
        if len(out) >= budget.max_strategies:
            return out

    for c in const_pool:
        _maybe_add(ConstExpr(value=c))
        if len(out) >= budget.max_strategies:
            return out

    # 2) Composites (seeded RNG, budgeted attempts)
    rng = random.Random(int(budget.seed))

    # If there are no ops or pool is empty, we're done.
    if not ops or not pool:
        return out

    attempts = 0
    while len(out) < budget.max_strategies and attempts < budget.max_attempts:
        attempts += 1
        op = rng.choice(ops)

        # Select args from current pool.
        # We use choice with replacement to keep it simple & stable.
        args: List[Expr] = [rng.choice(pool) for _ in range(op.arity)]

        # Canonicalize commutative args to reduce duplicates.
        if op.arity >= 2 and op.commutative:
            args_t = canonicalize_commutative_args(args)
        else:
            args_t = tuple(args)

        expr = OpExpr(op=op.name, args=args_t)
        _maybe_add(expr)

    return out


# ---------------------------
# Minimal evaluation (optional adapter)
# ---------------------------

def eval_expr(
    expr: Expr,
    *,
    leaf_resolver: Callable[[str, Mapping[str, Any]], float],
    leaf_ctx: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Minimal numeric evaluation adapter for Expr trees.

    This is NOT a backtester. It's a small interpreter that:
    - delegates leaf values to a provided resolver function
    - implements a minimal set of operator semantics
    - keeps behavior deterministic and offline

    Later files can:
    - register new operator semantics
    - replace this interpreter
    - vectorize evaluation against time series (backtests)

    leaf_resolver(leaf_id, leaf_ctx) must return a float.
    """
    leaf_ctx = leaf_ctx or {}

    d = expr.to_dict()
    t = d.get("t")
    if t == "leaf":
        return float(leaf_resolver(str(d.get("id")), leaf_ctx))
    if t == "const":
        return float(d.get("v"))
    if t != "op":
        raise ValueError(f"Unknown expr node type: {t!r}")

    op = str(d.get("op"))
    args = [eval_expr(c, leaf_resolver=leaf_resolver, leaf_ctx=leaf_ctx) for c in expr.children()]

    if op == "add":
        return float(args[0] + args[1])
    if op == "sub":
        return float(args[0] - args[1])
    if op == "mul":
        return float(args[0] * args[1])
    if op == "div":
        # "safe" division to avoid inf/nan explosion.
        denom = float(args[1])
        if abs(denom) < 1e-12:
            return 0.0
        return float(args[0] / denom)
    if op == "neg":
        return float(-args[0])

    raise ValueError(f"Unknown operator semantics: {op!r}. Provide an extended evaluator in a later file.")


# ---------------------------
# Selftest
# ---------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _selftest_deterministic_enumeration() -> None:
    lp = StaticLeafProvider(
        leaves=[
            LeafSpec("feature.a"),
            LeafSpec("feature.b"),
        ]
    )
    budget = EnumBudget(max_strategies=32, max_depth=3, max_nodes=15, max_attempts=5000, seed=42)
    s1 = enumerate_strategies(leaf_provider=lp, budget=budget)
    s2 = enumerate_strategies(leaf_provider=lp, budget=budget)

    ids1 = [x.sid() for x in s1]
    ids2 = [x.sid() for x in s2]
    _assert(ids1 == ids2, "Enumeration must be deterministic for same seed/budget/leaves")
    _assert(len(ids1) == len(set(ids1)), "Strategy IDs must be unique in the enumeration output")


def _selftest_extensibility_via_leaf_provider() -> None:
    lp_small = StaticLeafProvider(leaves=[LeafSpec("feature.a"), LeafSpec("feature.b")])
    lp_big = StaticLeafProvider(leaves=[LeafSpec("feature.a"), LeafSpec("feature.b"), LeafSpec("feature.c")])

    budget = EnumBudget(max_strategies=10, max_depth=2, max_nodes=9, max_attempts=2000, seed=7)

    small = enumerate_strategies(leaf_provider=lp_small, budget=budget)
    big = enumerate_strategies(leaf_provider=lp_big, budget=budget)

    # Because atoms are yielded first, the new leaf must appear quickly and deterministically.
    big_leaf_union: List[str] = []
    for s in big:
        big_leaf_union.extend(expr_leaf_ids(s.expr))
    _assert("feature.c" in set(big_leaf_union), "Adding a leaf must expand reachable strategies without code changes")

    # The first few atom strategies should include the leaf itself.
    first_leaf_ids = [s.expr.to_dict().get("id") for s in big[:3] if s.expr.to_dict().get("t") == "leaf"]
    _assert("feature.c" in first_leaf_ids, "New leaf should appear among earliest atom outputs")


def _selftest_hash_invariance_to_key_order() -> None:
    # Ensure stable_hash is invariant to dict insertion order for strategy dicts.
    e = OpExpr(op="add", args=(ConstExpr(1.0), ConstExpr(2.0)))
    spec = StrategySpec(expr=e, meta={"x": 1, "y": 2}).to_dict()
    # Create a semantically identical dict with different insertion order
    spec2 = {
        "meta": {"y": 2, "x": 1},
        "expr": spec["expr"],
        "dsl": spec["dsl"],
        "v": spec["v"],
        "t": spec["t"],
    }
    h1 = k.stable_hash(spec, salt="strategy_spec")
    h2 = k.stable_hash(spec2, salt="strategy_spec")
    _assert(h1 == h2, "stable_hash must be invariant to dict key order in strategy specs")


def _selftest_registry_integration() -> None:
    # Discovery should work and ops must be resolvable.
    dr = k.discover(strict=False)
    _assert("d002_strategy_dsl.py" in dr.seen_files, "Discovery must see d002_strategy_dsl.py in repo root")

    # At least one known operator should resolve.
    add_item = k.REGISTRY.resolve(kind=REG_KIND_DSL_OP, name="add")
    _assert(isinstance(add_item.provider, OperatorSpec), "Registry provider for dsl_op/add must be an OperatorSpec")

    ops = collect_latest_operator_specs()
    _assert(any(o.name == "add" for o in ops), "collect_latest_operator_specs must include 'add'")


def _selftest_eval_adapter_smoke() -> None:
    # Basic deterministic evaluation smoke test
    expr = OpExpr(op="sub", args=(LeafExpr("feature.a"), ConstExpr(1.0)))

    def resolver(leaf_id: str, ctx: Mapping[str, Any]) -> float:
        return float(ctx.get(leaf_id, 0.0))

    v = eval_expr(expr, leaf_resolver=resolver, leaf_ctx={"feature.a": 2.5})
    _assert(abs(v - 1.5) < 1e-12, "eval_expr must compute correct value")


def selftest() -> int:
    print(f"[d002] Strategy DSL selftest — version {DSL_VERSION}")
    print(f"[d002] Using registry kind={REG_KIND_DSL_OP!r} (ops)")

    _selftest_registry_integration()
    print("[d002] OK: registry integration")

    _selftest_deterministic_enumeration()
    print("[d002] OK: deterministic enumeration")

    _selftest_extensibility_via_leaf_provider()
    print("[d002] OK: leaf-provider extensibility")

    _selftest_hash_invariance_to_key_order()
    print("[d002] OK: hash invariance to dict insertion order")

    _selftest_eval_adapter_smoke()
    print("[d002] OK: eval adapter smoke test")

    print("[d002] SELFTEST PASS")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        try:
            return selftest()
        except AssertionError as e:
            print("[d002] SELFTEST FAIL:", e)
            return 2
        except Exception as e:
            print("[d002] SELFTEST ERROR:", f"{type(e).__name__}: {e}")
            return 3

    print("d002_strategy_dsl.py")
    print("Usage:")
    print("  python d002_strategy_dsl.py --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
