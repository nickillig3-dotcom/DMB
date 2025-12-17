"""
d004_research_runner.py — Deterministic Offline Research Runner (Discovery -> Enum -> Eval -> Holdout)

Purpose (Contracts > Contents):
- Provide a minimal "end-to-end" offline research loop that is:
    - reproducible (stable hashes, deterministic budgets/seeds)
    - robust (holdout evaluation gate)
    - extensible (registry contracts; "latest wins" overrides without editing older files)

What this file DOES:
- Defines registry kinds for:
    - dataset providers (kind="dataset_provider")
    - research runners (kind="research_runner")
- Registers:
    - a synthetic offline dataset provider ("synth_v1") for testing & local development
    - a basic holdout research runner ("basic_holdout_v1")
    - a small backward-compatible override for the evaluator factory so callers can pass cfg explicitly
      (keeps d003 selftests working)

What this file DOES NOT:
- No external APIs
- No live trading / no orders
- No hardcoded market-data schema beyond the synthetic test provider (which is just a test harness)

Selftest:
    python d004_research_runner.py --selftest
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import d001_kernel as k
import d002_strategy_dsl as dsl
import d003_eval_engine as ev


RUNNER_VERSION = "0.1.0"

REG_KIND_DATASET_PROVIDER = "dataset_provider"
REG_KIND_RESEARCH_RUNNER = "research_runner"

# Reuse names from earlier components
DEFAULT_ENUMERATOR_NAME = "expr_enum_v1"             # from d002
DEFAULT_EVALUATOR_NAME = ev.REG_EVALUATOR_NAME       # from d003 ("simple_signal_pnl_v1")


# ---------------------------
# Contracts
# ---------------------------

class DatasetProvider(Protocol):
    """
    A dataset provider is a callable that returns a Dataset (see d003_eval_engine.Dataset).
    """
    def __call__(self, **kwargs: Any) -> ev.Dataset:
        ...


class ResearchRunner(Protocol):
    """
    A research runner executes an offline research pass and returns a ResearchReport.
    """
    def run(self) -> "ResearchReport":
        ...


# ---------------------------
# Utility: dataset leaf provider (no hardcoded features)
# ---------------------------

@dataclass(frozen=True)
class DatasetLeafProvider:
    """
    LeafProvider adapter backed by a Dataset.

    Filters out obvious target/leakage leaves by default using EvalConfig settings.
    This keeps enumeration efficient and prevents wasting budget on blocked strategies.

    IMPORTANT: This is a *filter*, not the only protection.
    The evaluator in d003 also enforces leakage guards.
    """
    ds: ev.Dataset
    eval_cfg: ev.EvalConfig

    def iter_leaf_specs(self) -> Iterable[dsl.LeafSpec]:
        for lid in self.ds.series_ids():
            if lid == self.eval_cfg.target_return_leaf and not self.eval_cfg.allow_target_leaf_in_expr:
                continue
            if lid in self.eval_cfg.forbidden_leaf_ids:
                continue
            if any(str(lid).startswith(pfx) for pfx in self.eval_cfg.forbidden_leaf_prefixes):
                # Exempt exact target leaf only when explicitly allowed
                if self.eval_cfg.allow_target_leaf_in_expr and lid == self.eval_cfg.target_return_leaf:
                    pass
                else:
                    continue
            yield dsl.LeafSpec(leaf_id=str(lid), meta={})


# ---------------------------
# Synthetic dataset provider (offline test harness)
# ---------------------------

def synth_dataset_provider(
    *,
    length: int = 512,
    seed: int = 123,
    target_return_leaf: str = "target.ret",
) -> ev.InMemoryDataset:
    """
    Deterministic synthetic dataset.

    Design:
    - feature.a ~ N(0,1)
    - target returns depend on sign(feature.a[t-1]) plus noise
      -> strategy using feature.a (with sign position) should score better than constant-0
    - feature.b is noise

    This is ONLY for functional testing and reproducibility (not a profit claim).
    """
    L = max(1, int(length))
    rng = random.Random(int(seed))

    feature_a: List[float] = [rng.gauss(0.0, 1.0) for _ in range(L)]
    feature_b: List[float] = [rng.gauss(0.0, 1.0) for _ in range(L)]

    def _sign(x: float) -> int:
        if x > 0.0:
            return 1
        if x < 0.0:
            return -1
        return 0

    target_ret: List[float] = [0.0] * L
    if L > 0:
        target_ret[0] = 0.0
    for t in range(1, L):
        drift = 0.002 * float(_sign(feature_a[t - 1]))
        noise = 0.001 * rng.gauss(0.0, 1.0)
        target_ret[t] = drift + noise

    return ev.InMemoryDataset(
        series={
            "feature.a": feature_a,
            "feature.b": feature_b,
            str(target_return_leaf): target_ret,
        },
        meta={"t": "synth_v1", "L": L, "seed": int(seed), "target_return_leaf": str(target_return_leaf)},
    )


# ---------------------------
# Holdout split (time-series)
# ---------------------------

def time_split_dataset(ds: ev.Dataset, split_index: int) -> Tuple[ev.InMemoryDataset, ev.InMemoryDataset]:
    """
    Create train/test datasets by slicing each series at split_index.
    Copies into InMemoryDataset (simple, deterministic).

    NOTE: For very large datasets, copying is expensive. A future domino file can
    introduce a "view" dataset that avoids copying while keeping the same contract.
    """
    L = int(ds.length())
    si = max(1, min(int(split_index), max(1, L - 1)))

    train_series: Dict[str, List[float]] = {}
    test_series: Dict[str, List[float]] = {}

    for lid in ds.series_ids():
        s = ds.get_series(lid)
        train_series[str(lid)] = [float(x) for x in list(s[:si])]
        test_series[str(lid)] = [float(x) for x in list(s[si:])]

    return (
        ev.InMemoryDataset(series=train_series, meta={"split": "train", "split_index": si}),
        ev.InMemoryDataset(series=test_series, meta={"split": "test", "split_index": si}),
    )


# ---------------------------
# Research config + report
# ---------------------------

@dataclass(frozen=True)
class ResearchConfig:
    """
    Deterministic research configuration.

    All fields are JSON-serializable via to_dict(), enabling stable hashes.
    """
    dataset_provider_name: str = "synth_v1"
    dataset_provider_kwargs: Mapping[str, Any] = field(default_factory=dict)

    holdout_fraction: float = 0.70  # train fraction, time-based

    enumerator_name: str = DEFAULT_ENUMERATOR_NAME
    enum_budget: dsl.EnumBudget = dsl.EnumBudget(max_strategies=32, max_depth=3, max_nodes=15, max_attempts=5000, seed=42)
    const_pool: Tuple[Any, ...] = (0.0, 1.0)

    evaluator_name: str = DEFAULT_EVALUATOR_NAME
    eval_config: ev.EvalConfig = ev.EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=64, min_coverage=0.98)

    preselect_m: int = 8
    final_k: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "research_config",
            "v": 1,
            "dataset_provider_name": self.dataset_provider_name,
            "dataset_provider_kwargs": dict(self.dataset_provider_kwargs),
            "holdout_fraction": float(self.holdout_fraction),
            "enumerator_name": self.enumerator_name,
            "enum_budget": {
                "max_strategies": int(self.enum_budget.max_strategies),
                "max_depth": int(self.enum_budget.max_depth),
                "max_nodes": int(self.enum_budget.max_nodes),
                "max_attempts": int(self.enum_budget.max_attempts),
                "seed": int(self.enum_budget.seed),
            },
            "const_pool": list(self.const_pool),
            "evaluator_name": self.evaluator_name,
            "eval_config": self.eval_config.to_dict(),
            "preselect_m": int(self.preselect_m),
            "final_k": int(self.final_k),
        }

    def cid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="research_config_v1")


@dataclass(frozen=True)
class ResearchReport:
    """
    Deterministic research report wrapper.
    """
    run_id: str
    payload: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.payload)
        d["run_id"] = self.run_id
        return d

    def rid(self) -> str:
        return self.run_id


def _sort_key_score(sc: ev.ScoreCard) -> Tuple[float, str]:
    # Desc score, then strategy_id for stability
    return (-float(sc.score), str(sc.strategy_id))


# ---------------------------
# Runner implementation
# ---------------------------

@dataclass(frozen=True)
class BasicHoldoutRunner:
    """
    Basic runner:
    - Load dataset from provider
    - Time split train/test
    - Enumerate strategies from leaves (excluding target/leakage leaves)
    - Evaluate on train -> preselect top-M
    - Evaluate on test -> select top-K
    """
    cfg: ResearchConfig

    def run(self) -> ResearchReport:
        # Ensure all domino modules are imported/registered (plugin discovery).
        # strict=False to keep runner robust to unrelated modules failing import.
        disc = k.discover(strict=False)

        # Registry snapshot (reproducibility / audit trail)
        reg_snap = k.REGISTRY.snapshot()
        reg_snap_hash = k.stable_hash(reg_snap, salt="registry_snapshot_v1")

        # Dataset provider
        dp_item = k.REGISTRY.resolve(kind=REG_KIND_DATASET_PROVIDER, name=self.cfg.dataset_provider_name)
        ds_full = dp_item.create(**dict(self.cfg.dataset_provider_kwargs))

        # Validate dataset length and split
        L = int(ds_full.length())
        hf = float(self.cfg.holdout_fraction)
        if not (0.0 < hf < 1.0):
            # Clamp but deterministic; prefer sane defaults over raising for runner usability
            hf = 0.70

        split_index = int(math.floor(L * hf))
        train_ds, test_ds = time_split_dataset(ds_full, split_index)

        # Guard: ensure both splits satisfy min_obs; otherwise evaluation will reject anyway.
        # We keep the report informative rather than throwing hard.
        min_obs = int(self.cfg.eval_config.min_obs)

        # LeafProvider derived from dataset and eval config
        leaf_provider = DatasetLeafProvider(train_ds, self.cfg.eval_config)

        # Enumerator
        enum_item = k.REGISTRY.resolve(kind=dsl.REG_KIND_STRATEGY_ENUMERATOR, name=self.cfg.enumerator_name)
        enum_fn = enum_item.create()  # returns a function like dsl.enumerate_strategies

        # Enumerate strategies (budgeted)
        strategies: List[dsl.StrategySpec] = list(
            enum_fn(
                leaf_provider=leaf_provider,
                operators=None,
                const_pool=tuple(self.cfg.const_pool),
                budget=self.cfg.enum_budget,
            )
        )

        # Evaluator (from registry; we override in this file to allow passing cfg explicitly)
        eval_item = k.REGISTRY.resolve(kind=ev.REG_KIND_EVALUATOR, name=self.cfg.evaluator_name)
        evaluator = eval_item.create(cfg=self.cfg.eval_config)

        # Evaluate train
        train_scores: List[ev.ScoreCard] = [evaluator.evaluate(s, train_ds) for s in strategies]
        train_ok = [sc for sc in train_scores if sc.ok]
        train_ok_sorted = sorted(train_ok, key=_sort_key_score)

        pre_m = max(1, int(self.cfg.preselect_m))
        preselected_ids = [sc.strategy_id for sc in train_ok_sorted[:pre_m]]

        id_to_spec: Dict[str, dsl.StrategySpec] = {s.sid(): s for s in strategies}
        preselected_specs: List[dsl.StrategySpec] = [id_to_spec[sid] for sid in preselected_ids if sid in id_to_spec]

        # Evaluate test only on preselected
        test_scores: List[ev.ScoreCard] = [evaluator.evaluate(s, test_ds) for s in preselected_specs]
        test_ok = [sc for sc in test_scores if sc.ok]
        test_ok_sorted = sorted(test_ok, key=_sort_key_score)

        final_k = max(1, int(self.cfg.final_k))
        final_ids = [sc.strategy_id for sc in test_ok_sorted[:final_k]]

        # Build report candidates (include both train+test dicts when available)
        train_by_id: Dict[str, ev.ScoreCard] = {sc.strategy_id: sc for sc in train_scores}
        test_by_id: Dict[str, ev.ScoreCard] = {sc.strategy_id: sc for sc in test_scores}

        final_candidates: List[Dict[str, Any]] = []
        for rank, sid in enumerate(final_ids, start=1):
            spec = id_to_spec.get(sid)
            if spec is None:
                continue
            final_candidates.append(
                {
                    "rank": int(rank),
                    "strategy_id": sid,
                    "strategy": spec.to_dict(),
                    "train": train_by_id.get(sid).to_dict() if sid in train_by_id else None,
                    "test": test_by_id.get(sid).to_dict() if sid in test_by_id else None,
                }
            )

        payload: Dict[str, Any] = {
            "t": "research_report",
            "v": 1,
            "runner": {"version": RUNNER_VERSION},
            "modules": {
                "kernel": k.KERNEL_VERSION,
                "dsl": dsl.DSL_VERSION,
                "eval_engine": ev.EVAL_ENGINE_VERSION,
            },
            "discovery": {
                "seen_files": list(disc.seen_files),
                "loaded_modules": list(disc.loaded_modules),
                "skipped_files": list(disc.skipped_files),
                "errors": list(disc.errors),
            },
            "config": self.cfg.to_dict(),
            "config_id": self.cfg.cid(),
            "registry_snapshot_hash": reg_snap_hash,
            "dataset": {
                "provider": {"name": self.cfg.dataset_provider_name, "kwargs": dict(self.cfg.dataset_provider_kwargs)},
                "full": {
                    "length": L,
                    "fingerprint": ev.dataset_fingerprint(ds_full),
                    "series_ids": list(ds_full.series_ids()),
                },
                "train": {
                    "length": int(train_ds.length()),
                    "fingerprint": ev.dataset_fingerprint(train_ds),
                    "min_obs_ok": bool(int(train_ds.length()) >= min_obs),
                },
                "test": {
                    "length": int(test_ds.length()),
                    "fingerprint": ev.dataset_fingerprint(test_ds),
                    "min_obs_ok": bool(int(test_ds.length()) >= min_obs),
                },
                "split": {
                    "holdout_fraction": float(hf),
                    "split_index": int(split_index),
                },
            },
            "enumeration": {
                "count": int(len(strategies)),
                "budget": {
                    "max_strategies": int(self.cfg.enum_budget.max_strategies),
                    "max_depth": int(self.cfg.enum_budget.max_depth),
                    "max_nodes": int(self.cfg.enum_budget.max_nodes),
                    "max_attempts": int(self.cfg.enum_budget.max_attempts),
                    "seed": int(self.cfg.enum_budget.seed),
                },
                "leaf_count": int(len(list(leaf_provider.iter_leaf_specs()))),
                "const_pool": list(self.cfg.const_pool),
            },
            "evaluation": {
                "train": {
                    "count": int(len(train_scores)),
                    "ok": int(sum(1 for sc in train_scores if sc.ok)),
                    "rejected": int(sum(1 for sc in train_scores if not sc.ok)),
                    "preselect_m": int(pre_m),
                },
                "test": {
                    "count": int(len(test_scores)),
                    "ok": int(sum(1 for sc in test_scores if sc.ok)),
                    "rejected": int(sum(1 for sc in test_scores if not sc.ok)),
                    "final_k": int(final_k),
                },
            },
            "final": {
                "count": int(len(final_candidates)),
                "candidates": final_candidates,
                "sorting": {"primary": "test.score desc", "tie_break": "strategy_id asc"},
            },
        }

        run_id = k.stable_hash(payload, salt="research_report_v1")
        return ResearchReport(run_id=run_id, payload=payload)


# ---------------------------
# nb_local defaults (optional)
# ---------------------------

def default_research_config_from_nb_local() -> ResearchConfig:
    """
    Build a ResearchConfig from nb_local.py overrides (optional), with safe defaults.

    Supported optional keys in nb_local.py:
      - RESEARCH_DATASET_PROVIDER_NAME (str)
      - RESEARCH_DATASET_LENGTH (int)
      - RESEARCH_DATASET_SEED (int)
      - RESEARCH_HOLDOUT_FRACTION (float)
      - RESEARCH_ENUMERATOR_NAME (str)
      - RESEARCH_ENUM_MAX_STRATEGIES (int)
      - RESEARCH_ENUM_MAX_DEPTH (int)
      - RESEARCH_ENUM_MAX_NODES (int)
      - RESEARCH_ENUM_MAX_ATTEMPTS (int)
      - RESEARCH_ENUM_SEED (int)
      - RESEARCH_CONST_POOL (tuple/list)
      - RESEARCH_EVALUATOR_NAME (str)
      - RESEARCH_PRESELECT_M (int)
      - RESEARCH_FINAL_K (int)

    Evaluation config is taken from d003.default_eval_config_from_nb_local().
    """
    nb = k.load_nb_local()

    eval_cfg = ev.default_eval_config_from_nb_local()

    provider_name = str(nb.get("RESEARCH_DATASET_PROVIDER_NAME", "synth_v1"))
    ds_len = int(nb.get("RESEARCH_DATASET_LENGTH", 512))
    ds_seed = int(nb.get("RESEARCH_DATASET_SEED", nb.get("RANDOM_SEED", 1337)))

    holdout_fraction = float(nb.get("RESEARCH_HOLDOUT_FRACTION", 0.70))

    enum_name = str(nb.get("RESEARCH_ENUMERATOR_NAME", DEFAULT_ENUMERATOR_NAME))
    enum_seed = int(nb.get("RESEARCH_ENUM_SEED", nb.get("RANDOM_SEED", 1337)))

    enum_budget = dsl.EnumBudget(
        max_strategies=int(nb.get("RESEARCH_ENUM_MAX_STRATEGIES", 32)),
        max_depth=int(nb.get("RESEARCH_ENUM_MAX_DEPTH", 3)),
        max_nodes=int(nb.get("RESEARCH_ENUM_MAX_NODES", 15)),
        max_attempts=int(nb.get("RESEARCH_ENUM_MAX_ATTEMPTS", 5000)),
        seed=enum_seed,
    )

    const_pool_raw = nb.get("RESEARCH_CONST_POOL", (0.0, 1.0))
    if isinstance(const_pool_raw, (list, tuple)):
        const_pool = tuple(const_pool_raw)
    else:
        const_pool = (const_pool_raw,)

    evaluator_name = str(nb.get("RESEARCH_EVALUATOR_NAME", DEFAULT_EVALUATOR_NAME))

    pre_m = int(nb.get("RESEARCH_PRESELECT_M", 8))
    fin_k = int(nb.get("RESEARCH_FINAL_K", 3))

    # Dataset kwargs include the target leaf to stay consistent if user changes TARGET_RETURN_LEAF.
    ds_kwargs = {
        "length": ds_len,
        "seed": ds_seed,
        "target_return_leaf": eval_cfg.target_return_leaf,
    }

    return ResearchConfig(
        dataset_provider_name=provider_name,
        dataset_provider_kwargs=ds_kwargs,
        holdout_fraction=holdout_fraction,
        enumerator_name=enum_name,
        enum_budget=enum_budget,
        const_pool=const_pool,
        evaluator_name=evaluator_name,
        eval_config=eval_cfg,
        preselect_m=pre_m,
        final_k=fin_k,
    )


# ---------------------------
# Registry registrations (plugins)
# ---------------------------

def evaluator_factory(cfg: Optional[ev.EvalConfig] = None) -> ev.SimpleSignalPnLEvaluator:
    """
    Backward-compatible evaluator factory override:
    - create() with no args still works (uses nb_local defaults)
    - callers can now pass cfg=... to create a configured evaluator

    This is registered with a higher version than d003's evaluator registration, so "latest wins".
    """
    if cfg is None:
        cfg = ev.default_eval_config_from_nb_local()
    return ev.SimpleSignalPnLEvaluator(cfg=cfg)


def _register_d004_once() -> None:
    sentinel = "_dmb_d004_registered_v1"
    if getattr(sys, sentinel, False):
        return
    setattr(sys, sentinel, True)

    # 1) Evaluator factory override (latest-wins) to allow passing cfg explicitly.
    #    Must remain backward compatible with d003 selftests (create() with no args).
    k.REGISTRY.register(
        kind=ev.REG_KIND_EVALUATOR,
        name=DEFAULT_EVALUATOR_NAME,
        version="1.1.0",
        provider=evaluator_factory,
        meta={
            "engine": ev.EVAL_ENGINE_VERSION,
            "notes": "Backward-compatible evaluator factory override: supports create(cfg=...).",
        },
        source="d004_research_runner.py",
    )

    # 2) Dataset provider (synthetic offline harness)
    k.REGISTRY.register(
        kind=REG_KIND_DATASET_PROVIDER,
        name="synth_v1",
        version="1.0.0",
        provider=synth_dataset_provider,
        meta={"offline": True, "deterministic": True, "notes": "Synthetic dataset for testing/dev."},
        source="d004_research_runner.py",
    )

    # 3) Research runner factory
    def runner_factory(cfg: Optional[ResearchConfig] = None) -> BasicHoldoutRunner:
        if cfg is None:
            cfg = default_research_config_from_nb_local()
        return BasicHoldoutRunner(cfg=cfg)

    k.REGISTRY.register(
        kind=REG_KIND_RESEARCH_RUNNER,
        name="basic_holdout_v1",
        version="1.0.0",
        provider=runner_factory,
        meta={"offline": True, "deterministic": True, "holdout": True},
        source="d004_research_runner.py",
    )


_register_d004_once()


# ---------------------------
# Selftest
# ---------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _selftest_registry_resolve() -> None:
    # Ensure discovery sees this file
    dr = k.discover(strict=False)
    _assert("d004_research_runner.py" in dr.seen_files, "Discovery must see d004_research_runner.py in repo root")

    # Ensure required components are resolvable
    _ = k.REGISTRY.resolve(kind=REG_KIND_DATASET_PROVIDER, name="synth_v1")
    _ = k.REGISTRY.resolve(kind=dsl.REG_KIND_STRATEGY_ENUMERATOR, name=DEFAULT_ENUMERATOR_NAME)
    _ = k.REGISTRY.resolve(kind=ev.REG_KIND_EVALUATOR, name=DEFAULT_EVALUATOR_NAME)
    _ = k.REGISTRY.resolve(kind=REG_KIND_RESEARCH_RUNNER, name="basic_holdout_v1")


def _selftest_end_to_end_determinism() -> None:
    # Explicit config avoids dependence on user's nb_local overrides.
    eval_cfg = ev.EvalConfig(
        target_return_leaf="target.ret",
        transaction_cost=0.0005,
        min_obs=64,
        min_coverage=0.98,
    )

    cfg = ResearchConfig(
        dataset_provider_name="synth_v1",
        dataset_provider_kwargs={"length": 512, "seed": 123, "target_return_leaf": "target.ret"},
        holdout_fraction=0.70,
        enumerator_name=DEFAULT_ENUMERATOR_NAME,
        enum_budget=dsl.EnumBudget(max_strategies=32, max_depth=3, max_nodes=15, max_attempts=5000, seed=42),
        const_pool=(0.0, 1.0),
        evaluator_name=DEFAULT_EVALUATOR_NAME,
        eval_config=eval_cfg,
        preselect_m=8,
        final_k=3,
    )

    runner_item = k.REGISTRY.resolve(kind=REG_KIND_RESEARCH_RUNNER, name="basic_holdout_v1")

    rep1 = runner_item.create(cfg=cfg).run()
    rep2 = runner_item.create(cfg=cfg).run()

    _assert(rep1.run_id == rep2.run_id, "ResearchReport run_id must be deterministic for same config")
    _assert(rep1.to_dict() == rep2.to_dict(), "ResearchReport payload must be deterministic for same config")

    final = rep1.to_dict()["final"]["candidates"]
    _assert(len(final) > 0, "Final candidate list should be non-empty on synth dataset")

    # Ensure final candidates are sorted by test.score desc then strategy_id asc
    def _key(c: Mapping[str, Any]) -> Tuple[float, str]:
        test = c.get("test") or {}
        return (-float(test.get("score", float("-inf"))), str(c.get("strategy_id")))

    sorted_final = sorted(final, key=_key)
    _assert(final == sorted_final, "Final candidates must be deterministically sorted")


def _selftest_holdout_flow_sanity() -> None:
    cfg = ResearchConfig(
        dataset_provider_name="synth_v1",
        dataset_provider_kwargs={"length": 256, "seed": 7, "target_return_leaf": "target.ret"},
        holdout_fraction=0.75,
        enum_budget=dsl.EnumBudget(max_strategies=16, max_depth=2, max_nodes=9, max_attempts=2000, seed=7),
        eval_config=ev.EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=32, min_coverage=0.95),
        preselect_m=6,
        final_k=2,
    )
    rep = BasicHoldoutRunner(cfg=cfg).run().to_dict()
    train_len = int(rep["dataset"]["train"]["length"])
    test_len = int(rep["dataset"]["test"]["length"])
    _assert(train_len > 0 and test_len > 0, "Train and test splits must be non-empty")


def selftest() -> int:
    print(f"[d004] Research runner selftest — version {RUNNER_VERSION}")

    _selftest_registry_resolve()
    print("[d004] OK: registry resolve + discovery")

    _selftest_end_to_end_determinism()
    print("[d004] OK: end-to-end determinism")

    _selftest_holdout_flow_sanity()
    print("[d004] OK: holdout flow sanity")

    print("[d004] SELFTEST PASS")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        try:
            return selftest()
        except AssertionError as e:
            print("[d004] SELFTEST FAIL:", e)
            return 2
        except Exception as e:
            print("[d004] SELFTEST ERROR:", f"{type(e).__name__}: {e}")
            return 3

    if "--run" in argv:
        cfg = default_research_config_from_nb_local()
        runner = BasicHoldoutRunner(cfg=cfg)
        rep = runner.run()
        # Print a compact deterministic summary (no timestamps).
        d = rep.to_dict()
        print(json.dumps(
            {
                "run_id": d["run_id"],
                "config_id": d["config_id"],
                "final_count": d["final"]["count"],
                "final": [
                    {
                        "rank": c["rank"],
                        "strategy_id": c["strategy_id"],
                        "test_score": None if not c.get("test") else c["test"].get("score"),
                    }
                    for c in d["final"]["candidates"]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ))
        return 0

    print("d004_research_runner.py")
    print("Usage:")
    print("  python d004_research_runner.py --selftest")
    print("  python d004_research_runner.py --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
