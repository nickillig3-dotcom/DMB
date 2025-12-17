"""
d005_portfolio_builder.py — Portfolio Builder (Diversification) + Holdout Portfolio Research Runner

Contracts > Contents:
- Adds a portfolio-building layer on top of the existing offline research flow.
- No hardcoded market-data schema (no "only OHLCV").
- Deterministic, offline, budgeted.
- No trading / no orders / no external APIs.

Provides:
1) Portfolio builder contract + implementation:
   - kind="portfolio_builder", name="simple_diversified_v1"
   - Greedy selection: maximize (score - corr_penalty * max_abs_corr_to_selected)
   - Weights: inverse volatility (long-only, sum to 1)

2) A new research runner:
   - kind="research_runner", name="holdout_portfolio_v1"
   - Flow:
       discovery -> dataset_provider -> time split -> enum -> eval(train) -> preselect
       -> eval(test) -> candidate_pool -> build portfolio on test
   - Produces a deterministic report with stable hashes.

Selftest:
    python d005_portfolio_builder.py --selftest
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import d001_kernel as k
import d002_strategy_dsl as dsl
import d003_eval_engine as ev


D005_VERSION = "0.1.0"

REG_KIND_PORTFOLIO_BUILDER = "portfolio_builder"
REG_KIND_RESEARCH_RUNNER = "research_runner"
REG_KIND_DATASET_PROVIDER = "dataset_provider"


_EPS = 1e-12


# ---------------------------
# Contracts
# ---------------------------

class PortfolioBuilder(Protocol):
    def build(
        self,
        candidates: Sequence[dsl.StrategySpec],
        ds: ev.Dataset,
        *,
        eval_cfg: ev.EvalConfig,
        scores_by_id: Optional[Mapping[str, float]] = None,
    ) -> "PortfolioReport":
        ...


# ---------------------------
# Small deterministic utilities
# ---------------------------

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


def series_fingerprint(series: Sequence[float], *, sample_n: int = 3) -> str:
    """
    PC-friendly fingerprint for a numeric series (not a full data dump).
    """
    L = len(series)
    head = [float(x) for x in list(series[:sample_n])]
    tail = [float(x) for x in list(series[-sample_n:])] if L >= sample_n else [float(x) for x in list(series)]
    payload = {"t": "series_fp_v1", "len": int(L), "sample_n": int(sample_n), "head": head, "tail": tail}
    return k.stable_hash(payload, salt="series_fp_v1")


def series_metrics(pnl_series: Sequence[float]) -> Dict[str, float]:
    """
    Metrics on a PnL series (already shifted if desired).
    Uses population stdev for determinism and stability on small n.
    """
    n = len(pnl_series)
    if n <= 0:
        return {"mean": 0.0, "stdev": 0.0, "info_ratio": 0.0, "max_drawdown": 0.0}

    mean = math.fsum(float(x) for x in pnl_series) / float(n)
    var = math.fsum((float(x) - mean) * (float(x) - mean) for x in pnl_series) / float(n)
    stdev = math.sqrt(max(0.0, var))
    info = mean / (stdev + _EPS)

    # drawdown on cumulative equity
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in pnl_series:
        eq += float(x)
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return {"mean": float(mean), "stdev": float(stdev), "info_ratio": float(info), "max_drawdown": float(max_dd)}


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Correlation with robust handling of near-constant series.
    Deterministic computation.
    """
    n = min(len(a), len(b))
    if n <= 1:
        return 0.0
    xa = [float(x) for x in a[:n]]
    xb = [float(x) for x in b[:n]]

    ma = math.fsum(xa) / float(n)
    mb = math.fsum(xb) / float(n)

    va = math.fsum((x - ma) * (x - ma) for x in xa) / float(n)
    vb = math.fsum((x - mb) * (x - mb) for x in xb) / float(n)

    sa = math.sqrt(max(0.0, va))
    sb = math.sqrt(max(0.0, vb))
    if sa < 1e-18 or sb < 1e-18:
        return 0.0

    cov = math.fsum((xa[i] - ma) * (xb[i] - mb) for i in range(n)) / float(n)
    r = cov / (sa * sb + _EPS)
    # Clamp for numerical safety
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return float(r)


# ---------------------------
# PnL series computation (mirrors d003 mechanics; no execution/trading)
# ---------------------------

def pnl_series_for_strategy(
    spec: dsl.StrategySpec,
    ds: ev.Dataset,
    *,
    eval_cfg: ev.EvalConfig,
) -> Tuple[List[float], float, float]:
    """
    Compute (pnl_series, turnover, coverage) deterministically, mirroring d003 evaluator logic.

    - signal[t] from DSL expression using dataset leaves
    - position[t] = sign(signal[t]) or raw(signal[t])
    - pnl[t] = position[t-1] * ret[t] - transaction_cost * abs(position[t-1]-position[t-2])
    - pnl[0] = 0

    coverage is min(signal_coverage, return_coverage).
    """
    L = int(ds.length())
    if L <= 0:
        return [], 0.0, 0.0

    # Leakage guard is enforced by the evaluator too, but we keep builder consistent.
    leak, _reason = ev.detect_leakage(spec.expr, eval_cfg)
    if leak:
        return [0.0] * L, 0.0, 0.0

    # target return series
    ret = ds.get_series(eval_cfg.target_return_leaf)

    # prefetch series needed by expr leaves
    leaf_ids = dsl.expr_leaf_ids(spec.expr)
    series_map: Dict[str, Sequence[float]] = {lid: ds.get_series(lid) for lid in leaf_ids}
    series_map[eval_cfg.target_return_leaf] = ret

    def resolver(leaf_id: str, leaf_ctx: Mapping[str, Any]) -> float:
        t = int(leaf_ctx["t"])
        return float(series_map[leaf_id][t])

    # signals
    clip = None if eval_cfg.signal_clip is None else float(eval_cfg.signal_clip)
    signals: List[float] = [0.0] * L
    bad_signal = 0
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

    # positions
    positions: List[float] = [0.0] * L
    if eval_cfg.position_mode == "sign":
        for t in range(L):
            positions[t] = float(_sign(signals[t]))
    elif eval_cfg.position_mode == "raw":
        for t in range(L):
            positions[t] = float(signals[t])
    else:
        # unknown mode -> treat as flat
        return [0.0] * L, 0.0, 0.0

    # pnl (shifted)
    pnl: List[float] = [0.0] * L
    turnover_sum = 0.0
    bad_ret = 0
    cost = float(eval_cfg.transaction_cost)

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

    coverage_signal = 1.0 - (bad_signal / float(L))
    coverage_ret = 1.0 - (bad_ret / float(L))
    coverage = min(coverage_signal, coverage_ret)
    turnover = turnover_sum / float(max(1, L - 1))

    return pnl, float(turnover), float(coverage)


# ---------------------------
# Portfolio config / reports
# ---------------------------

@dataclass(frozen=True)
class PortfolioBuildConfig:
    """
    Portfolio builder configuration (budgeted and deterministic).
    """
    max_n: int = 3
    corr_penalty: float = 0.25  # objective penalty per max_abs_corr
    min_candidate_score: float = float("-inf")
    min_coverage: float = 0.98

    weight_mode: str = "inv_vol"  # currently only inv_vol
    min_vol: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "portfolio_build_config",
            "v": 1,
            "max_n": int(self.max_n),
            "corr_penalty": float(self.corr_penalty),
            "min_candidate_score": float(self.min_candidate_score),
            "min_coverage": float(self.min_coverage),
            "weight_mode": self.weight_mode,
            "min_vol": float(self.min_vol),
        }

    def cid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="portfolio_build_config_v1")


@dataclass(frozen=True)
class PortfolioItem:
    strategy_id: str
    weight: float
    score: float
    pnl_fp: str
    metrics: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "weight": float(self.weight),
            "score": float(self.score),
            "pnl_fp": self.pnl_fp,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PortfolioSpec:
    items: Sequence[PortfolioItem]
    builder: str
    builder_version: str
    config_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "portfolio_spec",
            "v": 1,
            "builder": self.builder,
            "builder_version": self.builder_version,
            "config_id": self.config_id,
            "items": [it.to_dict() for it in self.items],
        }

    def pid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="portfolio_spec_v1")


@dataclass(frozen=True)
class PortfolioReport:
    ok: bool
    reason: str
    spec: PortfolioSpec
    portfolio_metrics: Mapping[str, float]
    portfolio_pnl_fp: str
    dataset_fp: str
    eval_config_id: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "portfolio_report",
            "v": 1,
            "ok": bool(self.ok),
            "reason": self.reason,
            "spec": self.spec.to_dict(),
            "portfolio_metrics": dict(self.portfolio_metrics),
            "portfolio_pnl_fp": self.portfolio_pnl_fp,
            "dataset_fp": self.dataset_fp,
            "eval_config_id": self.eval_config_id,
            "meta": dict(self.meta),
        }

    def rid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="portfolio_report_v1")


# ---------------------------
# Portfolio builder implementation
# ---------------------------

@dataclass(frozen=True)
class _Candidate:
    strategy_id: str
    spec: dsl.StrategySpec
    score: float
    pnl: List[float]             # full length (pnl[0]=0)
    pnl_shifted: List[float]     # pnl[1:]
    metrics: Mapping[str, float]
    turnover: float
    coverage: float
    pnl_fp: str
    vol: float


def _sort_key_score_desc_id(c: _Candidate) -> Tuple[float, str]:
    return (-float(c.score), str(c.strategy_id))


@dataclass(frozen=True)
class SimpleDiversifiedPortfolioBuilder:
    cfg: PortfolioBuildConfig

    def build(
        self,
        candidates: Sequence[dsl.StrategySpec],
        ds: ev.Dataset,
        *,
        eval_cfg: ev.EvalConfig,
        scores_by_id: Optional[Mapping[str, float]] = None,
    ) -> PortfolioReport:
        ds_fp = ev.dataset_fingerprint(ds)
        eval_cid = eval_cfg.cid()
        cfg_id = self.cfg.cid()

        if not candidates:
            empty_spec = PortfolioSpec(items=[], builder="simple_diversified_v1", builder_version=D005_VERSION, config_id=cfg_id)
            return PortfolioReport(
                ok=False,
                reason="No candidates provided",
                spec=empty_spec,
                portfolio_metrics={"mean": 0.0, "stdev": 0.0, "info_ratio": 0.0, "max_drawdown": 0.0},
                portfolio_pnl_fp=series_fingerprint([]),
                dataset_fp=ds_fp,
                eval_config_id=eval_cid,
                meta={"builder": D005_VERSION},
            )

        # Build candidate records (filtering by evaluator ok + coverage)
        recs: List[_Candidate] = []
        for spec in candidates:
            sid = spec.sid()
            score = None
            if scores_by_id is not None and sid in scores_by_id:
                score = float(scores_by_id[sid])
            else:
                # fallback: evaluate on ds
                sc = ev.evaluate_strategy(spec, ds, cfg=eval_cfg)
                score = float(sc.score)

            if score < float(self.cfg.min_candidate_score):
                continue

            pnl, turnover, coverage = pnl_series_for_strategy(spec, ds, eval_cfg=eval_cfg)
            if len(pnl) <= 1:
                continue
            if coverage < float(self.cfg.min_coverage):
                continue

            pnl_shifted = [float(x) for x in pnl[1:]]
            met = series_metrics(pnl_shifted)
            vol = float(met.get("stdev", 0.0))
            if not _isfinite(vol):
                vol = 0.0
            vol_eff = max(float(self.cfg.min_vol), vol)

            recs.append(
                _Candidate(
                    strategy_id=sid,
                    spec=spec,
                    score=float(score),
                    pnl=[float(x) for x in pnl],
                    pnl_shifted=pnl_shifted,
                    metrics=met,
                    turnover=float(turnover),
                    coverage=float(coverage),
                    pnl_fp=series_fingerprint(pnl_shifted),
                    vol=vol_eff,
                )
            )

        if not recs:
            empty_spec = PortfolioSpec(items=[], builder="simple_diversified_v1", builder_version=D005_VERSION, config_id=cfg_id)
            return PortfolioReport(
                ok=False,
                reason="No valid candidates after filtering",
                spec=empty_spec,
                portfolio_metrics={"mean": 0.0, "stdev": 0.0, "info_ratio": 0.0, "max_drawdown": 0.0},
                portfolio_pnl_fp=series_fingerprint([]),
                dataset_fp=ds_fp,
                eval_config_id=eval_cid,
                meta={"builder": D005_VERSION, "candidates_in": int(len(candidates)), "candidates_ok": 0},
            )

        # Deterministic base ordering
        recs_sorted = sorted(recs, key=_sort_key_score_desc_id)

        # Greedy diversified selection
        max_n = max(1, int(self.cfg.max_n))
        selected: List[_Candidate] = []
        remaining: List[_Candidate] = list(recs_sorted)

        while remaining and len(selected) < max_n:
            if not selected:
                best = remaining[0]
                selected.append(best)
                remaining = [r for r in remaining if r.strategy_id != best.strategy_id]
                continue

            best_obj = float("-inf")
            best_id = ""
            best_cand: Optional[_Candidate] = None

            for cand in remaining:
                # max abs correlation to already selected
                max_abs_r = 0.0
                for s in selected:
                    r = abs(corr(cand.pnl_shifted, s.pnl_shifted))
                    if r > max_abs_r:
                        max_abs_r = r
                obj = float(cand.score) - float(self.cfg.corr_penalty) * float(max_abs_r)

                # Deterministic tie-break: higher obj, then strategy_id lexicographically
                if obj > best_obj + 0.0:
                    best_obj = obj
                    best_id = cand.strategy_id
                    best_cand = cand
                elif abs(obj - best_obj) <= 0.0:
                    if best_cand is None or cand.strategy_id < best_id:
                        best_id = cand.strategy_id
                        best_cand = cand

            if best_cand is None:
                break
            selected.append(best_cand)
            remaining = [r for r in remaining if r.strategy_id != best_cand.strategy_id]

        # Weights
        if self.cfg.weight_mode != "inv_vol":
            # fallback to equal weights
            w = 1.0 / float(len(selected))
            weights = [w] * len(selected)
        else:
            inv = [1.0 / (float(c.vol) + _EPS) for c in selected]
            s = math.fsum(inv)
            if s <= 0.0:
                w = 1.0 / float(len(selected))
                weights = [w] * len(selected)
            else:
                weights = [float(x / s) for x in inv]

        # Portfolio pnl series (shifted)
        n = len(selected[0].pnl_shifted) if selected else 0
        port_pnl: List[float] = [0.0] * n
        for i, c in enumerate(selected):
            wi = float(weights[i])
            # align by min length
            m = min(n, len(c.pnl_shifted))
            for t in range(m):
                port_pnl[t] += wi * float(c.pnl_shifted[t])

        port_met = series_metrics(port_pnl)
        port_fp = series_fingerprint(port_pnl)

        items: List[PortfolioItem] = []
        for i, c in enumerate(selected):
            items.append(
                PortfolioItem(
                    strategy_id=c.strategy_id,
                    weight=float(weights[i]),
                    score=float(c.score),
                    pnl_fp=c.pnl_fp,
                    metrics=dict(c.metrics),
                )
            )

        spec = PortfolioSpec(
            items=items,
            builder="simple_diversified_v1",
            builder_version=D005_VERSION,
            config_id=cfg_id,
        )

        # Correlation summary (small, deterministic)
        max_pair_corr = 0.0
        if len(selected) >= 2:
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    r = abs(corr(selected[i].pnl_shifted, selected[j].pnl_shifted))
                    if r > max_pair_corr:
                        max_pair_corr = r

        # Weight invariants (best-effort; not raising)
        wsum = math.fsum(float(it.weight) for it in items)

        return PortfolioReport(
            ok=True,
            reason="OK",
            spec=spec,
            portfolio_metrics=dict(port_met),
            portfolio_pnl_fp=port_fp,
            dataset_fp=ds_fp,
            eval_config_id=eval_cid,
            meta={
                "builder": D005_VERSION,
                "selected_n": int(len(items)),
                "candidates_in": int(len(candidates)),
                "candidates_ok": int(len(recs_sorted)),
                "max_pair_abs_corr": float(max_pair_corr),
                "weight_sum": float(wsum),
            },
        )


# ---------------------------
# Holdout portfolio research runner
# ---------------------------

@dataclass(frozen=True)
class HoldoutPortfolioResearchConfig:
    """
    Deterministic, offline config for a holdout portfolio research run.
    """
    dataset_provider_name: str = "synth_v1"
    dataset_provider_kwargs: Mapping[str, Any] = field(default_factory=dict)

    holdout_fraction: float = 0.70

    enumerator_name: str = "expr_enum_v1"
    enum_budget: dsl.EnumBudget = dsl.EnumBudget(max_strategies=32, max_depth=3, max_nodes=15, max_attempts=5000, seed=42)
    const_pool: Tuple[Any, ...] = (0.0, 1.0)

    evaluator_name: str = ev.REG_EVALUATOR_NAME
    eval_config: ev.EvalConfig = ev.EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=64, min_coverage=0.98)

    preselect_m: int = 8
    candidate_pool_n: int = 10

    portfolio_builder_name: str = "simple_diversified_v1"
    portfolio_build_config: PortfolioBuildConfig = PortfolioBuildConfig(max_n=3, corr_penalty=0.25)

    base_final_k: int = 3  # included for reference (top-k by test score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": "holdout_portfolio_research_config",
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
            "candidate_pool_n": int(self.candidate_pool_n),
            "portfolio_builder_name": self.portfolio_builder_name,
            "portfolio_build_config": self.portfolio_build_config.to_dict(),
            "base_final_k": int(self.base_final_k),
        }

    def cid(self) -> str:
        return k.stable_hash(self.to_dict(), salt="holdout_portfolio_research_config_v1")


@dataclass(frozen=True)
class ResearchPortfolioReport:
    run_id: str
    payload: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.payload)
        d["run_id"] = self.run_id
        return d

    def rid(self) -> str:
        return self.run_id


@dataclass(frozen=True)
class _DatasetLeafProvider:
    ds: ev.Dataset
    eval_cfg: ev.EvalConfig

    def iter_leaf_specs(self) -> Iterable[dsl.LeafSpec]:
        for lid in self.ds.series_ids():
            if lid == self.eval_cfg.target_return_leaf and not self.eval_cfg.allow_target_leaf_in_expr:
                continue
            if lid in self.eval_cfg.forbidden_leaf_ids:
                continue
            if any(str(lid).startswith(pfx) for pfx in self.eval_cfg.forbidden_leaf_prefixes):
                if self.eval_cfg.allow_target_leaf_in_expr and lid == self.eval_cfg.target_return_leaf:
                    pass
                else:
                    continue
            yield dsl.LeafSpec(leaf_id=str(lid), meta={})


def _time_split_dataset(ds: ev.Dataset, split_index: int) -> Tuple[ev.InMemoryDataset, ev.InMemoryDataset]:
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


def _sort_key_scorecard(sc: ev.ScoreCard) -> Tuple[float, str]:
    return (-float(sc.score), str(sc.strategy_id))


@dataclass(frozen=True)
class HoldoutPortfolioRunner:
    cfg: HoldoutPortfolioResearchConfig

    def run(self) -> ResearchPortfolioReport:
        disc = k.discover(strict=False)

        reg_snap = k.REGISTRY.snapshot()
        reg_hash = k.stable_hash(reg_snap, salt="registry_snapshot_v1")

        # dataset
        dp_item = k.REGISTRY.resolve(kind=REG_KIND_DATASET_PROVIDER, name=self.cfg.dataset_provider_name)
        ds_full = dp_item.create(**dict(self.cfg.dataset_provider_kwargs))

        L = int(ds_full.length())
        hf = float(self.cfg.holdout_fraction)
        if not (0.0 < hf < 1.0):
            hf = 0.70
        split_index = int(math.floor(L * hf))
        train_ds, test_ds = _time_split_dataset(ds_full, split_index)

        # leaves from train
        leaf_provider = _DatasetLeafProvider(train_ds, self.cfg.eval_config)

        # enumerator
        enum_item = k.REGISTRY.resolve(kind=dsl.REG_KIND_STRATEGY_ENUMERATOR, name=self.cfg.enumerator_name)
        enum_fn = enum_item.create()

        strategies: List[dsl.StrategySpec] = list(
            enum_fn(
                leaf_provider=leaf_provider,
                operators=None,
                const_pool=tuple(self.cfg.const_pool),
                budget=self.cfg.enum_budget,
            )
        )
        id_to_spec: Dict[str, dsl.StrategySpec] = {s.sid(): s for s in strategies}

        # evaluator
        eval_item = k.REGISTRY.resolve(kind=ev.REG_KIND_EVALUATOR, name=self.cfg.evaluator_name)
        try:
            evaluator = eval_item.create(cfg=self.cfg.eval_config)
        except TypeError:
            # Fallback if evaluator factory does not support cfg kwarg
            evaluator = eval_item.create()

        # train eval
        train_scores: List[ev.ScoreCard] = [evaluator.evaluate(s, train_ds) for s in strategies]
        train_ok_sorted = sorted([sc for sc in train_scores if sc.ok], key=_sort_key_scorecard)

        pre_m = max(1, int(self.cfg.preselect_m))
        pre_ids = [sc.strategy_id for sc in train_ok_sorted[:pre_m]]
        pre_specs = [id_to_spec[sid] for sid in pre_ids if sid in id_to_spec]

        # test eval
        test_scores: List[ev.ScoreCard] = [evaluator.evaluate(s, test_ds) for s in pre_specs]
        test_ok_sorted = sorted([sc for sc in test_scores if sc.ok], key=_sort_key_scorecard)

        # candidate pool for portfolio
        pool_n = max(1, int(self.cfg.candidate_pool_n))
        pool = test_ok_sorted[:pool_n]
        pool_ids = [sc.strategy_id for sc in pool]
        pool_specs = [id_to_spec[sid] for sid in pool_ids if sid in id_to_spec]
        scores_by_id = {sc.strategy_id: float(sc.score) for sc in pool}

        # base final top-k (reference)
        base_k = max(1, int(self.cfg.base_final_k))
        base_final = [
            {"rank": i + 1, "strategy_id": sc.strategy_id, "test_score": float(sc.score)}
            for i, sc in enumerate(test_ok_sorted[:base_k])
        ]

        # portfolio builder
        pb_item = k.REGISTRY.resolve(kind=REG_KIND_PORTFOLIO_BUILDER, name=self.cfg.portfolio_builder_name)
        try:
            pb = pb_item.create(cfg=self.cfg.portfolio_build_config)
        except TypeError:
            pb = pb_item.create()

        port = pb.build(pool_specs, test_ds, eval_cfg=self.cfg.eval_config, scores_by_id=scores_by_id)

        payload: Dict[str, Any] = {
            "t": "research_portfolio_report",
            "v": 1,
            "runner": {"name": "holdout_portfolio_v1", "version": D005_VERSION},
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
            "registry_snapshot_hash": reg_hash,
            "dataset": {
                "provider": {"name": self.cfg.dataset_provider_name, "kwargs": dict(self.cfg.dataset_provider_kwargs)},
                "full": {
                    "length": L,
                    "fingerprint": ev.dataset_fingerprint(ds_full),
                    "series_ids": list(ds_full.series_ids()),
                },
                "split": {"holdout_fraction": float(hf), "split_index": int(split_index)},
                "train": {"length": int(train_ds.length()), "fingerprint": ev.dataset_fingerprint(train_ds)},
                "test": {"length": int(test_ds.length()), "fingerprint": ev.dataset_fingerprint(test_ds)},
            },
            "enumeration": {
                "count": int(len(strategies)),
                "leaf_count": int(len(list(leaf_provider.iter_leaf_specs()))),
                "const_pool": list(self.cfg.const_pool),
                "budget": {
                    "max_strategies": int(self.cfg.enum_budget.max_strategies),
                    "max_depth": int(self.cfg.enum_budget.max_depth),
                    "max_nodes": int(self.cfg.enum_budget.max_nodes),
                    "max_attempts": int(self.cfg.enum_budget.max_attempts),
                    "seed": int(self.cfg.enum_budget.seed),
                },
            },
            "evaluation": {
                "train": {"count": int(len(train_scores)), "ok": int(sum(1 for sc in train_scores if sc.ok))},
                "test": {"count": int(len(test_scores)), "ok": int(sum(1 for sc in test_scores if sc.ok))},
            },
            "candidates": {
                "pool_n": int(pool_n),
                "pool": [{"strategy_id": sc.strategy_id, "test_score": float(sc.score)} for sc in pool],
                "base_final_topk": base_final,
            },
            "portfolio": port.to_dict(),
        }

        run_id = k.stable_hash(payload, salt="research_portfolio_report_v1")
        return ResearchPortfolioReport(run_id=run_id, payload=payload)


# ---------------------------
# nb_local defaults (optional)
# ---------------------------

def default_holdout_portfolio_config_from_nb_local() -> HoldoutPortfolioResearchConfig:
    nb = k.load_nb_local()

    eval_cfg = ev.default_eval_config_from_nb_local()

    provider_name = str(nb.get("RESEARCH_DATASET_PROVIDER_NAME", "synth_v1"))
    ds_len = int(nb.get("RESEARCH_DATASET_LENGTH", 512))
    ds_seed = int(nb.get("RESEARCH_DATASET_SEED", nb.get("RANDOM_SEED", 1337)))
    holdout_fraction = float(nb.get("RESEARCH_HOLDOUT_FRACTION", 0.70))

    enum_name = str(nb.get("RESEARCH_ENUMERATOR_NAME", "expr_enum_v1"))
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

    evaluator_name = str(nb.get("RESEARCH_EVALUATOR_NAME", ev.REG_EVALUATOR_NAME))

    pre_m = int(nb.get("RESEARCH_PRESELECT_M", 8))
    pool_n = int(nb.get("PORTFOLIO_CANDIDATE_POOL_N", 10))

    builder_name = str(nb.get("PORTFOLIO_BUILDER_NAME", "simple_diversified_v1"))
    max_n = int(nb.get("PORTFOLIO_MAX_N", 3))
    corr_pen = float(nb.get("PORTFOLIO_CORR_PENALTY", 0.25))

    build_cfg = PortfolioBuildConfig(
        max_n=max_n,
        corr_penalty=corr_pen,
        min_candidate_score=float(nb.get("PORTFOLIO_MIN_CANDIDATE_SCORE", float("-inf"))),
        min_coverage=float(nb.get("PORTFOLIO_MIN_COVERAGE", 0.98)),
        weight_mode=str(nb.get("PORTFOLIO_WEIGHT_MODE", "inv_vol")),
        min_vol=float(nb.get("PORTFOLIO_MIN_VOL", 1e-6)),
    )

    base_final_k = int(nb.get("PORTFOLIO_BASE_FINAL_K", 3))

    ds_kwargs = {"length": ds_len, "seed": ds_seed, "target_return_leaf": eval_cfg.target_return_leaf}

    return HoldoutPortfolioResearchConfig(
        dataset_provider_name=provider_name,
        dataset_provider_kwargs=ds_kwargs,
        holdout_fraction=holdout_fraction,
        enumerator_name=enum_name,
        enum_budget=enum_budget,
        const_pool=const_pool,
        evaluator_name=evaluator_name,
        eval_config=eval_cfg,
        preselect_m=pre_m,
        candidate_pool_n=pool_n,
        portfolio_builder_name=builder_name,
        portfolio_build_config=build_cfg,
        base_final_k=base_final_k,
    )


# ---------------------------
# Registry registrations
# ---------------------------

def portfolio_builder_factory(cfg: Optional[PortfolioBuildConfig] = None) -> SimpleDiversifiedPortfolioBuilder:
    if cfg is None:
        cfg = PortfolioBuildConfig()
    return SimpleDiversifiedPortfolioBuilder(cfg=cfg)


def holdout_portfolio_runner_factory(cfg: Optional[HoldoutPortfolioResearchConfig] = None) -> HoldoutPortfolioRunner:
    if cfg is None:
        cfg = default_holdout_portfolio_config_from_nb_local()
    return HoldoutPortfolioRunner(cfg=cfg)


def _register_d005_once() -> None:
    sentinel = "_dmb_d005_registered_v1"
    if getattr(sys, sentinel, False):
        return
    setattr(sys, sentinel, True)

    # Portfolio builder
    k.REGISTRY.register(
        kind=REG_KIND_PORTFOLIO_BUILDER,
        name="simple_diversified_v1",
        version="1.0.0",
        provider=portfolio_builder_factory,
        meta={"offline": True, "deterministic": True, "weights": "inv_vol", "selection": "greedy_corr_penalty"},
        source="d005_portfolio_builder.py",
    )

    # Portfolio research runner (new name; does not override d004 runner)
    k.REGISTRY.register(
        kind=REG_KIND_RESEARCH_RUNNER,
        name="holdout_portfolio_v1",
        version="1.0.0",
        provider=holdout_portfolio_runner_factory,
        meta={"offline": True, "deterministic": True, "holdout": True, "portfolio": True},
        source="d005_portfolio_builder.py",
    )


_register_d005_once()


# ---------------------------
# Selftest
# ---------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _selftest_registry_integration() -> None:
    dr = k.discover(strict=False)
    _assert("d005_portfolio_builder.py" in dr.seen_files, "Discovery must see d005_portfolio_builder.py in repo root")

    _ = k.REGISTRY.resolve(kind=REG_KIND_PORTFOLIO_BUILDER, name="simple_diversified_v1")
    _ = k.REGISTRY.resolve(kind=REG_KIND_RESEARCH_RUNNER, name="holdout_portfolio_v1")
    # dataset provider should exist from d004; discover() will import it.
    _ = k.REGISTRY.resolve(kind=REG_KIND_DATASET_PROVIDER, name="synth_v1")


def _selftest_end_to_end_determinism_and_weights() -> None:
    cfg = HoldoutPortfolioResearchConfig(
        dataset_provider_name="synth_v1",
        dataset_provider_kwargs={"length": 512, "seed": 123, "target_return_leaf": "target.ret"},
        holdout_fraction=0.70,
        enumerator_name="expr_enum_v1",
        enum_budget=dsl.EnumBudget(max_strategies=32, max_depth=3, max_nodes=15, max_attempts=5000, seed=42),
        const_pool=(0.0, 1.0),
        evaluator_name=ev.REG_EVALUATOR_NAME,
        eval_config=ev.EvalConfig(target_return_leaf="target.ret", transaction_cost=0.0005, min_obs=64, min_coverage=0.98),
        preselect_m=10,
        candidate_pool_n=10,
        portfolio_builder_name="simple_diversified_v1",
        portfolio_build_config=PortfolioBuildConfig(max_n=3, corr_penalty=0.25, min_coverage=0.98),
        base_final_k=3,
    )

    runner_item = k.REGISTRY.resolve(kind=REG_KIND_RESEARCH_RUNNER, name="holdout_portfolio_v1")

    rep1 = runner_item.create(cfg=cfg).run()
    rep2 = runner_item.create(cfg=cfg).run()

    _assert(rep1.run_id == rep2.run_id, "Portfolio research run_id must be deterministic")
    _assert(rep1.to_dict() == rep2.to_dict(), "Portfolio research payload must be deterministic")

    d = rep1.to_dict()
    candidates_pool = d["candidates"]["pool"]
    port = d["portfolio"]
    items = port["spec"]["items"]

    max_n = int(cfg.portfolio_build_config.max_n)
    _assert(len(items) <= max_n, "Selected portfolio items must not exceed max_n")

    # If we have at least 2 candidates and max_n >= 2, we expect >= 2 selected (on synth dataset).
    if len(candidates_pool) >= 2 and max_n >= 2:
        _assert(len(items) >= 2, "Portfolio should select at least 2 strategies when available and allowed")

    wsum = math.fsum(float(it["weight"]) for it in items) if items else 0.0
    _assert(abs(wsum - 1.0) < 1e-9, "Portfolio weights must sum to 1")

    _assert(all(float(it["weight"]) >= -1e-12 for it in items), "Portfolio weights must be non-negative (long-only)")


def selftest() -> int:
    print(f"[d005] Portfolio builder selftest — version {D005_VERSION}")

    _selftest_registry_integration()
    print("[d005] OK: registry integration")

    _selftest_end_to_end_determinism_and_weights()
    print("[d005] OK: end-to-end determinism + weights invariants")

    print("[d005] SELFTEST PASS")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        try:
            return selftest()
        except AssertionError as e:
            print("[d005] SELFTEST FAIL:", e)
            return 2
        except Exception as e:
            print("[d005] SELFTEST ERROR:", f"{type(e).__name__}: {e}")
            return 3

    if "--run" in argv:
        cfg = default_holdout_portfolio_config_from_nb_local()
        runner = HoldoutPortfolioRunner(cfg=cfg)
        rep = runner.run().to_dict()

        # compact deterministic summary (no timestamps)
        summary = {
            "run_id": rep["run_id"],
            "config_id": rep["config_id"],
            "portfolio_ok": rep["portfolio"]["ok"],
            "selected_n": len(rep["portfolio"]["spec"]["items"]),
            "selected": [
                {
                    "strategy_id": it["strategy_id"],
                    "weight": it["weight"],
                    "score": it["score"],
                }
                for it in rep["portfolio"]["spec"]["items"]
            ],
            "portfolio_info_ratio": rep["portfolio"]["portfolio_metrics"].get("info_ratio"),
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0

    print("d005_portfolio_builder.py")
    print("Usage:")
    print("  python d005_portfolio_builder.py --selftest")
    print("  python d005_portfolio_builder.py --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
