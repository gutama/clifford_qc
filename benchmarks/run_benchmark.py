"""One-command benchmark matrix: models x methods x seeds -> JSONL.

Each output line is one complete ADAPT run with full outer-loop resource
accounting (shots, circuits, unique words, operators, selection statuses,
regret against the exact gradient). Configs are JSON files; see
``benchmarks/configs/spin_small.json``.

Run:
    python benchmarks/run_benchmark.py --config benchmarks/configs/spin_small.json \
        --out results.jsonl
    python benchmarks/summarize.py results.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clifford_qc.models import tfim, xxz, random_ising
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import UniformFixed, UniformDoubling, VarianceProportional
from clifford_qc.algorithms import (
    ConfidenceSelector, RandomSelector, all_words_pool, local_pool, odd_y_filter,
    run_adapt,
)

MODELS = {"tfim": tfim, "xxz": xxz, "random_ising": random_ising}
ALLOCATORS = {"uniform_fixed": UniformFixed, "uniform_doubling": UniformDoubling,
              "variance_proportional": VarianceProportional}


def build_model(cfg: dict, seed: int):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "random_ising":  # disorder realization follows the run seed
        cfg.setdefault("seed", seed)
    return MODELS[kind](**cfg)


def build_pool(cfg: dict, n: int):
    cfg = dict(cfg)
    kind = cfg.pop("type", "local")
    if kind == "local":
        return local_pool(n, **cfg)
    if kind == "odd_y_all_words":
        return odd_y_filter(all_words_pool(n, **cfg))
    raise ValueError(f"unknown pool type {kind!r}")


def build_run_kwargs(method: dict, seed: int) -> dict:
    method = dict(method)
    kind = method.pop("kind")
    kwargs = {k: method.pop(k) for k in
              ("grouping", "subpool_size", "layer_alpha", "allow_repeats",
               "max_operators", "threshold", "maxiter") if k in method}
    if kind == "exact":
        pass
    elif kind == "random":
        kwargs["selector"] = RandomSelector(seed=seed)
    elif kind == "confidence":
        kwargs["selector"] = ConfidenceSelector(**method.pop("selector", {}))
        if "allocator" not in method:
            raise ValueError("confidence methods need an 'allocator' entry")
        alloc = dict(method.pop("allocator"))
        alloc_type = alloc.pop("type", None)
        if alloc_type not in ALLOCATORS:
            raise ValueError(f"unknown allocator type {alloc_type!r}; "
                             f"known: {sorted(ALLOCATORS)}")
        kwargs["allocator"] = ALLOCATORS[alloc_type](**alloc)
        kwargs["backend"] = FiniteShotBackend(seed=seed)
        kwargs.setdefault("subpool_seed", seed)
    else:
        raise ValueError(f"unknown method kind {kind!r}")
    if method:
        raise ValueError(f"unused method keys: {sorted(method)}")
    return kwargs


def summarize_records(records) -> dict:
    selected = [r for r in records if r.selected_label]
    regret = [abs(r.exact_gradient_max) - abs(r.exact_gradient)
              for r in selected
              if r.exact_gradient is not None and r.exact_gradient_max is not None]
    statuses: dict[str, int] = {}
    for r in records:
        statuses[r.status.value] = statuses.get(r.status.value, 0) + 1
    return {
        "selection_steps": len(selected),
        "unique_words_total": sum(r.unique_words_measured for r in records),
        "status_counts": statuses,
        "mean_regret": sum(regret) / len(regret) if regret else None,
        "max_regret": max(regret) if regret else None,
        "near_optimal_rate": (sum(1 for r in selected
                                  if r.exact_gradient is not None
                                  and r.exact_gradient_max is not None
                                  and abs(r.exact_gradient) >= 0.95 * abs(r.exact_gradient_max))
                              / len(selected)) if selected else None,
    }


def run_one(model_cfg: dict, pool_cfg: dict, method_name: str, method: dict,
            seed: int) -> dict:
    model = build_model(model_cfg, seed)
    pool = build_pool(pool_cfg, model.n)
    kwargs = build_run_kwargs(method, seed)
    start = time.perf_counter()
    res = run_adapt(model, pool, **kwargs)
    wall = time.perf_counter() - start
    row = {
        "model": model.name, "method": method_name, "seed": seed,
        "n": model.n, "pool_size": len(pool),
        "energy": res.energy, "exact_ground_energy": res.exact_ground_energy,
        "relative_error": res.relative_error,
        "operators": len(res.labels), "labels": list(res.labels),
        "total_shots": res.total_shots, "total_circuits": res.total_circuits,
        "support_peak": res.support_peak, "stopped_reason": res.stopped_reason,
        "wall_seconds": wall,
    }
    row.update(summarize_records(res.records))
    return row


def parse_shard(text: str) -> tuple[int, int]:
    index, count = (int(x) for x in text.split("/"))
    if not (count >= 1 and 0 <= index < count):
        raise ValueError(f"shard must be i/k with 0 <= i < k, got {text!r}")
    return index, count


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=None,
                        help="override the config's seed count")
    parser.add_argument("--shard", default="0/1",
                        help="i/k with 0-indexed i (0 <= i < k): run every "
                             "k-th job starting at job i, for launching k "
                             "parallel workers writing separate files "
                             "(concatenate afterwards)")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    n_seeds = args.seeds if args.seeds is not None else config.get("seeds", 30)
    shard_index, shard_count = parse_shard(args.shard)

    jobs = [(model_entry, method_name, method, seed)
            for model_entry in config["models"]
            for method_name, method in config["methods"].items()
            for seed in range(n_seeds)]
    jobs = jobs[shard_index::shard_count]

    out = Path(args.out)
    done = 0
    with out.open("w") as fh:
        for model_entry, method_name, method, seed in jobs:
            row = run_one(model_entry["model"], model_entry.get("pool", {}),
                          method_name, method, seed)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            print(f"[{done}/{len(jobs)}] {row['model']} / {method_name} / seed {seed}: "
                  f"rel={row['relative_error']:.2e} shots={row['total_shots']:,}",
                  flush=True)
    print(f"wrote {done} runs to {out}")


if __name__ == "__main__":
    main()
