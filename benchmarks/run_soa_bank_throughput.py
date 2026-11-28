"""Throughput of the ``kernel="soa"`` bank route against the ``mv`` default.

Two measurements, both on the frozen Phase 2M inputs (FCIDUMP, determinant
singles+doubles pool, ``ACASEConfig`` options from
``benchmarks/configs/bank_policy_matrix.json``):

``pairs``
    Pair-product throughput. For each system, build the first ``--pairs``
    upper-triangle ``(A_i' A_j, A_i' H A_j)`` products of the candidate pool
    under each kernel, ``--repeats`` times, in this process. Reports pairs per
    second, output coefficients per second, and that both kernels produce the
    same words with coefficients equal to ``--atol``.

``solve``
    One full exact A-CASE solve per kernel on one system (default
    ``h2o_stretched``, the 14-qubit ``M = 31`` cell where Phase 2M measured
    444.9 s of an 850.4 s solve in first pair builds). Each kernel runs in a
    fresh interpreter so neither inherits the other's allocator state or word
    memo. Reports wall time, product time, selected labels and energy, and
    compares labels and energy with the frozen ``bank_policy_matrix.json`` cell
    of the same storage arm.

This is a timing record, not a scientific one: nothing here feeds an energy
claim, and wall times depend on the host recorded beside them. Run
``python benchmarks/run_soa_bank_throughput.py --help`` for options.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

CONFIG = ROOT / "benchmarks/configs/bank_policy_matrix.json"
FROZEN = ROOT / "benchmarks/reference_results/bank_policy_matrix.json"
OUTPUT = ROOT / "benchmarks/timing_results/soa_bank_throughput_current.json"
ARCHIVE = ROOT / "benchmarks/timing_results/soa_bank_throughput.json"
KERNELS = ("mv", "soa")
ARM_BY_STORAGE = {"object": "object_retain", "packed": "packed_retain"}


def load_config():
    return json.loads(CONFIG.read_text())


def system_spec(config, system_id):
    for spec in config["systems"]:
        if spec["id"] == system_id:
            return spec
    raise KeyError(system_id)


def build_problem(spec):
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.models.fcidump import fcidump_model
    from clifford_qc.subspace.fermionic_generators import (
        determinant_excitations, occupied_spin_orbitals,
    )
    model = fcidump_model(ROOT / spec["source"]["path"])
    candidates = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    if len(candidates) != spec["manifest"]["candidate_count"]:
        raise ValueError("candidate pool differs from the frozen manifest")
    rho = ExactMVBackend().state(model.reference, ())
    return model, rho, candidates


def host():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.machine(), "processor": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "kernel_cache_size_env": os.environ.get("CLIFFORD_QC_KERNEL_CACHE_SIZE"),
    }


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def source_identity(specs):
    """Bind the actual run to committed, retrievable code and frozen inputs."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    paths = sorted({CONFIG, FROZEN, Path(__file__).resolve(),
                    *(ROOT / spec["source"]["path"] for spec in specs)})
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--",
                "clifford_qc", *relative)
    if dirty:
        raise ValueError(f"timing needs clean committed sources and inputs:\n{dirty}")
    for path in relative:
        git("ls-files", "--error-unmatch", "--", path)
    return {
        "status": "clean_committed_source",
        "commit": git("rev-parse", "HEAD"),
        "package_tree": git("rev-parse", "HEAD:clifford_qc"),
        "inputs_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in relative},
    }


# ------------------------------------------------------------------ pairs

def measure_pairs(spec, pairs, repeats, atol):
    from clifford_qc import clear_kernel_caches
    from clifford_qc.subspace.projection import MatrixElementBank

    model, rho, candidates = build_problem(spec)
    keys = [(i, j) for j in range(len(candidates)) for i in range(j + 1)][:pairs]
    timings = {kernel: [] for kernel in KERNELS}
    coefficients = {}
    products = {}
    for _ in range(repeats):
        for kernel in KERNELS:
            clear_kernel_caches()
            bank = MatrixElementBank(rho, model.hamiltonian, kernel=kernel)
            setup = time.perf_counter()
            bank.extend(candidates[: keys[-1][1] + 1])
            acted = time.perf_counter() - setup
            started = time.perf_counter()
            out = [bank._pair_products(i, j) for i, j in keys]
            timings[kernel].append({"h_acted_seconds": acted,
                                    "pair_seconds": time.perf_counter() - started})
            coefficients[kernel] = sum(a.nnz() + b.nnz() for a, b in out)
            products[kernel] = out
    max_diff = 0.0
    same_words = True
    for (a0, b0), (a1, b1) in zip(products["mv"], products["soa"]):
        for x, y in ((a0, a1), (b0, b1)):
            if set(x.terms) != set(y.terms):
                same_words = False
                continue
            for code, value in x.terms.items():
                max_diff = max(max_diff, abs(value - y.terms[code]))
    summary = {}
    for kernel in KERNELS:
        best = min(t["pair_seconds"] for t in timings[kernel])
        summary[kernel] = {
            "best_pair_seconds": best,
            "pairs_per_second": len(keys) / best,
            "output_coefficients_per_second": coefficients[kernel] / best,
            "best_h_acted_seconds": min(t["h_acted_seconds"] for t in timings[kernel]),
            "runs": timings[kernel],
        }
    return {
        "system": spec["id"], "n_qubits": model.n, "hamiltonian_terms": model.hamiltonian.nnz()
        if hasattr(model.hamiltonian, "nnz") else None,
        "pairs": len(keys), "repeats": repeats, "output_coefficients": coefficients["mv"],
        "kernels": summary,
        "speedup_pairs": summary["mv"]["best_pair_seconds"] / summary["soa"]["best_pair_seconds"],
        "parity": {"same_words": same_words, "max_abs_coefficient_difference": max_diff,
                   "atol": atol, "passed": same_words and max_diff <= atol},
    }


# ------------------------------------------------------------------ solve

def solve_worker(system_id, kernel, storage):
    """One full A-CASE solve; runs in a fresh interpreter."""
    from clifford_qc.subspace.adaptive import ACASEConfig, run_acase
    from clifford_qc.subspace.projection import MatrixElementBank

    config = load_config()
    spec = system_spec(config, system_id)
    started = time.perf_counter()
    model, rho, candidates = build_problem(spec)
    options = dict(config["options"])
    options.update(exact_ground_energy=spec.get("exact_ground_energy"))
    bank = MatrixElementBank(rho, model.hamiltonian, storage=storage, kernel=kernel)
    prepared = time.perf_counter()
    result = run_acase(rho, model.hamiltonian, candidates, bank=bank,
                       config=ACASEConfig(**options))
    finished = time.perf_counter()
    return {
        "system": system_id, "kernel": kernel, "storage": storage,
        "effective_kernel": bank.kernel,
        "basis_size": len(result.indices),
        "preparation_seconds": prepared - started,
        "solve_seconds": finished - prepared,
        "product_seconds": bank._seconds,
        "pairs_built": bank._pairs_built,
        "labels": list(result.labels), "energy": float(result.energy).hex(),
        "stopped_reason": result.stopped_reason,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def measure_solve(system_id, storage, kernels, energy_atol):
    runs = {}
    for kernel in kernels:
        print(f"solve {system_id} / {storage} / {kernel}: fresh interpreter", flush=True)
        out = subprocess.run(
            [sys.executable, __file__, "--worker", system_id, kernel, storage],
            cwd=ROOT, check=True, capture_output=True, text=True)
        runs[kernel] = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"  solve {runs[kernel]['solve_seconds']:.1f} s, products "
              f"{runs[kernel]['product_seconds']:.1f} s", flush=True)
    frozen = None
    arm = ARM_BY_STORAGE[storage]
    for cell in json.loads(FROZEN.read_text())["cells"]:
        if cell["system"] == system_id and cell["arm"][0] == arm:
            frozen = cell
    if frozen is None:
        raise ValueError(f"no frozen control cell for {system_id}/{arm}")
    checks = {}
    for kernel, run in runs.items():
        check = {}
        if frozen is not None:
            frozen_energy = float.fromhex(frozen["scientific"]["energy"])
            check = {
                "frozen_arm": arm,
                "labels_match_frozen": run["labels"] == frozen["scientific"]["labels"],
                "energy_abs_difference_from_frozen": abs(float.fromhex(run["energy"]) - frozen_energy),
                "frozen_solve_seconds": frozen["costs"]["solve_seconds"],
                "frozen_first_pair_build_seconds": frozen["costs"]["first_pair_build_seconds"],
            }
            check["passed"] = (check["labels_match_frozen"]
                               and check["energy_abs_difference_from_frozen"] <= energy_atol)
        checks[kernel] = check
    summary = {"system": system_id, "storage": storage, "runs": runs, "frozen_checks": checks,
               "energy_atol": energy_atol}
    if set(KERNELS).issubset(runs):
        summary["speedup_solve"] = runs["mv"]["solve_seconds"] / runs["soa"]["solve_seconds"]
        summary["speedup_products"] = runs["mv"]["product_seconds"] / runs["soa"]["product_seconds"]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worker", nargs=3, metavar=("SYSTEM", "KERNEL", "STORAGE"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--systems", nargs="+", default=["h4", "h2o_stretched"],
                        help="systems for the pair-throughput sweep")
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--solve-system", default="h2o_stretched")
    parser.add_argument("--solve-storage", choices=sorted(ARM_BY_STORAGE), default="packed",
                        help="packed keeps peak RSS near 2.6 GiB on h2o_stretched; "
                             "object needs about 10 GiB")
    parser.add_argument("--solve-kernels", nargs="+", choices=(*KERNELS, "auto"),
                        default=list(KERNELS))
    parser.add_argument("--energy-atol", type=float, default=1e-10)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    if args.worker:
        print(json.dumps(solve_worker(*args.worker)))
        return 0

    if args.output.resolve() == ARCHIVE.resolve():
        parser.error("the historical timing archive is immutable; use another output path")
    if args.pairs < 1 or args.repeats < 1:
        parser.error("--pairs and --repeats must be positive")
    if not np.isfinite(args.atol) or args.atol < 0 or not np.isfinite(args.energy_atol) or args.energy_atol < 0:
        parser.error("tolerances must be finite and nonnegative")
    config = load_config()
    specs = [system_spec(config, name) for name in args.systems]
    if not args.skip_solve:
        specs.append(system_spec(config, args.solve_system))
    source = source_identity(specs)
    record = {
        "schema": "clifford_qc.soa_bank_throughput.v2",
        "claim_boundary": ("Timing only. Wall times are host-specific; parity and "
                           "frozen-label checks apply to the declared workloads."),
        "git_head": source["commit"], "source_identity": source, "host": host(),
        "pairs": [measure_pairs(system_spec(config, s), args.pairs, args.repeats, args.atol)
                  for s in args.systems],
    }
    for item in record["pairs"]:
        print(f"pairs {item['system']}: mv {item['kernels']['mv']['pairs_per_second']:.0f}/s, "
              f"soa {item['kernels']['soa']['pairs_per_second']:.0f}/s "
              f"({item['speedup_pairs']:.1f}x), parity {item['parity']['passed']}", flush=True)
    if not args.skip_solve:
        record["solve"] = measure_solve(args.solve_system, args.solve_storage,
                                        args.solve_kernels, args.energy_atol)
    passed = all(item["parity"]["passed"] for item in record["pairs"]) and all(
        check.get("passed", False) for check in record.get("solve", {}).get("frozen_checks", {}).values())
    record["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}; "
          f"passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
