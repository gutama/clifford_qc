"""Reproducible throughput & parity benchmark for the Rank 5a SoA Pauli kernel.

Compares the legacy Python-dict ``MV`` kernel (cold and warm process memo)
against :mod:`clifford_qc.soa_kernel` on CPU (NumPy) and, when available,
CUDA GPU (``cupy`` fused ``__popcll`` kernel and ``torch`` CUDA SWAR kernel).

Usage examples:
    python benchmarks/benchmark_soa_kernel.py --quick
    python benchmarks/benchmark_soa_kernel.py --backends numpy --json-out results.json
    python benchmarks/benchmark_soa_kernel.py --backends all --gpu-sweep --json-out gpu_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc import MV, clear_kernel_caches, kernel_cache_info
from clifford_qc.models import fcidump_model
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.soa_kernel import (
    PauliSoA,
    available_gpu_backends,
    build_soa_bank,
    soa_anticommute_mask,
    soa_commutator,
    soa_mul,
    soa_trace_pairing,
    synchronize_backend,
)
from clifford_qc.subspace import MatrixElementBank, identity_generator
from clifford_qc.subspace.symmetry_pools import (
    closed_shell_occupied_spatial,
    spin_adapted_excitations,
)


def _git_meta() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": None}


def _env_meta() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numpy_bitwise_count": hasattr(np, "bitwise_count"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_backends": available_gpu_backends(),
        **_git_meta(),
    }


def _random_soa_and_mv(
    rng: np.random.Generator, n: int, t: int, *, need_mv: bool = True
) -> tuple[PauliSoA, MV | None]:
    if n <= 12:
        codes = rng.choice(1 << (2 * n), size=min(t, 1 << (2 * n)), replace=False)
        coeffs = rng.normal(size=codes.size) + 1j * rng.normal(size=codes.size)
        soa = PauliSoA.from_arrays(n, codes, coeffs)
    else:
        w = (n + 31) // 32 if n > 32 else 1
        if n <= 32:
            hi = (1 << (2 * n)) if n < 32 else (1 << 64)
            # Sample slightly more and dedup to guarantee t distinct codes.
            pool = rng.integers(0, hi, size=int(t * 1.15) + 16, dtype=np.uint64)
            uniq = np.unique(pool)
            while uniq.size < t:
                extra = rng.integers(0, hi, size=t, dtype=np.uint64)
                uniq = np.unique(np.concatenate((uniq, extra)))
            codes = rng.choice(uniq, size=t, replace=False)
            coeffs = rng.normal(size=t) + 1j * rng.normal(size=t)
            soa = PauliSoA.from_arrays(n, codes, coeffs)
        else:
            limbs = rng.integers(0, np.iinfo(np.uint64).max, size=(t, w), dtype=np.uint64)
            rem = n - 32 * (w - 1)
            if rem < 32:
                limbs[:, -1] &= np.uint64((1 << (2 * rem)) - 1)
            coeffs = rng.normal(size=t) + 1j * rng.normal(size=t)
            soa = PauliSoA.from_arrays(n, limbs, coeffs)
    mv = soa.to_mv() if need_mv else None
    return soa, mv


def _time_fn(fn, *, repeats: int, warmup: int, backend: str = "numpy") -> tuple[Any, dict[str, float]]:
    out = None
    for _ in range(warmup):
        out = fn()
        synchronize_backend(backend)
    samples: list[float] = []
    for _ in range(repeats):
        synchronize_backend(backend)
        t0 = time.perf_counter()
        out = fn()
        synchronize_backend(backend)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    med = statistics.median(samples)
    q1 = float(np.percentile(samples, 25))
    q3 = float(np.percentile(samples, 75))
    return out, {
        "median_s": med,
        "min_s": samples[0],
        "max_s": samples[-1],
        "iqr_s": q3 - q1,
        "samples_s": samples,
    }


def _max_soa_diff(a: PauliSoA, b: PauliSoA) -> float:
    if a.nnz() != b.nnz():
        return float("inf")
    if not np.array_equal(a.codes, b.codes):
        return float("inf")
    if a.nnz() == 0:
        return 0.0
    return float(np.max(np.abs(a.coeffs - b.coeffs)))


def run_random_suite(
    backends: list[str],
    *,
    repeats: int,
    warmup: int,
    quick: bool,
    gpu_sweep: bool,
) -> list[dict[str, Any]]:
    cases = [(8, 300), (8, 1000), (12, 1000)] if quick else [
        (8, 300),
        (8, 1000),
        (8, 3000),
        (12, 1000),
        (12, 3000),
    ]
    if gpu_sweep:
        cases.extend([(12, 6000), (12, 10000), (16, 2000), (16, 4000), (16, 6000)])

    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for n, t in cases:
        pairs = t * t
        run_mv = pairs <= (1_000_000 if quick else 9_000_000)
        sa, mv_a = _random_soa_and_mv(rng, n, t, need_mv=run_mv)
        sb, mv_b = _random_soa_and_mv(rng, n, t, need_mv=run_mv)

        row: dict[str, Any] = {"n": n, "t": t, "pairs": pairs, "timings": {}}

        ref_soa: PauliSoA | None = None
        if run_mv and mv_a is not None and mv_b is not None:
            clear_kernel_caches()
            t0 = time.perf_counter()
            mv_prod = mv_a * mv_b
            t_cold = time.perf_counter() - t0
            memo_after_cold = kernel_cache_info()["word_products"]["currsize"]
            t0 = time.perf_counter()
            _ = mv_a * mv_b
            t_warm = time.perf_counter() - t0
            row["mv_cold_s"] = t_cold
            row["mv_warm_s"] = t_warm
            row["mv_memo_entries"] = memo_after_cold
            row["mv_cold_mpairs_s"] = pairs / (t_cold * 1e6)
            row["mv_warm_mpairs_s"] = pairs / (t_warm * 1e6)
            ref_soa = PauliSoA.from_mv(mv_prod)
            row["out_words"] = ref_soa.nnz()

        for backend in backends:
            if backend == "torch" and n > 31:
                continue
            # For very large CPU sweep cells (>16M pairs), run 1 repeat on NumPy to keep runtime reasonable.
            eff_rep = 1 if (backend == "numpy" and pairs > 16_000_000) else repeats
            eff_warm = 0 if (backend == "numpy" and pairs > 16_000_000) else warmup
            if n <= 12 and pairs > 16_000_000:
                modes = ["dense"]
            elif n <= 12:
                modes = ["dense", "sort"]
            else:
                modes = ["sort"]
            for mode in modes:
                key = f"{backend}_{mode}"
                clear_kernel_caches()
                prod, stats = _time_fn(
                    lambda b=backend, m=mode: soa_mul(sa, sb, accumulate=m, backend=b),
                    repeats=eff_rep,
                    warmup=eff_warm,
                    backend=backend,
                )
                memo_used = kernel_cache_info()["word_products"]["currsize"]
                if ref_soa is None:
                    ref_soa = prod
                    row["out_words"] = prod.nnz()
                max_diff = _max_soa_diff(prod, ref_soa)
                assert max_diff < 1e-8, f"parity failure on {key} n={n} t={t}: diff={max_diff}"
                entry: dict[str, Any] = {
                    **stats,
                    "mpairs_s": pairs / (stats["median_s"] * 1e6),
                    "max_abs_diff": max_diff,
                    "memo_entries": memo_used,
                }
                if "mv_cold_s" in row:
                    entry["speedup_vs_mv_cold"] = row["mv_cold_s"] / stats["median_s"]
                    entry["speedup_vs_mv_warm"] = row["mv_warm_s"] / stats["median_s"]
                row["timings"][key] = entry
        if "numpy_sort" in row["timings"]:
            np_best = min(
                v["median_s"]
                for k, v in row["timings"].items()
                if k.startswith("numpy_")
            )
            for k, v in row["timings"].items():
                if not k.startswith("numpy_"):
                    v["speedup_vs_numpy_best"] = np_best / v["median_s"]
        rows.append(row)
    return rows


def run_chemistry_suite(
    backends: list[str], *, repeats: int, warmup: int, quick: bool
) -> list[dict[str, Any]]:
    fcidumps = [
        ("h4_8q", ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP", 10),
        ("lih_8q", ROOT / "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP", 10),
        ("beh2_8q", ROOT / "benchmarks/data/beh2_sto3g_r1.3264.FCIDUMP", 10),
        ("h2o_12q", ROOT / "benchmarks/data/h2o_sto3g_cas8e6o.FCIDUMP", 6 if quick else 10),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, max_gens in fcidumps:
        if not path.is_file():
            continue
        model = fcidump_model(path)
        h_mv = model.hamiltonian.to_mv() if hasattr(model.hamiltonian, "to_mv") else model.hamiltonian
        h_soa = PauliSoA.from_mv(h_mv)
        pairs = h_soa.nnz() ** 2

        row: dict[str, Any] = {
            "system": name,
            "n": model.n,
            "h_nnz": h_soa.nnz(),
            "h2_pairs": pairs,
            "h2_timings": {},
            "bank_timings": {},
        }

        # 1. H * H (second moment)
        clear_kernel_caches()
        t0 = time.perf_counter()
        h2_mv = h_mv * h_mv
        t_mv_cold = time.perf_counter() - t0
        memo_cold = kernel_cache_info()["word_products"]["currsize"]
        t0 = time.perf_counter()
        _ = h_mv * h_mv
        t_mv_warm = time.perf_counter() - t0
        h2_ref = PauliSoA.from_mv(h2_mv)
        row["h2_out_words"] = h2_ref.nnz()
        row["h2_mv_cold_s"] = t_mv_cold
        row["h2_mv_warm_s"] = t_mv_warm
        row["h2_mv_memo_entries"] = memo_cold

        for backend in backends:
            prod, stats = _time_fn(
                lambda b=backend: soa_mul(h_soa, h_soa, accumulate="auto", backend=b),
                repeats=repeats,
                warmup=warmup,
                backend=backend,
            )
            max_diff = _max_soa_diff(prod, h2_ref)
            assert max_diff < 1e-8, f"H^2 parity failed on {name} {backend}: {max_diff}"
            row["h2_timings"][backend] = {
                **stats,
                "mpairs_s": pairs / (stats["median_s"] * 1e6),
                "speedup_vs_mv_cold": t_mv_cold / stats["median_s"],
                "speedup_vs_mv_warm": t_mv_warm / stats["median_s"],
                "max_abs_diff": max_diff,
            }

        # 2. Spin-adapted bank build + exact determinant evaluation
        n_spatial = model.n // 2
        occ_spin = tuple(model.metadata["reference_occupied_spin_orbitals"])
        hf_bits = "".join("1" if q in occ_spin else "0" for q in range(model.n))
        occ_spatial = closed_shell_occupied_spatial(
            occ_spin,
            n_spatial,
            spin_ordering="interleaved",
        )
        pool = spin_adapted_excitations(n_spatial, occ_spatial)
        gens = [identity_generator(model.n), *pool[: max_gens - 1]]
        det_ref = DeterminantReference(hf_bits)

        clear_kernel_caches()
        t0 = time.perf_counter()
        legacy = MatrixElementBank(det_ref, h_mv, gens)
        S_ref, H_ref = legacy.matrices()
        t_legacy_bank = time.perf_counter() - t0
        legacy_memo = kernel_cache_info()["word_products"]["currsize"]
        legacy_words = len(legacy.word_set())

        row["bank_m"] = len(gens)
        row["bank_words"] = legacy_words
        row["bank_legacy_s"] = t_legacy_bank
        row["bank_legacy_memo_entries"] = legacy_memo

        for backend in backends:
            def _run_bank(b=backend):
                bk = build_soa_bank(h_soa, gens, backend=b)
                return bk, bk.evaluate_determinant(det_ref)

            (bk, (S_soa, H_soa)), stats = _time_fn(
                _run_bank,
                repeats=max(1, repeats // 2),
                warmup=1 if warmup > 0 else 0,
                backend=backend,
            )
            assert bk.word_universe_size == legacy_words
            s_err = float(np.max(np.abs(S_soa - S_ref)))
            h_err = float(np.max(np.abs(H_soa - H_ref)))
            assert s_err < 1e-10 and h_err < 1e-10
            row["bank_timings"][backend] = {
                **stats,
                "speedup_vs_legacy": t_legacy_bank / stats["median_s"],
                "total_coefficients": bk.total_coefficients,
                "soa_bytes": bk.nbytes(),
                "bytes_per_coeff": bk.bytes_per_coefficient(),
                "max_S_diff": s_err,
                "max_H_diff": h_err,
            }
        rows.append(row)
    return rows


def run_commutator_suite(
    backends: list[str], *, repeats: int, warmup: int, quick: bool
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    cases: list[tuple[str, PauliSoA, PauliSoA]] = []
    for name, path in [
        ("h4_8q", ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"),
        ("h2o_12q", ROOT / "benchmarks/data/h2o_sto3g_cas8e6o.FCIDUMP"),
    ]:
        if path.is_file():
            model = fcidump_model(path)
            n_spatial = model.n // 2
            occ_spin = tuple(model.metadata["reference_occupied_spin_orbitals"])
            occ = closed_shell_occupied_spatial(
                occ_spin,
                n_spatial,
                spin_ordering="interleaved",
            )
            pool = spin_adapted_excitations(n_spatial, occ)
            # Combine first 8 pool operators into one multi-term operator to stress [H, A].
            combo = pool[0].mv
            for g in pool[1 : min(8, len(pool))]:
                combo = combo + g.mv
            cases.append((f"{name}_H_vs_pool8", PauliSoA.from_mv(model.hamiltonian), PauliSoA.from_mv(combo)))

    for n, t in ([(12, 800)] if quick else [(12, 1500), (16, 1500)]):
        sa, _ = _random_soa_and_mv(rng, n, t, need_mv=False)
        sb, _ = _random_soa_and_mv(rng, n, t, need_mv=False)
        cases.append((f"rand_n{n}_t{t}", sa, sb))

    rows: list[dict[str, Any]] = []
    for label, sa, sb in cases:
        pairs = sa.nnz() * sb.nnz()
        ac_frac = float(np.mean(soa_anticommute_mask(sa.n, sa.codes, sb.codes))) if pairs <= 4_000_000 else None
        row: dict[str, Any] = {
            "case": label,
            "n": sa.n,
            "na": sa.nnz(),
            "nb": sb.nnz(),
            "pairs": pairs,
            "anticommute_fraction": ac_frac,
            "timings": {},
        }
        ref: PauliSoA | None = None
        for backend in backends:
            comm, c_stats = _time_fn(
                lambda b=backend: soa_commutator(sa, sb, backend=b),
                repeats=repeats,
                warmup=warmup,
                backend=backend,
            )
            naive, n_stats = _time_fn(
                lambda b=backend: soa_mul(sa, sb, backend=b) - soa_mul(sb, sa, backend=b),
                repeats=repeats,
                warmup=warmup,
                backend=backend,
            )
            if ref is None:
                ref = comm
            assert _max_soa_diff(comm, naive) < 1e-8
            assert _max_soa_diff(comm, ref) < 1e-8
            row["timings"][backend] = {
                "commutator_median_s": c_stats["median_s"],
                "naive_ab_minus_ba_median_s": n_stats["median_s"],
                "prefilter_speedup": n_stats["median_s"] / c_stats["median_s"],
                "out_words": comm.nnz(),
            }
        rows.append(row)
    return rows


def run_multiword_suite(
    backends: list[str], *, repeats: int, warmup: int, quick: bool
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(99)
    cases = [(40, 400), (64, 400)] if quick else [(40, 800), (64, 800), (64, 1500)]
    rows: list[dict[str, Any]] = []
    for n, t in cases:
        sa, mv_a = _random_soa_and_mv(rng, n, t, need_mv=(t <= 400))
        sb, mv_b = _random_soa_and_mv(rng, n, t, need_mv=(t <= 400))
        pairs = t * t
        row: dict[str, Any] = {"n": n, "t": t, "pairs": pairs, "timings": {}}

        if mv_a is not None and mv_b is not None:
            clear_kernel_caches()
            t0 = time.perf_counter()
            mv_prod = mv_a * mv_b
            t_mv = time.perf_counter() - t0
            row["mv_cold_s"] = t_mv
            ref = PauliSoA.from_mv(mv_prod)
            # Also benchmark sorted intersection vs dict trace_pairing.
            _, tp_soa_stats = _time_fn(lambda: soa_trace_pairing(sa, sb), repeats=repeats, warmup=warmup)
            _, tp_mv_stats = _time_fn(lambda: mv_a.trace_pairing(mv_b), repeats=repeats, warmup=warmup)
            assert abs(soa_trace_pairing(sa, sb) - mv_a.trace_pairing(mv_b)) < 1e-10
            row["trace_pairing_soa_us"] = tp_soa_stats["median_s"] * 1e6
            row["trace_pairing_mv_us"] = tp_mv_stats["median_s"] * 1e6
        else:
            ref = None

        for backend in backends:
            if backend == "torch":
                continue  # Multiword GPU path is implemented on CuPy; NumPy covers CPU.
            prod, stats = _time_fn(
                lambda b=backend: soa_mul(sa, sb, backend=b),
                repeats=repeats,
                warmup=warmup,
                backend=backend,
            )
            if ref is None:
                ref = prod
            assert _max_soa_diff(prod, ref) < 1e-8
            entry: dict[str, Any] = {
                **stats,
                "mpairs_s": pairs / (stats["median_s"] * 1e6),
                "out_words": prod.nnz(),
            }
            if "mv_cold_s" in row:
                entry["speedup_vs_mv_cold"] = row["mv_cold_s"] / stats["median_s"]
            row["timings"][backend] = entry
        rows.append(row)
    return rows


def _print_summary(report: dict[str, Any]) -> None:
    env = report["environment"]
    print(
        f"Environment: Python {env['python']} | NumPy {env['numpy']} "
        f"(bitwise_count={env['numpy_bitwise_count']}) | CPU={env['processor']} ({env['cpu_count']} cores) | "
        f"GPUs={list(env['gpu_backends'].keys())}"
    )
    for name, info in env["gpu_backends"].items():
        print(f"  GPU [{name}]: {info['device_name']} (v{info['version']})")

    if "random" in report:
        print("\n=== Random Outer-Product Sweep ===")
        for r in report["random"]:
            mv_s = (
                f"MV cold={r['mv_cold_s']:.3f}s warm={r['mv_warm_s']:.3f}s (memo={r['mv_memo_entries']:,})"
                if "mv_cold_s" in r
                else "MV skipped"
            )
            parts = [f"n={r['n']:>2} t={r['t']:>5} ({r['pairs']:>11,} pairs, out={r['out_words']:,}) | {mv_s}"]
            for k, v in r["timings"].items():
                sp = f" [{v['speedup_vs_mv_cold']:.1f}x vs MV]" if "speedup_vs_mv_cold" in v else ""
                gsp = f" [{v['speedup_vs_numpy_best']:.1f}x vs NumPy]" if "speedup_vs_numpy_best" in v else ""
                parts.append(f"  {k:<12}: {v['median_s']:.4f}s ({v['mpairs_s']:>7.1f} Mpairs/s){sp}{gsp}")
            print("\n".join(parts))

    if "chemistry" in report:
        print("\n=== Chemistry Hamiltonians (H^2 & Global SoA Bank) ===")
        for r in report["chemistry"]:
            print(
                f"{r['system']} (n={r['n']}, nnz(H)={r['h_nnz']:,}, H^2={r['h2_out_words']:,} words): "
                f"MV cold={r['h2_mv_cold_s']:.3f}s warm={r['h2_mv_warm_s']:.3f}s | "
                f"Bank(m={r['bank_m']}, W={r['bank_words']:,}): legacy={r['bank_legacy_s']:.3f}s"
            )
            for b, v in r["h2_timings"].items():
                print(
                    f"  H^2 [{b:<5}]: {v['median_s']:.4f}s ({v['mpairs_s']:.1f} Mpairs/s, "
                    f"{v['speedup_vs_mv_cold']:.1f}x cold / {v['speedup_vs_mv_warm']:.1f}x warm)"
                )
            for b, v in r["bank_timings"].items():
                print(
                    f"  Bank[{b:<5}]: {v['median_s']:.4f}s ({v['speedup_vs_legacy']:.1f}x vs legacy, "
                    f"{v['bytes_per_coeff']:.1f} B/coeff, {v['total_coefficients']:,} coeffs)"
                )

    if "commutator" in report:
        print("\n=== Commutator Pre-Filter ([A, B] vs AB - BA) ===")
        for r in report["commutator"]:
            frac = f"{r['anticommute_fraction']:.3f}" if r["anticommute_fraction"] is not None else "n/a"
            print(f"{r['case']} (n={r['n']}, pairs={r['pairs']:,}, anticomm_frac={frac}):")
            for b, v in r["timings"].items():
                print(
                    f"  [{b:<5}]: comm={v['commutator_median_s']:.4f}s vs naive={v['naive_ab_minus_ba_median_s']:.4f}s "
                    f"({v['prefilter_speedup']:.2f}x faster, out={v['out_words']:,})"
                )

    if "multiword" in report:
        print("\n=== Multiword (>32q) Scaling ===")
        for r in report["multiword"]:
            mv_str = f"MV cold={r['mv_cold_s']:.3f}s" if "mv_cold_s" in r else "MV skipped"
            print(f"n={r['n']} t={r['t']} ({r['pairs']:,} pairs) | {mv_str}")
            for b, v in r["timings"].items():
                sp = f" ({v['speedup_vs_mv_cold']:.1f}x vs MV)" if "speedup_vs_mv_cold" in v else ""
                print(f"  [{b:<5}]: {v['median_s']:.4f}s ({v['mpairs_s']:.1f} Mpairs/s){sp}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "random", "chemistry", "commutator", "multiword"),
        default="all",
    )
    parser.add_argument(
        "--backends",
        default="auto",
        help="Comma-separated backends ('numpy,cupy,torch') or 'auto' / 'all'.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gpu-sweep", action="store_true", help="Include 4M–100M pair 16q sweep.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    gpus = available_gpu_backends()
    if args.backends in ("auto", "all"):
        backends = ["numpy", *sorted(gpus.keys())]
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    report: dict[str, Any] = {
        "schema": "clifford_qc.soa_kernel_benchmark.v1",
        "environment": _env_meta(),
        "backends_run": backends,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "quick": args.quick,
        "gpu_sweep": args.gpu_sweep,
    }

    if args.suite in ("all", "random"):
        report["random"] = run_random_suite(
            backends, repeats=args.repeats, warmup=args.warmup, quick=args.quick, gpu_sweep=args.gpu_sweep
        )
    if args.suite in ("all", "chemistry"):
        report["chemistry"] = run_chemistry_suite(
            backends, repeats=args.repeats, warmup=args.warmup, quick=args.quick
        )
    if args.suite in ("all", "commutator"):
        report["commutator"] = run_commutator_suite(
            backends, repeats=args.repeats, warmup=args.warmup, quick=args.quick
        )
    if args.suite in ("all", "multiword"):
        report["multiword"] = run_multiword_suite(
            backends, repeats=args.repeats, warmup=args.warmup, quick=args.quick
        )

    _print_summary(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote benchmark JSON to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
