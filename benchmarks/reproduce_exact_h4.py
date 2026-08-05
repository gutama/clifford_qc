"""Reproduce only the exact-gradient H4 chemistry baseline.

This isolates the expensive H4 calculation from the full chemistry matrix.
The PySCF memory-probe shim is needed only in restricted containers that do
not expose ``/proc/<pid>/statm``; it reports zero pre-existing process memory
and leaves PySCF's configured memory ceiling unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from clifford_qc.reproducibility import stamp_record


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _pin_threads(threads: int) -> None:
    """Fix the BLAS thread count before NumPy is imported.

    Multithreaded BLAS reductions sum in completion order, which perturbs the
    optimizer at the 1e-11 level. That is far below chemical accuracy and does
    not move the reported energy, but this pool is highly degenerate -- at the
    H4 reference state only 10 distinct gradient magnitudes span 160
    candidates, the largest tie group holding 72 -- so an exact tie is decided
    by whichever member the perturbed state happens to favour.

    Fixing the *count* does not fix the *order*. Two runs of this script at
    ``--threads 4``, same machine, same versions, same seed, still disagreed
    on 9 of 12 selected labels; completion order varies run to run whenever
    more than one thread participates.

    At ``--threads 1`` it does not: two runs agreed on every recorded field
    bitwise -- labels, parameters, per-step energies. That is why 1 is the
    default. It costs nothing here (1997 s against 2052 s over the same pair
    of runs), because this trajectory is dominated by the pure-Python
    multivector kernel rather than by BLAS, so extra threads were adding
    coordination overhead to a calculation that could not use them.

    Note that reproducing *this* script bitwise is not the same as matching
    the committed record, which was produced on four threads and whose labels
    are therefore one draw among the tied representatives. See Sec. V C.

    Must run before the first NumPy import: the BLAS layer reads these once,
    at load time, and ignores later changes.
    """
    for name in _THREAD_VARS:
        os.environ[name] = str(threads)


def _configure_restricted_container() -> None:
    try:
        import pyscf.lib
        import pyscf.lib.misc

        probe = Path(f"/proc/{os.getpid()}/statm")
        if not probe.exists():
            pyscf.lib.current_memory = lambda: (0.0, 0.0)
            pyscf.lib.misc.current_memory = pyscf.lib.current_memory
            print("[setup] PySCF /proc memory probe shim enabled", flush=True)
    except ImportError:
        pass


def _environment() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "openfermion", "openfermionpyscf",
                 "pyscf", "clifford-qc"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["thread_limits"] = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    return versions


def _compare(row: dict, ref: dict, e0: float) -> dict:
    """Agreement between two independent runs of the same trajectory.

    Reports the labels separately from the numbers because they fail
    differently. Under a strict argmax the two runs disagreed on nine of
    twelve labels while agreeing on the energies to 1e-15, since the pool is
    degenerate and a 1e-11 perturbation decides a tied step. With the argmax
    resolved over a tolerance the labels agree too, so
    ``alternate_numerical_tie_representatives`` should now be zero -- it is
    kept, rather than dropped as redundant, precisely so that a regression
    shows up as a nonzero count in the committed record.
    """
    ref_energy = e0 + ref["final_error_mha"] / 1000.0
    traj, ref_traj = row.get("trajectory", []), ref.get("trajectory", [])
    paired = list(zip(traj, ref_traj))

    def max_abs(key):
        vals = [abs((a.get(key) or 0.0) - (b.get(key) or 0.0)) for a, b in paired]
        return max(vals) if vals else 0.0

    labels, ref_labels = row["labels"], ref["labels"]
    matching = sum(1 for a, b in zip(labels, ref_labels) if a == b)
    return {
        "reference_final_energy_ha": ref_energy,
        "reference_final_error_mha": ref["final_error_mha"],
        "reference_wall_seconds": ref["wall_seconds"],
        "final_energy_abs_difference": abs(row["final_energy_ha"] - ref_energy),
        "max_trajectory_energy_abs_difference": max_abs("energy_ha"),
        "max_gradient_magnitude_abs_difference": max_abs("exact_gradient_max"),
        "labels_identical": labels == ref_labels,
        "labels_matching_by_position": matching,
        "alternate_numerical_tie_representatives": len(labels) - matching,
        "operator_count_matches": len(labels) == len(ref_labels),
        "ops_to_chemical_accuracy_matches":
            row["ops_to_accuracy"] == ref["ops_to_accuracy"],
        "optimizer_evaluations_match":
            row["optimizer_evaluations"] == ref["optimizer_evaluations"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--trajectory-out",
                        help="optional pretty-printed full trajectory JSON")
    parser.add_argument("--with-reference", action="store_true",
                        help="run the trajectory a second time in the same "
                             "process and embed the comparison, which is what "
                             "backs the reproducibility statement in the paper "
                             "(doubles the wall time)")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS thread count, pinned before NumPy loads "
                             "(default: 1, which reproduces bitwise and is "
                             "no slower here). Any count above 1 leaves "
                             "reduction order free, so tied operators can "
                             "break differently between runs")
    args = parser.parse_args(argv)

    # Before _configure_restricted_container(), which imports pyscf -> numpy.
    _pin_threads(args.threads)
    _configure_restricted_container()

    # Executed as a file from ``benchmarks/``; import its sibling directly.
    from run_chemistry import run_arm
    from clifford_qc.matrix import exact_ground
    from clifford_qc.models.chemistry import excitation_pool, h4_chain

    started = time.time()
    print("[stage] constructing H4/STO-3G Hamiltonian with PySCF", flush=True)
    model = h4_chain()
    print(
        f"[stage] model ready: n={model.n} HF={model.metadata['hf_energy']:.12f} ",
        flush=True,
    )
    print("[stage] diagonalizing the 8-qubit Hamiltonian", flush=True)
    e0, _ = exact_ground(model.hamiltonian.to_mv())
    pool = excitation_pool(model.n, model.metadata["n_electrons"])
    print(
        f"[stage] exact ADAPT-VQE: E0={e0:.12f}, pool={len(pool)}, max_ops=12",
        flush=True,
    )
    row = run_arm(model, pool, "exact", seed=0, E0=e0)
    # Derived here rather than left to a reader: the manuscript quotes the
    # absolute energy, and n_electrons identifies the active space.
    row["final_energy_ha"] = e0 + row["final_error_mha"] / 1000.0
    row["n_electrons"] = model.metadata["n_electrons"]
    row["environment"] = _environment()
    row["reproduction_started_unix"] = started
    row["reproduction_finished_unix"] = time.time()

    if args.with_reference:
        print("[stage] independent reference run for the comparison block",
              flush=True)
        ref = run_arm(model, pool, "exact", seed=0, E0=e0)
        row["reference_comparison"] = _compare(row, ref, e0)
        c = row["reference_comparison"]
        print(f"[stage] reference: labels identical={c['labels_identical']}, "
              f"final energy differs by {c['final_energy_abs_difference']:.3g} Ha",
              flush=True)

    row = stamp_record(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(out)
    if args.trajectory_out:
        trajectory_out = Path(args.trajectory_out)
        trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        trajectory_tmp = trajectory_out.with_suffix(trajectory_out.suffix + ".tmp")
        trajectory_tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        trajectory_tmp.replace(trajectory_out)
    print(
        f"[done] H4 exact: error={row['final_error_mha']:.6f} mHa, "
        f"ops@accuracy={row['ops_to_accuracy']}, wall={row['wall_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
