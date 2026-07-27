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


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _pin_threads(threads: int) -> None:
    """Fix the BLAS thread count before NumPy is imported.

    Multithreaded BLAS reductions sum in completion order, so an unpinned run
    perturbs the optimizer at the 1e-11 level. That is far below chemical
    accuracy and does not move the reported energy, but this pool is highly
    degenerate -- at the H4 reference state only 10 distinct gradient
    magnitudes span 160 candidates, the largest tie group holding 72 -- and
    an exact tie is decided by whichever member the perturbed state happens
    to favour. The trajectory's *labels* are therefore reproducible only at a
    fixed thread count, even though its energies are not sensitive to it.

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--trajectory-out",
                        help="optional pretty-printed full trajectory JSON")
    parser.add_argument("--threads", type=int, default=4,
                        help="BLAS thread count, pinned before NumPy loads so "
                             "that tied operators break the same way each run "
                             "(default: 4, matching the committed record)")
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
    row["environment"] = _environment()
    row["reproduction_started_unix"] = started
    row["reproduction_finished_unix"] = time.time()

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
