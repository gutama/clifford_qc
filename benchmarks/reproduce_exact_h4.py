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
    args = parser.parse_args(argv)

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
