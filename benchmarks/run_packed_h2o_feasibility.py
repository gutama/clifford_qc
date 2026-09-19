"""Phase 2M-D -- the stretched-H2O ``M = 31`` configuration under a 15 GiB ceiling.

Phase 2M's go/no-go names one end-to-end test: "the previously failing
stretched-H2O ``M = 31`` configuration completing below 15 GiB without shrinking
its candidate pool, word universe, or basis budget". ``molecular.simulation``
carries the failure in a comment beside the configuration that caused it --
``max_subspace`` is pinned at 20 because 30 "was killed by the OOM reaper" and
"M=31 needs ~18 GiB on a 15 GiB machine". This is that configuration, run.

**Both backends, because one of them is the control.** The object backend is
expected to cross the ceiling; that is what reproduces the recorded failure and
is what makes the packed arm's result mean something. A packed arm that finished
under 15 GiB with no failing control would only show that *this* machine is
bigger than the one that died.

**The ceiling is enforced here rather than by the OOM killer.** A run the kernel
kills reports nothing: not the peak it reached, not how far it got, and not
whether it died at 15 GiB or at whatever the host happened to have. A sampling
thread watches ``/proc/self/statm`` and interrupts the main thread the moment
resident memory crosses the declared ceiling, so an arm that fails says where it
failed. That also keeps the box alive to run the second arm.

**Nothing is shrunk.** The FCIDUMP is the one the failing run wrote, checked by
digest. The candidate pool is the pipeline's own ``determinant_excitations`` at
rank 2 with no cap -- 140 candidates. The basis budget is ``max_size = 30``,
which is ``M = 31`` once the identity is counted. The exact ground energy that
feeds the oracle stop is read from the committed record rather than recomputed,
because it is an input to the stopping rule and not a result this test produces.

    python benchmarks/run_packed_h2o_feasibility.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import threading
import time
import _thread
from pathlib import Path

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    MatrixElementBank,
    determinant_excitations,
    identity_generator,
    occupied_spin_orbitals,
    run_acase,
)
from clifford_qc.subspace.projection import STORAGE_BACKENDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = HERE / "reference_results" / "packed_h2o_feasibility.json"
SCHEMA = "clifford_qc.packed_h2o_feasibility.v1"

# Phase 2M's declared end-to-end ceiling, in bytes. Fifteen gibibytes, which is
# the machine size PLAN.md records the original run being killed on.
CEILING_BYTES = 15 * 2 ** 30

# The failing configuration, verbatim from ``molecular.catalog.MOLECULES``
# and its ``run_pipeline`` defaults. Restated here rather than imported because
# that module needs PySCF at import time to rebuild integrals this test reads
# from the FCIDUMP the failing run already wrote.
SOURCE = ROOT / "molecular/results" / "h2o_stretched.fcidump"
SOURCE_SHA256 = "7defd58c24020a1ffbd9ebe3d08c15e32b5636376a676df01451f94b1d2afa36"
COMMITTED_RECORD = ROOT / "molecular/results" / "h2o_stretched_results.json"
MAX_SIZE = 30           # M = 31 once the identity generator is counted
MAX_RANK = 2
LEAKAGE_TOL = 1e-10
TARGET_ERROR_MHA = 1.5936

# How often the watchdog reads resident memory. The bank grows by one
# element-operator row at a time and a row is megabytes, so a quarter second
# cannot miss the crossing by more than a rounding error -- and the headroom
# between the ceiling and this container's total is far larger than that.
SAMPLE_SECONDS = 0.25


class CeilingExceeded(RuntimeError):
    """Raised in the main thread when resident memory crosses the ceiling."""


def _rss() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class ResidentCeiling:
    """Sample resident memory, and interrupt the main thread if it crosses.

    ``PeakRSS`` in ``molecular.simulation`` samples for reporting; this also
    enforces. The distinction matters because the quantity under test *is* a
    ceiling: a run the kernel kills reports nothing, and one that merely records
    its peak after finishing cannot tell you that it would have finished on a
    smaller machine.

    ``_thread.interrupt_main`` is the mechanism because ``run_acase`` has no
    progress callback to poll. It raises ``KeyboardInterrupt`` in the main
    thread at the next bytecode boundary, which the caller converts into
    :class:`CeilingExceeded`.
    """

    def __init__(self, ceiling: int, interval: float = SAMPLE_SECONDS) -> None:
        self.ceiling = ceiling
        self.interval = interval
        self.baseline = 0
        self.peak = 0
        self.crossed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                resident = _rss()
            except (OSError, ValueError, IndexError):
                return
            self.peak = max(self.peak, resident)
            if resident >= self.ceiling and not self.crossed:
                self.crossed = True
                _thread.interrupt_main()
                return

    def __enter__(self) -> "ResidentCeiling":
        self.baseline = self.peak = _rss()
        if self.baseline >= self.ceiling:
            raise RuntimeError(
                f"resident memory is already {self.baseline} bytes, at or above "
                f"the {self.ceiling}-byte ceiling before the arm starts")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, _rss())
        except (OSError, ValueError, IndexError):
            pass

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)


def _frozen_inputs() -> dict:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{SOURCE.name} has changed ({digest[:12]}... is not the frozen "
            f"{SOURCE_SHA256[:12]}...). This test is only the *previously "
            "failing* configuration if it reads the integrals that run wrote")
    committed = json.loads(COMMITTED_RECORD.read_text(encoding="utf-8"))
    return {
        "fcidump": SOURCE.relative_to(ROOT).as_posix(),
        "fcidump_sha256": digest,
        "exact_ground_energy": committed["e_exact_fci"],
        "exact_ground_energy_source": COMMITTED_RECORD.name,
        "committed_basis_size": committed["subspace_size_m"],
        "committed_peak_rss_bytes": committed["adaptive_peak_rss_bytes"],
        "max_size": MAX_SIZE,
        "target_basis_size": MAX_SIZE + 1,
        "max_rank": MAX_RANK,
        "candidate_pool_cap": None,
        "leakage_tol": LEAKAGE_TOL,
        "target_error_mha": TARGET_ERROR_MHA,
        "statement": (
            "The candidate pool, word universe and basis budget are the "
            "pipeline's own: determinant excitations at rank 2 with no cap, the "
            "full Hamiltonian, and max_size 30 for M = 31. The exact ground "
            "energy feeds the oracle stop and is read from the committed record "
            "rather than recomputed, because it is an input to the stopping rule "
            "and not a result this test produces."),
    }


def _arm(backend: str, inputs: dict, ceiling: int) -> dict:
    """One storage backend, run to the budget or to the ceiling."""
    gc.collect()
    model = fcidump_model(SOURCE)
    occupied = occupied_spin_orbitals(model)
    candidates = determinant_excitations(model.n, occupied, max_rank=MAX_RANK)
    rho0 = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho0, model.hamiltonian, (), storage=backend)
    row = {
        "storage_backend": backend,
        "n_qubits": model.n,
        "candidate_pool_size": len(candidates),
        "hamiltonian_pauli_terms": len(model.hamiltonian.to_mv().terms),
    }
    started = time.perf_counter()
    result = None
    try:
        with ResidentCeiling(ceiling) as watch:
            result = run_acase(
                rho0, model.hamiltonian, candidates,
                initial=[identity_generator(model.n)],
                bank=bank,
                max_size=MAX_SIZE,
                leakage_tol=LEAKAGE_TOL,
                exact_ground_energy=inputs["exact_ground_energy"],
                target_error=TARGET_ERROR_MHA / 1000.0,
            )
    except KeyboardInterrupt:
        # Raised by the watchdog, not by a person: the arm crossed the ceiling.
        result = None
    seconds = time.perf_counter() - started

    row.update({
        "completed": result is not None,
        "crossed_ceiling": watch.crossed,
        "peak_rss_bytes": watch.peak,
        "peak_rss_delta_bytes": watch.delta,
        "baseline_rss_bytes": watch.baseline,
        "wall_seconds": seconds,
        "registered_generators": len(bank),
        "pairs_built": bank.resources()["pairs_built"],
    })
    if result is not None:
        resources = bank.resources(result.indices)
        row.update({
            "basis_size": resources["basis_size"],
            "coefficient_occurrences": resources["coefficient_occurrences"],
            "resident_word_universe": resources["resident_word_universe"],
            "selected_word_universe": resources["word_universe"],
            "coefficient_reuse": resources["coefficient_reuse"],
            "ground_energy": float(result.result.ground_energy),
            "stopped_reason": result.stopped_reason,
            "basis_labels": list(result.labels),
        })
    else:
        resources = bank.resources()
        row.update({
            "basis_size": None,
            "coefficient_occurrences": resources["coefficient_occurrences"],
            "resident_word_universe": resources["resident_word_universe"],
            "selected_word_universe": None,
            "coefficient_reuse": resources["coefficient_reuse"],
            "ground_energy": None,
            "stopped_reason": "resident memory crossed the declared ceiling",
            "basis_labels": None,
        })
    del bank, result, model, candidates, rho0
    gc.collect()
    return row


def _verdict(arms: list[dict], inputs: dict, ceiling: int) -> dict:
    by_backend = {arm["storage_backend"]: arm for arm in arms}
    packed = by_backend.get("packed")
    control = by_backend.get("object")
    target = inputs["target_basis_size"]
    packed_ok = bool(packed and packed["completed"]
                     and packed["basis_size"] == target
                     and packed["peak_rss_bytes"] < ceiling)
    control_failed = bool(control and not control["completed"])
    if packed_ok and control_failed:
        outcome = "packed_completes_under_the_ceiling_object_does_not"
    elif packed_ok:
        outcome = "packed_completes_under_the_ceiling_control_also_completed"
    elif packed and packed["crossed_ceiling"]:
        outcome = "packed_crossed_the_ceiling"
    else:
        outcome = "indeterminate"
    return {
        "clause": "stretched_h2o_m31_below_15_gib",
        "ceiling_bytes": ceiling,
        "target_basis_size": target,
        "outcome": outcome,
        "packed_completed_at_target": packed_ok,
        "object_control_failed": control_failed,
        "what_this_does_not_establish": (
            "Not that Phase 2M passes. Its go/no-go has three clauses: this is "
            "the end-to-end feasibility one, 2M-B graded the storage reduction, "
            "and the streaming-policy clause belongs to 2M-C, which does not "
            "exist -- no eviction policy was built or measured, and this run is "
            "entirely retain_all. Nor is a peak measured on one machine a "
            "portable bound: resident memory depends on the allocator and the "
            "interpreter as well as on the representation, and the ceiling here "
            "is enforced rather than predicted."),
    }


def build_record(ceiling: int = CEILING_BYTES,
                 backends: tuple[str, ...] = STORAGE_BACKENDS) -> dict:
    inputs = _frozen_inputs()
    arms = []
    for backend in backends:
        print(f"  [{backend}] starting", flush=True)
        arm = _arm(backend, inputs, ceiling)
        arms.append(arm)
        print(f"  [{backend}] completed={arm['completed']} "
              f"M={arm['basis_size']} peak={arm['peak_rss_bytes'] / 2**30:.2f} GiB "
              f"in {arm['wall_seconds'] / 60:.1f} min", flush=True)
    record = {
        "schema": SCHEMA,
        "phase": "2M-D_stretched_h2o_m31_feasibility",
        "evidence_tier": "structural",
        "ceiling_bytes": ceiling,
        "ceiling_statement": (
            "Enforced by a sampling thread that interrupts the main thread when "
            "resident memory crosses it, not by the OOM killer. A run the kernel "
            "kills reports neither the peak it reached nor how far it got, and "
            "cannot say whether it died at the ceiling or at whatever the host "
            "happened to have."),
        "configuration": inputs,
        "arms": arms,
        "go_no_go": _verdict(arms, inputs, ceiling),
        "claim_boundary": (
            "Structural feasibility evidence on one configuration, one machine "
            "and one interpreter. Nothing is sampled and no resource price is "
            "produced. The peak reported for an arm is that arm's resident set "
            "on this host: resident memory depends on the allocator's behaviour "
            "and the interpreter's object layout as well as on the storage "
            "representation, so it is evidence that the configuration fits under "
            "the declared ceiling here rather than a portable bound. An arm that "
            "crossed the ceiling was stopped at the crossing, so its peak is the "
            "ceiling and not the peak it would have reached; what it reports is "
            "where it stopped, not how much it would have needed. The exact "
            "ground energy feeding the oracle stop is read from the committed "
            "record and is not re-derived here."),
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    parser.add_argument("--ceiling-bytes", type=int, default=CEILING_BYTES)
    parser.add_argument("--backends", default=",".join(STORAGE_BACKENDS),
                        help="comma-separated subset, for a dry run")
    parser.add_argument("--max-size", type=int, default=None,
                        help="override the basis budget; a smaller value does "
                             "NOT run the declared test and the record says so")
    args = parser.parse_args()

    global MAX_SIZE
    if args.max_size is not None:
        MAX_SIZE = args.max_size
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())

    print(f"stretched-H2O M={MAX_SIZE + 1} under a "
          f"{args.ceiling_bytes / 2**30:.0f} GiB ceiling, backends {backends}\n",
          flush=True)
    record = build_record(ceiling=args.ceiling_bytes, backends=backends)
    if MAX_SIZE != 30 or backends != STORAGE_BACKENDS:
        record["declared_test"] = False
        record["declared_test_note"] = (
            f"Run with max_size={MAX_SIZE} and backends {list(backends)}, which "
            "is not Phase 2M's declared configuration (max_size 30, both "
            "backends). This record is a dry run and grades nothing.")
    else:
        record["declared_test"] = True
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    for arm in record["arms"]:
        print(f"{arm['storage_backend']:8s} completed={str(arm['completed']):5s} "
              f"M={arm['basis_size']} peak={arm['peak_rss_bytes'] / 2**30:6.2f} GiB "
              f"pairs={arm['pairs_built']:6d} T={arm['coefficient_occurrences']:11d} "
              f"{arm['wall_seconds'] / 60:6.1f} min")
    print(f"\n{record['go_no_go']['clause']}: {record['go_no_go']['outcome']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
