"""One exact A-CASE storage arm; invoked only in a fresh Python process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import resource
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.bank_policy_contract import (
    ARMS, atomic_json, digest, matrix_record, read_json,
)
from clifford_qc.backends import ExactMVBackend
from clifford_qc.subspace import adaptive
from clifford_qc.subspace.adaptive import ACASEConfig
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.streaming import StreamingMatrixElementBank


def rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def scientific_value(value):
    """Keep IEEE floats exactly, including a distinction between signed zeros."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("non-finite scientific diagnostic")
        return float(value).hex()
    if isinstance(value, dict):
        return {k: scientific_value(v) for k, v in value.items() if k != "step_seconds"}
    if isinstance(value, (tuple, list)):
        return [scientific_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_problem(spec):
    if spec.get("kind") == "toy":
        from clifford_qc.algorithms.pools import local_pool, odd_y_filter
        from clifford_qc.models.spin import tfim
        from clifford_qc.subspace.generators import pauli_orbit
        model = tfim(spec["n_qubits"])
        candidates = pauli_orbit([op.word for op in odd_y_filter(local_pool(model.n))])
    else:
        from clifford_qc.models.fcidump import fcidump_model
        from clifford_qc.subspace.fermionic_generators import (
            determinant_excitations, occupied_spin_orbitals,
        )
        model = fcidump_model(ROOT / spec["source"]["path"])
        candidates = determinant_excitations(
            model.n, occupied_spin_orbitals(model), max_rank=2)
    manifest = {
        "n_qubits": model.n, "candidate_count": len(candidates),
        "candidate_labels_sha256": digest([g.label for g in candidates]),
        "candidate_operators_sha256": digest([
            [[int(code), complex(value).real.hex(), complex(value).imag.hex()]
             for code, value in g.mv.terms.items()] for g in candidates]),
    }
    if spec.get("kind") != "toy" and manifest != spec["manifest"]:
        raise ValueError("full candidate manifest differs")
    return model, candidates, manifest


class ObservationMixin:
    """Observe existing methods; do not alter generators, scoring or lifetime."""

    def __init__(self, *args, **kwargs):
        self.pair_ledger = []
        self.physical_builds = 0
        self._observed_keys = set()
        self._observed_retained = ()
        self.row_observations = []
        self.solves = []
        self.first_build_seconds = self.recompute_seconds = 0.0
        super().__init__(*args, **kwargs)

    def _observe_rows(self):
        rows = self._resident_row_count()
        size = len(self._observed_retained)
        limit = (2 * (size * (size + 1) // 2 + self.frontier_pairs)
                 if isinstance(self, StreamingMatrixElementBank) else None)
        self.row_observations.append([rows, limit])

    def _build_pair(self, i, j, *, evaluate=False):
        key = (i, j)
        new = key not in self._observed_keys
        started = time.perf_counter()
        super()._build_pair(i, j, evaluate=evaluate)
        elapsed = time.perf_counter() - started
        self.physical_builds += 1
        if new:
            self.first_build_seconds += elapsed
            self._observed_keys.add(key)
            self.pair_ledger.append([i, j, self._row_nnz("overlap", key),
                                     self._row_nnz("element", key)])
        else:
            self.recompute_seconds += elapsed
        self._observe_rows()

    def retain_basis(self, indices):
        self._observed_retained = tuple(self.resolve(indices))
        super().retain_basis(indices)
        self._observe_rows()

    def matrices(self, indices=None):
        order = self.resolve(indices)
        s, h = super().matrices(order)
        self.solves.append({"indices": list(order), "S": matrix_record(s), "H": matrix_record(h)})
        return s, h


class ObservedRetainBank(ObservationMixin, MatrixElementBank):
    pass


class ObservedStreamingBank(ObservationMixin, StreamingMatrixElementBank):
    pass


def run_arm(spec, arm, options, frontier_pairs, ready_path=None, sample_seconds=0.05):
    started = time.perf_counter()
    model, candidates, manifest = build_problem(spec)
    rho = ExactMVBackend().state(model.reference, ())
    cfg = dict(options)
    cfg.update(exact_ground_energy=spec.get("exact_ground_energy"))
    bank_class = ObservedRetainBank if arm[2] == "retain_all" else ObservedStreamingBank
    kwargs = {} if arm[2] == "retain_all" else {"frontier_pairs": frontier_pairs}
    bank = bank_class(rho, model.hamiltonian, storage=arm[1], **kwargs)
    baseline = rss_bytes()
    preparation_seconds = time.perf_counter() - started
    stop = threading.Event()
    sample = {"peak": baseline, "observations": 0, "errors": []}
    def observe_rss():
        try:
            while True:
                rss = rss_bytes()
                sample["peak"] = max(sample["peak"], rss)
                sample["observations"] += 1
                if ready_path is not None:
                    atomic_json(ready_path, {"baseline_rss_bytes": baseline, "pid": os.getpid(),
                                           "rss_bytes": rss, "peak_rss_bytes": sample["peak"]})
                if stop.wait(sample_seconds):
                    return
        except Exception as exc:
            sample["errors"].append(str(exc))
    observer = threading.Thread(target=observe_rss, daemon=True)
    observer.start()
    scores = []
    original = adaptive.score_candidate
    def observe_score(*args, **kw):
        score = original(*args, **kw)
        scores.append({"basis": list(args[1]), "candidate": scientific_value(asdict(score))})
        return score
    solve_started = time.perf_counter()
    try:
        with patch.object(adaptive, "score_candidate", observe_score):
            result = adaptive.run_acase(rho, model.hamiltonian, candidates, bank=bank,
                                        config=ACASEConfig(**cfg))
    finally:
        stop.set()
        observer.join()
    if sample["errors"]:
        raise RuntimeError("worker RSS observation failed: " + "; ".join(sample["errors"]))
    solve_seconds = time.perf_counter() - solve_started
    observed_end_rss = rss_bytes()
    resources = bank.resources(result.indices)
    logical = {
        "pair_ledger": bank.pair_ledger,
        "pairs_built": resources["pairs_built"],
        "coefficient_occurrences": sum(row[2] + row[3] for row in bank.pair_ledger),
        "word_universe": len(bank.word_set()),
        "selected_word_universe": resources["word_universe"],
        "selected_word_sha256": digest(sorted(bank.word_set(result.indices))),
        "pair_ownership": [[label, bank._new_words[i], bank._reused_words[i]]
                           for i, label in enumerate(bank.labels)],
        "retained_block_pairs": resources["retained_block_pairs"],
        "selection_pairs": resources["selection_pairs"],
    }
    scientific = {
        "manifest": manifest, "config": cfg, "labels": list(result.labels),
        "indices": list(result.indices), "energy": float(result.energy).hex(),
        "energy_history": scientific_value(result.energy_history),
        "growth": scientific_value([asdict(r) for r in result.records]),
        "scores": scores, "solves": bank.solves, "stopped_reason": result.stopped_reason,
        "logical": logical, "evidence_tier": "exact", "resource_scope": "S/H_bank_only",
    }
    packed_live = packed_reserved = 0
    if arm[1] == "packed":
        packed_live = sum(store.nbytes() for store in bank._packed.values()) + bank._word_table.live_bytes()
        packed_reserved = sum(store.reserved_bytes() for store in bank._packed.values()) + bank._word_table.nbytes()
    resident_coefficients = resources["coefficient_occurrences"]
    costs = {
        "preparation_seconds": preparation_seconds, "solve_seconds": solve_seconds,
        "first_pair_build_seconds": bank.first_build_seconds,
        "recompute_seconds": bank.recompute_seconds,
        "postprocess_seconds": time.perf_counter() - solve_started - solve_seconds,
        "baseline_rss_bytes": baseline, "solve_end_rss_bytes": observed_end_rss,
        "worker_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "worker_sampled_solve_peak_rss_bytes": max(sample["peak"], observed_end_rss),
        "worker_rss_observations": sample["observations"],
        "solve_end_delta_bytes_per_resident_coefficient": (
            max(0, observed_end_rss - baseline) / resident_coefficients
            if resident_coefficients else None),
        "physical_pair_builds": bank.physical_builds,
        "recomputed_rows": resources["recomputed_rows"], "evicted_rows": resources["evicted_rows"],
        "resident_rows": resources["resident_operator_rows"],
        "resident_coefficients": resident_coefficients,
        "resident_word_universe": resources["resident_word_universe"],
        "packed_live_bytes": packed_live, "packed_reserved_bytes": packed_reserved,
        "coefficient_payload_estimate_bytes": 24 * resident_coefficients,
        "spill_bytes": resources["spill_bytes"], "row_observations": bank.row_observations,
        "product_temporary_row_limit": 2, "instrumentation_in_process_costs": True,
    }
    import scipy
    return {"system": spec["id"], "arm": list(arm), "status": "completed",
            "scientific": scientific, "costs": costs,
            "worker": {"pid": os.getpid(), "parent_pid": os.getppid(),
                       "start_ticks": int(Path("/proc/self/stat").read_text().split()[21]),
                       "python": sys.version.split()[0], "numpy": np.__version__,
                       "scipy": scipy.__version__,
                       "threads": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    arm = tuple(request["arm"])
    if arm not in ARMS:
        raise ValueError("unknown storage arm")
    cell = run_arm(request["system"], arm, request["options"], request["frontier_pairs"],
                   args.ready, request["sample_seconds"])
    atomic_json(args.output, cell, exclusive=True)


if __name__ == "__main__":
    main()
