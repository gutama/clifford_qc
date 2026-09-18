"""Measure and certify the Krylov arm's Pauli-word universe.

For ``A_k = H^k`` with Hermitian ``H``,

    A_i^dag A_j   = H^(i+j),
    A_i^dag H A_j = H^(i+j+1),

so the matrix-element word universe is the union of the ``2m+2`` powers
``H^0, ..., H^(2m+1)``.  This avoids the quadratic element-operator route.

Floating-point multiplication leaves round-off-level Pauli coefficients in
high powers.  A coefficient threshold is therefore an approximation, not an
exact support identity.  This script never feeds a thresholded power into the
next multiplication: every cutoff is applied to the same unpruned powers, so
the support sweep is nested.  More importantly, it rebuilds the full Hankel
overlap and Hamiltonian pencils from the thresholded moments and records the
resulting matrix, energy, rank, and conditioning errors.  A word count is
reportable only when the predeclared cutoff passes that certificate.

    python benchmarks/run_krylov_width.py
    python benchmarks/run_krylov_width.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend
from clifford_qc.multivector import MV
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.solver import solve_projected

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "reference_results" / "krylov_width.json"
CONFIG = ROOT / "configs" / "acase_ladder.json"

KRYLOV_ORDER = 8
THRESHOLDS = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
REPORTED_THRESHOLD = 1e-8

# A reported resource count must describe a pencil numerically equivalent to
# the unpruned double-precision construction.  These are validation tolerances,
# not claims about physical or chemical accuracy.
MAX_NORMALIZED_PENCIL_ERROR = 1e-10
MAX_ENERGY_ERROR = 1e-9
MAX_CONDITION_RELATIVE_ERROR = 5e-2


def _unpruned_powers(hamiltonian, order: int) -> list[MV]:
    """Return H^0 ... H^(2*order+1), without recursive thresholding."""
    H = hamiltonian if isinstance(hamiltonian, MV) else hamiltonian.to_mv()
    powers = [MV.scalar(H.n, 1.0)]
    for _ in range(1, 2 * order + 2):
        powers.append(powers[-1] * H)
    return powers


def _truncate(power: MV, tol: float) -> MV:
    if tol <= 0.0:
        return power
    return MV(power.n, {
        code: value for code, value in power.terms.items()
        if abs(value) > tol
    })


def _pencil(powers: list[MV], rho: MV, order: int,
            tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the Krylov Hankel pencil from consistently truncated moments."""
    scale = float(2 ** rho.n)
    moments = [
        float((scale * _truncate(power, tol).trace_pairing(rho)).real)
        for power in powers
    ]
    size = order + 1
    overlap = np.empty((size, size), dtype=float)
    projected_h = np.empty((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            overlap[i, j] = moments[i + j]
            projected_h[i, j] = moments[i + j + 1]
    return overlap, projected_h


def _normalized_matrix_error(approx: np.ndarray, exact: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(exact))), 1.0)
    return float(np.max(np.abs(approx - exact)) / scale)


def krylov_word_diagnostics(hamiltonian, rho: MV, order: int,
                            thresholds=THRESHOLDS) -> dict:
    """Support and pencil errors for cuts applied to one unpruned power chain."""
    powers = _unpruned_powers(hamiltonian, order)
    exact_s, exact_h = _pencil(powers, rho, order, 0.0)
    exact_result = solve_projected(exact_s, exact_h)

    by_threshold: dict[str, dict] = {}
    previous_words: int | None = None
    for tol in thresholds:
        truncated = [_truncate(power, tol) for power in powers]
        words = len(set().union(*(set(power.terms) for power in truncated)))
        if previous_words is not None and words > previous_words:
            raise RuntimeError(
                f"non-nested support sweep: threshold {tol:g} has {words} "
                f"words after {previous_words}")
        previous_words = words

        overlap, projected_h = _pencil(powers, rho, order, tol)
        result = solve_projected(overlap, projected_h)
        condition_scale = max(abs(exact_result.condition_number), 1.0)
        row = {
            "word_universe": words,
            "overlap_normalized_max_error":
                _normalized_matrix_error(overlap, exact_s),
            "hamiltonian_normalized_max_error":
                _normalized_matrix_error(projected_h, exact_h),
            "ground_energy": result.ground_energy,
            "ground_energy_absolute_error":
                abs(result.ground_energy - exact_result.ground_energy),
            "condition_number": result.condition_number,
            "condition_number_relative_error":
                abs(result.condition_number - exact_result.condition_number)
                / condition_scale,
            "effective_rank": result.effective_rank,
            "rank_matches_unpruned":
                result.effective_rank == exact_result.effective_rank,
        }
        row["certificate_passed"] = bool(
            row["overlap_normalized_max_error"]
            <= MAX_NORMALIZED_PENCIL_ERROR
            and row["hamiltonian_normalized_max_error"]
            <= MAX_NORMALIZED_PENCIL_ERROR
            and row["ground_energy_absolute_error"] <= MAX_ENERGY_ERROR
            and row["condition_number_relative_error"]
            <= MAX_CONDITION_RELATIVE_ERROR
            and row["rank_matches_unpruned"]
        )
        by_threshold[repr(tol)] = row

    return {
        "unpruned": {
            "ground_energy": exact_result.ground_energy,
            "condition_number": exact_result.condition_number,
            "effective_rank": exact_result.effective_rank,
        },
        "by_threshold": by_threshold,
    }


def krylov_word_universe(hamiltonian, order: int,
                         thresholds=THRESHOLDS) -> dict[str, int]:
    """Compatibility helper used by the direct-enumeration tests."""
    powers = _unpruned_powers(hamiltonian, order)
    return {
        repr(tol): len(set().union(*(
            set(_truncate(power, tol).terms) for power in powers)))
        for tol in thresholds
    }


def systems():
    """Every validation-ladder rung whose Krylov arm was configured."""
    import run_acase_ladder as ladder

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    order = int(config["methods"]["krylov"].get("size", KRYLOV_ORDER))
    for rung in config["ladder"]:
        if "krylov" not in rung["methods"]:
            continue
        try:
            model = ladder.build_system(rung["system"])
        except ImportError as exc:
            yield rung["name"], kind_of(rung), None, order, str(exc)
            continue
        yield rung["name"], model.metadata["kind"], model, order, None


def kind_of(rung: dict) -> str:
    return str(rung["system"].get("type", "unknown"))


def build_record() -> dict:
    rows = []
    skipped = []
    for name, kind, model, order, error in systems():
        if model is None:
            skipped.append({"rung": name, "kind": kind, "reason": error})
            print(f"{name}: SKIPPED ({error})", flush=True)
            continue
        started = time.perf_counter()
        rho = ExactMVBackend().state(model.reference, ())
        diagnostics = krylov_word_diagnostics(
            model.hamiltonian, rho, order)
        reported = diagnostics["by_threshold"][repr(REPORTED_THRESHOLD)]
        if not reported["certificate_passed"]:
            raise RuntimeError(
                f"{model.name}: the {REPORTED_THRESHOLD:g} Krylov width "
                "does not reproduce the unpruned pencil")
        rows.append({
            "system": model.name,
            "rung": name,
            "kind": kind,
            "n": model.n,
            "hamiltonian_terms": len(model.hamiltonian.terms),
            "krylov_order": order,
            "basis_size": order + 1,
            "word_universe_by_threshold": {
                key: value["word_universe"]
                for key, value in diagnostics["by_threshold"].items()
            },
            "pencil_diagnostics_by_threshold":
                diagnostics["by_threshold"],
            "unpruned_pencil": diagnostics["unpruned"],
            "word_universe": reported["word_universe"],
            "word_universe_certificate_passed": True,
        })
        raw = diagnostics["by_threshold"][repr(0.0)]["word_universe"]
        print(f"{model.name}: W={reported['word_universe']} "
              f"(raw {raw}, dE={reported['ground_energy_absolute_error']:.2e}) "
              f"[{time.perf_counter() - started:.1f}s]", flush=True)
    return {
        "schema": "clifford_qc.krylov_width.v2",
        "method": "union of supp(H^k), k = 0 .. 2*order+1",
        "reported_threshold": REPORTED_THRESHOLD,
        "certificate": {
            "cuts_are_nonrecursive": True,
            "max_normalized_pencil_error": MAX_NORMALIZED_PENCIL_ERROR,
            "max_energy_error": MAX_ENERGY_ERROR,
            "max_condition_relative_error":
                MAX_CONDITION_RELATIVE_ERROR,
            "same_effective_rank_required": True,
        },
        "threshold_note": (
            "Every cutoff is applied independently to the same unpruned "
            "double-precision powers. The reported count is accepted only if "
            "the resulting Hankel overlap and Hamiltonian pencil reproduces "
            "the unpruned effective rank, ground energy, conditioning, and "
            "normalized matrix entries within the recorded tolerances."),
        "rows": rows,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    record = stamp_record(build_record())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
