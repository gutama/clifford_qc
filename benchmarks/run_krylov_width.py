"""Measure the Krylov arm's word universe W, the ladder's one missing column.

The ladder reports W for every A-CASE arm and leaves it blank for Krylov.  That
is not a policy choice: the tracked element-operator route costs
``O(|A_i| |H| |A_j|)`` per pair, and deep Krylov generators carry thousands of
words each, so ``run_acase_ladder.py`` falls back to the untracked contraction
and records the support columns as unavailable.  The consequence is that the
paper's three-way trade -- energy against conditioning against measurable
support -- had no width data for the arm that wins on energy.

The quadratic route is avoidable.  For the Krylov family ``A_0 = I`` and
``A_k = H^k`` with Hermitian ``H``,

    A_i^dag A_j   = H^(i+j),
    A_i^dag H A_j = H^(i+j+1),

so the union in the manuscript's Eq. (9) collapses from ``O(M^2)`` distinct
element operators to the ``2m+2`` powers ``H^0 ... H^(2m+1)``.  That is linear
in the basis size and runs in seconds where the pair enumeration does not
finish.  ``tests/test_krylov_width.py`` checks the identity against direct
enumeration on cases small enough for both.

Round-off matters here in a way it does not for A-CASE.  ``MV.__mul__`` keeps
every product term, and seventeen successive multiplications leave thousands of
words carrying coefficients at the 1e-11 level -- words that exist in the
floating-point object and not in the operator.  The A-CASE universe is
insensitive to this (every one of its 7371 words on H4 survives a 1e-8 cut);
the Krylov universe is not.  Reporting the raw count would therefore overstate
the baseline's width, so the record carries a threshold sweep and the paper
quotes a stated cut rather than the raw number.

    python benchmarks/run_krylov_width.py
    python benchmarks/run_krylov_width.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from clifford_qc.multivector import MV

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "reference_results" / "krylov_width.json"
CONFIG = ROOT / "configs" / "acase_ladder.json"

# the ladder's default Krylov arm is size=8, i.e. I, H, ..., H^8 -> M=9
KRYLOV_ORDER = 8
THRESHOLDS = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
# the cut the manuscript quotes: the largest threshold under which the A-CASE
# universe is unchanged, so it cannot flatter A-CASE
REPORTED_THRESHOLD = 1e-8


def krylov_word_universe(hamiltonian, order: int,
                         thresholds=THRESHOLDS) -> dict[str, int]:
    """|union_k supp(H^k)| for k = 0 .. 2*order+1, at each threshold.

    Pruning is applied to each power before it is accumulated *and* before it
    is multiplied again, so a coefficient that only exists as round-off cannot
    seed further round-off in the next power.
    """
    H = hamiltonian if isinstance(hamiltonian, MV) else hamiltonian.to_mv()
    out: dict[str, int] = {}
    for tol in thresholds:
        words: set[int] = set()
        power = MV.scalar(H.n, 1.0)
        words |= set(power.terms)
        for _ in range(1, 2 * order + 2):
            power = power * H
            if tol > 0.0:
                power = MV(H.n, {code: value for code, value in power.terms.items()
                                 if abs(value) > tol})
            words |= set(power.terms)
        out[repr(tol)] = len(words)
    return out


def systems():
    """Every ladder rung whose Krylov arm was actually run, built the same way.

    The rungs come from the ladder's own config through the ladder's own
    ``build_system``, so the Hamiltonian whose width is measured here is the
    one the CSV row was produced from rather than a lookalike rebuilt from
    remembered parameters.  Rungs needing chemistry extras are skipped with a
    note when those extras are absent; the committed record carries them.
    """
    import run_acase_ladder as ladder

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    order = int(config["methods"]["krylov"].get("size", KRYLOV_ORDER))
    for rung in config["ladder"]:
        if "krylov" not in rung["methods"]:
            continue
        try:
            model, kind = ladder.build_system(rung["system"])
        except ImportError as exc:  # chemistry extras absent
            yield rung["name"], kind_of(rung), None, order, str(exc)
            continue
        yield rung["name"], kind, model, order, None


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
        counts = krylov_word_universe(model.hamiltonian, order)
        rows.append({
            "system": model.name,
            "rung": name,
            "kind": kind,
            "n": model.n,
            "hamiltonian_terms": len(model.hamiltonian.terms),
            "krylov_order": order,
            "basis_size": order + 1,
            "word_universe_by_threshold": counts,
            "word_universe": counts[repr(REPORTED_THRESHOLD)],
        })
        # Timing stays on stdout: CI gates this file bit-for-bit and a
        # duration is the one field two correct runs will disagree on.
        print(f"{model.name}: W={rows[-1]['word_universe']} "
              f"(raw {counts[repr(0.0)]}) "
              f"[{time.perf_counter() - started:.1f}s]", flush=True)
    return {
        "schema": "clifford_qc.krylov_width.v1",
        "method": "union of supp(H^k), k = 0 .. 2*order+1",
        "reported_threshold": REPORTED_THRESHOLD,
        "threshold_note": (
            "Counts at the reported threshold. The raw count includes words "
            "that exist only as accumulated round-off; the A-CASE universe is "
            "unchanged by any threshold up to 1e-8, the Krylov universe is not. "
            "Entries below 1e-8 are not reproducible run to run on the "
            "chemistry rungs: the SCF settles on orbital coefficients differing "
            "in the last bits, and H^17 of a Hamiltonian perturbed at 1e-16 has "
            "a different round-off-level support. Observed on h4_chain(r=0.9): "
            "8184 raw on one run, 8180 on the next, 4224 reported on both. "
            "benchmarks/check_krylov_width.py therefore gates the reported "
            "counts rather than the file bytes."),
        "rows": rows,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    record = build_record()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
