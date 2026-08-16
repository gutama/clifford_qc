"""Recompute and gate the R3 protocol-axis (``mapping x k``) record.

Two independent jobs, as the project's other checkers do. The record is
rebuilt from scratch and compared field by field, and the contracts it must
satisfy are re-derived from the record's own contents rather than trusted:

* the ``k = 1`` column reproduces the setting count R2b froze, per arm -- the
  condition that makes the grid an extension of that record rather than a
  different experiment;
* group counts are non-increasing in ``k``, since dyadic coarsening only ever
  merges blocks and can never force two words apart;
* every rung reads at least the words it assigned, so pooled coverage is never
  below the assigned share;
* ``k = 1`` emits nothing entangling, and only there;
* the ``mapping_spread`` summary compares arms of equal measured width only.

    python benchmarks/check_protocol_axis.py
"""

from __future__ import annotations

import json
import sys

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct script execution
    from benchmarks.run_protocol_axis import (
        REFERENCE,
        SCHEMA,
        build_record,
        frozen_qwc_settings,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_protocol_axis import (
        REFERENCE,
        SCHEMA,
        build_record,
        frozen_qwc_settings,
    )


# ``error_millihartree`` is (subspace energy - exact energy) in millihartree: a
# residue of order 1e-3 mHa left by two energies of order 1e4 mHa, so seven
# significant digits are gone to cancellation before the comparison starts. The
# reproducible scale is set by the energies, not by the residue. This is the
# same tolerance, for the same reason, that check_mapping_axis.py documents.
ENERGY_DIFFERENCE_TOLERANCES = {"error_millihartree": (1e-10, 1e-8)}


def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    tier = record.get("protocol", {}).get("evidence_tier")
    if tier != "structural":
        problems.append(
            f"evidence tier is {tier!r}; this record prices no accuracy-matched "
            "cost and must not claim a stronger tier"
        )
    frozen = frozen_qwc_settings()

    for system in record.get("systems", []):
        key = system["system"]
        for arm in system["arms"]:
            name = arm["mapping"]
            rungs = {row["block_size"]: row for row in arm["rungs"]}
            width = arm["measured_qubits"]

            expected = frozen.get((key, name))
            if expected is not None and rungs[1]["settings"] != expected:
                problems.append(
                    f"{key}/{name}: k=1 gives {rungs[1]['settings']} settings, "
                    f"R2b froze {expected}"
                )

            sizes = sorted(rungs)
            counts = [rungs[k]["settings"] for k in sizes]
            if counts != sorted(counts, reverse=True):
                problems.append(
                    f"{key}/{name}: settings not non-increasing in k: "
                    f"{dict(zip(sizes, counts))}"
                )

            for block_size, row in rungs.items():
                if block_size > width:
                    problems.append(
                        f"{key}/{name}: rung k={block_size} exceeds the arm's "
                        f"{width} measured qubits"
                    )
                two_qubit = row["resource_metrics"]["N_2q"]["sum"]
                if block_size == 1 and two_qubit:
                    problems.append(
                        f"{key}/{name}: k=1 emitted {two_qubit} two-qubit gates; "
                        "a qubit-wise setting is a per-qubit basis rotation"
                    )
                if block_size > 1 and two_qubit == 0 and row["settings"] < rungs[1]["settings"]:
                    problems.append(
                        f"{key}/{name}: k={block_size} merged settings without "
                        "emitting any entangling gate"
                    )
                assigned = 1.0 / row["settings"]
                if row["coverage"]["min_fraction_of_words_read"] < 0:
                    problems.append(f"{key}/{name}: negative coverage at k={block_size}")
                if row["coverage"]["max_fraction_of_words_read"] > 1:
                    problems.append(
                        f"{key}/{name}: coverage above 1 at k={block_size}"
                    )
                if row["coverage"]["mean_fraction_of_words_read"] < assigned * 0.5:
                    problems.append(
                        f"{key}/{name}: mean coverage {row['coverage']} at "
                        f"k={block_size} is below the assigned share"
                    )
                if row["z_only_restrictions_checked"] <= 0:
                    problems.append(
                        f"{key}/{name}: k={block_size} checked no Z-only "
                        "restrictions, so the diagonalizer invariant never ran"
                    )

    for key, widths in record.get("mapping_spread", {}).items():
        for width, payload in widths.items():
            arms = payload["arms"]
            system = next(s for s in record["systems"] if s["system"] == key)
            measured = {
                arm["mapping"]: arm["measured_qubits"] for arm in system["arms"]
            }
            if any(measured[name] != int(width) for name in arms):
                problems.append(
                    f"{key}: spread group {width} mixes arms of different "
                    f"measured widths"
                )
            for block_size, entry in payload["by_block_size"].items():
                counts = entry["settings"]
                if abs(entry["spread"] - max(counts) / min(counts)) > 1e-12:
                    problems.append(
                        f"{key}: spread at k={block_size} is not max/min of "
                        f"{counts}"
                    )
    return problems


def main() -> int:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(
        expected, actual, atol=1e-10, rtol=1e-10,
        key_tolerances=ENERGY_DIFFERENCE_TOLERANCES,
    )
    problems += contract_problems(expected)
    problems += [
        f"rebuilt record: {problem}" for problem in contract_problems(actual)
    ]
    for problem in problems:
        print(f"  {problem}")
    print("protocol axis:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
