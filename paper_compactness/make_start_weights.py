"""Capture the start-span ground-state weights the compactness paper quotes.

The budget-convergence record holds every trajectory but not the weight of the
reference ground space in the shared three-column start span: that diagnostic
was added to the independent checker after observation
(``check_packet_budget_convergence.py --start-weights``) and needs each sector
Hamiltonian rebuilt, which takes seconds and hundreds of megabytes on the
4,900-determinant inputs.  Rebuilding it on every manuscript check would make
the paper gate as heavy as the benchmark replay, so it is computed here, once,
by the checker's own functions, and committed as ``data/start_weights.json``.

The file binds the Git blob of the record it was computed from.  The table
generator refuses to read it once that record changes, so a stale diagnostic
cannot reach the paper.

    python paper_compactness/make_start_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = HERE / "data" / "start_weights.json"
SCHEMA = "clifford_qc.paper_compactness.start_weights.v1"


def main() -> Path:
    from benchmarks.check_packet_budget_convergence import (
        independent_data, independent_reference, start_ground_weight)
    from benchmarks.metadata_io import read_json
    from benchmarks.run_packet_budget_convergence import RECORD
    from make_tables import BUDGET, git_blob_sha

    record = read_json(ROOT / RECORD)
    config = record["config"]
    weights = {}
    for case in config["inputs"]:
        data = independent_data(case, config)
        reference = independent_reference(case, config, data)
        weights[case["name"]] = {
            "start_columns": len(data["initial"]),
            "ground_space_weight": start_ground_weight(data, reference),
        }
    payload = {
        "schema": SCHEMA,
        "record": str(BUDGET.relative_to(ROOT)),
        "record_git_blob_sha": git_blob_sha(BUDGET),
        "producer": "benchmarks/check_packet_budget_convergence.py "
                    "(independent_data, independent_reference, start_ground_weight)",
        "weights": weights,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    print(main())
