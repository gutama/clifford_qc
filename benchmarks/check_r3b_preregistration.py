"""Gate the R3b preregistration before any sampling happens under it.

A preregistration earns its name from commit order, not from the word: the
commit that declares a rule has to precede the commit that first reports a
result under it. This checker is what makes that worth something. It runs
against a config carrying no results, and it fails a config whose declared bank
is not the bank its own selection rule picks.

What it re-derives, in deterministic double-precision arithmetic with nothing
sampled:

* the committed FCIDUMP and its provenance still hash to the declared digests,
  and the provenance still binds the FCIDUMP;
* the lineage digests match the R3S record and config that admitted this bank,
  and the QR3b config whose protocol it inherits;
* the declared labels really are a prefix of QR3b's frozen greedy ordering, so
  this is a shorter draw from a decided ordering rather than a new selection;
* the R3S margin rule, re-run on that ordering, selects exactly this prefix --
  the check that the bank was chosen by the accuracy target and not by its
  measurement cost;
* the bank's bias, its per-arm word universes and settings, and its binding
  word universe reproduce both the declared values and the R3S record;
* the protocol is QR3b's, endpoint for endpoint, with seed roots that cannot
  alias QR3b's streams;
* nothing in the config is a result, a verdict, or a cost.

    python benchmarks/check_r3b_preregistration.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:  # package import in tests versus direct script execution
    from benchmarks.r3_environment_migration import record_successor_problems
    from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from benchmarks.run_priceability_screen import (
        WORD_UNIVERSE_CONVENTION,
        _arm_rows,
        _solved,
    )
    from benchmarks.run_mapping_axis import _build_model, _selected_generators
    from benchmarks.run_qr3b_instance_preflight import CONFIG as QR3B_CONFIG
    from benchmarks.run_priceability_screen import CONFIG as SCREEN_CONFIG
    from benchmarks.run_priceability_screen import REFERENCE as SCREEN_RECORD
except ImportError:  # pragma: no cover - direct script execution
    from r3_environment_migration import record_successor_problems
    from run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from run_priceability_screen import (
        CONFIG as SCREEN_CONFIG,
        REFERENCE as SCREEN_RECORD,
        WORD_UNIVERSE_CONVENTION,
        _arm_rows,
        _solved,
    )
    from run_mapping_axis import _build_model, _selected_generators
    from run_qr3b_instance_preflight import CONFIG as QR3B_CONFIG

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.subspace.elements import MatrixElementBank

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "r3b_margin_stop_probe.json"
CONFIG_SCHEMA = "clifford_qc.r3b_margin_stop_probe_config.v1"

# A preregistration that carries any of these is a record wearing the wrong
# label, which is the one failure that would make the commit-order argument
# meaningless.
RESULT_KEYS = {
    "C_time",
    "cost_bracket",
    "decision",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
    "qr3b_verdict",
    "scoping_probe",
    "shot_to_target",
    "verdict",
}

# ``provenance`` is a record's execution stamp when it is an object and a path
# to a committed provenance file when it is a string. Only the first makes a
# config into a record, and this config legitimately carries the second.
STAMP_KEYS = {"provenance"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_key_paths(node, path: str = "$") -> list[str]:
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            if key in RESULT_KEYS or (key in STAMP_KEYS and isinstance(value, dict)):
                found.append(f"{path}.{key}")
            found += _result_key_paths(value, f"{path}.{key}")
        return found
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _result_key_paths(v, f"{path}[{i}]")]
    return []


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported R3b preregistration config schema")
    if config.get("status") != "preregistration_only_no_run_yet":
        raise ValueError(
            "this config declares a status other than preregistration; a record "
            "belongs in reference_results with its own producer and checker"
        )
    return config


def digest_problems(config: dict) -> list[str]:
    problems: list[str] = []
    root = HERE.parent
    candidate = config["candidate"]

    source = root / candidate["source"]
    provenance = root / candidate["provenance"]
    if _sha256(source) != candidate["source_sha256"]:
        problems.append("LiH FCIDUMP digest drifted from the preregistration")
    if _sha256(provenance) != candidate["provenance_sha256"]:
        problems.append("LiH provenance digest drifted from the preregistration")
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("fcidump_sha256") != candidate["source_sha256"]:
        problems.append("LiH provenance does not bind the committed FCIDUMP")

    lineage = config["parent_lineage"]
    historical_screen = lineage["admitted_by"]["record_sha256"]
    if _sha256(SCREEN_RECORD) != historical_screen:
        migration = [
            f"R3S record migration: {problem}"
            for problem in record_successor_problems(
                "priceability_screen", historical_sha256=historical_screen
            )
        ]
        if migration:
            problems.append(
                "R3S record digest drifted; this preregistration no longer binds "
                "the historical artifact or its declared migrated successor"
            )
            problems += migration
    for declared, actual, what in (
        (lineage["admitted_by"]["config_sha256"], SCREEN_CONFIG, "R3S config"),
        (lineage["extends"]["config_sha256"], QR3B_CONFIG, "QR3b config"),
    ):
        if _sha256(actual) != declared:
            problems.append(
                f"{what} digest drifted; this preregistration no longer binds the "
                "artifact it claims to inherit from"
            )
    return problems


def protocol_problems(config: dict) -> list[str]:
    """QR3b's protocol, endpoint for endpoint, with non-aliasing seeds."""
    problems: list[str] = []
    protocol = config["protocol"]
    frozen = json.loads(QR3B_CONFIG.read_text(encoding="utf-8"))["protocol"]

    if tuple(protocol["search_endpoints_effective_shots_per_setting"]) != tuple(
        SEARCH_ENDPOINTS
    ):
        problems.append("R3b may not change the frozen R1/R3 search grid")
    for field in ("block_sizes", "estimators", "exploratory_replicas",
                  "confirmatory_replicas", "search_grid_policy"):
        if protocol[field] != frozen[field]:
            problems.append(
                f"protocol.{field} differs from QR3b's, which this probe declares "
                "it inherits unchanged"
            )
    if list(protocol["estimators"]) != list(ESTIMATORS):
        problems.append("the declared estimators are not the frozen pair")
    if protocol.get("probe_is_a_cost_record") is not False:
        problems.append("the R3b probe must disclaim cost-record status")

    roots = protocol.get("seed_roots", {})
    if set(roots) != {"exploratory", "confirmatory", "bootstrap"}:
        problems.append("the probe must declare all three seed roots")
    elif any(not isinstance(v, int) or v < 0 for v in roots.values()):
        problems.append("seed roots must be non-negative integers")
    elif len(set(roots.values())) != 3:
        problems.append("seed roots must be distinct from each other")
    elif set(roots.values()) & set(frozen["seed_roots"].values()):
        problems.append(
            "a seed root collides with QR3b's, so the two probes' streams could alias"
        )
    if not protocol.get("seed_roots_basis"):
        problems.append("fresh seed roots must declare why they are fresh")
    return problems


def gate_problems(config: dict) -> list[str]:
    problems: list[str] = []
    gates = config["acceptance_gates"]
    screen_gates = json.loads(SCREEN_CONFIG.read_text(encoding="utf-8"))["acceptance_gates"]

    if gates.get("all_probe_cells_must_resolve_strictly_before") != SEARCH_ENDPOINTS[-1]:
        problems.append("the resolution gate must name the frozen search ceiling")
    if gates.get("preferred_headroom_endpoint") not in SEARCH_ENDPOINTS:
        problems.append("the headroom endpoint must be on the frozen grid")
    if gates.get("full_30_plus_100_run_authorized_by_this_config") is not False:
        problems.append("a preflight may not authorize a full run")
    if gates.get("word_universe_ceiling") != screen_gates["word_universe_ceiling"]:
        problems.append("the ceiling differs from the one R3S admitted this bank under")
    if gates.get("accuracy_target_millihartree") != screen_gates["accuracy_target_millihartree"]:
        problems.append("the accuracy target differs from R3S's")
    if gates.get("selection_may_not_use_mapping_cost_direction") is not True:
        problems.append("R3b may not relax the QR3b cost-direction gate")
    # QR3b required the intrinsic stop and this probe drops it. That is the one
    # substantive difference between the two, so it is required to be explicit.
    if gates.get("selection_must_stop_intrinsically") is not False:
        problems.append(
            "the margin-stop bank does not stop intrinsically; the gate must say so"
        )
    if not gates.get("selection_must_stop_intrinsically_basis"):
        problems.append("dropping QR3b's intrinsic-stop gate must declare its reason")
    if config["candidate"].get("word_universe_convention") != WORD_UNIVERSE_CONVENTION:
        problems.append(
            f"the bank must be counted in {WORD_UNIVERSE_CONVENTION!r}, the "
            "convention the ceiling was calibrated in"
        )
    if config["candidate"].get("selection_rule_is_preregistered_here") is not False:
        problems.append(
            "the margin rule was declared in R3S, not here; this config is its "
            "first preregistered use and may not claim to originate it"
        )
    return problems


def lineage_problems(config: dict) -> list[str]:
    """The declared bank is a prefix of the ordering QR3b already froze."""
    problems: list[str] = []
    declared = list(config["candidate"]["selected_labels"])
    frozen = list(
        json.loads(QR3B_CONFIG.read_text(encoding="utf-8"))["candidate"]["selected_labels"]
    )
    if frozen[: len(declared)] != declared:
        problems.append(
            f"the declared labels {declared} are not a prefix of QR3b's frozen "
            "ordering, so this is a new selection rather than a shorter draw "
            "from a decided one"
        )
    if len(declared) != config["candidate"]["expected_basis_size"]:
        problems.append("expected_basis_size disagrees with the declared labels")
    return problems


def rederivation_problems(config: dict) -> list[str]:
    """Rebuild the bank and check every declared structural number.

    This is the check that stops the preregistration being a wish. It also
    re-runs the R3S margin rule over the frozen ordering, so a hand-picked
    prefix -- one chosen because it is cheap to measure rather than because the
    accuracy target selects it -- fails here rather than passing quietly.
    """
    problems: list[str] = []
    candidate = config["candidate"]
    gates = config["acceptance_gates"]
    screen_gates = json.loads(SCREEN_CONFIG.read_text(encoding="utf-8"))["acceptance_gates"]
    screen_rule = json.loads(SCREEN_CONFIG.read_text(encoding="utf-8"))["selection_rule"]

    target = float(gates["accuracy_target_millihartree"])
    admissible_bias = target / float(screen_gates["margin_factor"])
    minimum_prefix = int(screen_rule["minimum_prefix_size"])

    # The full frozen ordering, so the margin rule can be re-run over it.
    frozen_labels = json.loads(QR3B_CONFIG.read_text(encoding="utf-8"))["candidate"][
        "selected_labels"
    ]
    spec = dict(candidate, key="lih_cas4e4o", selected_labels=frozen_labels)
    model, _ = _build_model(spec)
    _, ordering = _selected_generators(model, spec)
    reference = ExactMVBackend().state(model.reference, ())
    backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact = float(backend.ground_state(model.hamiltonian, k=1, method="dense")[0][0])

    # Which prefix does the declared rule actually select?
    chosen = None
    for size in range(minimum_prefix, len(ordering) + 1):
        bias, _ = _solved(
            MatrixElementBank(reference, model.hamiltonian, list(ordering[:size])), exact
        )
        if bias <= admissible_bias:
            chosen = size
            break
    if chosen != candidate["expected_basis_size"]:
        problems.append(
            f"the margin rule selects M={chosen}, not the declared "
            f"M={candidate['expected_basis_size']}; the declared bank is not the "
            "one the rule picks"
        )
        return problems

    prefix = ordering[:chosen]
    if [g.label for g in prefix] != list(candidate["selected_labels"]):
        problems.append("the selected prefix's labels differ from the declared ones")
        return problems

    arms = _arm_rows(
        model, reference, prefix, exact,
        config["mapping_arms"], "qwc_groups", count_settings=True,
    )
    tol = float(candidate["bias_tolerance_millihartree"])
    for arm in arms:
        name = arm["mapping"]
        if abs(arm["bias_millihartree"] - candidate["expected_bias_millihartree"]) > tol:
            problems.append(f"{name}: re-derived bias differs from the declared value")
        if arm["word_universe"] != candidate["expected_arm_word_universe"][name]:
            problems.append(
                f"{name}: re-derived W={arm['word_universe']} against declared "
                f"{candidate['expected_arm_word_universe'][name]}"
            )
        if arm["settings"] != candidate["expected_arm_settings"][name]:
            problems.append(
                f"{name}: re-derived {arm['settings']} settings against declared "
                f"{candidate['expected_arm_settings'][name]}"
            )

    binding = max(a["word_universe"] for a in arms)
    if binding != candidate["expected_binding_word_universe"]:
        problems.append("the binding word universe differs from the declared value")
    if binding > gates["word_universe_ceiling"]:
        problems.append(
            f"binding W={binding} exceeds the ceiling {gates['word_universe_ceiling']}; "
            "R3S would not have admitted this bank"
        )

    # And the same numbers as the record that admitted it.
    record = json.loads(SCREEN_RECORD.read_text(encoding="utf-8"))
    stop = next(
        c for c in record["candidates"] if c["candidate"] == "lih_cas4e4o"
    )["margin_stop"]
    if stop["basis_size"] != chosen or list(stop["labels"]) != list(
        candidate["selected_labels"]
    ):
        problems.append("the declared bank is not the one the R3S record admitted")
    if stop["binding_word_universe"] != binding:
        problems.append("binding W disagrees with the R3S record")
    return problems


def main() -> int:
    config = load_config()
    problems = digest_problems(config)
    problems += protocol_problems(config)
    problems += gate_problems(config)
    problems += lineage_problems(config)
    problems += rederivation_problems(config)
    problems += [
        f"result field in a preregistration: {path}"
        for path in _result_key_paths(config)
    ]
    for problem in problems:
        print(f"  {problem}")
    print("R3b preregistration:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
