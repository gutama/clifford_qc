"""Gate Q18-S4's declaration: the measured residual's cost under fully commuting grouping.

Q18-S1 closed the uniform single-assignment QWC estimator negative on the five
licensed banks. Q18-S4 asks Q18-S1's question at the other grouping endpoint
the package ships, fully commuting settings, on the four 8-qubit banks. It is
declared after every Q18 and Q18-S1 ratio was seen, so it is post-hoc, and its
commit order protects execution identity, not inference. This gate checks that
it can answer its question as frozen, and that nothing outcome-bearing was
frozen with it:

* completeness, the absence of any result-shaped key, and a tenseless claim
  boundary;
* that the estimand, linearization, derivative validator, statistic, threshold,
  ladder and combination rule are Q18-S1's, unchanged, and that only the
  partition protocols and the bank set differ;
* the two fully commuting protocols on each bank, and H2O's exclusion with its
  reason;
* the predecessor records, bound by SHA-256, and Q18-S1's NONE;
* a recomputation of every frozen number from committed inputs and first
  moments: each bank's ``(S, H)`` solve and both fully commuting partitions of
  its ``(S, H)`` universe. No second-moment row, residual functional, grouping
  of the combined universe or setting variance is formed;
* input hashes always, implementation hashes until the record names its own
  commit, and, once a record exists, commit order.

    python benchmarks/check_phase15_fc_residual_declaration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

from benchmarks import check_phase15_certifying_screen_declaration as q18s2
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import check_phase15_residual_successor_declaration as q18s1
from benchmarks import phase15_fc_residual as fc

CONFIG = HERE / "configs" / "phase15_fc_residual.json"
RECORD = HERE / "reference_results" / "phase15_fc_residual.json"
SCHEMA = "clifford_qc.phase15_fc_residual_config.v1"
SYSTEMS = ("h4", "h4_converged", "beh2", "hubbard_2x2")
EXCLUDED = ("h2o_cas8e6o",)
ARTIFACT_PATHS = {
    "schema": "clifford_qc.phase15_fc_residual.v1",
    "producer": "benchmarks/run_phase15_fc_residual.py",
    "checker": "benchmarks/check_phase15_fc_residual.py",
    "record": "benchmarks/reference_results/phase15_fc_residual.json",
}
REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "estimand", "linearization", "derivative_validation",
    "grouping", "statistic", "decision_rule", "consequences", "banks",
    "measured_before_freezing", "reachable_verdicts", "deterministic_checks", "evidence",
    "excluded_from_cost", "exploratory_disclosure", "prespecified_followup", "successor",
    "implementation_lineage", "record_requirements", "revisions",
)
# Q18-S1's own words for the parts this successor keeps.
UNCHANGED = ("estimand", "linearization", "derivative_validation", "statistic",
             "decision_rule")
UNCHANGED_GROUPING = ("estimator", "allocation", "energy_universe", "residual_universe",
                      "universe_convention", "variance")
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/reference_results/mapping_axis.json",
    "benchmarks/configs/phase15_h2_preflight.json",
    "benchmarks/reference_results/phase15_h2_preflight.json",
    "benchmarks/reference_results/bank_storage_ledger.json",
    "benchmarks/reference_results/second_moment_validation.json",
    "benchmarks/configs/phase15_measured_residual_preflight.json",
    "benchmarks/reference_results/phase15_measured_residual_preflight.json",
    "benchmarks/configs/phase15_residual_successor.json",
    "benchmarks/reference_results/phase15_residual_successor.json",
)
BOUND_IMPLEMENTATIONS = (
    *q18.BOUND_IMPLEMENTATIONS,
    "benchmarks/run_phase15_h2_preflight.py",
    "benchmarks/phase15_residual_successor.py",
    "clifford_qc/measurement/block_commuting.py",
    "clifford_qc/measurement/regrouping.py",
    "clifford_qc/pauli_action.py",
    "benchmarks/phase15_fc_residual.py",
    "benchmarks/check_phase15_fc_residual_declaration.py",
    "benchmarks/run_phase15_fc_residual.py",
    "benchmarks/check_phase15_fc_residual.py",
)
RESULT_KEYS = q18.RESULT_KEYS | frozenset({
    "statuses", "decision", "estimator_values", "predecessor_anchor",
    "finite_differences", "richardson", "partitions_valid", "statevector_agreement",
    "statevector_worst_relative_deviation", "energy_partition_sha256",
    "residual_partition_sha256", "energy_setting_variances", "residual_setting_variances",
    "deterministic_check_values",
})
UNTENSED = q18.UNTENSED
INT_FIELDS = ("n_qubits", "basis_size", "effective_rank", "raw_sh_word_universe",
              "raw_combined_word_universe", "measured_sh_words", "measured_combined_words")
FLOAT_FIELDS = ("ground_energy", "subspace_gap", "variance", "residual_norm",
                "cancellation_scale", "matched_precision_half")
PREMISES = ("full_rank", "ground_root_nondegenerate", "variance_resolved",
            "identity_in_raw_universes", "ground_energy_matches_validation",
            "sh_universe_matches_records", "fc_energy_partitions_cover_measured_words")
FROZEN_FIELDS = (*INT_FIELDS, *FLOAT_FIELDS, "raw_combined_words_sha256",
                 "fc_energy_settings", *PREMISES)


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("contains_results") is not False:
        raise ValueError("the declaration must state contains_results: false")
    found = sorted(result_keys(config))
    if found:
        raise ValueError(f"the declaration carries result-shaped keys: {found}")
    return config


def result_keys(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RESULT_KEYS:
                yield f"{path}.{key}"
            yield from result_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from result_keys(item, f"{path}[{index}]")


# ------------------------------------------------------------------ static

def predecessor_problems(config: dict) -> list[str]:
    """Everything Q18-S4 keeps from Q18-S1 must still be Q18-S1's."""
    problems = []
    predecessor = q18s1.load_config()
    for key in UNCHANGED:
        if config[key] != predecessor[key]:
            problems.append(f"{key} must be Q18-S1's, unchanged")
    for key in UNCHANGED_GROUPING:
        if config["grouping"].get(key) != predecessor["grouping"][key]:
            problems.append(f"grouping {key} must be Q18-S1's, unchanged")
    successor = config["successor"]
    records = {
        "q18s1_record": q18s1.RECORD,
        "q18_record": q18.RECORD,
    }
    for key, path in records.items():
        row = successor.get(key, {})
        if row.get("path") != str(path.relative_to(ROOT)):
            problems.append(f"successor {key} must name {path.relative_to(ROOT)}")
        elif not path.is_file() or row.get("sha256") != q18._sha256(path):
            problems.append(f"successor {key} changed since the declaration")
    outcome = json.loads(q18s1.RECORD.read_text(encoding="utf-8"))["decision"]["verdict"]
    if outcome != "NONE" or successor.get("q18s1_outcome") != "NONE":
        problems.append("Q18-S1 must read NONE, and the declaration must say so")
    if successor.get("design_status") != "post_hoc" or not successor.get("disclosure"):
        problems.append("the successor must disclose its post-hoc design")
    if successor.get("estimator_licensed") is not False:
        problems.append("the declaration licenses no estimator by itself")
    shared = q18s1.load_config()["measured_before_freezing"]
    for name in SYSTEMS:
        mine = config["measured_before_freezing"].get(name, {})
        for key in (*INT_FIELDS, *FLOAT_FIELDS, "raw_combined_words_sha256"):
            if mine.get(key) != shared[name][key]:
                problems.append(f"{name}: {key} differs from Q18-S1's freeze")
    return problems


def grouping_problems(config: dict) -> list[str]:
    problems = []
    grouping = config["grouping"]
    declared = grouping.get("declared_protocol_by_system")
    alternative = grouping.get("alternative_protocol_by_system")
    if declared != {name: "fc_greedy" for name in SYSTEMS}:
        problems.append("the declared protocol must be fc_greedy on every bank")
    if alternative != {name: "fc_isotropic_cover" for name in SYSTEMS}:
        problems.append("the alternative protocol must be fc_isotropic_cover on every bank")
    if set(grouping.get("protocol_definitions", {})) != set(fc.PROTOCOLS):
        problems.append("both fully commuting protocols must be defined")
    banks = config["banks"]
    if tuple(banks.get("systems", ())) != SYSTEMS:
        problems.append(f"the banks must be {list(SYSTEMS)}, in that order")
    excluded = banks.get("excluded", {})
    if tuple(excluded) != EXCLUDED or not all(
            isinstance(reason, str) and reason for reason in excluded.values()):
        problems.append("H2O must be excluded by name, with its reason")
    if banks.get("source_config") != q18s1.load_config()["banks"]["source_config"]:
        problems.append("the banks must come from Q18-S1's source config")
    if set(config["measured_before_freezing"]) != set(SYSTEMS):
        problems.append("measured_before_freezing must cover exactly the declared banks")
    for name, frozen in config["measured_before_freezing"].items():
        if tuple(frozen) != FROZEN_FIELDS:
            problems.append(f"{name}: measured_before_freezing fields differ from the schema")
        elif set(frozen["fc_energy_settings"]) != set(fc.PROTOCOLS):
            problems.append(f"{name}: both protocols' energy settings must be frozen")
    reachable = q18.reachable_verdicts([True] * len(SYSTEMS))
    if config["reachable_verdicts"] != sorted(reachable):
        problems.append(f"reachable_verdicts must be {sorted(reachable)}")
    if set(config["consequences"]) != {"FULL", "RESTRICTED", "NONE", "INVALID"} or not all(
            isinstance(text, str) and text for text in config["consequences"].values()):
        problems.append("every verdict needs its consequence")
    return problems


def lineage_problems(config: dict, *, record_exists: bool | None = None) -> list[str]:
    problems = []
    exists = RECORD.exists() if record_exists is None else record_exists
    bindings = config["implementation_lineage"]
    for group in ("inputs", "implementation"):
        if group == "implementation" and exists:
            continue
        for row in bindings[group]:
            path = ROOT / row["path"]
            if not path.is_file():
                problems.append(f"lineage file missing: {row['path']}")
            elif q18._sha256(path) != row["sha256"]:
                problems.append(f"declared sha256 drifted for {row['path']}")
    inputs = {row["path"] for row in bindings["inputs"]}
    implementation = {row["path"] for row in bindings["implementation"]}
    problems += [f"input {path} must be bound" for path in BOUND_INPUTS if path not in inputs]
    problems += [f"implementation {path} must be bound" for path in BOUND_IMPLEMENTATIONS
                 if path not in implementation]
    return problems


def static_problems(config: dict, *, record_exists: bool | None = None) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    if config["question_id"] != "Q18-S4":
        problems.append("wrong question identifier")
    if config["contains_results"] is not False:
        problems.append("contains_results must be false")
    problems += [f"result-shaped key {key}" for key in result_keys(config)]
    boundary = config["claim_boundary"].lower()
    problems += [f"claim boundary is tensed: {phrase!r}" for phrase in UNTENSED
                 if phrase in boundary]
    evidence = config["evidence"]
    if evidence.get("label") != "asymptotic" or evidence.get("design_status") != "post_hoc":
        problems.append("evidence must be labelled asymptotic and post_hoc")
    if config["prespecified_followup"].get("permitted") != "none":
        problems.append("the declaration permits no follow-up")
    for key, value in ARTIFACT_PATHS.items():
        if config["record_requirements"].get(key) != value:
            problems.append(f"record_requirements {key} must name this declaration's artifact")
    if config["record_requirements"].get("must_carry", {}).get(
            "quantum_advantage_claim") is not False:
        problems.append("the record must carry quantum_advantage_claim: false")
    revisions = config["revisions"]
    if [r.get("revision") for r in revisions] != list(range(len(revisions))):
        problems.append("revisions must be numbered 0, 1, 2, ... in order")
    problems += predecessor_problems(config)
    problems += grouping_problems(config)
    problems += lineage_problems(config, record_exists=record_exists)
    return problems


# ------------------------------------------------------------------ structural

def quantities_for(model, selected, committed: dict) -> dict:
    """Every frozen number of one bank. First moments only: no second-moment
    row, no residual functional, no grouping of the combined universe and no
    setting variance. ``committed`` is :func:`q18.committed_rows` of the bank."""
    bank = lineage.first_moment_bank(model, selected)
    result = bank.solve()
    energies = [float(value) for value in result.energies]
    ground = energies[0]
    gap = energies[1] - ground if len(energies) > 1 else math.inf
    sh_words = bank.word_set()
    identity_in = 0 in sh_words
    partitions = {protocol: fc.fc_partition(sh_words, protocol, model.n)
                  for protocol in fc.PROTOCOLS}
    variance = committed["variance"]
    residual = math.sqrt(max(variance, 0.0))
    measured = len(sh_words) - int(identity_in)
    return {
        "n_qubits": model.n,
        "basis_size": len(selected),
        "effective_rank": int(result.effective_rank),
        "raw_sh_word_universe": len(sh_words),
        "raw_combined_word_universe": committed["raw_combined_word_universe"],
        "measured_sh_words": measured,
        "measured_combined_words": committed["raw_combined_word_universe"] - int(identity_in),
        "ground_energy": ground,
        "subspace_gap": gap,
        "variance": variance,
        "residual_norm": residual,
        "cancellation_scale": committed["cancellation_scale"],
        "matched_precision_half": residual / 4.0,
        "raw_combined_words_sha256": committed["raw_combined_words_sha256"],
        "fc_energy_settings": {protocol: len(groups)
                               for protocol, groups in partitions.items()},
        "full_rank": int(result.effective_rank) == len(selected),
        "ground_root_nondegenerate": gap > q18.GAP_FLOOR * max(1.0, abs(ground)),
        "variance_resolved": committed["variance_resolved"],
        "identity_in_raw_universes": identity_in,
        "ground_energy_matches_validation": q18._close(
            ground, committed["validation_energy"], abs_=1e-10),
        "sh_universe_matches_records": (
            len(sh_words) == committed["preflight_sh_word_universe"]
            == committed["ledger_sh_word_universe"]),
        "fc_energy_partitions_cover_measured_words": all(
            not fc.partition_problems(groups, sh_words, model.n)
            for groups in partitions.values()),
    }


def structural_quantities(name: str) -> dict:
    model, selected = lineage.bank_inputs(name)
    return quantities_for(model, selected, q18.committed_rows(name))


def compare(declared: dict, got: dict) -> list[str]:
    problems = []
    if tuple(declared) != tuple(got):
        return ["fields differ from the recomputation"]
    for key, value in got.items():
        want = declared[key]
        if key in FLOAT_FIELDS:
            if not q18._close(want, value, rel=1e-9, abs_=1e-12):
                problems.append(f"{key} declared {want!r}, recomputed {value!r}")
        elif want != value or type(want) is not type(value):
            problems.append(f"{key} declared {want!r}, recomputed {value!r}")
    return problems


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None, *, quantities=None) -> list[str]:
    quantities = structural_quantities if quantities is None else quantities
    problems = []
    for name in config["banks"]["systems"]:
        got = quantities(name)
        if computed is not None:
            computed[name] = got
        problems += [f"{name}: {problem}"
                     for problem in compare(config["measured_before_freezing"][name], got)]
        problems += [f"{name}: premise {key} fails" for key in PREMISES if not got[key]]
        if notes is not None:
            notes.append(f"  {name}: n={got['n_qubits']} M={got['basis_size']} "
                         f"sigma={got['residual_norm']:.4g} raw |U_SH|="
                         f"{got['raw_sh_word_universe']} raw |U|="
                         f"{got['raw_combined_word_universe']} fully commuting G_SH="
                         f"{got['fc_energy_settings']}")
    return problems


# ------------------------------------------------------------------ order

def commit_order_problems(notes: list[str], *, require_config: bool = False) -> list[str]:
    """The config's last commit strictly precedes the record's first, and the
    record's own execution commit sits between them."""
    config_commit = lineage._last_commit(CONFIG)
    if config_commit is None:
        if require_config:
            return ["commit the declaration before executing it"]
        notes.append("  commit order: SKIP (declaration not committed, or no git history)")
        return []
    if not RECORD.exists():
        notes.append("  record absent; the declaration carries no result")
        return []
    record_commit = lineage._first_commit(RECORD)
    if record_commit is None:
        notes.append("  record not committed yet")
        return []
    try:
        provenance = json.loads(RECORD.read_text(encoding="utf-8")).get("provenance")
    except (OSError, ValueError, AttributeError):
        return ["the record cannot be read for its execution commit"]
    execution = provenance.get("git_sha") if isinstance(provenance, dict) else None
    if not isinstance(execution, str):
        return ["the record names no execution commit"]
    if {config_commit, record_commit, execution} & lineage._shallow_boundary():
        notes.append("  commit order: SKIP (shallow history)")
        return []
    if config_commit == record_commit or not q18s2._is_ancestor(config_commit, record_commit):
        return ["the declaration's last commit must strictly precede the record's first"]
    problems = q18s2.execution_order_problems(config_commit, execution, record_commit)
    if not problems:
        notes.append(f"  declaration {config_commit[:12]} <= execution {execution[:12]} "
                     f"< record {record_commit[:12]}")
    return problems


def main() -> int:
    try:
        config = load_config()
        problems = static_problems(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}")
        return 1
    notes: list[str] = []
    if not problems:
        problems += structural_problems(config, notes)
        problems += commit_order_problems(notes)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("OK Q18-S4 declaration: Q18-S1's estimator, statistic and threshold under "
              "fully commuting grouping on the four 8-qubit banks, post_hoc design")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
