"""Gate Q18-S2's declaration: the oracle screen on certifying Hubbard 2x2 bases.

The declaration selects its system post hoc, from Q18-S1's oracle diagnostic,
and is declared before its own execution. This gate checks that it can answer
its question as frozen, and that nothing outcome-bearing was frozen with it:

* completeness, the absence of any result-shaped key, and a tenseless claim
  boundary;
* that the estimator, groupings, derivative validator, allocation diagnostic
  and threshold are Q18-S1's own, unchanged;
* the decision ladder's status and verdict names, exactly as the helper module
  states them, and Q18's ``permitted: none`` follow-up and evidence label;
* both trajectories: labels, energy histories and prefixes frozen in the config
  and still carried by the committed ladder rows;
* a recomputation of every frozen structural number from committed inputs:
  the exact sector spectrum, the sector premises and storage bound, and each
  trajectory's rebuilt ``(S, H)`` energies. No second-moment row and no ratio
  is formed;
* input hashes always, implementation hashes until the record names its own
  commit, and, once a record exists, commit order: the declaration's last
  commit precedes or is the commit the record names as its execution, which
  strictly precedes the record's first commit.

    python benchmarks/check_phase15_certifying_screen_declaration.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import check_phase15_residual_successor_declaration as q18s1
from benchmarks import phase15_certifying_screen as screen

CONFIG = HERE / "configs" / "phase15_certifying_screen.json"
RECORD = HERE / "reference_results" / "phase15_certifying_screen.json"
LADDER = HERE / "reference_results" / "acase_ladder.jsonl"
H2_PREFLIGHT = HERE / "configs" / "phase15_h2_preflight.json"
SCHEMA = "clifford_qc.phase15_certifying_screen_config.v1"
SYSTEM = "hubbard_2x2"
ARTIFACT_PATHS = {
    "schema": "clifford_qc.phase15_certifying_screen.v1",
    "producer": "benchmarks/run_phase15_certifying_screen.py",
    "checker": "benchmarks/check_phase15_certifying_screen.py",
    "record": "benchmarks/reference_results/phase15_certifying_screen.json",
}
TRAJECTORIES = {"acase_exact_m25": False, "acase_level4": True}
REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "disclosure", "evidence", "system", "trajectories",
    "domain", "statistic", "grouping", "estimator", "derivative_validation",
    "allocation_diagnostic", "deterministic_checks", "lineage_check", "decision_rule",
    "consequences", "excluded", "prespecified_followup", "measured_before_freezing",
    "implementation_lineage", "record_requirements", "revisions",
)
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/configs/phase15_h2_preflight.json",
    "benchmarks/configs/phase15_residual_successor.json",
    "benchmarks/reference_results/phase15_residual_successor.json",
)
BOUND_IMPLEMENTATIONS = (
    *q18.BOUND_IMPLEMENTATIONS,
    "clifford_qc/subspace/symmetry.py",
    "clifford_qc/subspace/generators.py",
    "clifford_qc/backends/sector_statevector.py",
    "benchmarks/run_acase_ladder.py",
    "benchmarks/run_phase15_measured_residual_preflight.py",
    "benchmarks/run_phase15_h2_preflight.py",
    "benchmarks/phase15_residual_successor.py",
    "benchmarks/check_phase15_extrapolation_preregistration.py",
    "benchmarks/phase15_certifying_screen.py",
    "benchmarks/check_phase15_certifying_screen_declaration.py",
    "benchmarks/run_phase15_certifying_screen.py",
    "benchmarks/check_phase15_certifying_screen.py",
)
# Q18's result keys, plus every outcome-bearing key this screen's record writes.
RESULT_KEYS = q18.RESULT_KEYS | frozenset({
    "status", "statuses", "certifying", "in_domain", "weinstein_upper",
    "temple_lower_bound", "ground_dominated", "ground_weight_lower_bound", "decision", "role",
    "finite_differences", "energy_setting_variances", "residual_setting_variances",
    "energy_neyman_sum", "residual_neyman_sum", "production_ratio", "total_cost_ratio",
    "variance_resolved", "lineage",
})
UNTENSED = q18.UNTENSED
DECISION_STATUSES = list(screen.STATUSES)
VERDICTS = list(screen.VERDICTS)
FROZEN_BANK_SIZE = 9
FLOAT_TOLERANCE = 1e-9
# Q18's own rebuild tolerance for Neyman sums, which add square roots of
# rounding-level setting variances.
NEYMAN_TOLERANCE = 1e-3


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("contains_results") is not False:
        raise ValueError("the declaration must state contains_results: false")
    found = sorted(_result_keys(config))
    if found:
        raise ValueError(f"the declaration carries result-shaped keys: {found}")
    return config


def _result_keys(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RESULT_KEYS:
                yield f"{path}.{key}"
            yield from _result_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _result_keys(item, f"{path}[{index}]")


def ladder_rows(path: Path = LADDER) -> dict:
    """The committed Hubbard 2x2 rows of the two declared trajectories."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rung_name") == SYSTEM and row.get("method") in TRAJECTORIES:
            if row["method"] in rows:
                raise ValueError(f"the ladder repeats {row['method']} on {SYSTEM}")
            rows[row["method"]] = row
    return rows


def frozen_labels() -> list[str]:
    from benchmarks.run_mapping_axis import load_config as load_mapping

    spec = next(s for s in load_mapping()["systems"] if s["key"] == SYSTEM)
    return list(spec["selected_labels"])


# ------------------------------------------------------------------ static

def predecessor_problems(config: dict) -> list[str]:
    """Everything that is Q18-S1's must still be Q18-S1's."""
    problems = []
    predecessor = q18s1.load_config()
    finite = predecessor["linearization"]["finite_difference"]
    expected_rule = {**predecessor["derivative_validation"],
                     "directions": finite["directions"], "seed": finite["seed"],
                     "direction_definitions": finite["direction_definitions"]}
    if config["derivative_validation"] != expected_rule:
        problems.append("derivative_validation must be Q18-S1's rule with its directions and seed")
    if config["allocation_diagnostic"] != predecessor["allocation_diagnostic"]:
        problems.append("allocation_diagnostic must be Q18-S1's, unchanged")
    if config["estimator"] != {"estimand": predecessor["estimand"],
                               "linearization": {key: predecessor["linearization"][key]
                                                 for key in ("energy", "residual",
                                                             "implementation", "properties")}}:
        problems.append("estimator must quote Q18-S1's estimand and linearization unchanged")
    grouping = config["grouping"]
    pg = predecessor["grouping"]
    if (grouping.get("declared_protocol") != pg["declared_protocol_by_system"][SYSTEM]
            or grouping.get("alternative_protocol") != pg["alternative_protocol_by_system"][SYSTEM]
            or grouping.get("estimator") != pg["estimator"]):
        problems.append("grouping must be Q18-S1's Hubbard 2x2 protocols and estimator")
    for key in ("energy_universe", "residual_universe", "universe_convention", "variance"):
        if grouping.get(key) != pg[key]:
            problems.append(f"grouping {key} must be Q18-S1's")
    statistic = config["statistic"]
    if statistic.get("max_ratio") != predecessor["statistic"]["max_ratio"]:
        problems.append("the threshold must be Q18's")
    if statistic.get("name") != "neyman_ratio":
        problems.append("the decision statistic must be the Neyman ratio")
    if statistic.get("uniform_definition") != predecessor["statistic"]["definition"]:
        problems.append("the reported uniform R must be Q18's definition")
    return problems


def rule_problems(config: dict) -> list[str]:
    problems = []
    rule = config["decision_rule"]
    if rule.get("prefix_statuses") != DECISION_STATUSES:
        problems.append("prefix statuses must be the helper's ladder")
    if rule.get("verdict_ladder") != VERDICTS:
        problems.append("verdicts must be the helper's")
    if set(config["consequences"]) != set(VERDICTS):
        problems.append("every verdict needs exactly one consequence")
    domain = config["domain"]
    if domain.get("rounding_ulps") != screen.SPECTRAL_ROUNDING_ULPS:
        problems.append("domain rounding must be the helper's")
    if domain.get("degeneracy_tolerance") != screen.DEGENERACY_TOLERANCE:
        problems.append("degeneracy tolerance must be the helper's")
    if config["deterministic_checks"].get("history_tolerance") != screen.HISTORY_TOLERANCE:
        problems.append("history tolerance must be the helper's")
    if config["lineage_check"].get("relative_tolerance") != FLOAT_TOLERANCE or config[
            "lineage_check"].get("neyman_relative_tolerance") != NEYMAN_TOLERANCE:
        problems.append("lineage tolerances must be the gate's")
    allocation = config["allocation_diagnostic"]
    if allocation.get("decision_role") != "diagnostic_only":
        problems.append("the integer allocation must stay outside the decision")
    return problems


def trajectory_problems(config: dict, rows: dict | None = None) -> list[str]:
    problems = []
    rows = ladder_rows() if rows is None else rows
    declared = config["trajectories"]
    if declared.get("source") != str(LADDER.relative_to(ROOT)):
        problems.append("trajectories must come from the committed ladder")
    if sorted(declared.get("rows", {})) != sorted(TRAJECTORIES):
        return problems + [f"trajectories must be exactly {sorted(TRAJECTORIES)}"]
    frozen = frozen_labels()
    for method, level4 in TRAJECTORIES.items():
        spec = declared["rows"][method]
        labels, history = spec["labels"], spec["energy_history"]
        if spec.get("level4") is not level4:
            problems.append(f"{method}: level4 flag differs")
        if labels[:FROZEN_BANK_SIZE] != frozen:
            problems.append(f"{method}: the first nine labels are not the frozen bank")
        if spec.get("prefixes") != list(range(FROZEN_BANK_SIZE, len(labels) + 1)):
            problems.append(f"{method}: prefixes must be every M from 9 to the final size")
        if len(history) != len(labels):
            problems.append(f"{method}: one energy per prefix")
        row = rows.get(method)
        if row is None:
            problems.append(f"{method}: absent from the committed ladder")
            continue
        if row["labels"] != labels:
            problems.append(f"{method}: the ladder's labels changed")
        if len(row["energy_history"]) != len(history) or any(
                abs(a - b) > screen.HISTORY_TOLERANCE
                for a, b in zip(row["energy_history"], history)):
            problems.append(f"{method}: the ladder's energy history changed")
        family = "levels 0-4" if level4 else "levels 0-3"
        if row.get("candidate_family") != family or spec.get("candidate_family") != family:
            problems.append(f"{method}: candidate family differs")
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


def static_problems(config: dict, *, record_exists: bool | None = None,
                    rows: dict | None = None) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    if config["question_id"] != "Q18-S2":
        problems.append("wrong question identifier")
    if config["contains_results"] is not False:
        problems.append("contains_results must be false")
    problems += [f"result-shaped key {key}" for key in _result_keys(config)]
    boundary = config["claim_boundary"].lower()
    problems += [f"claim boundary is tensed: {phrase!r}" for phrase in UNTENSED
                 if phrase in boundary]
    evidence = config["evidence"]
    if evidence.get("design_status") != "declared_before_execution" or evidence.get(
            "system_selection") != "post_hoc":
        problems.append("evidence must label a pre-execution declaration with post-hoc selection")
    if evidence.get("label") != "asymptotic_oracle":
        problems.append("evidence label must be asymptotic_oracle")
    if config["prespecified_followup"].get("permitted") != "none":
        problems.append("the declaration permits no follow-up")
    if config["system"].get("key") != SYSTEM or config["system"].get(
            "source_config") != "benchmarks/configs/mapping_axis.json":
        problems.append("the system must be the mapping-axis Hubbard 2x2")
    for key, value in ARTIFACT_PATHS.items():
        if config["record_requirements"].get(key) != value:
            problems.append(f"record_requirements {key} must name this screen's artifact")
    if config["record_requirements"].get("must_carry", {}).get(
            "quantum_advantage_claim") is not False:
        problems.append("the record must carry quantum_advantage_claim: false")
    revisions = config["revisions"]
    if [row.get("revision") for row in revisions] != list(range(len(revisions))):
        problems.append("revisions must be numbered 0, 1, 2, ... in order")
    problems += predecessor_problems(config)
    problems += rule_problems(config)
    problems += trajectory_problems(config, rows)
    problems += lineage_problems(config, record_exists=record_exists)
    return problems


# ------------------------------------------------------------------ structural

def system_model():
    from benchmarks.run_mapping_axis import _build_model
    from benchmarks.run_mapping_axis import load_config as load_mapping

    spec = next(s for s in load_mapping()["systems"] if s["key"] == SYSTEM)
    model, _ = _build_model(spec)
    return model


def trajectory_inputs(config: dict, model=None):
    """``{method: (generators, pool_size)}`` rebuilt from the frozen labels."""
    model = system_model() if model is None else model
    return {method: screen.trajectory_generators(model, spec["labels"],
                                                 level4=spec["level4"])
            for method, spec in config["trajectories"]["rows"].items()}


def structural_quantities(config: dict) -> dict:
    """Every frozen number, from committed inputs. First moments only."""
    from benchmarks.check_phase15_preregistration import first_moment_bank
    from benchmarks.run_mapping_axis import _selected_generators
    from benchmarks.run_mapping_axis import load_config as load_mapping

    model = system_model()
    spec = next(s for s in load_mapping()["systems"] if s["key"] == SYSTEM)
    _, frozen = _selected_generators(model, spec)
    inputs = trajectory_inputs(config, model)
    spectrum = screen.sector_spectrum(model)
    premises = screen.sector_premises(model, [gens for gens, _ in inputs.values()])
    out = {"spectrum": spectrum, "premises": premises, "trajectories": {}}
    for method, (generators, pool) in inputs.items():
        declared = config["trajectories"]["rows"][method]
        bank = first_moment_bank(model, generators)
        energies = [float(bank.solve(list(range(size))).ground_energy)
                    for size in range(1, len(generators) + 1)]
        same = all(a.label == b.label and a.mv.terms == b.mv.terms
                   for a, b in zip(generators[:FROZEN_BANK_SIZE], frozen))
        out["trajectories"][method] = {
            "pool_size": pool,
            "final_basis_size": len(generators),
            "first_nine_are_frozen_bank": bool(same and len(frozen) == FROZEN_BANK_SIZE),
            "history_reproduces": all(abs(a - b) <= screen.HISTORY_TOLERANCE
                                      for a, b in zip(energies, declared["energy_history"])),
            "prefixes_below_midpoint": [size for size in declared["prefixes"]
                                        if energies[size - 1] < spectrum["midpoint"]],
        }
    return out


def _compare(declared, got, path="") -> list[str]:
    if isinstance(declared, dict) and isinstance(got, dict):
        if sorted(declared) != sorted(got):
            return [f"{path or '.'}: fields differ"]
        out = []
        for key in declared:
            out += _compare(declared[key], got[key], f"{path}.{key}")
        return out
    if isinstance(declared, float) or isinstance(got, float):
        if isinstance(declared, bool) or isinstance(got, bool) or not math.isclose(
                float(declared), float(got), rel_tol=FLOAT_TOLERANCE, abs_tol=1e-12):
            return [f"{path}: declared {declared!r}, recomputed {got!r}"]
        return []
    return [] if declared == got else [f"{path}: declared {declared!r}, recomputed {got!r}"]


def premise_problems(quantities: dict) -> list[str]:
    problems = []
    premises, spectrum = quantities["premises"], quantities["spectrum"]
    if (premises["reference_image_sector_leakage"] > 1e-12
            or premises["hamiltonian_sector_leakage"] > 1e-12):
        problems.append("a generator's image of the reference, or the Hamiltonian, leaves the "
                        "(N, S_z) sector; the sector spectrum is not the reference")
    if premises["spin_parity_sector_violations"]:
        problems.append("a word leaves the spin-parity sector; the storage bound fails")
    anchor = json.loads(H2_PREFLIGHT.read_text(encoding="utf-8"))["clauses"]["storage"][
        "max_coefficient_occurrences"]
    if premises["coefficient_occurrence_bound"] > anchor:
        problems.append("the storage bound exceeds the Phase 2M anchor")
    if spectrum["ground_degeneracy"] != 1:
        problems.append("the sector ground state is degenerate")
    for method, row in quantities["trajectories"].items():
        if not row["first_nine_are_frozen_bank"] or not row["history_reproduces"]:
            problems.append(f"{method}: does not rebuild the committed trajectory")
    return problems


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None) -> list[str]:
    got = structural_quantities(config)
    if computed is not None:
        computed.update(got)
    problems = _compare(config["measured_before_freezing"], got)
    problems += premise_problems(got)
    if notes is not None:
        spectrum = got["spectrum"]
        notes.append(f"  sector: E0={spectrum['ground_energy']:.12f} "
                     f"E1={spectrum['first_excited_energy']:.12f} "
                     f"midpoint={spectrum['midpoint']:.12f}")
        for method, row in got["trajectories"].items():
            notes.append(f"  {method}: M<={row['final_basis_size']} pool={row['pool_size']} "
                         f"below midpoint {row['prefixes_below_midpoint']}")
    return problems


# ------------------------------------------------------------------ order

def _is_ancestor(older: str, newer: str) -> bool | None:
    """``older`` is ``newer`` or an ancestor of it; None when git cannot say."""
    try:
        out = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer],
                             cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(out.returncode)


def execution_order_problems(config_commit: str, execution_commit: str | None,
                             record_commit: str, is_ancestor=_is_ancestor) -> list[str]:
    """Declaration, then execution, then the record, by the record's own provenance.

    The execution commit is the one the record's provenance names. It may be
    the declaration's last commit itself, and it must strictly precede the
    commit that first adds the record.
    """
    if not execution_commit:
        return ["the record names no execution commit"]
    after_declaration = is_ancestor(config_commit, execution_commit)
    before_record = is_ancestor(execution_commit, record_commit)
    if after_declaration is None or before_record is None:
        return ["the record's execution commit is not in this history"]
    problems = []
    if not after_declaration:
        problems.append("the record was executed at a commit that does not contain the "
                        "declaration's last change")
    if not before_record or execution_commit == record_commit:
        problems.append("the record's execution commit must strictly precede its first commit")
    return problems


def commit_order_problems(notes: list[str], *, require_config: bool = False) -> list[str]:
    """The config's last commit must strictly precede the record's first, and the
    record's own execution commit must sit between them."""
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
    execution = json.loads(RECORD.read_text(encoding="utf-8")).get("provenance", {}).get("git_sha")
    if {config_commit, record_commit, execution} & lineage._shallow_boundary():
        notes.append("  commit order: SKIP (shallow history)")
        return []
    if config_commit == record_commit or not _is_ancestor(config_commit, record_commit):
        return ["the declaration's last commit must strictly precede the record's first"]
    problems = execution_order_problems(config_commit, execution, record_commit)
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
        print("OK Q18-S2 declaration: Q18-S1's estimator and threshold, frozen trajectories, "
              "certifying domain, post-hoc system selection")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
