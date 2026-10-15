"""Gate Q18-S3's declaration: certify on the ladder's Krylov basis, then allocate.

The declaration selects its basis post hoc, after Q18-S2's record and its
post-record Krylov check, and is declared before its own execution. This gate
checks that it can answer its question as frozen, and that nothing
outcome-bearing was frozen with it:

* completeness, the absence of any result-shaped key, and a tenseless claim
  boundary;
* that the estimator, groupings, threshold, fixed-step validator and oracle
  allocation diagnostic are Q18-S2's (so Q18-S1's), unchanged;
* the helper module's constants: rank margin, eighty-digit validator,
  agreement tolerances, status and verdict ladders;
* the practical-allocation rule: seeds, replicates, pilot, floor, target,
  confidence family and validation thresholds, typed and in range;
* the basis: the committed ladder's Krylov row, its labels and prefixes;
* a recomputation of every frozen structural number from committed inputs:
  the sector spectrum, the sector premises and storage bound, the exactness of
  the power table, and each prefix's first-moment ``(S, H)`` solve. No
  second-moment pairing and no ratio is formed;
* input hashes always, implementation hashes until the record names its own
  commit, and, once a record exists, commit order.

    python benchmarks/check_phase15_certifying_krylov_declaration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_certifying_screen_declaration as q18s2
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import phase15_certifying_krylov as krylov
from benchmarks import phase15_certifying_screen as screen

CONFIG = HERE / "configs" / "phase15_certifying_krylov.json"
RECORD = HERE / "reference_results" / "phase15_certifying_krylov.json"
LADDER = HERE / "reference_results" / "acase_ladder.jsonl"
H2_PREFLIGHT = HERE / "configs" / "phase15_h2_preflight.json"
SCHEMA = "clifford_qc.phase15_certifying_krylov_config.v1"
SYSTEM = q18s2.SYSTEM
METHOD = "krylov"
KRYLOV_ORDER = 8
PREFIXES = list(range(1, KRYLOV_ORDER + 2))
ARTIFACT_PATHS = {
    "schema": "clifford_qc.phase15_certifying_krylov.v1",
    "producer": "benchmarks/run_phase15_certifying_krylov.py",
    "checker": "benchmarks/check_phase15_certifying_krylov.py",
    "record": "benchmarks/reference_results/phase15_certifying_krylov.json",
}
REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "disclosure", "evidence", "system", "basis",
    "certification", "rank_margin", "statistic", "grouping", "estimator",
    "derivative_validation", "fixed_step_validation", "allocation_diagnostic",
    "practical_allocation", "deterministic_checks", "lineage_check", "decision_rule",
    "consequences", "excluded", "prespecified_followup", "measured_before_freezing",
    "implementation_lineage", "record_requirements", "revisions",
)
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/configs/acase_ladder.json",
    "benchmarks/reference_results/acase_ladder.jsonl",
    "benchmarks/configs/phase15_h2_preflight.json",
    "benchmarks/configs/phase15_residual_successor.json",
    "benchmarks/reference_results/phase15_residual_successor.json",
    "benchmarks/configs/phase15_certifying_screen.json",
    "benchmarks/reference_results/phase15_certifying_screen.json",
)
BOUND_IMPLEMENTATIONS = (
    *q18s2.BOUND_IMPLEMENTATIONS,
    "clifford_qc/subspace/projection.py",
    "clifford_qc/subspace/contracts.py",
    "clifford_qc/backends/finite_shot.py",
    "clifford_qc/pauli_action.py",
    "benchmarks/run_phase15_certifying_screen.py",
    "benchmarks/check_phase15_certifying_screen.py",
    "benchmarks/phase15_certifying_krylov.py",
    "benchmarks/check_phase15_certifying_krylov_declaration.py",
    "benchmarks/run_phase15_certifying_krylov.py",
    "benchmarks/check_phase15_certifying_krylov.py",
)
# Q18-S2's result keys, plus every outcome-bearing key this record writes and
# its config never uses.
RESULT_KEYS = q18s2.RESULT_KEYS | frozenset({
    "rank", "rank_stable", "precise", "dense", "precise_derivatives", "fixed_step", "practical", "replicates", "summary",
    "validation_checks", "budget", "failures", "certification_rate",
    "expected_effective_cost_ratio", "median_effective_cost_ratio", "oracle_verdict",
    "final_verdict", "precise_energy_agrees", "precise_variance_agrees",
    "dense_energy_agrees", "dense_variance_agrees", "fixed_step_validated",
})
UNTENSED = q18.UNTENSED
FLOAT_TOLERANCE = 1e-9
# The normalized overlap's eigenvalues carry absolute rounding near 1e-15, so the
# smallest of them, down to 6e-12 here, and the condition number built on it
# are frozen to this relative tolerance rather than to FLOAT_TOLERANCE.
SPECTRAL_TOLERANCE = 1e-2
SPECTRAL_FIELDS = frozenset({"smallest_overlap_eigenvalue", "overlap_condition_number",
                             "ladder_condition_number"})
NEYMAN_TOLERANCE = q18s2.NEYMAN_TOLERANCE


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("contains_results") is not False:
        raise ValueError("the declaration must state contains_results: false")
    found = sorted(q18s2._result_keys(config))
    found += sorted(_own_result_keys(config))
    if found:
        raise ValueError(f"the declaration carries result-shaped keys: {found}")
    return config


def _own_result_keys(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RESULT_KEYS - q18s2.RESULT_KEYS:
                yield f"{path}.{key}"
            yield from _own_result_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _own_result_keys(item, f"{path}[{index}]")


def ladder_row(path: Path = LADDER) -> dict:
    """The committed Hubbard 2x2 fixed Krylov row."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    found = [row for row in rows
             if row.get("rung_name") == SYSTEM and row.get("method") == METHOD]
    if len(found) != 1:
        raise ValueError(f"the ladder must carry exactly one {METHOD} row on {SYSTEM}")
    return found[0]


# ------------------------------------------------------------------ static

def predecessor_problems(config: dict) -> list[str]:
    """Everything that is Q18-S2's (so Q18-S1's) must still be."""
    problems = []
    predecessor = q18s2.load_config()
    for key in ("estimator", "grouping", "allocation_diagnostic"):
        if config[key] != predecessor[key]:
            problems.append(f"{key} must be Q18-S2's, unchanged")
    if config["fixed_step_validation"] != predecessor["derivative_validation"]:
        problems.append("fixed_step_validation must be Q18-S1's rule as Q18-S2 froze it")
    statistic = config["statistic"]
    for key in ("name", "definition", "max_ratio", "uniform_definition", "derivation"):
        if statistic.get(key) != predecessor["statistic"][key]:
            problems.append(f"statistic {key} must be Q18-S2's")
    if config["system"] != predecessor["system"]:
        problems.append("the system must be Q18-S2's")
    rule = config["derivative_validation"]
    fixed = predecessor["derivative_validation"]
    if (rule.get("directions") != fixed["directions"] or rule.get("seed") != fixed["seed"]
            or rule.get("direction_definitions") != fixed["direction_definitions"]
            or rule.get("relative_tolerance") != fixed["relative_tolerance"]):
        problems.append("the eighty-digit validator must keep Q18-S1's directions, seed and "
                        "relative tolerance")
    return problems


def rule_problems(config: dict) -> list[str]:
    problems = []
    rule = config["derivative_validation"]
    if rule.get("decimal_digits") != krylov.DECIMAL_DIGITS or rule.get(
            "step") != krylov.DECIMAL_STEP:
        problems.append("the eighty-digit validator's precision and step must be the helper's")
    if config["rank_margin"].get("factor") != krylov.RANK_MARGIN:
        problems.append("the rank margin must be the helper's")
    certification = config["certification"]
    if (certification.get("variance_agreement") != krylov.VARIANCE_AGREEMENT
            or certification.get("energy_agreement") != krylov.ENERGY_AGREEMENT):
        problems.append("the independent-route tolerances must be the helper's")
    if (certification.get("rounding_ulps") != screen.SPECTRAL_ROUNDING_ULPS
            or certification.get("degeneracy_tolerance") != screen.DEGENERACY_TOLERANCE):
        problems.append("rounding and degeneracy tolerances must be the helper's")
    decision = config["decision_rule"]
    if decision.get("prefix_statuses") != list(krylov.STATUSES):
        problems.append("prefix statuses must be the helper's ladder")
    if decision.get("unpriced_statuses") != list(krylov.UNPRICED):
        problems.append("unpriced statuses must be the helper's")
    if decision.get("oracle_verdicts") != list(krylov.ORACLE_VERDICTS):
        problems.append("oracle verdicts must be the helper's")
    if decision.get("practical_statuses") != list(krylov.PRACTICAL_STATUSES):
        problems.append("practical statuses must be the helper's")
    if decision.get("final_verdicts") != list(krylov.FINAL_VERDICTS):
        problems.append("final verdicts must be the helper's")
    if set(config["consequences"]) != set(krylov.FINAL_VERDICTS):
        problems.append("every final verdict needs exactly one consequence")
    if config["allocation_diagnostic"].get("decision_role") != "diagnostic_only":
        problems.append("the oracle integer allocation must stay outside the decision")
    if config["lineage_check"].get("relative_tolerance") != FLOAT_TOLERANCE or config[
            "lineage_check"].get("neyman_relative_tolerance") != NEYMAN_TOLERANCE:
        problems.append("lineage tolerances must be Q18-S2's")
    return problems


def _int(value, low) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def _fraction(value) -> bool:
    return (isinstance(value, float) and not isinstance(value, bool)
            and math.isfinite(value) and 0.0 < value <= 1.0)


def practical_problems(config: dict) -> list[str]:
    """The practical-allocation rule is complete, typed and in range."""
    rule = config["practical_allocation"]
    problems = []
    for key, low in (("seed", 0), ("replicate_count", 2), ("minimum_pilot_shots_per_setting", 2),
                     ("minimum_shots_per_setting", 1)):
        if not _int(rule.get(key), low):
            problems.append(f"practical_allocation {key} must be an integer >= {low}")
    for key in ("standard_error_fraction_of_sigma", "pilot_fraction"):
        if not _fraction(rule.get(key)):
            problems.append(f"practical_allocation {key} must be a float in (0, 1]")
    if rule.get("max_ratio") != config["statistic"]["max_ratio"]:
        problems.append("the practical threshold must be the oracle's")
    if rule.get("protocol") != config["grouping"]["declared_protocol"]:
        problems.append("the practical stage runs under the declared protocol")
    if rule.get("standard_error_fraction_of_sigma") != config["allocation_diagnostic"][
            "standard_error_fraction_of_sigma"]:
        problems.append("the practical target must be the oracle diagnostic's sigma/100")
    if rule.get("minimum_shots_per_setting") != config["allocation_diagnostic"][
            "minimum_shots_per_setting"] or rule.get(
            "minimum_pilot_shots_per_setting") != config["allocation_diagnostic"][
            "pilot_shots_per_setting"]:
        problems.append("the practical floors must be Q18-S1's")
    family = rule.get("confidence_family", {})
    if (family.get("two_sided_z") != krylov.Z_TWO_SIDED
            or family.get("one_sided_z") != krylov.Z_ONE_SIDED):
        problems.append("the confidence family's quantiles must be the helper's")
    validation = rule.get("validation", {})
    if not (_int(validation.get("max_failures"), 0)
            and _fraction(validation.get("min_coverage"))
            and isinstance(validation.get("max_z_spread"), float)
            and validation["max_z_spread"] >= 1.0):
        problems.append("practical validation thresholds are malformed")
    return problems


def basis_problems(config: dict, row: dict | None = None) -> list[str]:
    problems = []
    row = ladder_row() if row is None else row
    basis = config["basis"]
    if basis.get("source") != str(LADDER.relative_to(ROOT)) or basis.get("method") != METHOD:
        problems.append("the basis must be the committed ladder's Krylov row")
    expected = ["I", *(f"H^{k}" for k in range(1, KRYLOV_ORDER + 1))]
    if basis.get("labels") != expected or row.get("labels") != expected:
        problems.append("the basis labels must be the ladder's I, H^1 ... H^8")
    if basis.get("krylov_order") != KRYLOV_ORDER or basis.get("prefixes") != PREFIXES:
        problems.append("the basis is the order-8 sequence and every prefix of it")
    for key, field in (("committed_energy", "energy"),
                       ("committed_retained_rank", "retained_rank"),
                       ("committed_condition_number", "condition_number")):
        if basis.get(key) != row.get(field):
            problems.append(f"basis {key} differs from the committed ladder row")
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
                    row: dict | None = None) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    if config["question_id"] != "Q18-S3":
        problems.append("wrong question identifier")
    if config["contains_results"] is not False:
        problems.append("contains_results must be false")
    problems += [f"result-shaped key {key}" for key in q18s2._result_keys(config)]
    problems += [f"result-shaped key {key}" for key in _own_result_keys(config)]
    boundary = config["claim_boundary"].lower()
    problems += [f"claim boundary is tensed: {phrase!r}" for phrase in UNTENSED
                 if phrase in boundary]
    evidence = config["evidence"]
    if evidence.get("design_status") != "declared_before_execution" or evidence.get(
            "basis_selection") != "post_hoc":
        problems.append("evidence must label a pre-execution declaration with post-hoc "
                        "basis selection")
    if evidence.get("label") != "asymptotic_oracle_then_sampled":
        problems.append("evidence label must be asymptotic_oracle_then_sampled")
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
    problems += rule_problems(config)
    problems += practical_problems(config)
    problems += basis_problems(config, row)
    problems += lineage_problems(config, record_exists=record_exists)
    return problems


# ------------------------------------------------------------------ structural

def system_model():
    return q18s2.system_model()


def first_moment_structure(model, order: int, prefixes) -> dict:
    """Spectrum, premises and every prefix's ``(S, H)`` solve on ``model``.

    Each prefix is solved on its first-moment pencil alone;
    ``SecondMomentBank`` is never constructed, so no ``H^2`` row is paired
    with the reference.
    """
    from clifford_qc.subspace.linalg import solve_projected

    spectrum = screen.sector_spectrum(model)
    bank = krylov.krylov_bank(model, order)
    generators = [bank.generator(i) for i in range(len(bank))]
    premises = screen.sector_premises(model, [generators])
    limit = -1
    for p, power in enumerate(bank.powers):
        if max(abs(complex(c)) for c in power.terms.values()) >= 2.0 ** 53:
            break
        limit = p
    premises["exact_power_limit"] = limit
    basis = {}
    for size in prefixes:
        S, H = bank.matrices(list(range(size)))
        result = solve_projected(S, H)
        margin = krylov.rank_margin(result, size)
        energy = float(result.ground_energy)
        basis[str(size)] = {
            "ground_energy": energy,
            "effective_rank": margin["effective_rank"],
            "smallest_overlap_eigenvalue": margin["smallest_overlap_eigenvalue"],
            "overlap_condition_number": margin["overlap_condition_number"],
            "retains_whole_block": margin["full_rank"],
            "passes_rank_margin": margin["rank_stable"],
            "below_midpoint": energy < spectrum["midpoint"],
            "highest_power": 2 * size,
        }
    return {"spectrum": spectrum, "premises": premises, "basis": basis, "bank": bank}


def structural_quantities(config: dict | None = None) -> dict:
    """Every frozen number, from committed inputs. First moments only."""
    out = first_moment_structure(system_model(), KRYLOV_ORDER, PREFIXES)
    bank = out.pop("bank")
    row = ladder_row()
    final = bank.solve(list(range(len(bank))))
    out["ladder"] = {
        "energy_reproduces": abs(float(final.ground_energy) - row["energy"]) <= FLOAT_TOLERANCE,
        "retained_rank_reproduces": int(final.effective_rank) == row["retained_rank"],
        "ladder_condition_number": float(final.condition_number),
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
    if isinstance(declared, bool) or isinstance(got, bool):
        return [] if declared is got else [f"{path}: declared {declared!r}, recomputed {got!r}"]
    if (isinstance(declared, float) or isinstance(got, float)) and None not in (declared, got):
        rel = SPECTRAL_TOLERANCE if path.rsplit(".", 1)[-1] in SPECTRAL_FIELDS else FLOAT_TOLERANCE
        if not math.isclose(float(declared), float(got), rel_tol=rel, abs_tol=1e-12):
            return [f"{path}: declared {declared!r}, recomputed {got!r}"]
        return []
    return [] if declared == got else [f"{path}: declared {declared!r}, recomputed {got!r}"]


def premise_problems(quantities: dict, config: dict | None = None) -> list[str]:
    problems = []
    premises, spectrum = quantities["premises"], quantities["spectrum"]
    if (premises["reference_image_sector_leakage"] > 1e-12
            or premises["hamiltonian_sector_leakage"] > 1e-12):
        problems.append("a Krylov vector or the Hamiltonian leaves the (N, S_z) sector; the "
                        "sector spectrum is not the reference")
    if premises["spin_parity_sector_violations"]:
        problems.append("a word leaves the spin-parity sector; the storage bound fails")
    anchor = json.loads(H2_PREFLIGHT.read_text(encoding="utf-8"))["clauses"]["storage"][
        "max_coefficient_occurrences"]
    if premises["coefficient_occurrence_bound"] > anchor:
        problems.append("the storage bound exceeds the Phase 2M anchor")
    if spectrum["ground_degeneracy"] != 1:
        problems.append("the sector ground state is degenerate")
    priceable = [int(size) for size, row in quantities["basis"].items()
                 if row["passes_rank_margin"]]
    if priceable and 2 * max(priceable) > premises["exact_power_limit"]:
        problems.append("a rank-stable prefix's second moments need a power whose "
                        "coefficients are not exact in double precision")
    ladder = quantities["ladder"]
    if not ladder["energy_reproduces"] or not ladder["retained_rank_reproduces"]:
        problems.append("the order-8 Krylov solve does not reproduce the committed ladder row")
    if config is not None and not math.isclose(
            ladder["ladder_condition_number"], config["basis"]["committed_condition_number"],
            rel_tol=SPECTRAL_TOLERANCE):
        problems.append("the order-8 condition number does not reproduce the ladder's")
    return problems


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None) -> list[str]:
    got = structural_quantities(config)
    if computed is not None:
        computed.update(got)
    problems = _compare(config["measured_before_freezing"], got)
    problems += premise_problems(got, config)
    if notes is not None:
        spectrum = got["spectrum"]
        notes.append(f"  sector: E0={spectrum['ground_energy']:.12f} "
                     f"E1={spectrum['first_excited_energy']:.12f} "
                     f"midpoint={spectrum['midpoint']:.12f}")
        stable = [s for s, r in got["basis"].items() if r["passes_rank_margin"]]
        below = [s for s, r in got["basis"].items() if r["below_midpoint"]]
        notes.append(f"  Krylov prefixes: rank-stable {stable}; below the midpoint {below}")
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
        print("OK Q18-S3 declaration: Q18-S2's estimator, groupings and threshold, the "
              "ladder's Krylov prefixes, rank margin, eighty-digit validator, conditional "
              "practical allocation, post-hoc basis selection")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
