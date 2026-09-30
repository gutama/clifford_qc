"""Gate the result-free Phase 15 preregistration: energy-variance extrapolation.

The declaration asks one question per bank in its domain. Each frozen basis
is grown in a fixed order, so its prefixes form a trajectory of Ritz ground
energies ``E_M`` and variances ``sigma^2_M``. Does a straight line through the
last three points, read at ``sigma^2 = 0``, at least halve the final Ritz
energy's error against the exact sector ground energy? This gate checks that
the declaration can answer that as frozen, and that it was frozen before any
answer existed:

* completeness, and the absence of any result-shaped value;
* the SHA-256 of every input the run reads: the FCIDUMPs and their provenance,
  the mapping-axis config that fixes each basis and its order, and the
  committed SecondMomentBank validation record that supplies each final
  variance. The second-moment and eigensolver code stay bound until a record
  exists;
* an exhaustive status ladder and a total verdict rule;
* the clauses against each other, the Phase 16B v3 lesson: the window fits
  every required basis, the improvement factor lies strictly inside (0, 1),
  every required bank has an error to improve, and the reachable verdicts
  are the ones the config claims;
* a recomputation, from committed inputs, of every number in
  ``measured_before_freezing``. That covers the exact sector ground and first
  excited energies and their gap; the final Ritz energy from ``(S, H)``,
  which must match the validation record; and the final variance, read from
  that record. It then applies the domain criterion, ``sigma_f <= gap / 2``,
  which decides the required banks;
* commit order once a record exists, and the revision log.

It forms no prefix second moment, computes no intermediate variance, fits
nothing, and computes no status or verdict.

    python benchmarks/check_phase15_extrapolation_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_preregistration as preflight_gate
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as preflight_gate

CONFIG = HERE / "configs" / "phase15_variance_extrapolation.json"
RECORD = HERE / "reference_results" / "phase15_variance_extrapolation.json"
VALIDATION = HERE / "reference_results" / "second_moment_validation.json"
MAPPING = HERE / "configs" / "mapping_axis.json"
SCHEMA = "clifford_qc.phase15_variance_extrapolation_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "trajectory", "fit", "domain", "decision_rule",
    "consequences", "banks", "measured_before_freezing", "required_banks",
    "diagnostic_banks", "reachable_verdicts", "deterministic_checks", "evidence",
    "excluded", "exploratory_disclosure", "prespecified_followup",
    "implementation_lineage", "record_requirements", "revisions",
)
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/reference_results/second_moment_validation.json",
)
BOUND_IMPLEMENTATIONS = ("clifford_qc/subspace/second_moment.py",
                         "clifford_qc/subspace/linalg.py")
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "bank_status",
    "bank_statuses", "extrapolated_energy", "extrapolated_error", "prefix_energies",
    "prefix_variances", "fit_slope", "fit_intercept", "gain", "conclusion",
    "elapsed_seconds",
})
BANK_STATUSES = ("IMPROVES", "NO_GAIN", "WORSENS", "NOT_EXTRAPOLABLE")
VERDICTS = ("GO", "NO_GO", "CONDITIONAL")
UNTENSED = ("has been", "was run", "were run", "no fit has been")

_INT_FIELDS = ("n_qubits", "basis_size", "sector_dimension")
_FLOAT_FIELDS = ("exact_ground_energy", "exact_first_excited_energy", "exact_gap",
                 "final_ritz_energy", "final_error", "final_variance", "final_residual")
_BOOL_FIELDS = ("in_domain", "exact_matches_provenance", "final_matches_validation")
MEASURED_FIELDS = _INT_FIELDS + _FLOAT_FIELDS + _BOOL_FIELDS


# ------------------------------------------------------------------ loading

def _walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def load_config(path: Path = CONFIG) -> dict:
    """Read the declaration, refusing one that carries a result."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for where, key, value in _walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            raise ValueError(f"config carries a result field {where}.{key}")
    if config.get("contains_results") is not False:
        raise ValueError("contains_results must be literally false")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ------------------------------------------------------------------ the rule

def extrapolate(variances, energies, *, resolved, scales=None) -> dict:
    """The frozen fit: least squares ``E = a + b sigma^2`` over the window.

    Extrapolable only when every window variance is resolved, the variances
    are distinguishable, and the slope is positive, as
    ``E - E_0 ~ sigma^2/gap`` requires. "Distinguishable" means their spread
    exceeds the resolution times the largest cancellation scale among them
    (revision 1). Each variance carries rounding of about that size, so a
    smaller spread yields a slope made of rounding. Anything else is a
    legitimate outcome, NOT_EXTRAPOLABLE, not an error.
    """
    import numpy as np

    from clifford_qc.subspace.second_moment import RESOLUTION

    x = np.asarray(variances, dtype=float)
    y = np.asarray(energies, dtype=float)
    if not all(resolved):
        return {"extrapolable": False, "reason": "a window variance is not resolved"}
    floor = RESOLUTION * max(scales) if scales else 0.0
    if float(np.max(x) - np.min(x)) <= floor:
        return {"extrapolable": False,
                "reason": "the window variances differ only by rounding"}
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = intercept + slope * x
    out = {"intercept": float(intercept), "slope": float(slope),
           "residual_rms": float(np.sqrt(np.mean((y - fitted) ** 2)))}
    if slope <= 0.0:
        return {**out, "extrapolable": False, "reason": "the slope is not positive"}
    return {**out, "extrapolable": True, "reason": None}


def status_of(extrapolable: bool, extrapolated_error: float | None,
              final_error: float, factor: float) -> str:
    """The frozen ladder, read on absolute errors against the exact energy."""
    if not extrapolable:
        return "NOT_EXTRAPOLABLE"
    if abs(extrapolated_error) <= factor * abs(final_error):
        return "IMPROVES"
    if abs(extrapolated_error) <= abs(final_error):
        return "NO_GAIN"
    return "WORSENS"


def verdict_of(statuses) -> str:
    statuses = tuple(statuses)
    if "INVALID" in statuses:
        return "INVALID"
    improves = sum(value == "IMPROVES" for value in statuses)
    if improves == len(statuses):
        return "GO"
    return "NO_GO" if improves == 0 else "CONDITIONAL"


def reachable_verdicts(required_count: int) -> set[str]:
    reached = set()
    for statuses in itertools.product(BANK_STATUSES, repeat=required_count):
        reached.add(verdict_of(statuses))
    return reached


def ladder_problems(config: dict) -> list[str]:
    problems = []
    rule = config["decision_rule"]
    order = [row["status"] for row in rule["bank_status_order"]]
    if order != list(BANK_STATUSES):
        problems.append(f"status ladder order moved: {order}")
    factor = float(config["fit"]["improvement_factor"])
    reached = {status_of(False, None, 1.0, factor)}
    for error in (0.0, 0.5 * (1 + factor), 2.0):
        reached.add(status_of(True, error, 1.0, factor))
    if reached != set(BANK_STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(BANK_STATUSES) - reached}")
    if rule.get("combination_rule") != (
            "every required bank IMPROVES gives GO; no required bank IMPROVES gives "
            "NO_GO; anything else is CONDITIONAL; any INVALID required bank gives INVALID"):
        problems.append("combination_rule text moved from the rule this gate enforces")
    if set(config["consequences"]) != set(VERDICTS) | {"INVALID"}:
        problems.append("every verdict, INVALID included, must declare its consequence")
    return problems


def population_problems(config: dict) -> list[str]:
    problems = []
    systems = list(config["banks"]["systems"])
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    validated = sorted(name for name, entry in validation["banks"].items()
                       if entry["status"] == "VALIDATED")
    if sorted(systems) != validated:
        problems.append(f"the banks must be the validated SecondMomentBank banks "
                        f"{validated}, not {sorted(systems)}")
    if set(config["measured_before_freezing"]) != set(systems):
        problems.append("measured_before_freezing must cover exactly the declared banks")
    required, diagnostic = config["required_banks"], config["diagnostic_banks"]
    if sorted(required + diagnostic) != sorted(systems) or set(required) & set(diagnostic):
        problems.append("required and diagnostic banks must partition the declared banks")
    return problems


def consistency_problems(config: dict) -> list[str]:
    """The declaration's clauses against each other (the Phase 16B v3 lesson)."""
    problems = []
    fit = config["fit"]
    window = fit["window"]
    if not isinstance(window, int) or window < 3:
        problems.append("the window must be an integer of at least three, so the "
                        "line has a residual to report")
    factor = fit["improvement_factor"]
    if not 0.0 < float(factor) < 1.0:
        problems.append("the improvement factor must lie strictly inside (0, 1)")
    measured = config["measured_before_freezing"]
    in_domain = sorted(name for name, row in measured.items() if row["in_domain"])
    if sorted(config["required_banks"]) != in_domain:
        problems.append(f"required_banks must be the banks in the domain: {in_domain}")
    if not in_domain:
        problems.append("no bank is in the domain, so the preregistration decides nothing")
    for name in config["required_banks"]:
        row = measured.get(name)
        if row is None:
            continue  # population_problems reports the missing bank
        if isinstance(window, int) and row["basis_size"] < window:
            problems.append(f"{name}: a basis of {row['basis_size']} cannot fill the window")
        if not row["final_error"] > 1e-10:
            problems.append(f"{name}: the final Ritz energy has no error to improve")
    reachable = reachable_verdicts(len(config["required_banks"]))
    if sorted(config["reachable_verdicts"]) != sorted(reachable):
        problems.append(f"reachable_verdicts must be {sorted(reachable)}")
    return problems


def claim_problems(config: dict) -> list[str]:
    problems = []
    for phrase in UNTENSED:
        if phrase in config["claim_boundary"]:
            problems.append(f"claim boundary is tensed ({phrase!r})")
    if config["record_requirements"].get("must_carry", {}).get(
            "quantum_advantage_claim") is not False:
        problems.append("records must carry quantum_advantage_claim: false")
    if config["prespecified_followup"].get("permitted") != "none":
        problems.append("this declaration permits no follow-up; say so literally")
    if config["evidence"].get("label") != "exact_oracle":
        problems.append("the evidence label must be exact_oracle: the verdict reads "
                        "an exact energy no device run has")
    return problems


def lineage_problems(config: dict, *, record_exists: bool | None = None) -> list[str]:
    """Inputs stay bound; the implementation is bound until a record names it."""
    problems = []
    record_exists = RECORD.exists() if record_exists is None else record_exists
    lineage = config["implementation_lineage"]
    for group in ("inputs", "implementation"):
        if group == "implementation" and record_exists:
            continue
        for row in lineage[group]:
            path = ROOT / row["path"]
            if not path.is_file():
                problems.append(f"lineage file missing: {row['path']}")
            elif _sha256(path) != row["sha256"]:
                problems.append(f"declared sha256 drifted for {row['path']}")
    inputs = {row["path"] for row in lineage["inputs"]}
    implementation = {row["path"] for row in lineage["implementation"]}
    for path in BOUND_INPUTS:
        if path not in inputs:
            problems.append(f"input {path} must be bound in implementation_lineage")
    for path in BOUND_IMPLEMENTATIONS:
        if path not in implementation:
            problems.append(f"implementation {path} must be bound in implementation_lineage")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    by_key = {spec["key"]: spec for spec in mapping["systems"]}
    for name in config["banks"]["systems"]:
        for key in ("source", "provenance"):
            value = by_key.get(name, {}).get(key)
            if value is not None and value not in inputs:
                problems.append(f"{name}: {key} is not bound in implementation_lineage")
    return problems


def revision_problems(config: dict) -> list[str]:
    problems = []
    revisions = config["revisions"]
    if not revisions or [row.get("revision") for row in revisions] != list(
            range(len(revisions))):
        return ["revisions must be numbered 0, 1, 2, ... in order"]
    for row in revisions[1:]:
        if not str(row.get("prefix_variances_at_revision", "")).startswith("none"):
            problems.append(f"revision {row['revision']} does not state that no prefix "
                            "variance existed when it was made")
        if not row.get("changes"):
            problems.append(f"revision {row['revision']} lists no changes")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    for check in (ladder_problems, population_problems, consistency_problems,
                  claim_problems, lineage_problems, revision_problems):
        problems.extend(check(config))
    return problems


# ------------------------------------------------------------------ structural

def sector_spectrum(model, count: int = 2):
    """The lowest ``count`` eigenvalues of the reference's ``(N, S_z)`` sector."""
    import numpy as np

    from clifford_qc.backends import SectorStatevectorBackend

    metadata = model.metadata
    backend = SectorStatevectorBackend(model.n, metadata["n_electrons"], metadata["sz"],
                                       spin_ordering=metadata["spin_convention"])
    operator = backend.operator(model.hamiltonian)
    dense = np.column_stack([operator.matvec(column)
                             for column in np.eye(backend.dimension)])
    values = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
    return [float(v) for v in values[:count]], backend.dimension


def quantities_for(name: str, model, selected, validation_root: dict,
                   provenance_fci: float | None) -> dict:
    """Every frozen number of one bank. No prefix and no second moment is formed."""
    (ground, first), dimension = sector_spectrum(model)
    bank = preflight_gate.first_moment_bank(model, selected)
    final = float(bank.solve().energies[0])
    variance = float(validation_root["variance"])
    residual = math.sqrt(max(variance, 0.0))
    gap = first - ground
    return {
        "n_qubits": model.n,
        "basis_size": len(selected),
        "sector_dimension": int(dimension),
        "exact_ground_energy": ground,
        "exact_first_excited_energy": first,
        "exact_gap": gap,
        "final_ritz_energy": final,
        "final_error": final - ground,
        "final_variance": variance,
        "final_residual": residual,
        "in_domain": residual <= 0.5 * gap,
        "exact_matches_provenance": (provenance_fci is None
                                     or abs(ground - provenance_fci) <= 1e-8),
        "final_matches_validation": _close(final, validation_root["energy"], abs_=1e-10),
    }


def structural_quantities(name: str) -> dict:
    model, selected = preflight_gate.bank_inputs(name)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    spec = next(s for s in mapping["systems"] if s["key"] == name)
    fci = None
    if spec.get("provenance"):
        provenance = json.loads((ROOT / spec["provenance"]).read_text(encoding="utf-8"))
        fci = float(provenance["reference_energies"]["fci"])
    return quantities_for(name, model, selected, validation["banks"][name]["roots"][0], fci)


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None) -> list[str]:
    notes = [] if notes is None else notes
    problems = []
    for name in config["banks"]["systems"]:
        declared = config["measured_before_freezing"][name]
        got = structural_quantities(name)
        if computed is not None:
            computed[name] = got
        for key in _INT_FIELDS + _BOOL_FIELDS:
            if declared.get(key) != got[key]:
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        for key in _FLOAT_FIELDS:
            if not _close(declared.get(key, float("nan")), got[key], rel=1e-9, abs_=1e-10):
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        if not got["exact_matches_provenance"]:
            problems.append(f"{name}: the sector ground energy is not the provenance FCI")
        if not got["final_matches_validation"]:
            problems.append(f"{name}: the final Ritz energy is not the validation record's")
        notes.append(f"  {name}: M={got['basis_size']} gap={got['exact_gap']:.4g} "
                     f"sigma_f={got['final_residual']:.4g} "
                     f"{'in' if got['in_domain'] else 'outside'} the domain")
    return problems


# ------------------------------------------------------------------ order

def commit_order_problems(notes: list[str]) -> list[str]:
    """The config's last change must strictly precede the record's first commit."""
    if not RECORD.exists():
        notes.append("  commit order: no record yet, nothing to order against")
        return []
    config_commit = preflight_gate._last_commit(CONFIG)
    record_commit = preflight_gate._first_commit(RECORD)
    if config_commit is None or record_commit is None:
        notes.append("  commit order: SKIP (not committed yet, or no git history)")
        return []
    if {config_commit, record_commit} & preflight_gate._shallow_boundary():
        notes.append("  commit order: SKIP (the clone is too shallow to reach "
                     "the commits that order them)")
        return []
    if config_commit == record_commit:
        return ["the config's last change and the record share a commit"]
    import subprocess
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config_commit, record_commit],
        cwd=ROOT, capture_output=True, timeout=30)
    if ancestor.returncode != 0:
        return [f"config commit {config_commit[:12]} is not an ancestor of record "
                f"commit {record_commit[:12]}"]
    notes.append(f"  commit order: config's last change {config_commit[:12]} precedes "
                 f"record {record_commit[:12]}")
    return []


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    notes: list[str] = []
    problems = static_problems(config)
    if not problems:
        problems.extend(structural_problems(config, notes))
        problems.extend(commit_order_problems(notes))
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print("OK Phase 15 extrapolation preregistration is complete, fit-free, jointly "
          "satisfiable, and bound to its inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
