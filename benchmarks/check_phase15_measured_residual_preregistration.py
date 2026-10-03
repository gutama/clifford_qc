"""Gate the result-free Phase 15 preregistration: the measured-residual cost.

The declaration asks one question per bank. Estimate the ground Ritz root's
residual norm ``sigma`` from shared QWC word means, to the same standard error
as its energy. Does that cost at most ten times the energy's shots? The
statistic is a ratio of asymptotic shot counts at matched precision,

    R = (G_U * V_sigma2) / (4 sigma^2 * G_SH * V_E),

where ``G`` counts settings and ``V`` is a linearized estimator's variance at
one shot per setting. This gate checks that the declaration can answer that
as frozen, and that it was frozen before any answer existed:

* completeness, and the absence of any result-shaped value;
* the SHA-256 of every input the run reads, and of the package code that
  defines a row, a grouping, a Ritz solve and a group variance, until a record
  names its own commit;
* an exhaustive status ladder and a total verdict rule;
* the clauses against each other: a threshold above one, a protocol for every
  bank, an alternative protocol exactly where the declared one is the
  quadratic greedy, and the reachable verdicts the config claims;
* a recomputation, from committed inputs, of every number in
  ``measured_before_freezing``. That covers the basis, its rank and Ritz gap,
  the ground energy, which must equal the validation record's, and the
  ``(S, H)`` word universe. It regroups that universe under the declared
  protocol, and requires the setting count to equal the mapping-axis record's.
  It reads the variance and the combined word universe from their committed
  records;
* commit order once a record exists, and the revision log.

It also holds the declared estimator: the second-moment row block, the
nonlinear residual pipeline, its first-order functional, and the
finite-difference check. The tests exercise them on the undeclared Hubbard
dimer. On a declared bank this gate forms no second-moment row, no residual
functional, no grouping of the combined universe and no group variance, and
it computes no ratio, status or verdict.

    python benchmarks/check_phase15_measured_residual_preregistration.py

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

CONFIG = HERE / "configs" / "phase15_measured_residual_preflight.json"
RECORD = HERE / "reference_results" / "phase15_measured_residual_preflight.json"
VALIDATION = HERE / "reference_results" / "second_moment_validation.json"
PREFLIGHT = HERE / "reference_results" / "phase15_h2_preflight.json"
LEDGER = HERE / "reference_results" / "bank_storage_ledger.json"
MAPPING = HERE / "configs" / "mapping_axis.json"
MAPPING_RECORD = HERE / "reference_results" / "mapping_axis.json"
SCHEMA = "clifford_qc.phase15_measured_residual_preflight_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "estimand", "linearization", "grouping",
    "statistic", "decision_rule", "consequences", "banks", "measured_before_freezing",
    "reachable_verdicts", "deterministic_checks", "evidence", "excluded_from_cost",
    "exploratory_disclosure", "prespecified_followup", "implementation_lineage",
    "record_requirements", "revisions",
)
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/reference_results/mapping_axis.json",
    "benchmarks/reference_results/second_moment_validation.json",
    "benchmarks/reference_results/phase15_h2_preflight.json",
    "benchmarks/reference_results/bank_storage_ledger.json",
)
BOUND_IMPLEMENTATIONS = (
    "clifford_qc/multivector.py",
    "clifford_qc/pauli_kernel.py",
    "clifford_qc/subspace/projection.py",
    "clifford_qc/subspace/second_moment.py",
    "clifford_qc/subspace/linalg.py",
    "clifford_qc/measurement/grouping.py",
    "benchmarks/run_mapping_axis.py",
)
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "bank_status",
    "bank_statuses", "cost_ratio", "cost_ratios", "residual_settings",
    "residual_variance_one_shot", "energy_variance_one_shot", "neyman_ratio",
    "group_variances", "conclusion", "elapsed_seconds",
})
BANK_STATUSES = ("AFFORDABLE", "PROHIBITIVE", "GROUPING_SENSITIVE")
VERDICTS = ("FULL", "RESTRICTED", "NONE")
PROTOCOLS = ("qwc_groups", "qwc_basis_cover")
UNTENSED = ("has been", "was run", "were run", "no ratio has been")
COMBINATION_RULE = (
    "every bank AFFORDABLE gives FULL; no bank AFFORDABLE gives NONE; anything "
    "else is RESTRICTED; any INVALID bank gives INVALID")

_INT_FIELDS = ("n_qubits", "basis_size", "effective_rank", "sh_word_universe",
               "combined_word_universe", "energy_settings")
_FLOAT_FIELDS = ("ground_energy", "subspace_gap", "variance", "residual_norm",
                 "cancellation_scale", "matched_precision_half")
_STR_FIELDS = ("energy_grouping_protocol", "combined_words_sha256")
_OPTIONAL_FIELDS = ("alternative_protocol", "mapping_axis_linearized_energy_variance")
_BOOL_FIELDS = ("full_rank", "ground_root_nondegenerate", "variance_resolved",
                "ground_energy_matches_validation", "sh_universe_matches_records",
                "energy_settings_match_mapping_axis")
MEASURED_FIELDS = (_INT_FIELDS + _FLOAT_FIELDS + _STR_FIELDS + _OPTIONAL_FIELDS
                   + _BOOL_FIELDS)

# The ground root must stand clear of the next Ritz root by this much, relative
# to max(1, |E_0|): the first-order Ritz-vector term divides by that gap.
GAP_FLOOR = 1e-6


# ------------------------------------------------------------------ loading

def load_config(path: Path = CONFIG) -> dict:
    """Read the declaration, refusing one that carries a result."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for where, key, value in preflight_gate._walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            raise ValueError(f"config carries a result field {where}.{key}")
    if config.get("contains_results") is not False:
        raise ValueError("contains_results must be literally false")
    return config


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ------------------------------------------------------------------ the rule

def cost_ratio(*, residual_settings: int, residual_variance: float,
               energy_settings: int, energy_variance: float, variance: float) -> float:
    """Shots for ``sigma`` over shots for ``E`` at one matched standard error.

    Uniform shots per setting make a linearized estimator's variance
    ``V / N`` at ``N`` shots per setting, so a standard error ``s`` costs
    ``G V / s^2`` shots in total. ``sigma`` is read from ``sigma^2`` through
    ``d sigma = d sigma^2 / (2 sigma)``, so its variance is ``V_sigma2 /
    (4 sigma^2)``. The common ``s^2`` cancels.
    """
    if variance <= 0.0:
        raise ValueError("a residual norm needs a positive variance to linearize")
    if energy_settings <= 0 or energy_variance <= 0.0:
        raise ValueError("the energy denominator must be positive")
    return (residual_settings * residual_variance) / (
        4.0 * variance * energy_settings * energy_variance)


def status_of(declared_ratio: float, alternative_ratio: float | None,
              threshold: float) -> str:
    """The frozen ladder. The alternative protocol exists only where the
    declared one is the greedy, and a disagreement between them is not
    affordability."""
    declared = declared_ratio <= threshold
    if alternative_ratio is None:
        return "AFFORDABLE" if declared else "PROHIBITIVE"
    alternative = alternative_ratio <= threshold
    if declared and alternative:
        return "AFFORDABLE"
    if not declared and not alternative:
        return "PROHIBITIVE"
    return "GROUPING_SENSITIVE"


def verdict_of(statuses) -> str:
    statuses = tuple(statuses)
    if "INVALID" in statuses:
        return "INVALID"
    affordable = sum(value == "AFFORDABLE" for value in statuses)
    if affordable == len(statuses):
        return "FULL"
    return "NONE" if affordable == 0 else "RESTRICTED"


def reachable_verdicts(alternatives) -> set[str]:
    """Verdicts the ladder can reach, given which banks carry an alternative."""
    choices = [BANK_STATUSES if alternative else BANK_STATUSES[:2]
               for alternative in alternatives]
    return {verdict_of(statuses) for statuses in itertools.product(*choices)}


# ------------------------------------------------------------------ the estimator

def block_rows(bank, second, order):
    """``(a, b) -> (S row, H row, K row)`` word terms over the retained block.

    The upper triangle only, ``a <= b``, in the bank's own product order; the
    lower triangle is the Hermitian conjugate and is never stored.
    """
    rows = {}
    for b, j in enumerate(order):
        for a in range(b + 1):
            i = order[a]
            rows[(a, b)] = (bank.overlap_operator(i, j).terms,
                            bank.element_operator(i, j).terms,
                            second.row(i, j).terms)
    return rows


def row_words(rows) -> set[int]:
    words: set[int] = set()
    for terms in rows.values():
        for part in terms:
            words.update(part)
    return words


def projected_matrices(rows, size: int, means):
    """``(S, H, K)`` reconstructed from word means, Hermitian by construction."""
    import numpy as np

    S = np.zeros((size, size), dtype=complex)
    H = np.zeros((size, size), dtype=complex)
    K = np.zeros((size, size), dtype=complex)
    for (a, b), parts in rows.items():
        for matrix, terms in zip((S, H, K), parts):
            value = sum(complex(c) * means.get(w, 0.0) for w, c in terms.items())
            matrix[a, b] = value
            if a != b:
                matrix[b, a] = value.conjugate()
    return S, H, K


def residual_variance(rows, size: int, means) -> float:
    """The nonlinear estimator: solve the measured pencil, read ``sigma^2``.

    The ground Ritz vector of ``(S, H)`` under the package's default
    thresholds, normalized ``c'Sc = 1``, gives ``sigma^2 = c'Kc - E^2``.
    """
    from clifford_qc.subspace.linalg import solve_projected

    S, H, K = projected_matrices(rows, size, means)
    result = solve_projected(S, H)
    c = result.coefficients[:, 0]
    energy = float(result.energies[0])
    norm = float((c.conj() @ S @ c).real)
    return float((c.conj() @ K @ c).real) / norm - energy * energy


def residual_functional(rows, size: int, means) -> tuple[dict[int, float], float]:
    """First-order weights of ``sigma^2`` in every word mean, and ``sigma^2``.

    With ``c'Sc = 1``, ``Q = c'Kc``, ``B = sum c_i A_i`` and
    ``V = sum v_i A_i``, the Jacobian is the word expansion of

        G = B'(H - E)^2 B - sigma^2 B'B - [V'(H - E)B + B'(H - E)V],

    where ``v = sum_{k>0} c_k (c_k'(K - QS)c) / (E_k - E)`` carries the Ritz
    vector's own movement. Unlike the energy, ``sigma^2`` is not stationary in
    ``c``, so that term is required. ``G`` is built from ``(H - E)``, so a
    constant shift of ``H`` leaves every weight unchanged. The identity
    weight is returned too, under code ``0``; it multiplies a mean that is
    not measured.
    """
    import numpy as np

    from clifford_qc.subspace.linalg import solve_projected

    S, H, K = projected_matrices(rows, size, means)
    result = solve_projected(S, H)
    if result.effective_rank != size:
        raise ValueError("the linearization assumes the whole block is retained")
    C = result.coefficients
    energies = np.asarray(result.energies, dtype=float)
    c, energy = C[:, 0], float(energies[0])
    second = float((c.conj() @ K @ c).real)
    variance = second - energy * energy
    pull = (K - second * S) @ c
    v = np.zeros(size, dtype=complex)
    for k in range(1, C.shape[1]):
        v += C[:, k] * (C[:, k].conj() @ pull) / (energies[k] - energy)
    weights: dict[int, float] = {}
    for (a, b), (s, h, q) in rows.items():
        outer = np.conj(c[a]) * c[b]
        mixed = np.conj(v[a]) * c[b] + np.conj(c[a]) * v[b]
        for word in set(s) | set(h) | set(q):
            sw = complex(s.get(word, 0.0))
            hw = complex(h.get(word, 0.0))
            qw = complex(q.get(word, 0.0))
            value = (outer * (qw - 2.0 * energy * hw + (energy * energy - variance) * sw)
                     - mixed * (hw - energy * sw))
            # a < b pairs with its adjoint row, so the two add to twice the real part.
            weights[word] = weights.get(word, 0.0) + (
                value.real if a == b else 2.0 * value.real)
    return weights, variance


def finite_difference_check(rows, size: int, means, weights, direction, *,
                            step: float, relative_tolerance: float,
                            cancellation_scale: float, resolution: float) -> dict:
    """Central difference of the nonlinear pipeline against ``g . d``.

    The tolerance has a relative part, against ``||g||_2``, and a rounding
    floor: ``sigma^2`` is a difference of numbers of the cancellation scale,
    so each evaluation carries about ``resolution * scale`` of rounding, which
    the difference divides by the step.
    """
    plus = {w: means.get(w, 0.0) + step * direction.get(w, 0.0)
            for w in set(means) | set(direction)}
    minus = {w: means.get(w, 0.0) - step * direction.get(w, 0.0)
             for w in set(means) | set(direction)}
    numeric = (residual_variance(rows, size, plus)
               - residual_variance(rows, size, minus)) / (2.0 * step)
    analytic = sum(weights.get(w, 0.0) * d for w, d in direction.items())
    norm = math.sqrt(sum(value * value for w, value in weights.items() if w != 0))
    tolerance = relative_tolerance * norm + resolution * cancellation_scale / step
    return {"numeric": numeric, "analytic": analytic, "tolerance": tolerance,
            "passes": abs(numeric - analytic) <= tolerance}


def check_directions(weights, sh_words, *, seed: int):
    """The two frozen directions: the functional's own, and a seeded Gaussian
    over the ``(S, H)`` words, where the Ritz-vector term lives."""
    import numpy as np

    live = {w: value for w, value in weights.items() if w != 0 and value != 0.0}
    norm = math.sqrt(sum(value * value for value in live.values()))
    along = {w: value / norm for w, value in live.items()} if norm else {}
    words = sorted(w for w in sh_words if w != 0)
    draw = np.random.default_rng(seed).standard_normal(len(words))
    draw = draw / np.linalg.norm(draw) if len(words) else draw
    return {"functional": along, "sh_gaussian": dict(zip(words, draw.tolist()))}


def reference_means(reference, words) -> dict[int, float]:
    """Exact reference expectations of each word, identity included."""
    from clifford_qc.multivector import MV

    scale = float(2 ** reference.n)
    return {w: float((scale * MV(reference.n, {w: 1.0}).trace_pairing(reference)).real)
            for w in words}


def group_words(words, protocol: str, n: int):
    """The declared QWC partition of a universe on ``n`` qubits, identity excluded.

    Words are listed in ascending code order, as the mapping-axis producer
    lists them, so the partition is the one that record committed.
    """
    from clifford_qc.ir import PauliWord
    from clifford_qc.measurement import qwc_basis_cover, qwc_groups

    listed = [PauliWord(n, code) for code in sorted(words) if code != 0]
    if not listed:
        return []
    if protocol == "qwc_groups":
        return qwc_groups(listed)
    if protocol == "qwc_basis_cover":
        return qwc_basis_cover(listed)
    raise ValueError(f"unsupported grouping protocol {protocol!r}")


def group_variances(reference, groups, weights) -> list[float]:
    """Single-assignment per-setting variances ``Var_psi(sum_g w P)``.

    Computed by the mapping-axis record's own ``_functional_variance``, the
    function that produced its committed energy variances.
    """
    try:
        from benchmarks.run_mapping_axis import _functional_variance
    except ImportError:  # pragma: no cover - script execution
        from run_mapping_axis import _functional_variance

    out = []
    for group in groups:
        live = {word.code: float(weights[word.code]) for word in group
                if word.code in weights and weights[word.code] != 0.0}
        out.append(_functional_variance(reference, live))
    return out


# ------------------------------------------------------------------ static

def ladder_problems(config: dict) -> list[str]:
    problems = []
    rule = config["decision_rule"]
    order = [row["status"] for row in rule["bank_status_order"]]
    if order != list(BANK_STATUSES):
        problems.append(f"status ladder order moved: {order}")
    threshold = float(config["statistic"]["max_ratio"])
    reached = {status_of(r, a, threshold)
               for r in (0.5 * threshold, 2.0 * threshold)
               for a in (None, 0.5 * threshold, 2.0 * threshold)}
    if reached != set(BANK_STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(BANK_STATUSES) - reached}")
    if status_of(threshold, None, threshold) != "AFFORDABLE":
        problems.append("a ratio exactly at the threshold must be affordable")
    if rule.get("combination_rule") != COMBINATION_RULE:
        problems.append("combination_rule text moved from the rule this gate enforces")
    if set(config["consequences"]) != set(VERDICTS) | {"INVALID"}:
        problems.append("every verdict, INVALID included, must declare its consequence")
    return problems


def population_problems(config: dict) -> list[str]:
    problems = []
    systems = list(config["banks"]["systems"])
    validated = sorted(name for name, entry in _json(VALIDATION)["banks"].items()
                       if entry["status"] == "VALIDATED")
    if sorted(systems) != validated:
        problems.append(f"the banks must be the validated SecondMomentBank banks "
                        f"{validated}, not {sorted(systems)}")
    eligible = sorted(name for name, entry in _json(PREFLIGHT)["banks"].items()
                      if entry["decision"]["status"] == "ELIGIBLE")
    if sorted(systems) != eligible:
        problems.append(f"the banks must be the preflight's ELIGIBLE banks {eligible}")
    if set(config["measured_before_freezing"]) != set(systems):
        problems.append("measured_before_freezing must cover exactly the declared banks")
    return problems


def consistency_problems(config: dict) -> list[str]:
    """The declaration's clauses against each other (the Phase 16B v3 lesson)."""
    problems = []
    systems = config["banks"]["systems"]
    threshold = config["statistic"]["max_ratio"]
    if not isinstance(threshold, (int, float)) or not float(threshold) > 1.0:
        problems.append("max_ratio must exceed one: at one the residual may cost "
                        "nothing beyond the energy, which no estimator achieves")
    grouping = config["grouping"]
    declared = grouping["declared_protocol_by_system"]
    alternative = grouping["alternative_protocol_by_system"]
    contract = _json(MAPPING_RECORD)["measurement_contract"]["grouping_protocol_by_system"]
    for name in systems:
        if declared.get(name) not in PROTOCOLS:
            problems.append(f"{name}: no declared grouping protocol")
        elif declared[name] != contract.get(name):
            problems.append(f"{name}: the declared protocol must be the mapping-axis "
                            f"record's {contract.get(name)!r}, so the energy "
                            "denominator reproduces its committed setting count")
        other = alternative.get(name, "missing")
        if other == "missing":
            problems.append(f"{name}: alternative_protocol_by_system must name it, "
                            "null included")
        elif declared.get(name) == "qwc_groups" and other != "qwc_basis_cover":
            problems.append(f"{name}: the greedy needs the scalable cover as its "
                            "alternative, so a status cannot rest on one heuristic")
        elif declared.get(name) == "qwc_basis_cover" and other is not None:
            problems.append(f"{name}: the greedy is quadratic in this universe, so "
                            "the cover has no feasible alternative; declare null")
        row = config["measured_before_freezing"].get(name, {})
        if row and row.get("alternative_protocol") != alternative.get(name):
            problems.append(f"{name}: measured_before_freezing names another "
                            "alternative protocol than the grouping block")
        if row and row.get("energy_grouping_protocol") != declared.get(name):
            problems.append(f"{name}: measured_before_freezing names another "
                            "declared protocol than the grouping block")
    if grouping.get("estimator") != "single_assignment":
        problems.append("the estimator must be single_assignment, as the mapping-axis "
                        "energy variances it reproduces are")
    if grouping.get("allocation") != "uniform":
        problems.append("allocation must be uniform: the ratio's s^2 cancels only "
                        "under one shot count per setting")
    check = config["linearization"]["finite_difference"]
    if not 0.0 < float(check["step"]) < 1.0:
        problems.append("the finite-difference step must lie in (0, 1): word means do")
    if not 0.0 < float(check["relative_tolerance"]) < 1e-3:
        problems.append("the finite-difference tolerance must be positive and below "
                        "1e-3, or it cannot see a dropped Ritz-vector term")
    if check.get("directions") != ["functional", "sh_gaussian"]:
        problems.append("the finite-difference directions moved")
    if not isinstance(check.get("seed"), int):
        problems.append("the Gaussian direction needs an integer seed")
    for name, row in config["measured_before_freezing"].items():
        if not row.get("full_rank"):
            problems.append(f"{name}: the linearization needs the whole block retained")
        if not row.get("ground_root_nondegenerate"):
            problems.append(f"{name}: the Ritz-vector term needs a nondegenerate root")
        if not row.get("variance_resolved"):
            problems.append(f"{name}: an unresolved variance has no norm to linearize")
    alternatives = [alternative.get(name) is not None for name in systems]
    reachable = reachable_verdicts(alternatives)
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
    if config["evidence"].get("label") != "asymptotic":
        problems.append("the evidence label must be asymptotic: first-order variances "
                        "at large shot counts, not a finite-sample statement")
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
    by_key = {spec["key"]: spec for spec in _json(MAPPING)["systems"]}
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
        if not str(row.get("ratios_at_revision", "")).startswith("none"):
            problems.append(f"revision {row['revision']} does not state that no ratio "
                            "existed when it was made")
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

def committed_rows(name: str) -> dict:
    """What the committed records already say about one bank."""
    validation = _json(VALIDATION)["banks"][name]
    preflight = _json(PREFLIGHT)["banks"][name]["counts"]
    digests = _json(PREFLIGHT)["banks"][name]["digests"]
    ledger = preflight_gate.ledger_baseline()[name]
    mapping = next(system for system in _json(MAPPING_RECORD)["systems"]
                   if system["system"] == name)
    jw = next(arm for arm in mapping["arms"] if arm["mapping"] == "jw")
    root = validation["roots"][0]
    return {
        "validation_energy": float(root["energy"]),
        "variance": float(root["variance"]),
        "cancellation_scale": float(root["cancellation_scale"]),
        "variance_resolved": bool(root["resolved"]),
        "preflight_sh_word_universe": int(preflight["sh_word_universe"]),
        "ledger_sh_word_universe": int(ledger["word_universe"]),
        "combined_word_universe": int(preflight["combined_word_universe"]),
        "combined_words_sha256": digests["combined_words_sha256"],
        "mapping_protocol": mapping["grouping_protocol"],
        "mapping_settings": int(jw["measurement"]["settings"]),
        "mapping_energy_variance": jw["accuracy_matched_cost"].get(
            "linearized_variance_ha2_at_one_shot_per_setting"),
    }


def quantities_for(model, selected, committed: dict, protocol: str,
                   alternative: str | None) -> dict:
    """Every frozen number of one bank. First moments only: no second-moment
    row, no residual functional, no grouping of the combined universe."""
    bank = preflight_gate.first_moment_bank(model, selected)
    result = bank.solve()
    energies = [float(value) for value in result.energies]
    ground = energies[0]
    gap = energies[1] - ground if len(energies) > 1 else math.inf
    sh_words = bank.word_set()
    settings = len(group_words(sh_words, protocol, model.n))
    variance = committed["variance"]
    residual = math.sqrt(max(variance, 0.0))
    return {
        "n_qubits": model.n,
        "basis_size": len(selected),
        "effective_rank": int(result.effective_rank),
        "sh_word_universe": len(sh_words),
        "combined_word_universe": committed["combined_word_universe"],
        "energy_settings": settings,
        "ground_energy": ground,
        "subspace_gap": gap,
        "variance": variance,
        "residual_norm": residual,
        "cancellation_scale": committed["cancellation_scale"],
        "matched_precision_half": residual / 4.0,
        "energy_grouping_protocol": protocol,
        "combined_words_sha256": committed["combined_words_sha256"],
        "alternative_protocol": alternative,
        "mapping_axis_linearized_energy_variance": committed["mapping_energy_variance"],
        "full_rank": int(result.effective_rank) == len(selected),
        "ground_root_nondegenerate": gap > GAP_FLOOR * max(1.0, abs(ground)),
        "variance_resolved": committed["variance_resolved"],
        "ground_energy_matches_validation": _close(ground, committed["validation_energy"],
                                                   abs_=1e-10),
        "sh_universe_matches_records": (
            len(sh_words) == committed["preflight_sh_word_universe"]
            == committed["ledger_sh_word_universe"]),
        "energy_settings_match_mapping_axis": (
            protocol == committed["mapping_protocol"]
            and settings == committed["mapping_settings"]),
    }


def structural_quantities(config: dict, name: str) -> dict:
    model, selected = preflight_gate.bank_inputs(name)
    grouping = config["grouping"]
    return quantities_for(model, selected, committed_rows(name),
                          grouping["declared_protocol_by_system"][name],
                          grouping["alternative_protocol_by_system"][name])


def compare(declared: dict, got: dict) -> list[str]:
    problems = []
    for key in _INT_FIELDS + _STR_FIELDS + _BOOL_FIELDS:
        if declared.get(key) != got[key]:
            problems.append(f"{key} declared {declared.get(key)!r}, recomputed {got[key]!r}")
    for key in _FLOAT_FIELDS:
        if not _close(declared.get(key, float("nan")), got[key], rel=1e-9, abs_=1e-12):
            problems.append(f"{key} declared {declared.get(key)!r}, recomputed {got[key]!r}")
    for key in _OPTIONAL_FIELDS:
        want, have = declared.get(key, "missing"), got[key]
        if have is None or want is None or isinstance(have, str):
            if want != have:
                problems.append(f"{key} declared {want!r}, recomputed {have!r}")
        elif not _close(want, have, rel=1e-12, abs_=0.0):
            problems.append(f"{key} declared {want!r}, recomputed {have!r}")
    return problems


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None) -> list[str]:
    notes = [] if notes is None else notes
    problems = []
    for name in config["banks"]["systems"]:
        got = structural_quantities(config, name)
        if computed is not None:
            computed[name] = got
        problems.extend(f"{name}: {problem}"
                        for problem in compare(config["measured_before_freezing"][name], got))
        for key in ("full_rank", "ground_root_nondegenerate", "variance_resolved",
                    "ground_energy_matches_validation", "sh_universe_matches_records",
                    "energy_settings_match_mapping_axis"):
            if not got[key]:
                problems.append(f"{name}: premise {key} fails")
        notes.append(f"  {name}: n={got['n_qubits']} M={got['basis_size']} "
                     f"gap={got['subspace_gap']:.4g} sigma={got['residual_norm']:.4g} "
                     f"|U_SH|={got['sh_word_universe']} |U|={got['combined_word_universe']} "
                     f"G_SH={got['energy_settings']} ({got['energy_grouping_protocol']})")
    return problems


# ------------------------------------------------------------------ order

def commit_order_problems(notes: list[str]) -> list[str]:
    """The config's last change must strictly precede the record's first commit."""
    import subprocess

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
    print("OK Phase 15 measured-residual preregistration is complete, ratio-free, "
          "jointly satisfiable, and bound to its inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
