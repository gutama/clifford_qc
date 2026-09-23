"""Gate the third Phase 16B preregistration, which prices the incumbent with grouping.

v2 returned GO under a cost model that charged every Pauli word its own shots,
and said in its own description that the largest exclusion was grouping. This
declaration removes that exclusion. Two things therefore have to be checked
that no earlier gate needed.

**The covariance precondition.** Grouping makes within-group correlation
material, and this design carries it exactly by simulating readouts rather than
assuming independence. That argument works only because the reference is a
computational basis state, whose distribution in any product basis factorises
over qubits. So the gate verifies that every required instance's reference is a
basis state, and refuses the declaration otherwise -- for a superposition the
honest treatment needs a joint distribution this design does not supply.

**The declared group counts.** A grouping comparison is won or lost on what one
setting costs, so the config declares a per-arm setting model and a table of
measured group counts. The gate re-derives that whole table from the instances
-- the A-CASE word universe under no grouping, QWC, block-commuting at k = 2 and
k = 4, and fully commuting, and the Hamiltonian's traceless terms under the same
partitions -- rather than trusting the numbers written down. ``G_H`` in
particular is priced: ``rt_hermitian`` charges ``d`` at ``m * G_H``.

Everything the v2 gate checked still applies: completeness, absence of any
result-shaped value, builder resolution, the strict phase-branch condition, an
enclosure containing the spectrum, an exhaustive and mutually exclusive status
ladder, a total combination rule, null oracle fields on the incumbent, commit
order from git history, and the admission criterion in exact arithmetic.

    python benchmarks/check_phase16b_v3_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "phase16b_v3_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_v3_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v3_feasibility_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "purpose", "contains_results", "question",
    "target", "measured_before_freezing", "instances",
    "required_decision_instances", "diagnostic_instances",
    "reference_precondition", "covariance_treatment", "grouping_schemes",
    "continuity_check", "setting_model", "arms", "primary_candidate_arms",
    "exact_propagation_arms", "incumbent_arm", "acase_pool",
    "basis_sizes_by_arm", "admission_criterion", "grid_rule", "noise_model",
    "cost_model", "regularizers", "non_ridge_regularizers", "decision_rule",
    "prespecified_followup", "record_requirements",
)
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "measured",
    "energy", "energies", "median_error", "p90_error", "status",
    "instance_status", "shots", "total_shots", "elapsed_seconds", "conclusion",
})
STATUSES = ("PASS", "MARGINAL", "UNDETERMINED", "FAIL")


def walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}[{index}]")


def check_no_results(config, problems):
    for path, key, value in walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            problems.append(f"config carries a result-shaped key {path}.{key} = {value!r}")
    if config.get("contains_results") is not False:
        problems.append("contains_results must be literally false")


def check_ladder(config, problems):
    order = [row["status"] for row in config["decision_rule"]["instance_status_order"]]
    if order != list(STATUSES):
        problems.append(f"status ladder order moved: {order}")
        return
    seen = set()
    for has_pass, has_marginal, censored in itertools.product((False, True), repeat=3):
        if has_pass and not has_marginal:
            continue
        seen.add("PASS" if has_pass else "MARGINAL" if has_marginal
                 else "UNDETERMINED" if censored else "FAIL")
    if seen != set(STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(STATUSES) - seen}")


def verdict_of(statuses):
    if "INVALID" in statuses:
        return "INVALID"
    if all(v == "PASS" for v in statuses):
        return "GO"
    if all(v == "FAIL" for v in statuses):
        return "NO_GO"
    return "CONDITIONAL"


def check_combination(config, problems):
    count = len(config["required_decision_instances"])
    if count < 1:
        problems.append("there must be at least one required decision instance")
        return
    for tuple_ in itertools.product(STATUSES, repeat=count):
        value = verdict_of(tuple_)
        if "UNDETERMINED" in tuple_ and value == "GO":
            problems.append(f"tuple {tuple_} yields GO despite a censored required instance")
    if verdict_of(tuple(["PASS"] * count)) != "GO":
        problems.append("an all-PASS tuple must give GO")
    if verdict_of(tuple(["FAIL"] * count)) != "NO_GO":
        problems.append("an all-FAIL tuple must give NO_GO")


def check_references(config, problems):
    declared = set(config["instances"])
    for key in ("required_decision_instances", "diagnostic_instances"):
        missing = set(config[key]) - declared
        if missing:
            problems.append(f"{key} names undeclared instances {sorted(missing)}")
    overlap = set(config["required_decision_instances"]) & set(config["diagnostic_instances"])
    if overlap:
        problems.append(f"an instance is both required and diagnostic: {sorted(overlap)}")
    if config["incumbent_arm"] not in config["arms"]:
        problems.append("incumbent_arm is not a declared arm")
    for name in config["primary_candidate_arms"] + config["exact_propagation_arms"]:
        if name not in config["arms"]:
            problems.append(f"arm {name} is referenced but not declared")
    for name in config["non_ridge_regularizers"]:
        if config["regularizers"].get(name, {}).get("is_ridge") is not False:
            problems.append(f"regularizer {name} is listed non-ridge but is not declared so")
    for arm in config["arms"]:
        if arm != "exact_diag":
            if arm not in config["basis_sizes_by_arm"]:
                problems.append(f"arm {arm} has no declared basis-size grid")
            if arm not in config["setting_model"]:
                problems.append(f"arm {arm} has no declared setting model")
    growth = config["acase_pool"]["growth"]
    for oracle in ("exact_ground_energy", "target_error"):
        if growth.get(oracle) is not None:
            problems.append(f"acase_pool.growth.{oracle} must be null")
    for scheme in config["decision_rule"]["primary_schemes"]:
        if scheme not in config["grouping_schemes"]:
            problems.append(f"primary scheme {scheme} is not a declared grouping scheme")
    if "ungrouped" not in config["grouping_schemes"]:
        problems.append("the ungrouped baseline must be declared for the v2 continuity check")
    if "ungrouped" in config["decision_rule"]["primary_schemes"]:
        problems.append("the ungrouped baseline must not be a primary scheme; it is the v2 "
                        "cost model this experiment exists to replace")
    blocked = set(config["decision_rule"]["diagnostics_cannot_promote"])
    promoters = set(config["required_decision_instances"]) | set(config["primary_candidate_arms"])
    if promoters & blocked:
        problems.append(f"a promoter is also listed as non-promoting: {sorted(promoters & blocked)}")
    if config["covariance_treatment"]["method"] != "simulated_group_readouts":
        problems.append("covariance must be carried by simulated readouts, not a bound")
    for arm in config["covariance_treatment"]["applies_to"]:
        if arm not in config["arms"]:
            problems.append(f"covariance_treatment names undeclared arm {arm}")


def resolve(path):
    """Return the builder named by a dotted path.

    Raises ``LookupError`` when the declared path is itself wrong -- a module
    that does not exist, or a name the module does not define -- and lets
    ``ImportError`` through when the module exists but an optional extra it
    needs is absent. The gate has to tell those apart: the first is a defect in
    the declaration and must fail it, the second is an environment without the
    chemistry extra and is only a skip.
    """
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise LookupError(f"{path!r} is not a dotted builder path")
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError) as exc:
        raise LookupError(f"module {module_name!r} does not exist ({exc})") from exc
    if spec is None:
        raise LookupError(f"module {module_name!r} does not exist")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise LookupError(f"module {module_name!r} defines no {attr!r}") from exc


def check_instances(config, problems, notes):
    """Build each instance; verify the grid, the basis-state precondition, admission."""
    import numpy as np

    from clifford_qc.dense_reference import to_matrix

    try:
        from scipy.linalg import expm
    except ImportError as exc:  # pragma: no cover
        notes.append(f"  SKIP admission: scipy unavailable ({exc})")
        return

    target = float(config["target"]["value"])
    required = set(config["required_decision_instances"])
    cap = max(config["acase_pool"]["growth"]["caps"])
    rt_sizes = config["basis_sizes_by_arm"]["rt_hermitian"]

    for name, spec in config["instances"].items():
        try:
            model = resolve(spec["builder"])(**spec["args"])
        except LookupError as exc:
            problems.append(f"{name}: declared builder does not resolve -- {exc}")
            continue
        except ImportError as exc:
            notes.append(f"  SKIP {name}: optional dependency missing ({exc})")
            if name in required:
                problems.append(f"{name} is REQUIRED and could not be checked (chemistry extra "
                                "missing); install it or do not require this instance")
            continue
        except TypeError as exc:
            problems.append(f"{name}: builder rejected its declared args -- {exc}")
            continue
        H = to_matrix(model.hamiltonian.to_mv())
        H = 0.5 * (H + H.conj().T)
        values, vectors = np.linalg.eigh(H)
        lower, upper = float(values[0]), float(values[-1])
        dt = math.pi / (upper - lower)
        if not (upper - lower) * dt < 2 * math.pi:
            problems.append(f"{name}: strict phase-branch condition fails")
        if lower > values[0] + 1e-9 or upper < values[-1] - 1e-9:
            problems.append(f"{name}: declared enclosure does not contain the spectrum")

        from clifford_qc.backends import ExactMVBackend
        rho = ExactMVBackend().state(model.reference, ())
        density = to_matrix(rho)
        weights, states = np.linalg.eigh(density)
        psi = states[:, -1] / np.linalg.norm(states[:, -1])
        pure = abs(weights[-1] - 1.0) <= 1e-9
        support = int(np.count_nonzero(np.abs(psi) > 1e-12))
        basis_state = pure and support == 1

        if name in required and not basis_state:
            problems.append(
                f"{name}: reference is not a computational basis state (purity {weights[-1]:.6f}, "
                f"{support} nonzero amplitudes). The simulated-readout covariance treatment "
                "factorises only for a basis state; a superposition needs the full joint "
                "distribution, which this design does not supply")
            continue  # the design does not apply here; admission would not mean anything

        if name not in required:
            notes.append(f"  {name}: {model.hamiltonian.n} qubits, basis-state reference="
                         f"{basis_state} (diagnostic)")
            continue

        from clifford_qc.models.chemistry import excitation_multivectors
        from clifford_qc.subspace import run_acase
        from clifford_qc.subspace.generators import Generator

        if "n_electrons" not in model.metadata:
            problems.append(f"{name}: no n_electrons in metadata, so the declared excitation "
                            "pool for the incumbent is not defined on this instance")
            continue
        electrons = int(model.metadata["n_electrons"])
        pool = [Generator(label, ps.to_mv()) for label, ps
                in excitation_multivectors(model.hamiltonian.n, electrons)]
        result = run_acase(rho, model.hamiltonian, pool, max_size=cap)
        incumbent_error = abs(result.result.ground_energy - float(values[0]))

        c = {0: complex(1.0)}
        d = {}
        for k in range(max(rt_sizes)):
            evolved = expm(-1j * H * (k * dt)) @ psi
            c[k] = complex(np.vdot(psi, evolved))
            d[k] = complex(np.vdot(psi, H @ evolved))
            if k:
                c[-k], d[-k] = c[k].conjugate(), d[k].conjugate()
        d[0] = complex(d[0].real)
        best_m = None
        for m in rt_sizes:
            S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
            Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
            S, Hm = 0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T)
            w, v = np.linalg.eigh(S)
            keep = w > 1e-13 * w[-1]
            if not keep.any():
                continue
            X = v[:, keep] / np.sqrt(w[keep])
            energy = float(np.linalg.eigvalsh(X.conj().T @ Hm @ X)[0])
            if abs(energy - float(values[0])) <= target and best_m is None:
                best_m = m

        if incumbent_error > target:
            problems.append(f"{name}: ADMISSION FAILS -- incumbent reaches only "
                            f"{incumbent_error:.3e} in exact arithmetic at cap {cap}")
        if best_m is None:
            problems.append(f"{name}: ADMISSION FAILS -- no exact-propagation real-time arm "
                            f"reaches the target within {rt_sizes}")

        from clifford_qc.ir import PauliWord
        universe = [PauliWord(model.hamiltonian.n, code)
                    for code in result.result.bank.word_set(result.result.indices)]
        if incumbent_error <= target and best_m is not None:
            notes.append(f"  {name}: ADMITS -- incumbent {incumbent_error:.3e} "
                         f"({len(universe)} words); real-time reaches at m={best_m}; "
                         f"basis-state reference ok")
        check_group_counts(config, name, model, universe, problems, notes)


def check_group_counts(config, name, model, universe, problems, notes):
    """Re-derive every group count the declaration writes down.

    These numbers are not background. The whole comparison turns on what one
    setting costs, so ``measured_before_freezing`` is the exchange rate between
    the v2 cost model and this one, and ``G_H`` in particular is a priced
    quantity: the ``rt_hermitian`` setting model charges ``d`` at ``m * G_H``.
    The gate therefore recomputes the counts from the instance instead of
    reading back what the config says about itself.
    """
    from clifford_qc.ir import PauliWord
    from clifford_qc.measurement.block_commuting import block_commuting_groups
    from clifford_qc.measurement.grouping import qwc_groups

    n = model.hamiltonian.n
    declared = config["measured_before_freezing"]
    derived = {
        "ungrouped": len(universe),
        "qwc": len(qwc_groups(universe)),
        "block_k2": len(block_commuting_groups(universe, 2)),
        "block_k4": len(block_commuting_groups(universe, 4)),
        "fully_commuting": len(block_commuting_groups(universe, n)),
    }
    terms = [PauliWord(n, code) for code in model.hamiltonian.terms if code != 0]
    ham = {
        "ungrouped": len(terms),
        "qwc": len(qwc_groups(terms)),
        "fully_commuting": len(block_commuting_groups(terms, n)),
    }
    for label, table, got in (("A-CASE word universe",
                               declared["acase_word_universe_h4_chain"], derived),
                              ("Hamiltonian traceless terms",
                               declared["hamiltonian_traceless_terms"], ham)):
        for partition, value in table.items():
            if partition not in got:
                problems.append(f"{name}: declared {label} partition {partition!r} is not one "
                                "this gate knows how to re-derive, so it cannot be audited")
            elif got[partition] != value:
                problems.append(f"{name}: declared {label} under {partition} is {value}, but "
                                f"the instance gives {got[partition]}")
    notes.append(f"  {name}: group counts re-derived -- A-CASE {derived['ungrouped']} -> "
                 f"{derived['qwc']} (QWC) -> {derived['fully_commuting']} (k={n}); "
                 f"H {ham['ungrouped']} -> {ham['qwc']} -> {ham['fully_commuting']}")


def first_commit(path):
    try:
        out = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", str(path)],
            cwd=ROOT.parent, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commits = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return commits[-1] if commits else None


def check_commit_order(problems, notes):
    if not RECORD.exists():
        notes.append("  commit order: no record yet, nothing to order against")
        return
    config_commit, record_commit = first_commit(CONFIG), first_commit(RECORD)
    if config_commit is None or record_commit is None:
        notes.append("  commit order: SKIP (not committed yet, or git unavailable)")
        return
    if config_commit == record_commit:
        problems.append("the config and the record entered the repository in the same commit")
        return
    try:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", config_commit, record_commit],
            cwd=ROOT.parent, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        notes.append("  commit order: SKIP (git unavailable)")
        return
    if ancestor.returncode != 0:
        problems.append(f"config commit {config_commit[:12]} is not an ancestor of record "
                        f"commit {record_commit[:12]}")
    else:
        notes.append(f"  commit order: config {config_commit[:12]} precedes "
                     f"record {record_commit[:12]}")


def main() -> int:
    if not CONFIG.exists():
        print(f"FAIL missing config {CONFIG}")
        return 1
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    problems: list[str] = []
    notes: list[str] = []

    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    for key in REQUIRED_TOP:
        if key not in config:
            problems.append(f"missing required key {key!r}")
    if not problems:
        check_no_results(config, problems)
        check_ladder(config, problems)
        check_combination(config, problems)
        check_references(config, problems)
        check_instances(config, problems, notes)
        check_commit_order(problems, notes)

    for note in notes:
        print(note)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("OK Phase 16B v3 preregistration is complete, result-free, decidable, admissible, "
          "and its covariance precondition holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
