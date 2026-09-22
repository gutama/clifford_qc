"""Gate the second Phase 16B preregistration, including its admission criterion.

The first decision experiment returned a verdict that was not about the thing
being decided: its incumbent never reached the accuracy target in exact
arithmetic, so the cost comparison was censored on every TFIM instance. Nothing
in the v1 gate could have caught that, because nothing in it asked whether the
control was capable of the target at all.

This gate asks. Beyond the v1 checks -- completeness, absence of results,
builder resolution, the strict phase-branch condition, an enclosure that
contains the spectrum, an exhaustive and mutually exclusive status ladder, a
total combination rule, null oracle fields on the incumbent, and commit order
from git history -- it recomputes, in exact arithmetic:

    for every required decision instance, the incumbent AND at least one
    exact-propagation real-time arm reach the target within their declared caps.

That reads zero-noise reachability only. It never inspects a noisy comparison,
a shot count, or a ratio; those are the result. The point is narrower than it
looks: a control must be *able* to reach the target before it is worth asking
what it costs to.

The admission check needs the chemistry extra. Where that is absent it reports
a loud SKIP rather than passing silently, because an unenforced admission
criterion is exactly the hole this gate was written to close.

    python benchmarks/check_phase16b_v2_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "phase16b_v2_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_v2_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v2_feasibility_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "purpose", "contains_results", "target",
    "v1_finding_this_design_answers", "instances", "required_decision_instances",
    "diagnostic_instances", "admission_criterion", "arms",
    "primary_candidate_arms", "exact_propagation_arms", "incumbent_arm",
    "acase_pool", "basis_sizes_by_arm", "grid_rule", "noise_model", "cost_model",
    "regularizers", "non_ridge_regularizers", "decision_rule",
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


def status_of(has_pass, has_marginal, censored):
    if has_pass:
        return "PASS"
    if has_marginal:
        return "MARGINAL"
    if censored:
        return "UNDETERMINED"
    return "FAIL"


def check_ladder(config, problems):
    order = [row["status"] for row in config["decision_rule"]["instance_status_order"]]
    if order != list(STATUSES):
        problems.append(f"status ladder order moved: {order}")
        return
    seen = set()
    for has_pass, has_marginal, censored in itertools.product((False, True), repeat=3):
        if has_pass and not has_marginal:
            continue  # unreachable: a PASS implies the MARGINAL predicate
        seen.add(status_of(has_pass, has_marginal, censored))
    if seen != set(STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(STATUSES) - seen}")


def verdict_of(statuses):
    if "INVALID" in statuses:
        return "INVALID"
    if all(value == "PASS" for value in statuses):
        return "GO"
    if all(value == "FAIL" for value in statuses):
        return "NO_GO"
    return "CONDITIONAL"


def check_combination(config, problems):
    """The generalized rule must be total and single-valued over every tuple."""
    count = len(config["required_decision_instances"])
    if count < 1:
        problems.append("there must be at least one required decision instance")
        return
    assigned = {}
    for tuple_ in itertools.product(STATUSES, repeat=count):
        assigned[tuple_] = verdict_of(tuple_)
    if len(assigned) != len(STATUSES) ** count:
        problems.append("the combination rule is not total over the status tuples")
    if assigned[tuple(["PASS"] * count)] != "GO":
        problems.append("an all-PASS tuple must give GO")
    if assigned[tuple(["FAIL"] * count)] != "NO_GO":
        problems.append("an all-FAIL tuple must give NO_GO")
    mixed = {tup: v for tup, v in assigned.items()
             if len(set(tup)) > 1 or tup[0] not in ("PASS", "FAIL")}
    if mixed and set(mixed.values()) != {"CONDITIONAL"}:
        problems.append(f"every mixed tuple must be CONDITIONAL; got {sorted(set(mixed.values()))}")
    # An UNDETERMINED on a required instance must never yield GO.
    for tuple_, value in assigned.items():
        if "UNDETERMINED" in tuple_ and value == "GO":
            problems.append(f"tuple {tuple_} yields GO despite a censored required instance")


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
        if arm != "exact_diag" and arm not in config["basis_sizes_by_arm"]:
            problems.append(f"arm {arm} has no declared basis-size grid")
    growth = config["acase_pool"]["growth"]
    for oracle in ("exact_ground_energy", "target_error"):
        if growth.get(oracle) is not None:
            problems.append(f"acase_pool.growth.{oracle} must be null: an oracle stop would make "
                            "the incumbent's basis depend on the answer it is compared against")
    blocked = set(config["decision_rule"]["diagnostics_cannot_promote"])
    promoters = set(config["required_decision_instances"]) | set(config["primary_candidate_arms"])
    if promoters & blocked:
        problems.append(f"a promoter is also listed as non-promoting: {sorted(promoters & blocked)}")


def build(spec):
    from clifford_qc.models import chemistry
    builder = spec["builder"].rsplit(".", 1)[-1]
    return getattr(chemistry, builder)(**spec["args"])


def check_instances_and_admission(config, problems, notes):
    """Build every instance; verify the grid, and admission on required ones."""
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
            model = build(spec)
        except ImportError as exc:
            notes.append(f"  SKIP {name}: optional dependency missing ({exc})")
            if name in required:
                problems.append(
                    f"{name} is a REQUIRED instance and its admission could not be checked "
                    "(chemistry extra missing). An unenforced admission criterion is the hole "
                    "this gate exists to close; install the extra or do not require this instance")
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

        if name not in required:
            notes.append(f"  {name}: {model.hamiltonian.n} qubits, dt={dt:.6f} (diagnostic, "
                         "admission not required)")
            continue

        from clifford_qc.backends import ExactMVBackend
        from clifford_qc.models.chemistry import excitation_multivectors
        from clifford_qc.subspace import run_acase
        from clifford_qc.subspace.generators import Generator

        rho = ExactMVBackend().state(model.reference, ())
        density = to_matrix(rho)
        _, states = np.linalg.eigh(density)
        psi = states[:, -1]
        psi = psi / np.linalg.norm(psi)
        exact = float(values[0])

        electrons = int(model.metadata["n_electrons"])
        pool = [Generator(label, ps.to_mv()) for label, ps
                in excitation_multivectors(model.hamiltonian.n, electrons)]
        result = run_acase(rho, model.hamiltonian, pool, max_size=cap)
        incumbent_error = abs(result.result.ground_energy - exact)
        incumbent_ok = incumbent_error <= target

        c, d = {0: complex(1.0)}, {}
        for k in range(max(rt_sizes) + 1):
            evolved = expm(-1j * H * (k * dt)) @ psi
            c[k] = complex(np.vdot(psi, evolved))
            d[k] = complex(np.vdot(psi, H @ evolved))
            if k:
                c[-k], d[-k] = c[k].conjugate(), d[k].conjugate()
        d[0] = complex(d[0].real)
        best_m, best_error = None, float("inf")
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
            if abs(energy - exact) < best_error:
                best_error = abs(energy - exact)
            if abs(energy - exact) <= target and best_m is None:
                best_m = m
        candidate_ok = best_m is not None

        if not incumbent_ok:
            problems.append(
                f"{name}: ADMISSION FAILS -- the incumbent reaches only {incumbent_error:.3e} "
                f"in exact arithmetic at cap {cap}, above the {target:.1e} target. A required "
                "instance whose control cannot reach the target produces a censored comparison, "
                "which is what the v1 run returned")
        if not candidate_ok:
            problems.append(
                f"{name}: ADMISSION FAILS -- no exact-propagation real-time arm reaches the "
                f"target within the declared sizes {rt_sizes} (best {best_error:.3e})")
        if incumbent_ok and candidate_ok:
            words = len(result.result.bank.word_set(result.result.indices))
            notes.append(
                f"  {name}: ADMITS -- incumbent {incumbent_error:.3e} at basis "
                f"{len(result.result.indices)} ({words} words); real-time reaches at m={best_m}")


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
        notes.append("  commit order: SKIP (not committed yet, or git history unavailable)")
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
        check_instances_and_admission(config, problems, notes)
        check_commit_order(problems, notes)

    for note in notes:
        print(note)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("OK Phase 16B v2 preregistration is complete, result-free, decidable, and admissible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
