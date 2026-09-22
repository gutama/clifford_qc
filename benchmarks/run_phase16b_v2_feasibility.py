"""Execute the second frozen Phase 16B decision experiment.

Inputs come from `configs/phase16b_v2_feasibility.json`, gated by
`check_phase16b_v2_preregistration.py`. Nothing here chooses a threshold, a
grid, or a rule; this module evaluates them.

Estimands are arrays rather than dicts, and the incumbent's pencil is rebuilt
by one matrix-vector product against precomputed coefficient matrices. That is
not a micro-optimization: A-CASE on `h4_chain` carries a 7 927-word universe,
and a per-replica Python loop over its terms would put the budget sweep out of
reach entirely.

Two sweeps, kept apart because they answer different questions: an
absolute-error stress sweep over the `eps` grid, and a budget sweep mapping a
total shot budget to per-component `eps` through the declared variance bounds.
`shots_to_target` is read only off tested budgets.

    python benchmarks/run_phase16b_v2_feasibility.py
    python benchmarks/run_phase16b_v2_feasibility.py --replicas 20 --out draft.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.reproducibility import stamp_record

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "phase16b_v2_feasibility.json"
DEFAULT_OUT = ROOT / "reference_results" / "phase16b_v2_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v2_feasibility.v1"
PENCIL_IDENTITY_TOL = 1e-10


class Estimands:
    """Independent scalar estimands as arrays, with their variance weights."""

    def __init__(self, values, weights, complexity):
        self.values = np.asarray(values, dtype=complex)
        self.weights = np.asarray(weights, dtype=float)
        self.complexity = np.asarray(complexity, dtype=bool)

    @property
    def real_components(self):
        return int(self.complexity.sum() * 2 + (~self.complexity).sum())

    def draw(self, rng):
        """Standard normals, one pair per estimand; imaginary parts unused if real."""
        return rng.standard_normal((self.values.size, 2))

    def perturb(self, variates, eps_scale):
        """`eps_scale` is a scalar multiplier applied to each estimand's weight."""
        eps = self.weights * eps_scale
        out = self.values + eps * variates[:, 0]
        out = out + 1j * eps * variates[:, 1] * self.complexity
        return np.where(self.complexity, out, out.real + 0j)


def build_instance(name, spec):
    from clifford_qc.models import chemistry
    builder = spec["builder"].rsplit(".", 1)[-1]
    model = getattr(chemistry, builder)(**spec["args"])
    mv = model.hamiltonian.to_mv()
    H = 0.5 * (to_matrix(mv) + to_matrix(mv).conj().T)
    values, vectors = np.linalg.eigh(H)

    from clifford_qc.backends import ExactMVBackend
    rho = ExactMVBackend().state(model.reference, ())
    density = to_matrix(rho)
    weights, states = np.linalg.eigh(density)
    if abs(weights[-1] - 1.0) > 1e-9:
        raise RuntimeError(f"{name}: reference density is not a pure state")
    psi = states[:, -1] / np.linalg.norm(states[:, -1])

    lower, upper = float(values[0]), float(values[-1])
    dt = math.pi / (upper - lower)
    lam = float(sum(abs(v) for code, v in mv.terms.items() if code != 0))
    return {
        "name": name, "model": model, "mv": mv, "H": H, "psi": psi, "rho": rho,
        "values": values, "vectors": vectors, "exact": float(values[0]),
        "dt": dt, "energy_shift": float((lower + upper) / 2),
        "lower": lower, "upper": upper,
        "branch_ok": bool((upper - lower) * dt < 2 * math.pi),
        "enclosure_contains": bool(lower <= values[0] + 1e-9 and upper >= values[-1] - 1e-9),
        "lambda_one": lam, "n_qubits": model.hamiltonian.n,
        "dimension": int(H.shape[0]),
        "support": distinct_energy_support(values, vectors, psi),
        "ground_weight": float(abs(vectors[:, 0].conj() @ psi) ** 2),
    }


def distinct_energy_support(values, vectors, psi, floor=1e-6):
    weights = np.abs(vectors.conj().T @ psi) ** 2
    grouped, anchor = [], None
    for value, weight in zip(values, weights):
        if anchor is None or not np.isclose(value, anchor, atol=1e-10, rtol=1e-12):
            grouped.append(float(weight))
            anchor = value
        else:
            grouped[-1] += float(weight)
    return int(sum(weight > floor for weight in grouped))


def whiten(S, cutoff_relative=0.0, floor_absolute=0.0, ridge=0.0):
    norms = np.sqrt(np.clip(np.real(np.diag(S)), 0.0, None))
    live = norms > 1e-300
    if not live.any():
        return None, 0
    scale = np.where(live, norms, 1.0)
    bar = (S / scale[:, None]) / scale[None, :]
    index = np.flatnonzero(live)
    bar = bar[np.ix_(index, index)]
    bar = 0.5 * (bar + bar.conj().T)
    values, vectors = np.linalg.eigh(bar)
    if ridge:
        values = values + ridge
    largest = max(float(values[-1]), 1e-300)
    keep = values > max(cutoff_relative * largest, floor_absolute, 1e-300)
    if not keep.any():
        return None, 0
    transform = np.zeros((S.shape[0], int(keep.sum())), dtype=complex)
    transform[index, :] = vectors[:, keep] / np.sqrt(values[keep])
    return transform / scale[:, None], int(keep.sum())


def solve_hermitian(S, Hm, **policy):
    X, rank = whiten(S, **policy)
    if X is None:
        return float("nan"), 0
    reduced = X.conj().T @ Hm @ X
    reduced = 0.5 * (reduced + reduced.conj().T)
    if not np.all(np.isfinite(reduced)):
        return float("nan"), rank
    return float(np.linalg.eigvalsh(reduced)[0]), rank


def solve_unitary(c, m, dt, energy_shift, **policy):
    shifted = {k: value * np.exp(1j * energy_shift * k * dt) for k, value in c.items()}
    S0 = np.array([[shifted[j - i] for j in range(m)] for i in range(m)])
    S1 = np.array([[shifted[j - i + 1] for j in range(m)] for i in range(m)])
    S0 = 0.5 * (S0 + S0.conj().T)
    X, rank = whiten(S0, **policy)
    if X is None:
        return float("nan"), 0
    lam = np.linalg.eigvals(X.conj().T @ S1 @ X)
    if not np.all(np.isfinite(lam)) or np.any(np.abs(lam) <= 1e-12):
        return float("nan"), rank
    return float(energy_shift + np.min(-np.angle(lam) / dt)), rank


# -------------------------------------------------------------------- arms

def exact_lags(inst, max_lag, want_d=True):
    c = {0: complex(1.0)}
    d = {}
    for k in range(max_lag + 1):
        evolved = expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
        c[k] = complex(np.vdot(inst["psi"], evolved))
        if want_d:
            d[k] = complex(np.vdot(inst["psi"], inst["H"] @ evolved))
        if k:
            c[-k] = c[k].conjugate()
            if want_d:
                d[-k] = d[k].conjugate()
    if want_d:
        d[0] = complex(d[0].real)
    return c, d


def trotter2_step_matrix(inst, microsteps):
    """Dense image of ``gates.trotter2_unitary(terms, dt, microsteps)``.

    Identical object, different arithmetic. Building it in the ``MV`` basis
    costs about five minutes per 8-qubit instance, because a Trotterized
    propagator's Pauli support fills in; the same product of
    ``rotor(P, theta) = cos(theta/2) I - i sin(theta/2) P`` factors in dense
    256x256 arithmetic takes under a second. `verify_trotter_step` checks the
    two against each other on a small instance so the substitution is a
    measured identity rather than an assumption.
    """
    n = inst["mv"].n
    dimension = 1 << n
    dt = inst["dt"] / microsteps
    terms = [(float(np.real(v)), code) for code, v in inst["mv"].terms.items() if code != 0]
    matrices = {code: to_matrix(MV(n, {code: 1.0})) for _, code in terms}
    identity = np.eye(dimension, dtype=complex)

    def rotor_matrix(coefficient, code):
        theta = coefficient * dt
        return (math.cos(theta / 2) * identity
                - 1j * math.sin(theta / 2) * matrices[code])

    half = identity
    for coefficient, code in terms:
        half = half @ rotor_matrix(coefficient, code)
    step = half
    for coefficient, code in reversed(terms):
        step = step @ rotor_matrix(coefficient, code)
    # Sequential, not matrix_power: the reference multiplies the step in one
    # by one, and binary squaring accumulates round-off differently. Matching
    # its order is what keeps the identity check at round-off instead of 1e-9.
    total = identity
    for _ in range(microsteps):
        total = total @ step
    return total


def verify_trotter_step(inst, microsteps, floor=1e-12, slack=10.0):
    """Check the dense step against the MV construction it stands in for.

    The two cannot be required to agree more tightly than the reference agrees
    with itself. On `h4_chain` the MV path's own unitarity defect is about
    9e-10 after 184 terms x 2 x 8 microsteps of sparse products, while the
    dense path's is about 6e-14 -- so the gap between them is the reference's
    accumulated round-off, and the dense construction is the more accurate of
    the two. The admissible gap is therefore scaled to the reference's measured
    defect rather than fixed, and both defects go into the record so the
    substitution is a reported measurement, not a silent swap.
    """
    from clifford_qc.gates import trotter2_unitary
    terms = [(float(np.real(v)), MV(inst["mv"].n, {code: 1.0}))
             for code, v in inst["mv"].terms.items() if code != 0]
    reference = to_matrix(trotter2_unitary(terms, inst["dt"], microsteps))
    dense = trotter2_step_matrix(inst, microsteps)
    identity = np.eye(reference.shape[0], dtype=complex)
    reference_defect = float(np.max(np.abs(reference.conj().T @ reference - identity)))
    dense_defect = float(np.max(np.abs(dense.conj().T @ dense - identity)))
    residual = float(np.max(np.abs(reference - dense)))
    tolerance = max(floor, slack * reference_defect)
    report = {"residual": residual, "tolerance": tolerance,
              "reference_unitarity_defect": reference_defect,
              "dense_unitarity_defect": dense_defect,
              "dense_is_more_accurate": bool(dense_defect < reference_defect)}
    if residual > tolerance:
        raise RuntimeError(f"dense Trotter step disagrees with trotter2_unitary by "
                           f"{residual:.3e}, beyond the reference's own {tolerance:.3e}")
    return report


# The dense-vs-MV identity is a property of the construction, not of the
# instance: the same code path, the same rotor factors, the same ordering. It
# is therefore checked once per (qubit count, microstep count) and reused. The
# check costs about five minutes at eight qubits because it has to build the
# MV reference it is standing in for, so paying it per instance would be paying
# for the slow path this function exists to avoid.
_TROTTER_VERIFIED: dict[tuple[int, int], dict] = {}


def verified_construction(inst, microsteps):
    key = (inst["mv"].n, microsteps)
    if key not in _TROTTER_VERIFIED:
        report = verify_trotter_step(inst, microsteps)
        report["verified_on_instance"] = inst["name"]
        report["shared_by_qubit_count"] = key[0]
        _TROTTER_VERIFIED[key] = report
    return _TROTTER_VERIFIED[key]


def arm_rt_hermitian(inst, m, cache):
    c, d = cache["lags"]
    lam = inst["lambda_one"]
    values = [c[k] for k in range(1, m)] + [d[0]] + [d[k] for k in range(1, m)]
    weights = [1.0] * (m - 1) + [lam] * m
    complexity = [True] * (m - 1) + [False] + [True] * (m - 1)
    est = Estimands(values, weights, complexity)

    def assemble(noisy):
        cc = {0: complex(1.0)}
        dd = {0: complex(noisy[m - 1].real)}
        for k in range(1, m):
            cc[k] = noisy[k - 1]; cc[-k] = cc[k].conjugate()
            dd[k] = noisy[m - 1 + k]; dd[-k] = dd[k].conjugate()
        S = np.array([[cc[j - i] for j in range(m)] for i in range(m)])
        Hm = np.array([[dd[j - i] for j in range(m)] for i in range(m)])
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": est, "assemble": assemble, "c_slots": list(range(m - 1))}


def arm_rt_unitary(inst, m, cache):
    c, _ = cache["lags"]
    est = Estimands([c[k] for k in range(1, m + 1)], [1.0] * m, [True] * m)

    def assemble(noisy):
        cc = {0: complex(1.0)}
        for k in range(1, m + 1):
            cc[k] = noisy[k - 1]; cc[-k] = cc[k].conjugate()
        return "unitary", cc

    return {"estimands": est, "assemble": assemble, "c_slots": list(range(m))}


def arm_rt_trotter(inst, m, cache):
    step = cache.get("trotter_step")
    if step is None:
        step = trotter2_step_matrix(inst, cache["microsteps"])
        cache["trotter_step"] = step
        cache["trotter_verification"] = verified_construction(inst, cache["microsteps"])
    columns, current = [], inst["psi"].copy()
    for _ in range(m):
        columns.append(current.copy())
        current = step @ current
    V = np.column_stack(columns)
    lam = inst["lambda_one"]
    S_exact = V.conj().T @ V
    H_exact = V.conj().T @ (inst["H"] @ V)
    fidelity = float(abs(np.vdot(expm(-1j * inst["H"] * ((m - 1) * inst["dt"])) @ inst["psi"],
                                 V[:, m - 1])) ** 2)
    values = [S_exact[0, k] for k in range(1, m)]
    weights = [1.0] * (m - 1)
    complexity = [True] * (m - 1)
    pairs = []
    for i in range(m):
        values.append(complex(H_exact[i, i].real)); weights.append(lam)
        complexity.append(False); pairs.append((i, i))
        for j in range(i + 1, m):
            values.append(complex(H_exact[i, j])); weights.append(lam)
            complexity.append(True); pairs.append((i, j))
    est = Estimands(values, weights, complexity)

    def assemble(noisy):
        ss = {0: complex(1.0)}
        for k in range(1, m):
            ss[k] = noisy[k - 1]; ss[-k] = ss[k].conjugate()
        S = np.array([[ss[j - i] for j in range(m)] for i in range(m)])
        Hm = np.zeros((m, m), dtype=complex)
        for offset, (i, j) in enumerate(pairs):
            value = noisy[m - 1 + offset]
            if i == j:
                Hm[i, i] = value.real
            else:
                Hm[i, j] = value; Hm[j, i] = np.conj(value)
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": est, "assemble": assemble, "propagation_fidelity": fidelity,
            "trotter_step_verification": cache.get("trotter_verification")}


def arm_power_krylov(inst, m, cache):
    powers = cache.setdefault("powers", {})
    need = 2 * m - 1
    if len(powers) < need:
        current = MV.scalar(inst["mv"].n, 1.0)
        for p in range(1, need + 1):
            current = current * inst["mv"]
            if p not in powers:
                weight = float(sum(abs(v) for code, v in current.terms.items() if code != 0))
                value = complex(np.vdot(inst["psi"], to_matrix(current) @ inst["psi"]).real)
                powers[p] = (value, max(weight, 1e-300))
    values = [powers[p][0] for p in range(1, need + 1)]
    weights = [powers[p][1] for p in range(1, need + 1)]
    est = Estimands(values, weights, [False] * need)

    def assemble(noisy):
        mu = {0: 1.0}
        for p in range(1, need + 1):
            mu[p] = float(noisy[p - 1].real)
        S = np.array([[mu[i + j] for j in range(m)] for i in range(m)], dtype=complex)
        Hm = np.array([[mu[i + j + 1] for j in range(m)] for i in range(m)], dtype=complex)
        return "hermitian", (S, Hm)

    return {"estimands": est, "assemble": assemble}


def arm_acase(inst, cap, cache):
    """Frozen A-CASE basis; the pencil is one matvec against coefficient matrices."""
    from clifford_qc.models.chemistry import excitation_multivectors
    from clifford_qc.subspace import run_acase
    from clifford_qc.subspace.generators import Generator

    pool = cache.get("pool")
    if pool is None:
        electrons = int(inst["model"].metadata["n_electrons"])
        pool = [Generator(label, ps.to_mv()) for label, ps
                in excitation_multivectors(inst["n_qubits"], electrons)]
        cache["pool"] = pool

    result = run_acase(inst["rho"], inst["model"].hamiltonian, pool, max_size=cap,
                       exact_ground_energy=None, target_error=None)
    bank, indices = result.result.bank, result.result.indices
    size = len(indices)
    words = sorted(bank.word_set(indices))
    position = {code: slot for slot, code in enumerate(words)}

    density = to_matrix(inst["rho"])
    exact_words = np.array(
        [float(np.trace(density @ to_matrix(MV(inst["mv"].n, {code: 1.0}))).real)
         for code in words])

    # S_ij and H_ij are linear in the word expectations, so the whole pencil is
    # two dense matrices applied to one vector -- built once, reused per replica.
    coeff_s = np.zeros((size * size, len(words)), dtype=complex)
    coeff_h = np.zeros((size * size, len(words)), dtype=complex)
    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            row = a * size + b
            for code, value in bank.iter_operator_terms("overlap", i, j):
                coeff_s[row, position[code]] += value
            for code, value in bank.iter_operator_terms("element", i, j):
                coeff_h[row, position[code]] += value

    identity_slot = position.get(0)
    free = [slot for slot in range(len(words)) if slot != identity_slot]
    est = Estimands(exact_words[free], [1.0] * len(free), [False] * len(free))

    def assemble(noisy):
        full = exact_words.copy()
        full[free] = np.real(noisy)
        if identity_slot is not None:
            full[identity_slot] = 1.0
        S = (coeff_s @ full).reshape(size, size)
        Hm = (coeff_h @ full).reshape(size, size)
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": est, "assemble": assemble, "basis_size": size,
            "word_count": len(words),
            "exact_energy": float(result.result.ground_energy)}


# ------------------------------------------------------------------ sweeps

def evaluate(arm, noisy, inst, m, policy):
    kind, data = arm["assemble"](noisy)
    if kind == "unitary":
        return solve_unitary(data, m, inst["dt"], inst["energy_shift"], **policy)
    return solve_hermitian(*data, **policy)


def summarize(energies, exact):
    values = np.asarray(energies, dtype=float)
    errors = np.where(np.isfinite(values), np.abs(values - exact), np.inf)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.9, method="higher")),
        "failures": int((~np.isfinite(values)).sum()),
        "replicas": int(errors.size),
    }


def policies(config):
    ridge = float(config["regularizers"]["ridge"]["coefficient"])
    return {
        "hard_truncation": lambda eps: {"cutoff_relative": max(eps, 1e-13)},
        "per_mode_floor": lambda eps: {"floor_absolute": max(eps, 1e-13)},
        "ridge": lambda eps: {"cutoff_relative": 1e-13, "ridge": ridge},
    }


def run_cell(arm, inst, m, eps_scale, policy_fn, policy_eps, replicas, rng,
             shared=None):
    est = arm["estimands"]
    energies, ranks = [], []
    for replica in range(replicas):
        variates = est.draw(rng)
        if shared is not None and "c_slots" in arm:
            slots = arm["c_slots"]
            variates[: len(slots)] = shared[replica][: len(slots)]
        noisy = est.perturb(variates, eps_scale)
        energy, rank = evaluate(arm, noisy, inst, m, policy_fn(policy_eps))
        energies.append(energy)
        ranks.append(rank)
    stats = summarize(energies, inst["exact"])
    stats["median_rank"] = float(np.median(ranks))
    return stats


def shots_to_target(arm_entry, reg_name, target):
    table = arm_entry.get("budget", {}).get(reg_name, {})
    best = None
    for size_key, budgets in table.items():
        for budget_key, stats in budgets.items():
            if stats["median"] <= target:
                budget = float(budget_key)
                if best is None or budget < best["budget"]:
                    best = {"budget": budget, "basis_size": int(size_key),
                            "eps_bounded": stats["eps_bounded"],
                            "median": stats["median"], "p90": stats["p90"],
                            "failures": stats["failures"]}
    return best


def instance_status(entry, config, target):
    if entry.get("status") == "INVALID":
        return "INVALID", {"reason": "a deterministic check failed"}
    non_ridge = list(config["non_ridge_regularizers"])
    all_regs = list(config["regularizers"])
    incumbent = config["incumbent_arm"]
    arms = entry["arms"]

    found = {}
    for reg in all_regs:
        row = shots_to_target(arms.get(incumbent, {}), reg, target)
        if row:
            found[reg] = row
    incumbent_best = min(found.values(), key=lambda r: r["budget"]) if found else None

    candidates = []
    for arm_name in config["exact_propagation_arms"]:
        for reg in all_regs:
            row = shots_to_target(arms.get(arm_name, {}), reg, target)
            if not row:
                continue
            ratio = (incumbent_best["budget"] / row["budget"]) if incumbent_best else None
            candidates.append({
                "arm": arm_name, "regularizer": reg, "is_ridge": reg not in non_ridge,
                "rt_budget": row["budget"], "rt_basis_size": row["basis_size"],
                "c_noise": row["eps_bounded"],
                "incumbent_budget": incumbent_best["budget"] if incumbent_best else None,
                "ratio": ratio,
                "qualifying": bool(ratio is not None and ratio >= 10.0),
            })

    qualifying = [row for row in candidates if row["qualifying"]]
    passes = [row for row in qualifying if not row["is_ridge"] and row["c_noise"] >= 1e-5]
    marginals = [row for row in qualifying if row["c_noise"] >= 1e-6]
    evidence = {"incumbent_shots_to_target": incumbent_best,
                "incumbent_censored": incumbent_best is None,
                "candidates": candidates}
    if passes:
        return "PASS", {**evidence, "basis": passes}
    if marginals:
        return "MARGINAL", {**evidence, "basis": marginals}
    if incumbent_best is None or not candidates:
        return "UNDETERMINED", {**evidence,
                                "reason": "a required cost comparison is censored on the tested grid"}
    return "FAIL", {**evidence,
                    "reason": "all comparisons complete; no qualifying candidate at c noise >= 1e-6"}


def combine(statuses, config):
    values = [statuses.get(name) for name in config["required_decision_instances"]]
    if any(value is None for value in values):
        return "INCOMPLETE"
    if "INVALID" in values:
        return "INVALID"
    if all(value == "PASS" for value in values):
        return "GO"
    if all(value == "FAIL" for value in values):
        return "NO_GO"
    return "CONDITIONAL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--instances", nargs="*", default=None)
    args = parser.parse_args()

    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    target = float(config["target"]["value"])
    replicas = args.replicas or int(config["noise_model"]["replicas"])
    seed_root = int(config["noise_model"]["seed_root"])
    eps_grid = list(config["noise_model"]["eps_grid"])
    budgets = [10.0 ** d for d in config["cost_model"]["budget_grid_decades"]]
    grids = config["basis_sizes_by_arm"]
    caps = list(config["acase_pool"]["growth"]["caps"])
    policy_map = policies(config)
    incumbent = config["incumbent_arm"]
    required = set(config["required_decision_instances"])

    names = args.instances or list(config["instances"])
    record = {"schema": SCHEMA, "config_digest": digest,
              "config_path": str(CONFIG.relative_to(ROOT.parent)),
              "evidence": "heuristic", "quantum_advantage_claim": False,
              "cost_contract": config["cost_model"]["contract"],
              "replicas": replicas, "target": target,
              "instances": {}, "decision": {}}
    started = time.time()

    for position, name in enumerate(names):
        spec = config["instances"][name]
        try:
            inst = build_instance(name, spec)
        except ImportError as exc:
            record["instances"][name] = {"skipped": f"optional dependency missing: {exc}"}
            print(f"SKIP {name}: {exc}")
            continue
        cache = {"microsteps": int(config["arms"]["rt_trotter"]["microsteps_per_dt"])}
        max_lag = max(max(grids["rt_unitary"]), max(grids["rt_hermitian"]))
        cache["lags"] = exact_lags(inst, max_lag)

        entry = {
            "n_qubits": inst["n_qubits"], "dimension": inst["dimension"],
            "exact_ground_energy": inst["exact"], "dt": inst["dt"],
            "energy_shift": inst["energy_shift"], "lambda_one": inst["lambda_one"],
            "distinct_energy_support": inst["support"],
            "ground_space_weight": inst["ground_weight"],
            "enclosure": [inst["lower"], inst["upper"]],
            "role": spec["role"], "arms": {},
            "checks": {"branch_condition": inst["branch_ok"],
                       "enclosure_contains_spectrum": inst["enclosure_contains"]},
        }
        print(f"\n=== {name}: {inst['n_qubits']}q dim={inst['dimension']} "
              f"support={inst['support']} dt={inst['dt']:.5f} Lambda={inst['lambda_one']:.3f} ===")

        residual = 0.0
        c, d = cache["lags"]
        for m in grids["rt_hermitian"]:
            S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
            Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
            V = np.column_stack([expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
                                 for k in range(m)])
            residual = max(residual, float(np.max(np.abs(S - V.conj().T @ V))),
                           float(np.max(np.abs(Hm - V.conj().T @ (inst["H"] @ V)))
                                 / max(inst["lambda_one"], 1.0)))
        entry["checks"]["pencil_identity"] = residual
        entry["checks"]["pencil_identity_ok"] = bool(residual <= PENCIL_IDENTITY_TOL)
        if not all((inst["branch_ok"], inst["enclosure_contains"],
                    entry["checks"]["pencil_identity_ok"])):
            entry["status"] = "INVALID"
            record["instances"][name] = entry
            print("  INVALID: a deterministic check failed")
            continue

        builders = {
            "rt_hermitian": (grids["rt_hermitian"], arm_rt_hermitian),
            "rt_unitary": (grids["rt_unitary"], arm_rt_unitary),
            "rt_trotter": (grids["rt_trotter"], arm_rt_trotter),
            "power_krylov_control": (grids["power_krylov_control"], arm_power_krylov),
            incumbent: (caps, arm_acase),
        }

        for arm_name, (sizes, builder) in builders.items():
            arm_entry = {"stress": {}, "budget": {}, "sizes": {}}
            for m in sizes:
                built = (builder(inst, m, cache) if arm_name != incumbent
                         else builder(inst, m, cache))
                est = built["estimands"]
                n_comp = est.real_components
                zero, zero_rank = evaluate(built, est.values, inst, m,
                                           policy_map["hard_truncation"](0.0))
                info = {"real_components": n_comp, "estimands": int(est.values.size),
                        "zero_noise_error": abs(zero - inst["exact"]) if np.isfinite(zero) else None,
                        "zero_noise_rank": zero_rank}
                for extra in ("basis_size", "word_count", "propagation_fidelity",
                              "trotter_step_verification"):
                    if extra in built:
                        info[extra] = built[extra]
                arm_entry["sizes"][str(m)] = info

                for cell, eps in enumerate(eps_grid):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed_root, position, 1, cell, m]))
                    reps = 1 if eps == 0 else replicas
                    shared = ([np.random.default_rng(
                        np.random.SeedSequence([seed_root, position, 1, cell, m, 99])
                    ).standard_normal((max_lag + 1, 2)) for _ in range(reps)]
                        if "c_slots" in built else None)
                    arm_entry["stress"].setdefault(str(m), {})[repr(eps)] = run_cell(
                        built, inst, m, eps, policy_map["hard_truncation"], eps, reps,
                        rng, shared)

                for cell, budget in enumerate(budgets):
                    scale = math.sqrt(n_comp / budget)
                    for reg_index, (reg_name, policy_fn) in enumerate(policy_map.items()):
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [seed_root, position, 2, cell, m, reg_index]))
                        shared = ([np.random.default_rng(np.random.SeedSequence(
                            [seed_root, position, 2, cell, m, reg_index, 99])
                        ).standard_normal((max_lag + 1, 2)) for _ in range(replicas)]
                            if "c_slots" in built else None)
                        stats = run_cell(built, inst, m, scale, policy_fn, scale,
                                         replicas, rng, shared)
                        stats["eps_bounded"] = scale
                        bucket = (arm_entry["budget"].setdefault(reg_name, {})
                                  .setdefault(str(m), {}))
                        bucket[f"1e{int(round(math.log10(budget)))}"] = stats
            entry["arms"][arm_name] = arm_entry
            print(f"  {arm_name}: done")

        if name in required:
            inc = min((v["zero_noise_error"] for v in entry["arms"][incumbent]["sizes"].values()
                       if v["zero_noise_error"] is not None), default=float("inf"))
            cand = min((v["zero_noise_error"]
                        for arm_name in config["exact_propagation_arms"]
                        for v in entry["arms"][arm_name]["sizes"].values()
                        if v["zero_noise_error"] is not None), default=float("inf"))
            entry["admission"] = {"incumbent_zero_noise_error": inc,
                                  "candidate_zero_noise_error": cand,
                                  "admits": bool(inc <= target and cand <= target)}
        record["instances"][name] = entry

    statuses = {}
    for name, entry in record["instances"].items():
        if "skipped" in entry:
            continue
        status, evidence = instance_status(entry, config, target)
        entry["status"] = status
        entry["status_evidence"] = evidence
        statuses[name] = status
        marker = "required" if name in required else "diagnostic"
        print(f"  {name:16} {status:13} ({marker})")
    record["decision"] = {
        "instance_statuses": statuses,
        "required_instances": config["required_decision_instances"],
        "verdict": combine(statuses, config),
        "rule": "frozen in the config; see decision_rule",
    }
    print(f"\nVERDICT: {record['decision']['verdict']}")

    record["elapsed_seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record), indent=1, sort_keys=True),
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
