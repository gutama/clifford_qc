"""Execute the frozen Phase 16B decision experiment.

Inputs come from `configs/phase16b_feasibility.json`, which
`check_phase16b_preregistration.py` gates before this runs. Nothing here
chooses a threshold, a grid, or a rule; this module only evaluates them.

Two sweeps, deliberately kept apart because they answer different questions:

* the **stress sweep** puts an absolute Gaussian error `eps` on every
  independent component, which measures noise tolerance and nothing else;
* the **budget sweep** maps a total shot budget to per-component `eps` through
  the config's declared variance bounds, which is what `shots_to_target` and
  the `N_acase / N_rt` ratio are read from.

Real-time arms share one `c` draw per cell so their comparison is paired.
Every arm's basis is frozen in exact arithmetic first, so this is fixed-basis
estimation only: adaptive construction and selection costs are not priced, and
the record says so.

    python benchmarks/run_phase16b_feasibility.py
    python benchmarks/run_phase16b_feasibility.py --replicas 20 --out draft.json
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
CONFIG = ROOT / "configs" / "phase16b_feasibility.json"
DEFAULT_OUT = ROOT / "reference_results" / "phase16b_feasibility.json"
SCHEMA = "clifford_qc.phase16b_feasibility.v1"
PENCIL_IDENTITY_TOL = 1e-10


# --------------------------------------------------------------- instances

def build_instance(name, spec):
    """Dense Hamiltonian, reference, exact spectrum, and grid for one instance."""
    if spec["builder"].endswith("chemistry.h2"):
        from clifford_qc.models import chemistry
        model = chemistry.h2(**spec["args"])
    else:
        from clifford_qc.models import tfim
        model = tfim(**spec["args"])
    mv = model.hamiltonian.to_mv()
    H = to_matrix(mv)
    H = 0.5 * (H + H.conj().T)
    values, vectors = np.linalg.eigh(H)

    bits = spec.get("reference_bitstring")
    if bits is not None:
        psi = np.zeros(H.shape[0], dtype=complex)
        psi[int(bits, 2)] = 1.0
    else:
        from clifford_qc.backends import ExactMVBackend
        rho = to_matrix(ExactMVBackend().state(model.reference, ()))
        weights, states = np.linalg.eigh(rho)
        if abs(weights[-1] - 1.0) > 1e-9:
            raise RuntimeError(f"{name}: reference density is not a pure state")
        psi = states[:, -1]
    psi = psi / np.linalg.norm(psi)

    # The enclosure rule is frozen per qubit count in grid_rule: exact extremal
    # eigenvalues at n=4, the Pauli one-norm about the identity shift beyond it.
    identity = float(np.real(mv.terms.get(0, 0.0)))
    one_norm = float(sum(abs(v) for code, v in mv.terms.items() if code != 0))
    if model.hamiltonian.n <= 4:
        lower, upper = float(values[0]), float(values[-1])
        enclosure_rule = "exact_extremal_eigenvalues"
    else:
        lower, upper = identity - one_norm, identity + one_norm
        enclosure_rule = "pauli_one_norm_about_identity_shift"
    dt = math.pi / (upper - lower)
    branch_ok = bool((upper - lower) * dt < 2 * math.pi)
    # One-norm of the traceless part: the d-component variance weight.
    lam = one_norm
    return {
        "name": name, "model": model, "mv": mv, "H": H, "psi": psi,
        "values": values, "vectors": vectors, "exact": float(values[0]),
        "dt": dt, "energy_shift": float((lower + upper) / 2),
        "lower": lower, "upper": upper, "branch_ok": branch_ok, "lambda_one": lam,
        "enclosure_rule": enclosure_rule,
        "n_qubits": model.hamiltonian.n, "dimension": int(H.shape[0]),
        "support": distinct_energy_support(values, vectors, psi),
        "ground_weight": float(abs(vectors[:, 0].conj() @ psi) ** 2),
    }


def distinct_energy_support(values, vectors, psi, floor=1e-6):
    """Distinct energies carrying weight above the floor, grouping degeneracies."""
    weights = np.abs(vectors.conj().T @ psi) ** 2
    grouped, anchor = [], None
    for value, weight in zip(values, weights):
        if anchor is None or not np.isclose(value, anchor, atol=1e-10, rtol=1e-12):
            grouped.append(float(weight))
            anchor = value
        else:
            grouped[-1] += float(weight)
    return int(sum(weight > floor for weight in grouped))


# ------------------------------------------------------------------ solves

def whiten(S, cutoff_relative=0.0, floor_absolute=0.0, ridge=0.0):
    """Norm-normalized whitening, matching solve_projected's normalization."""
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
    """Shifted Prony pencil; energies decoded on the declared branch."""
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
# Every arm exposes the same contract: a dict of independent estimands, each
# with its exact value, its variance weight (1 for a bounded [-1,1]
# observable, the relevant one-norm for a Hamiltonian-weighted one), and
# whether it is complex; plus an assemble() turning a noisy draw into a pencil.

def real_component_count(estimands):
    return sum(2 if spec["complex"] else 1 for spec in estimands.values())


def exact_c(inst, max_lag):
    c = {0: complex(1.0)}
    for k in range(1, max_lag + 1):
        c[k] = complex(np.vdot(inst["psi"], expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]))
        c[-k] = c[k].conjugate()
    return c


def exact_d(inst, max_lag):
    d = {}
    for k in range(max_lag + 1):
        evolved = expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
        d[k] = complex(np.vdot(inst["psi"], inst["H"] @ evolved))
        if k:
            d[-k] = d[k].conjugate()
    d[0] = complex(d[0].real)
    return d


def arm_rt_hermitian(inst, m):
    c, d = exact_c(inst, m - 1), exact_d(inst, m - 1)
    lam = inst["lambda_one"]
    estimands = {f"c{k}": {"value": c[k], "weight": 1.0, "complex": True} for k in range(1, m)}
    estimands["d0"] = {"value": d[0], "weight": lam, "complex": False}
    estimands.update({f"d{k}": {"value": d[k], "weight": lam, "complex": True} for k in range(1, m)})

    def assemble(noisy):
        cc = {0: complex(1.0)}
        dd = {0: complex(noisy["d0"].real)}
        for k in range(1, m):
            cc[k] = noisy[f"c{k}"]; cc[-k] = cc[k].conjugate()
            dd[k] = noisy[f"d{k}"]; dd[-k] = dd[k].conjugate()
        S = np.array([[cc[j - i] for j in range(m)] for i in range(m)])
        Hm = np.array([[dd[j - i] for j in range(m)] for i in range(m)])
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": estimands, "assemble": assemble, "shares_c": True}


def arm_rt_unitary(inst, m):
    c = exact_c(inst, m)
    estimands = {f"c{k}": {"value": c[k], "weight": 1.0, "complex": True} for k in range(1, m + 1)}

    def assemble(noisy):
        cc = {0: complex(1.0)}
        for k in range(1, m + 1):
            cc[k] = noisy[f"c{k}"]; cc[-k] = cc[k].conjugate()
        return "unitary", cc

    return {"estimands": estimands, "assemble": assemble, "shares_c": True}


def arm_rt_trotter(inst, m, microsteps):
    """Repeated fixed Trotter step. S stays Toeplitz; V^dag H V does not."""
    from clifford_qc.gates import trotter2_unitary
    terms = [(float(np.real(v)), MV(inst["mv"].n, {code: 1.0}))
             for code, v in inst["mv"].terms.items() if code != 0]
    step = to_matrix(trotter2_unitary(terms, inst["dt"], microsteps))
    columns, current = [], inst["psi"].copy()
    for _ in range(m):
        columns.append(current.copy())
        current = step @ current
    V = np.column_stack(columns)
    lam = inst["lambda_one"]
    S_exact, H_exact = V.conj().T @ V, V.conj().T @ (inst["H"] @ V)
    fidelity = float(abs(np.vdot(expm(-1j * inst["H"] * ((m - 1) * inst["dt"])) @ inst["psi"],
                                 V[:, m - 1])) ** 2)

    estimands = {f"s{k}": {"value": complex(S_exact[0, k]), "weight": 1.0, "complex": True}
                 for k in range(1, m)}
    for i in range(m):
        estimands[f"h{i}_{i}"] = {"value": complex(H_exact[i, i].real), "weight": lam, "complex": False}
        for j in range(i + 1, m):
            estimands[f"h{i}_{j}"] = {"value": complex(H_exact[i, j]), "weight": lam, "complex": True}

    def assemble(noisy):
        ss = {0: complex(1.0)}
        for k in range(1, m):
            ss[k] = noisy[f"s{k}"]; ss[-k] = ss[k].conjugate()
        S = np.array([[ss[j - i] for j in range(m)] for i in range(m)])
        Hm = np.zeros((m, m), dtype=complex)
        for i in range(m):
            Hm[i, i] = noisy[f"h{i}_{i}"].real
            for j in range(i + 1, m):
                Hm[i, j] = noisy[f"h{i}_{j}"]
                Hm[j, i] = np.conj(noisy[f"h{i}_{j}"])
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": estimands, "assemble": assemble, "shares_c": False,
            "propagation_fidelity": fidelity}


def arm_power_krylov(inst, m):
    """Identity plus H^1..H^(m-1). Moments mu_p carry the one-norm of H^p."""
    mv = inst["mv"]
    powers, current = [], MV.scalar(mv.n, 1.0)
    for _ in range(2 * m - 1):
        current = current * mv
        powers.append(current)
    estimands = {}
    for p, power in enumerate(powers, start=1):
        weight = float(sum(abs(v) for code, v in power.terms.items() if code != 0))
        value = complex(np.vdot(inst["psi"], to_matrix(power) @ inst["psi"]).real)
        estimands[f"mu{p}"] = {"value": value, "weight": max(weight, 1e-300), "complex": False}

    def assemble(noisy):
        mu = {0: 1.0}
        for p in range(1, 2 * m):
            mu[p] = float(noisy[f"mu{p}"].real)
        S = np.array([[mu[i + j] for j in range(m)] for i in range(m)], dtype=complex)
        Hm = np.array([[mu[i + j + 1] for j in range(m)] for i in range(m)], dtype=complex)
        return "hermitian", (S, Hm)

    return {"estimands": estimands, "assemble": assemble, "shares_c": False}


def arm_acase(inst, m, pool_rule, growth):
    """Frozen A-CASE basis; the pencil is rebuilt from shared Pauli estimands."""
    from clifford_qc.states import ket_density
    from clifford_qc.subspace import pauli_orbit, run_acase

    rho = None
    if inst.get("reference_bitstring"):
        rho = ket_density(inst["n_qubits"], inst["reference_bitstring"])
    else:
        from clifford_qc.backends import ExactMVBackend
        rho = ExactMVBackend().state(inst["model"].reference, ())

    if pool_rule.startswith("clifford_qc.models.chemistry.excitation_pool"):
        from clifford_qc.models.chemistry import excitation_multivectors
        from clifford_qc.subspace.generators import Generator
        candidates = [Generator(label, ps.to_mv())
                      for label, ps in excitation_multivectors(
                          inst["n_qubits"], int(inst["model"].metadata["n_electrons"]))]
    else:
        from clifford_qc import PauliWord
        codes = sorted({code for code in inst["mv"].terms if code != 0})
        candidates = pauli_orbit([PauliWord(inst["mv"].n, code) for code in codes])

    result = run_acase(rho, inst["model"].hamiltonian, candidates,
                       max_size=m, criterion=growth["criterion"],
                       exact_ground_energy=growth["exact_ground_energy"],
                       target_error=growth["target_error"])
    bank, indices = result.result.bank, result.result.indices
    size = len(indices)
    words = sorted(bank.word_set(indices))
    rho_matrix = to_matrix(rho)
    exact_words = {code: complex(np.trace(rho_matrix @ to_matrix(MV(inst["mv"].n, {code: 1.0}))).real)
                   for code in words}
    terms = {(kind, a, b): tuple(bank.iter_operator_terms(kind, i, j))
             for kind in ("overlap", "element")
             for a, i in enumerate(indices) for b, j in enumerate(indices)}

    estimands = {f"w{code}": {"value": exact_words[code], "weight": 1.0, "complex": False}
                 for code in words if code != 0}

    def assemble(noisy):
        values = {code: (1.0 if code == 0 else float(noisy[f"w{code}"].real)) for code in words}
        S = np.zeros((size, size), dtype=complex)
        Hm = np.zeros((size, size), dtype=complex)
        for a in range(size):
            for b in range(size):
                S[a, b] = sum(co * values[cd] for cd, co in terms[("overlap", a, b)])
                Hm[a, b] = sum(co * values[cd] for cd, co in terms[("element", a, b)])
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"estimands": estimands, "assemble": assemble, "shares_c": False,
            "basis_size": size, "word_count": len(words)}


# ------------------------------------------------------------------ sweeps

def standard_normals(estimands, rng):
    """One standard-normal variate per real component, drawn in key order."""
    return {name: (rng.normal(), rng.normal() if spec["complex"] else 0.0)
            for name, spec in sorted(estimands.items())}


def apply_noise(estimands, variates, eps_of):
    out = {}
    for name, spec in estimands.items():
        eps = eps_of(name, spec)
        z_real, z_imag = variates[name]
        if spec["complex"]:
            out[name] = spec["value"] + eps * z_real + 1j * eps * z_imag
        else:
            out[name] = complex(spec["value"].real + eps * z_real)
    return out


def evaluate(arm, noisy, inst, m, policy):
    kind, data = arm["assemble"](noisy)
    if kind == "unitary":
        return solve_unitary(data, m, inst["dt"], inst["energy_shift"], **policy)
    return solve_hermitian(*data, **policy)


def summarize(energies, exact):
    """Failed solves are infinite error and stay in every statistic."""
    values = np.asarray(energies, dtype=float)
    errors = np.where(np.isfinite(values), np.abs(values - exact), np.inf)
    finite = np.isfinite(errors)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.9, method="higher")),
        "failures": int((~finite).sum()),
        "replicas": int(errors.size),
    }


def policies(config):
    reg = config["regularizers"]
    return {
        "hard_truncation": lambda eps: {"cutoff_relative": max(eps, 1e-13)},
        "per_mode_floor": lambda eps: {"floor_absolute": max(eps, 1e-13)},
        "ridge": lambda eps: {"cutoff_relative": 1e-13,
                              "ridge": float(reg["ridge"]["coefficient"])},
    }


def run_cell(arm, inst, m, eps_of, policy_fn, eps_for_policy, replicas, rng):
    energies = []
    ranks = []
    for _ in range(replicas):
        variates = standard_normals(arm["estimands"], rng)
        noisy = apply_noise(arm["estimands"], variates, eps_of)
        energy, rank = evaluate(arm, noisy, inst, m, policy_fn(eps_for_policy))
        energies.append(energy)
        ranks.append(rank)
    stats = summarize(energies, inst["exact"])
    stats["median_rank"] = float(np.median(ranks))
    return stats


# ------------------------------------------------------- decision evaluation

def shots_to_target(arm_entry, reg_name, target):
    """Smallest TESTED budget whose median error meets the target.

    No interpolation and no extrapolation: a budget grid that never meets the
    target returns a censored result, which the decision rule routes to
    UNDETERMINED rather than letting it stand in for a pass.
    """
    table = arm_entry.get("budget", {}).get(reg_name, {})
    best = None
    for m_key, budgets in table.items():
        for budget_key, stats in budgets.items():
            if stats["median"] <= target:
                budget = float(budget_key)
                if best is None or budget < best["budget"]:
                    best = {"budget": budget, "basis_size": int(m_key),
                            "eps_bounded": stats["eps_bounded"],
                            "median": stats["median"], "p90": stats["p90"],
                            "failures": stats["failures"]}
    return best


def instance_status(entry, config, target):
    """Apply the frozen ladder; return the status and the evidence behind it."""
    if entry.get("status") == "INVALID":
        return "INVALID", {"reason": "a deterministic check failed"}

    non_ridge = list(config["non_ridge_regularizers"])
    all_regs = list(config["regularizers"])
    exact_prop = list(config["exact_propagation_arms"])
    arms = entry["arms"]

    acase = {}
    for reg in all_regs:
        found = shots_to_target(arms.get("acase_control", {}), reg, target)
        if found:
            acase[reg] = found
    # The incumbent is given its best regularizer, which is conservative
    # against the candidate the experiment is trying to promote.
    acase_best = min(acase.values(), key=lambda row: row["budget"]) if acase else None

    candidates = []
    for arm_name in exact_prop:
        for reg in all_regs:
            found = shots_to_target(arms.get(arm_name, {}), reg, target)
            if not found:
                continue
            ratio = (acase_best["budget"] / found["budget"]) if acase_best else None
            candidates.append({
                "arm": arm_name, "regularizer": reg, "is_ridge": reg not in non_ridge,
                "rt_budget": found["budget"], "rt_basis_size": found["basis_size"],
                "c_noise": found["eps_bounded"], "acase_budget":
                    acase_best["budget"] if acase_best else None,
                "ratio": ratio,
                "qualifying": bool(ratio is not None and ratio >= 10.0),
            })

    qualifying = [row for row in candidates if row["qualifying"]]
    passes = [row for row in qualifying
              if not row["is_ridge"] and row["c_noise"] >= 1e-5]
    marginals = [row for row in qualifying if row["c_noise"] >= 1e-6]

    evidence = {"acase_shots_to_target": acase_best,
                "acase_censored": acase_best is None,
                "candidates": candidates}
    if passes:
        return "PASS", {**evidence, "basis": passes}
    if marginals:
        return "MARGINAL", {**evidence, "basis": marginals}
    if acase_best is None or not candidates:
        return "UNDETERMINED", {**evidence,
                                "reason": "a required cost comparison is censored on the tested grid"}
    return "FAIL", {**evidence,
                    "reason": "all comparisons complete; no qualifying candidate at c noise >= 1e-6"}


def combine(statuses, config):
    required = config["required_decision_instances"]
    values = [statuses.get(name) for name in required]
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
    sizes = list(config["basis_sizes"])
    replicas = args.replicas or int(config["noise_model"]["replicas"])
    seed_root = int(config["noise_model"]["seed_root"])
    eps_grid = list(config["noise_model"]["eps_grid"])
    budgets = [10.0 ** d for d in config["cost_model"]["budget_grid_decades"]]
    policy_map = policies(config)
    non_ridge = set(config["non_ridge_regularizers"])
    exact_prop = set(config["exact_propagation_arms"])
    microsteps = int(config["arms"]["rt_trotter"]["microsteps_per_dt"])
    growth = config["acase_pool"]["growth"]

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
        inst["reference_bitstring"] = spec.get("reference_bitstring")
        pool_rule = config["acase_pool"][name]["rule"]
        checks = {"branch_condition": inst["branch_ok"], "pencil_identity": None}

        entry = {
            "n_qubits": inst["n_qubits"], "dimension": inst["dimension"],
            "exact_ground_energy": inst["exact"], "dt": inst["dt"],
            "energy_shift": inst["energy_shift"], "lambda_one": inst["lambda_one"],
            "distinct_energy_support": inst["support"],
            "ground_space_weight": inst["ground_weight"],
            "enclosure": [inst["lower"], inst["upper"]],
            "enclosure_rule": inst["enclosure_rule"],
            "true_spectrum": [float(inst["values"][0]), float(inst["values"][-1])],
            "arms": {}, "checks": checks,
        }
        print(f"\n=== {name}: {inst['n_qubits']} qubits, support={inst['support']}, "
              f"dt={inst['dt']:.6f}, Lambda={inst['lambda_one']:.4f} ===")

        # Deterministic pencil identity: Toeplitz assembly vs direct propagation.
        identity_residual = 0.0
        for m in sizes:
            c, d = exact_c(inst, m - 1), exact_d(inst, m - 1)
            S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
            Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
            V = np.column_stack([expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
                                 for k in range(m)])
            identity_residual = max(identity_residual,
                                    float(np.max(np.abs(S - V.conj().T @ V))),
                                    float(np.max(np.abs(Hm - V.conj().T @ (inst["H"] @ V)))
                                          / max(inst["lambda_one"], 1.0)))
        checks["pencil_identity"] = identity_residual
        checks["pencil_identity_ok"] = bool(identity_residual <= PENCIL_IDENTITY_TOL)
        if not (checks["branch_condition"] and checks["pencil_identity_ok"]):
            entry["status"] = "INVALID"
            record["instances"][name] = entry
            print(f"  INVALID: branch={checks['branch_condition']} "
                  f"pencil_residual={identity_residual:.2e}")
            continue

        builders = {
            "rt_hermitian": lambda m: arm_rt_hermitian(inst, m),
            "rt_unitary": lambda m: arm_rt_unitary(inst, m),
            "rt_trotter": lambda m: arm_rt_trotter(inst, m, microsteps),
            "power_krylov_control": lambda m: arm_power_krylov(inst, m),
            "acase_control": lambda m: arm_acase(inst, m, pool_rule, growth),
        }

        for arm_name, builder in builders.items():
            arm_entry = {"stress": {}, "budget": {}, "sizes": {}}
            for m in sizes:
                try:
                    arm = builder(m)
                except Exception as exc:  # a control that cannot be built is censored, not fatal
                    arm_entry["sizes"][str(m)] = {"unavailable": repr(exc)}
                    continue
                n_comp = real_component_count(arm["estimands"])
                zero, zero_rank = evaluate(
                    arm, {k: v["value"] for k, v in arm["estimands"].items()},
                    inst, m, policy_map["hard_truncation"](0.0))
                info = {"real_components": n_comp,
                        "zero_noise_error": abs(zero - inst["exact"]) if np.isfinite(zero) else None,
                        "zero_noise_rank": zero_rank}
                for extra in ("basis_size", "word_count", "propagation_fidelity"):
                    if extra in arm:
                        info[extra] = arm[extra]
                arm_entry["sizes"][str(m)] = info

                # Stress sweep: absolute eps on every component.
                for cell, eps in enumerate(eps_grid):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed_root, position, 1, cell, m]))
                    stats = run_cell(arm, inst, m, lambda n, s: eps,
                                     policy_map["hard_truncation"], eps,
                                     1 if eps == 0 else replicas, rng)
                    arm_entry["stress"].setdefault(str(m), {})[repr(eps)] = stats

                # Budget sweep: eps derived from the declared variance bounds.
                for cell, budget in enumerate(budgets):
                    scale = math.sqrt(n_comp / budget)
                    for reg_index, (reg_name, policy_fn) in enumerate(policy_map.items()):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed_root, position, 2, cell, m,
                                                    reg_index]))
                        stats = run_cell(arm, inst, m,
                                         lambda n, s: s["weight"] * scale,
                                         policy_fn, scale, replicas, rng)
                        stats["eps_bounded"] = scale
                        bucket = (arm_entry["budget"].setdefault(reg_name, {})
                                  .setdefault(str(m), {}))
                        bucket[f"1e{int(round(math.log10(budget)))}"] = stats
            entry["arms"][arm_name] = arm_entry
            print(f"  {arm_name}: done")

        record["instances"][name] = entry

    statuses = {}
    for name, entry in record["instances"].items():
        if "skipped" in entry:
            continue
        status, evidence = instance_status(entry, config, target)
        entry["status"] = status
        entry["status_evidence"] = evidence
        statuses[name] = status
        print(f"  {name}: {status}")
    record["decision"] = {
        "instance_statuses": statuses,
        "required_instances": config["required_decision_instances"],
        "verdict": combine(statuses, config),
        "rule": "frozen in the config; see decision_rule",
        "diagnostics_cannot_promote": config["decision_rule"]["diagnostics_cannot_promote"],
    }
    print(f"\nVERDICT: {record['decision']['verdict']}")
    record["elapsed_seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record), indent=1, sort_keys=True),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
