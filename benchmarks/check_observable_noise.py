"""Independent dense check of the committed observable-noise record.

Only problem definitions are shared with the producer: the H4 Hamiltonian
comes from the FCIDUMP through ``clifford_qc.models`` and
``dense_reference.to_matrix``. Plus/GHZ, TFIM and Hubbard Hamiltonians are
rebuilt here and must equal the library models, which anchors the qubit and
Jordan-Wigner conventions. Pauli and fermion operators, reference states,
Kraus channels in both pictures, S/H pencils, the Ritz solve, Pauli-l1 bias
bounds and span geometry are all recomputed with dense NumPy. This file never
imports the producer, its sparse Heisenberg map or ``clifford_qc.diagnostics``.
Sparse support counts (``support_rows``) are producer bookkeeping and are
only range-checked.

The declared grid, the 224-outcome count and the headline numbers quoted in
docs/restart/OBSERVABLE_NOISE.md are pinned here, so editing the design,
record and document together still fails unless this checker changes too.
"""
from __future__ import annotations

import argparse
from functools import reduce
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models import tfim
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard

RECORD = ROOT / "docs/restart/observable_noise_results.json.gz"
DESIGN = ROOT / "docs/restart/observable_noise_design.json"
DOC = ROOT / "docs/restart/OBSERVABLE_NOISE.md"
FCIDUMP = ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"

GRID = {"cases": ["plus4", "plus8", "ghz4", "ghz8", "tfim4", "tfim6", "hubbard2", "h4"],
        "channels": ["dephasing", "depolarizing", "amplitude_damping", "z_rotation"],
        "strengths": [0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4]}
OUTCOMES = 224
HOPS = {"hubbard2": (0, 2), "h4": (0, 4)}
TAU_S, MAX_CONDITION, RTOL_SPAN = 1e-8, 1e8, 1e-10
ATOL, RTOL = 1e-9, 1e-8
MIN_GAP = 1e-6

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|: occupied |1> -> empty
PAULIS = np.stack([I2, X, Y, Z])


def site(n, factors):
    """Dense tensor product with qubit 0 as the leftmost factor."""
    return reduce(np.kron, [factors.get(j, I2) for j in range(n)])


def annihilate(n, j):
    return site(n, {**{k: Z for k in range(j)}, j: LOWER})


def number(n, j):
    return site(n, {j: (I2-Z)/2})


def hop(n, a, b):
    term = annihilate(n, a).conj().T @ annihilate(n, b)
    return term + term.conj().T


def kraus(kind, p):
    if kind == "dephasing":
        return [math.sqrt(1-p)*I2, math.sqrt(p)*Z]
    if kind == "depolarizing":
        return [math.sqrt(1-p)*I2] + [math.sqrt(p/3)*o for o in (X, Y, Z)]
    if kind == "amplitude_damping":
        return [np.diag([1., math.sqrt(1-p)]).astype(complex), math.sqrt(p)*LOWER]
    if kind == "z_rotation":
        return [np.diag([np.exp(-.5j*p), np.exp(.5j*p)])]
    raise ValueError(f"unknown channel {kind!r}")


def schrodinger(rho, kind, p, n):
    """Full-register Kronecker Kraus operators, one qubit after another."""
    for j in range(n):
        ops = [site(n, {j: k}) for k in kraus(kind, p)]
        rho = sum(k @ rho @ k.conj().T for k in ops)
    return rho


def heisenberg(ops, kind, p, n):
    """Apply sum_k K† O K on every qubit to a stack of dense operators.

    The one-qubit adjoint is the 4x4 matrix C[(a,b),(c,d)] =
    sum_k conj(K_k[c,a]) K_k[d,b], applied to the (row, column) index pair.
    """
    stack = np.asarray(ops)
    m, dim = stack.shape[0], 2**n
    adjoint = sum(np.einsum("ca,db->abcd", k.conj(), k) for k in kraus(kind, p)).reshape(4, 4)
    tensor = stack.reshape([m] + [2]*(2*n))
    for j in range(n):
        moved = np.moveaxis(tensor, (1+j, 1+n+j), (1, 2))
        shape = moved.shape
        flat = adjoint @ moved.reshape(m, 4, -1)
        tensor = np.moveaxis(flat.reshape(shape), (1, 2), (1+j, 1+n+j))
    return tensor.reshape(m, dim, dim)


def pauli_l1(ops, n):
    """Sum of |Tr(P O)|/2^n over Pauli words P, for a stack of operators."""
    stack = np.asarray(ops)
    m, dim = stack.shape[0], 2**n
    # Row k holds sigma_k[b,a]/2 against the (a,b) index pair: Tr(sigma_k O)/2.
    project = np.stack([s.T.reshape(-1) for s in PAULIS])/2
    block = stack.reshape(m, 1, dim, dim)
    for _ in range(n):
        level, d = block.shape[1], block.shape[2]//2
        pairs = block.reshape(m, level, 2, d, 2, d).transpose(0, 1, 2, 4, 3, 5)
        block = (project @ pairs.reshape(m, level, 4, d*d)).reshape(m, 4*level, d, d)
    return np.abs(block.reshape(m, -1)).sum(axis=1)


def hs_norm(op, n):
    return float(np.linalg.norm(op)/math.sqrt(2**n))


def ritz(s, h):
    """Normalized, thresholded generalized eigenproblem, written from scratch."""
    scale = np.sqrt(np.diag(s).real)
    if np.any(scale <= 1e-14):
        raise ValueError("generator annihilates the state")
    sb = s/scale[:, None]/scale[None, :]
    hb = h/scale[:, None]/scale[None, :]
    values, vectors = np.linalg.eigh((sb+sb.conj().T)/2)
    keep = values > max(TAU_S, values.max()/MAX_CONDITION)
    basis = vectors[:, keep]/np.sqrt(values[keep])
    reduced = basis.conj().T @ hb @ basis
    energy = np.linalg.eigvalsh((reduced+reduced.conj().T)/2)[0]
    return {"energy": float(energy), "retained_rank": int(keep.sum()),
            "normalized_overlap_eigenvalues": [float(v) for v in values],
            "retained_condition": float(values[keep].max()/values[keep].min())}


def span(ops, kind, p, n):
    """Span geometry in normalized HS coordinates of dense operators."""
    vectors = np.stack([o.reshape(-1) for o in ops], axis=1)/math.sqrt(2**n)
    norms = np.linalg.norm(vectors, axis=0)
    vectors = vectors[:, norms > 0]/norms[norms > 0]
    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.count_nonzero(s > RTOL_SPAN*s[0]))
    q = u[:, :rank]
    dim = 2**n
    images = heisenberg((q.T*math.sqrt(dim)).reshape(rank, dim, dim), kind, p, n)
    mapped = images.reshape(rank, -1).T/math.sqrt(dim)
    transfer = q.conj().T @ mapped
    defect_s = np.linalg.svd(mapped-q, compute_uv=False)
    image_u, image_s, _ = np.linalg.svd(mapped, full_matrices=False)
    image_rank = int(np.count_nonzero(image_s > RTOL_SPAN*max(1., image_s[0])))
    if image_rank < rank:
        angle = math.pi/2
    else:
        basis = image_u[:, :image_rank]
        angle = math.atan2(float(np.linalg.norm(basis-q @ (q.conj().T @ basis), 2)),
                           float(np.linalg.svd(q.conj().T @ basis, compute_uv=False)[-1]))
    return {"rank": rank, "image_rank": image_rank,
            "leakage": float(np.linalg.norm(mapped-q @ transfer, 2)),
            "pointwise_defect": float(defect_s[0]),
            "fixed_dimension": rank-int(np.count_nonzero(defect_s > RTOL_SPAN)),
            "restricted_singular_values": [float(v) for v in np.linalg.svd(transfer, compute_uv=False)],
            "max_principal_angle_radians": angle, "rtol": RTOL_SPAN}


def ising(n):
    return -sum(site(n, {j: Z, j+1: Z}) for j in range(n-1))


def ground(h, mask=None):
    index = np.flatnonzero(np.ones(h.shape[0], bool) if mask is None else mask)
    values, vectors = np.linalg.eigh(h[np.ix_(index, index)])
    if values[1]-values[0] < MIN_GAP:
        raise ValueError("reference eigenvector is not isolated")
    psi = np.zeros(h.shape[0], complex)
    psi[index] = vectors[:, 0]
    if np.linalg.norm(h @ psi-values[0]*psi) > 1e-9:
        raise ValueError("reference sector is not invariant under H")
    return psi


def case(name):
    """Independent case construction; library models are only compared against."""
    if name.startswith(("plus", "ghz")):
        n = {"plus4": 4, "plus8": 8, "ghz4": 4, "ghz8": 8}[name]
        h = ising(n)
        psi = np.ones(2**n, complex)/math.sqrt(2**n)
        if name.startswith("ghz"):
            psi = np.zeros(2**n, complex)
            psi[[0, -1]] = 1/math.sqrt(2)
    elif name in ("tfim4", "tfim6"):
        n = int(name[4:])
        h = ising(n)-1.2*sum(site(n, {j: X}) for j in range(n))
        if not np.allclose(h, to_matrix(tfim(n, h=1.2).hamiltonian.to_mv()), atol=1e-12):
            raise ValueError("TFIM convention differs from the library model")
        psi = ground(h)
    elif name in ("hubbard2", "h4"):
        if name == "hubbard2":
            n, u, mu = 4, 4., 2.
            h = -(hop(n, 0, 2)+hop(n, 1, 3))
            h = h + u*(number(n, 0) @ number(n, 1) + number(n, 2) @ number(n, 3))
            h = h - mu*sum(number(n, j) for j in range(n))
            if not np.allclose(h, to_matrix(hubbard(2, U=4.).hamiltonian.to_mv()), atol=1e-12):
                raise ValueError("Hubbard/Jordan-Wigner convention differs from the library")
        else:
            model = fcidump_model(FCIDUMP)
            n = model.n
            h = to_matrix(model.hamiltonian.to_mv())
        alpha = sum(number(n, j) for j in range(0, n, 2)).diagonal().real
        beta = sum(number(n, j) for j in range(1, n, 2)).diagonal().real
        psi = ground(h, np.isclose(alpha, n/4) & np.isclose(beta, n/4))
        raise_spin = sum(annihilate(n, j).conj().T @ annihilate(n, j+1) for j in range(0, n, 2))
        sz = (np.diag(alpha)-np.diag(beta))/2
        total_spin = raise_spin.conj().T @ raise_spin + sz @ sz + sz
        if abs(np.vdot(psi, total_spin @ psi)) > 1e-9:
            raise ValueError("fermionic reference is not a spin singlet")
    else:
        raise ValueError(f"unknown case {name!r}")
    one, z0, z1 = np.eye(2**n, dtype=complex), site(n, {0: Z}), site(n, {1: Z})
    observables = {"Z0": z0, "X0": site(n, {0: X}), "Y0": site(n, {0: Y}), "Z0Z1": z0 @ z1,
                   "X0X1": site(n, {0: X, 1: X}), "Z_parity": site(n, {j: Z for j in range(n)}),
                   "energy": h}
    third = site(n, {0: X})
    if name in HOPS:
        observables["particle_number"] = sum(number(n, j) for j in range(n))
        third = hop(n, *HOPS[name])
    generators = {"diagonal": [one, z0, z1], "mixed": [one, z0, third]}
    return n, h, psi, observables, generators


class Checker:
    def __init__(self):
        self.problems: list[str] = []

    def close(self, path, recorded, fresh):
        if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                or not math.isfinite(recorded)
                or not math.isclose(recorded, fresh, rel_tol=RTOL, abs_tol=ATOL)):
            self.problems.append(f"{path}: recorded {recorded!r}, independent {fresh!r}")

    def exact(self, path, recorded, fresh):
        if type(recorded) is not type(fresh) or recorded != fresh:
            self.problems.append(f"{path}: recorded {recorded!r}, independent {fresh!r}")

    def keys(self, path, value, expected):
        if not isinstance(value, dict) or set(value) != set(expected):
            got = sorted(value) if isinstance(value, dict) else type(value).__name__
            self.problems.append(f"{path}: fields {got}, expected {sorted(expected)}")
            return False
        return True

    def compare(self, path, recorded, fresh):
        if isinstance(fresh, dict):
            if self.keys(path, recorded, fresh):
                for key, value in fresh.items():
                    self.compare(f"{path}/{key}", recorded[key], value)
        elif isinstance(fresh, list):
            if not isinstance(recorded, list) or len(recorded) != len(fresh):
                self.problems.append(f"{path}: length differs from independent value")
            else:
                for j, (a, b) in enumerate(zip(recorded, fresh)):
                    self.compare(f"{path}/{j}", a, b)
        elif isinstance(fresh, float):
            self.close(path, recorded, fresh)
        else:
            self.exact(path, recorded, fresh)


def check(record, design, *, doc_text=None, select=None, pinned=True):
    """Return problems; ``select(case, channel, strength)`` limits recomputation."""
    c = Checker()
    c.keys("record", record, {"schema", "design", "evidence", "source_sha256",
                              "environment", "cases", "rows"})
    if c.problems:
        return c.problems
    c.exact("record/schema", record["schema"], "clifford_qc.observable_noise.v1")
    if record["design"] != design:
        c.problems.append("record design differs from the design file")
    if pinned and any(design.get(k) != v for k, v in GRID.items()):
        c.problems.append("design grid differs from the pinned grid")
    grid = [(a, b, s) for a in design["cases"] for b in design["channels"] for s in design["strengths"]]
    if pinned and len(grid) != OUTCOMES:
        c.problems.append(f"design declares {len(grid)} outcomes, expected {OUTCOMES}")
    rows = record["rows"]
    observed = [(r.get("case"), r.get("channel"), r.get("strength")) for r in rows]
    if observed != grid:
        c.problems.append("record rows are not exactly the declared grid, in order")
        return c.problems
    if [x.get("case") for x in record["cases"]] != list(design["cases"]):
        c.problems.append("record case entries differ from the declared cases")
        return c.problems
    by_case = {x["case"]: x for x in record["cases"]}
    for name in design["cases"]:
        chosen = [r for r in rows if r["case"] == name
                  and (select is None or select(r["case"], r["channel"], r["strength"]))]
        if not chosen and select is not None:
            continue
        n, h, psi, observables, generators = case(name)
        rho = np.outer(psi, psi.conj())
        ideal = {k: float(np.vdot(psi, o @ psi).real) for k, o in observables.items()}
        spectrum = np.linalg.eigvalsh(h)
        products = {arm: {"S": [[a.conj().T @ b for b in gens] for a in gens],
                          "H": [[a.conj().T @ h @ b for b in gens] for a in gens]}
                    for arm, gens in generators.items()}
        def pencil(state, arm, label):
            return np.array([[np.einsum("ij,ji->", state, p) for p in line]
                             for line in products[arm][label]])
        baseline = {arm: ritz(pencil(rho, arm, "S"), pencil(rho, arm, "H")) for arm in generators}
        c.compare(f"cases/{name}", by_case[name], {
            "case": name, "n": n, "ideal_observables": ideal,
            "hamiltonian_extreme_eigenvalues": [float(spectrum[0]), float(spectrum[-1])],
            "ideal_projections": baseline})
        names = list(observables)
        stack = np.stack([observables[k] for k in names])
        for row in chosen:
            kind, p = row["channel"], row["strength"]
            where = f"rows/{name}/{kind}/{p}"
            noisy = schrodinger(rho, kind, p, n)
            mapped = heisenberg(stack, kind, p, n)
            bounds = pauli_l1(mapped-stack, n)
            fresh_observables = {}
            for j, key in enumerate(names):
                value = np.einsum("ij,ji->", noisy, observables[key])
                if abs(value-np.vdot(psi, mapped[j] @ psi)) > 1e-9:
                    c.problems.append(f"{where}/{key}: independent pictures disagree")
                fresh_observables[key] = {
                    "value": float(value.real), "absolute_error": abs(float(value.real)-ideal[key]),
                    "hs_norm": hs_norm(stack[j], n), "hs_defect": hs_norm(mapped[j]-stack[j], n),
                    "relative_hs_defect": hs_norm(mapped[j]-stack[j], n)/hs_norm(stack[j], n),
                    "bias_bound_pauli_l1": float(bounds[j])}
            if c.keys(where+"/observables", row["observables"], fresh_observables):
                for key, fresh in fresh_observables.items():
                    recorded = dict(row["observables"][key])
                    duality = recorded.pop("duality_defect", None)
                    if not isinstance(duality, float) or not 0 <= duality <= 1e-9:
                        c.problems.append(f"{where}/{key}: recorded duality defect {duality!r}")
                    c.compare(f"{where}/observables/{key}", recorded, fresh)
                    if fresh["absolute_error"] > fresh["bias_bound_pauli_l1"]+1e-9:
                        c.problems.append(f"{where}/{key}: error exceeds its l1 bound")
            fresh_spans, fresh_projections = {}, {}
            for arm in generators:
                drifts, entry_bounds = {}, {}
                ideal_pencil = {label: pencil(rho, arm, label) for label in ("S", "H")}
                noisy_pencil = {label: pencil(noisy, arm, label) for label in ("S", "H")}
                for label in ("S", "H"):
                    entries = np.stack([o for line in products[arm][label] for o in line])
                    fresh_spans[f"{arm}_{label}"] = span(entries, kind, p, n)
                    entry_l1 = pauli_l1(heisenberg(entries, kind, p, n)-entries, n)
                    drift = np.abs(noisy_pencil[label]-ideal_pencil[label]).reshape(-1)
                    if np.any(drift > entry_l1+1e-9):
                        c.problems.append(f"{where}/{arm}/{label}: entry drift exceeds its l1 bound")
                    drifts[label], entry_bounds[label] = float(drift.max()), float(entry_l1.max())
                projected = ritz(noisy_pencil["S"], noisy_pencil["H"])
                signed = projected["energy"]-baseline[arm]["energy"]
                projected.update({
                    "signed_energy_drift": signed, "absolute_energy_drift": abs(signed),
                    "retained_rank_changed": projected["retained_rank"] != baseline[arm]["retained_rank"],
                    "overlap_frobenius_drift": float(np.linalg.norm(noisy_pencil["S"]-ideal_pencil["S"])),
                    "hamiltonian_frobenius_drift": float(np.linalg.norm(noisy_pencil["H"]-ideal_pencil["H"])),
                    "max_entry_drift": drifts, "max_entry_bias_bound_pauli_l1": entry_bounds})
                fresh_projections[arm] = projected
            if c.keys(where+"/spans", row["spans"], fresh_spans):
                for key, fresh in fresh_spans.items():
                    recorded = dict(row["spans"][key])
                    support = recorded.pop("support_rows", None)
                    if type(support) is not int or not 1 <= support <= 4**n:
                        c.problems.append(f"{where}/spans/{key}: invalid support_rows {support!r}")
                    c.compare(f"{where}/spans/{key}", recorded, fresh)
            c.compare(where+"/projections", row["projections"], fresh_projections)
            fidelity = float(np.vdot(psi, noisy @ psi).real)
            c.close(where+"/fidelity_squared_pure_target", row["fidelity_squared_pure_target"], fidelity)
            for field in ("trace_defect", "hermiticity_defect"):
                if not isinstance(row[field], float) or not 0 <= row[field] <= 1e-10:
                    c.problems.append(f"{where}/{field}: {row[field]!r}")
            if abs(np.trace(noisy)-1) > 1e-10 or np.linalg.eigvalsh(noisy)[0] < -1e-10:
                c.problems.append(f"{where}: independent channel output is not a density matrix")
            c.keys(where, row, {"case", "channel", "strength", "fidelity_squared_pure_target",
                                "trace_defect", "hermiticity_defect", "observables", "spans",
                                "projections"})
    if doc_text is not None:
        c.problems.extend(check_document(record, doc_text))
    return c.problems


def _row(record, name, channel, strength):
    return next(r for r in record["rows"]
                if (r["case"], r["channel"], r["strength"]) == (name, channel, strength))


def _case(record, name):
    return next(x for x in record["cases"] if x["case"] == name)


def headlines(record):
    """Every number the investigation document quotes, read from the record."""
    t = _row(record, "tfim6", "dephasing", .4)
    u = _row(record, "hubbard2", "dephasing", .4)
    g = _row(record, "ghz8", "z_rotation", .4)
    d = _row(record, "h4", "dephasing", .005)
    z = _row(record, "h4", "depolarizing", .2)
    q = _row(record, "plus8", "depolarizing", .02)
    m = _row(record, "h4", "dephasing", .4)
    h4 = _case(record, "h4")
    width = h4["hamiltonian_extreme_eigenvalues"][1]-h4["ideal_observables"]["energy"]
    projections = [p for r in record["rows"] for p in r["projections"].values()]
    return [
        ("outcomes", f"{len(record['rows'])}"),
        ("TFIM6 F", f"{t['fidelity_squared_pure_target']:.7f}"),
        ("TFIM6 <Z0Z1>", f"{t['observables']['Z0Z1']['value']:.7f}"),
        ("TFIM6 condition", f"{t['projections']['diagonal']['retained_condition']:.5f}"),
        ("TFIM6 energy error", f"{t['observables']['energy']['absolute_error']:.5f}"),
        ("TFIM6 Ritz drift", f"{t['projections']['diagonal']['signed_energy_drift']:.5f}"),
        ("Hubbard F", f"{u['fidelity_squared_pure_target']:.7f}"),
        ("Hubbard <Z0Z1>", f"{u['observables']['Z0Z1']['value']:.7f}"),
        ("Hubbard condition", f"{u['projections']['diagonal']['retained_condition']:.5f}"),
        ("Hubbard energy error", f"{u['observables']['energy']['absolute_error']:.5f}"),
        ("Hubbard Ritz drift", f"{u['projections']['diagonal']['signed_energy_drift']:.6f}"),
        ("GHZ8 F", f"{g['fidelity_squared_pure_target']:.7f}"),
        ("H4 dephasing F", f"{d['fidelity_squared_pure_target']:.7f}"),
        ("H4 dephasing energy error", f"{1000*d['observables']['energy']['absolute_error']:.5f}"),
        ("H4 dephasing Ritz drift", f"{1000*d['projections']['diagonal']['signed_energy_drift']:.5f}"),
        ("H4 trivial bound", f"{1000*(1-d['fidelity_squared_pure_target'])*width:.1f} mHa"),
        ("H4 depolarizing F", f"{z['fidelity_squared_pure_target']:.7f}"),
        ("H4 depolarizing energy error", f"{z['observables']['energy']['absolute_error']:.6f}"),
        ("H4 depolarizing Ritz drift", f"{z['projections']['diagonal']['signed_energy_drift']:.6f}"),
        ("H4 ideal condition", f"{h4['ideal_projections']['diagonal']['retained_condition']:.3f}"),
        ("H4 depolarizing condition", f"{z['projections']['diagonal']['retained_condition']:.4f}"),
        ("plus8 F", f"{q['fidelity_squared_pure_target']:.6f}"),
        ("plus8 <X0>", f"{q['observables']['X0']['value']:.6f}"),
        ("H4 mixed S pointwise defect", f"{m['spans']['mixed_S']['pointwise_defect']:.2f}"),
        ("H4 mixed S entry drift", f"{m['projections']['mixed']['max_entry_drift']['S']:.4f}"),
        ("rank-changed arm-rows", f"{sum(p['retained_rank_changed'] for p in projections)} arm-rows"),
        ("negative signed drifts", f"{sum(p['signed_energy_drift'] < -1e-12 for p in projections)} arm-rows"),
        ("maximum duality defect", f"{max(v['duality_defect'] for r in record['rows'] for v in r['observables'].values()):.2e}"),
    ]


def check_document(record, text):
    return [f"{label}: document does not quote record value {value!r}"
            for label, value in headlines(record) if value not in text]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    record = read_json(args.record)
    problems = check(record, read_json(DESIGN), doc_text=DOC.read_text(encoding="utf-8"))
    for problem in problems[:25]:
        print("FAIL", problem)
    if problems:
        print(f"{len(problems)} independent observable-noise mismatch(es)")
        return 1
    print(f"OK: {len(record['rows'])} outcomes independently rebuilt with dense Kraus channels "
          f"(atol={ATOL:g}, rtol={RTOL:g}); document headlines match the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
