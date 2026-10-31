"""Exploratory exact-channel study; no hardware, shot, or advantage claim.

Run with BLAS threads set to one. Dense density matrices are an explicitly
bounded assessment oracle (at most eight qubits), never a production backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json, write_json
from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.models import tfim
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.noise_stability import (
    local_noise_adjoint, observable_defect, operator_span_diagnostics,
)
from clifford_qc.subspace.linalg import solve_projected

CONFIG = ROOT / "docs/restart/observable_noise_design.json"
RESULT = ROOT / "docs/restart/observable_noise_results.json.gz"
SOURCES = ["clifford_qc/noise_stability.py", "benchmarks/run_observable_noise.py",
           "docs/restart/observable_noise_design.json", "clifford_qc/subspace/linalg.py",
           "clifford_qc/dense_reference.py", "clifford_qc/models/spin.py",
           "clifford_qc/models/lattice.py", "clifford_qc/models/fcidump.py",
           "clifford_qc/multivector.py", "clifford_qc/fermion.py",
           "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"]


def word(n, **factors):
    letters = ["I"]*n
    for j, letter in factors.items():
        letters[int(j)] = letter
    return MV.from_label("".join(letters))


def dense_noise(rho, kind, strength):
    """Independent Kraus implementation, applying small matrices by tensor axes."""
    p = strength
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1., -1.])
    if kind == "dephasing":
        kraus = [np.sqrt(1-p)*i, np.sqrt(p)*z]
    elif kind == "depolarizing":
        kraus = [np.sqrt(1-p)*i] + [np.sqrt(p/3)*o for o in (x, y, z)]
    elif kind == "amplitude_damping":
        kraus = [np.diag([1., np.sqrt(1-p)]), np.array([[0., np.sqrt(p)], [0., 0.]])]
    elif kind == "z_rotation":
        kraus = [np.diag(np.exp(-.5j*p*np.array([1., -1.])))]
    else:
        raise ValueError("unknown channel")
    n = rho.shape[0].bit_length()-1
    tensor = np.asarray(rho).reshape([2]*(2*n))
    for j in range(n):
        axes = [j, n+j] + [a for a in range(2*n) if a not in (j, n+j)]
        paired = tensor.transpose(axes).reshape(2, 2, -1)
        transformed = sum(np.einsum("ab,bcx,dc->adx", k, paired, k.conj()) for k in kraus)
        tensor = transformed.reshape([2]*(2*n)).transpose(np.argsort(axes))
    return tensor.reshape(rho.shape)


def inputs(name):
    fermionic = name in {"hubbard2", "h4"}
    if name.startswith(("plus", "ghz")):
        n = int(name[-1])
        psi = np.ones(2**n, complex)/np.sqrt(2**n)
        if name.startswith("ghz"):
            psi[:] = 0
            psi[[0, -1]] = 1/np.sqrt(2)
        h = -sum((word(n, **{str(j): "Z", str(j+1): "Z"}) for j in range(n-1)), MV(n))
    else:
        model = (tfim(int(name[-1]), h=1.2) if name.startswith("tfim") else
                 hubbard(2, U=4.) if name == "hubbard2" else
                 fcidump_model(ROOT/"benchmarks/data/h4_sto3g_r0.9.FCIDUMP"))
        n = model.n
        h = model.hamiltonian.to_mv()
        matrix = to_matrix(h)
        indices = np.arange(2**n)
        if fermionic:
            indices = np.array([k for k in indices if
                sum(int(c) for c in f"{k:0{n}b}"[::2]) == n//4 and
                sum(int(c) for c in f"{k:0{n}b}"[1::2]) == n//4])
        _, vectors = np.linalg.eigh(matrix[np.ix_(indices, indices)])
        psi = np.zeros(2**n, complex)
        psi[indices] = vectors[:, 0]
    one = MV.scalar(n, 1)
    z0, z1 = word(n, **{"0": "Z"}), word(n, **{"1": "Z"})
    observables = {"Z0": z0, "X0": word(n, **{"0": "X"}), "Z0Z1": z0*z1,
                   "X0X1": word(n, **{"0": "X", "1": "X"}),
                   "Z_parity": MV.from_label("Z"*n), "energy": h}
    if fermionic:
        observables["particle_number"] = sum((.5*(one-word(n, **{str(j): "Z"}))
                                               for j in range(n)), MV(n))
        hop = cdag_op(n, 0)*c_op(n, 2) + cdag_op(n, 2)*c_op(n, 0)
    else:
        hop = word(n, **{"0": "X"})
    generators = {"diagonal": [one, z0, z1], "mixed": [one, z0, hop]}
    return n, h, psi, observables, generators


def pair(rho, operators):
    return np.array([[np.einsum("ij,ji->", rho, operator) for operator in row] for row in operators])


def blocks(h, generators):
    s = [[a.dagger()*b for b in generators] for a in generators]
    hm = [[a.dagger()*h*b for b in generators] for a in generators]
    return s, hm


def projection(s, h):
    result = solve_projected(s, h, tau_s=1e-8, max_condition=1e8)
    diagonal = np.sqrt(np.clip(np.diag(s).real, 1e-30, None))
    spectrum = np.linalg.eigvalsh(s/diagonal[:, None]/diagonal[None, :])
    retained = spectrum[spectrum > max(1e-8, spectrum[-1]/1e8)]
    return {"energy": float(result.ground_energy), "retained_rank": len(retained),
            "normalized_overlap_eigenvalues": spectrum.tolist(),
            "retained_condition": float(retained[-1]/retained[0])}


def run(config):
    rows, cases = [], []
    for name in config["cases"]:
        n, h, psi, observables, generators = inputs(name)
        rho = np.outer(psi, psi.conj())
        dense_observables = {key: to_matrix(o) for key, o in observables.items()}
        ideal = {key: float(np.vdot(psi, o @ psi).real) for key, o in dense_observables.items()}
        prepared = {}
        for arm, gens in generators.items():
            s, hm = blocks(h, gens)
            dense_s = [[to_matrix(o) for o in row] for row in s]
            dense_h = [[to_matrix(o) for o in row] for row in hm]
            prepared[arm] = (s, hm, dense_s, dense_h,
                             projection(pair(rho, dense_s), pair(rho, dense_h)))
        cases.append({"case": name, "n": n, "ideal_observables": ideal,
                      "ideal_projections": {a: data[-1] for a, data in prepared.items()}})
        for kind in config["channels"]:
            for p in config["strengths"]:
                noisy = dense_noise(rho, kind, p)
                image = lambda o: local_noise_adjoint(o, kind, p)
                row = {"case": name, "channel": kind, "strength": p,
                       "fidelity_squared_pure_target": float(np.vdot(psi, noisy @ psi).real),
                       "trace_defect": float(abs(np.trace(noisy)-1)),
                       "hermiticity_defect": float(np.linalg.norm(noisy-noisy.conj().T)),
                       "observables": {}, "spans": {}, "projections": {}}
                for key, op in observables.items():
                    value = float(np.einsum("ij,ji->", noisy, dense_observables[key]).real)
                    heisenberg = float(np.vdot(psi, to_matrix(image(op)) @ psi).real)
                    row["observables"][key] = {"value": value, "absolute_error": abs(value-ideal[key]),
                        "duality_defect": abs(value-heisenberg), **observable_defect(op, image(op))}
                for arm, (s, hm, ds, dh, baseline) in prepared.items():
                    for label, ops in (("S", s), ("H", hm)):
                        row["spans"][arm+"_"+label] = operator_span_diagnostics(
                            [o for group in ops for o in group], image)
                    projected = projection(pair(noisy, ds), pair(noisy, dh))
                    projected["absolute_energy_drift"] = abs(projected["energy"]-baseline["energy"])
                    projected["overlap_frobenius_drift"] = float(np.linalg.norm(pair(noisy, ds)-pair(rho, ds)))
                    projected["hamiltonian_frobenius_drift"] = float(np.linalg.norm(pair(noisy, dh)-pair(rho, dh)))
                    row["projections"][arm] = projected
                rows.append(row)
    return {"schema": "clifford_qc.observable_noise.v1", "design": config,
            "evidence": "exploratory exact finite-dimensional simulation",
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "cases": cases, "rows": rows}


def verify(record, *, design=None):
    """Replay all declared points and compare every scientific numeric field."""
    design = read_json(CONFIG) if design is None else design
    if record["design"] != design:
        raise ValueError("record differs from the required design")
    regenerated = run(design)
    def compare(a, b, path="record"):
        if isinstance(b, dict):
            if set(a) != set(b):
                raise ValueError("key mismatch at "+path)
            for k in b:
                if k != "environment":
                    compare(a[k], b[k], path+"/"+k)
        elif isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("length mismatch at "+path)
            for j, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+f"/{j}")
        elif isinstance(b, float):
            atol = 2e-7 if path.endswith("max_principal_angle_radians") else 2e-9
            if not isinstance(a, (float, int)) or not np.isfinite(a) or not np.isclose(a, b, atol=atol, rtol=1e-8):
                raise ValueError("numeric mismatch at "+path)
        elif a != b:
            raise ValueError("value mismatch at "+path)
    compare(record, regenerated)
    for row in record["rows"]:
        if row["trace_defect"] > 1e-10 or row["hermiticity_defect"] > 1e-10:
            raise ValueError("invalid noisy state")
        for value in row["observables"].values():
            if value["duality_defect"] > 1e-9:
                raise ValueError("Schrodinger/Heisenberg disagreement")
            if value["absolute_error"] > value["bias_bound_pauli_l1"]+1e-9:
                raise ValueError("observable error exceeds state-independent bound")
    return len(record["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        count = verify(read_json(args.output))
        print(f"Verified {count} exact-channel outcomes")
    else:
        result = run(read_json(CONFIG))
        write_json(args.output, result)
        print(f"Published {len(result['rows'])} exact-channel outcomes to {args.output}")


if __name__ == "__main__":
    main()
