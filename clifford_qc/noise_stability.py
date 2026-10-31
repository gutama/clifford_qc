"""Sparse Heisenberg diagnostics for declared observable spans.

Pauli coefficients use normalized Hilbert--Schmidt inner product Tr(A†B)/2**n.
No state, density matrix, or full 4**n transfer matrix is allocated here.
Span closure is deliberately separate from pointwise preservation.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np

from .multivector import MV


def local_noise_adjoint(operator: MV, kind: str, strength: float) -> MV:
    """Apply identical independent one-qubit channels to every qubit.

    ``dephasing``: (1-p) rho + p Z rho Z.
    ``depolarizing``: (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z).
    ``amplitude_damping``: decay |1> -> |0> with probability gamma.
    ``z_rotation``: U rho U†, U=exp(-i theta Z/2), theta in radians.
    The probability parameters match :mod:`clifford_qc.channels` exactly.
    """
    if kind not in {"dephasing", "depolarizing", "amplitude_damping", "z_rotation"}:
        raise ValueError("unknown local channel")
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
        raise ValueError("strength must be a finite real scalar")
    try:
        p = float(strength)
    except (ValueError, TypeError) as error:
        raise ValueError("strength must be a finite real scalar") from error
    if not math.isfinite(p) or (kind != "z_rotation" and not 0 <= p <= 1):
        raise ValueError("probability must be in [0,1]; angles must be finite")
    out = dict(operator.terms)
    for j in range(operator.n):
        updated: dict[int, complex] = {}
        shift = 2*j
        for code, coefficient in out.items():
            letter = (code >> shift) & 3
            images = [(code, 1.0)]
            if kind == "dephasing" and letter in (1, 2):
                images = [(code, 1-2*p)]
            elif kind == "depolarizing" and letter:
                images = [(code, 1-4*p/3)]
            elif kind == "amplitude_damping":
                if letter in (1, 2):
                    images = [(code, math.sqrt(1-p))]
                elif letter == 3:
                    images = [(code, 1-p), (code & ~(3 << shift), p)]
            elif kind == "z_rotation" and letter in (1, 2):
                other = (code & ~(3 << shift)) | ((3-letter) << shift)
                images = [(code, math.cos(p)),
                          (other, (-1 if letter == 1 else 1)*math.sin(p))]
            for target, factor in images:
                updated[target] = updated.get(target, 0j) + coefficient*factor
        out = {code: value for code, value in updated.items() if value != 0}
    return MV(operator.n, out)


def observable_defect(operator: MV, image: MV) -> dict:
    """State-independent operator-norm bias bound and normalized HS defect.

    The l1 Pauli-coefficient bound upper-bounds ||E†(O)-O||_op. For any density
    rho it therefore bounds |Tr[rho (E†(O)-O)]|, including zero-mean controls.
    """
    if operator.n != image.n:
        raise ValueError("operator registers differ")
    delta = image-operator
    norm = math.sqrt(sum(abs(c)**2 for c in operator.terms.values()))
    defect = math.sqrt(sum(abs(c)**2 for c in delta.terms.values()))
    return {"hs_norm": norm, "hs_defect": defect,
            "relative_hs_defect": defect/norm if norm else 0.0,
            "bias_bound_pauli_l1": float(sum(abs(c) for c in delta.terms.values()))}


def operator_span_diagnostics(operators: Sequence[MV],
                              image: Callable[[MV], MV], *, rtol: float = 1e-10) -> dict:
    """Measure image rank, leakage, attenuation and fixed directions.

    Q is an orthonormal basis of the *input* span. Metrics are spectral norms
    in normalized HS coordinates: leakage=||(I-QQ†)E†Q||, and
    pointwise_defect=||E†Q-Q||. Restricted singular values belong to Q†E†Q.
    Principal angles describe only geometry; image rank loss is reported even
    when leakage is zero. Inputs may be dependent and individually rescaled.
    ``fixed_dimension`` is a numerical kernel count, not an algebra theorem.
    """
    if not operators or not all(isinstance(o, MV) for o in operators):
        raise ValueError("a nonempty sequence of MV operators is required")
    if any(o.n != operators[0].n for o in operators):
        raise ValueError("operator registers differ")
    if (isinstance(rtol, (bool, np.bool_)) or not isinstance(rtol, Real)
            or not 0 < rtol < 1):
        raise ValueError("rtol must be in (0,1)")
    normalized = []
    for op in operators:
        if any(not np.isfinite(c) for c in op.terms.values()):
            raise ValueError("nonfinite operator coefficient")
        scale = max((abs(c) for c in op.terms.values()), default=0)
        if scale:
            normalized.append(op/scale)
    if not normalized:
        raise ValueError("operator span is zero")
    images = [image(op) for op in normalized]
    if any(not isinstance(o, MV) or o.n != operators[0].n for o in images):
        raise ValueError("channel image has a different register")
    if any(not np.isfinite(c) for o in images for c in o.terms.values()):
        raise ValueError("nonfinite channel image")
    support = sorted({c for op in normalized+images for c in op.terms})
    a = np.array([[op.terms.get(c, 0j) for op in normalized] for c in support])
    b = np.array([[op.terms.get(c, 0j) for op in images] for c in support])
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    rank = int(np.count_nonzero(s > rtol*s[0]))
    q = u[:, :rank]
    mapped = (b @ vh[:rank].conj().T)/s[:rank]
    transfer = q.conj().T @ mapped
    leakage = mapped-q @ transfer
    defect = mapped-q
    sv = np.linalg.svd(transfer, compute_uv=False)
    image_u, image_s, _ = np.linalg.svd(mapped, full_matrices=False)
    # Absolute resolution relative to a unit input direction detects annihilation.
    image_rank = int(np.count_nonzero(image_s > rtol*max(1., image_s[0])))
    cosines = np.linalg.svd(q.conj().T @ image_u[:, :image_rank], compute_uv=False)
    angle = (math.pi/2 if image_rank < rank else
             math.acos(float(np.clip(cosines[-1], 0, 1))))
    defect_s = np.linalg.svd(defect, compute_uv=False)
    return {"rank": rank, "image_rank": image_rank, "support_rows": len(support),
            "leakage": float(np.linalg.norm(leakage, 2)),
            "pointwise_defect": float(defect_s[0]),
            "fixed_dimension": rank-int(np.count_nonzero(defect_s > rtol)),
            "restricted_singular_values": [float(v) for v in sv],
            "max_principal_angle_radians": angle, "rtol": float(rtol)}
