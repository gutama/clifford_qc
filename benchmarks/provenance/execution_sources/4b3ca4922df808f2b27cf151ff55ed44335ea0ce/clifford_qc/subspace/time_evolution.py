r"""Time-evolved sampling states for QSCI (Phase 16A of ``PLAN.md``).

Time-evolved QSCI samples configurations from ``|psi(t)> = exp(-iHt)|psi_0>``
rather than from a variationally prepared state.  A short evolution of a
reference determinant spreads its weight over the configurations ``H`` couples
it to, order by order in ``t``, with no optimization loop.  Sampling at several
times and pooling the draws covers configurations one time can miss.  This
module supplies the states; :func:`~clifford_qc.subspace.qsci.sample_state_input`
and :func:`~clifford_qc.subspace.qsci.sample_state_inputs` sample them, and
:func:`~clifford_qc.subspace.qsci.run_qsci` solves on what was drawn.

Time is in inverse units of the model's energy (``hbar = 1``), and the
convention is ``exp(-iHt)``, the same sign as the rotors of :mod:`clifford_qc.ir`.

Two kinds of state, two evidence categories
-------------------------------------------
*Exact propagation* (``method`` ``"expm_multiply"`` or ``"lanczos"``) applies
``exp(-iHt)`` to working precision, matrix-free, in the sector when there is
one.  No finite circuit does that, so the input is labelled ``oracle``: it
validates the sampling method and puts no preparation cost on a resource axis.
``expm_multiply`` is SciPy's (the ``research`` extra), driven through the
operator's ``LinearOperator`` view.  The trace SciPy shifts by is passed in
exactly, from :meth:`SectorOperator.trace` or :meth:`PauliLinearOperator.trace`,
rather than left for SciPy to estimate from a matrix-free object.  ``lanczos``
is a short-iteration Krylov propagator with an a-posteriori step-error control,
and it needs NumPy only.

*A Trotter circuit* (``method="trotter"``) is an IR ``Program`` of Pauli rotors,
first or second order, that a device could run.  That input is
``implementable`` with one preparation circuit.  It is *validated*, not
assumed: the circuit's state is compared with exact propagation of the same
reference, and the fidelity, the energy drift and any sector leakage are
recorded on the input.  Single Pauli rotors do not conserve particle number
even when their sum does, so the circuit runs in the full ``2^n`` space.  A
leaking state is then sampled with post-selection, the path ``adapt_vqe_state``
already takes.

Nothing here claims that a time-evolved input beats another.  Which times to
sample, and against which inputs to compare, belongs to a declared experiment.
"""

from __future__ import annotations

import math

import numpy as np

from ..backends.sector_statevector import SectorOperator
from ..capabilities import available, require
from ..ir import PauliSum, PauliWord, Program, Rotor
from ..multivector import MV
from ..pauli_action import PauliLinearOperator, parity, word_masks
from .qsci import (
    IMPLEMENTABLE, ORACLE, StateInput, _determinant_amplitudes, _to_sampling_basis,
)

__all__ = [
    "apply_program",
    "propagate",
    "time_evolved_state",
    "trotter_program",
]

_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)
_EXACT_METHODS = ("expm_multiply", "lanczos")
_HERMITICITY_TOL = 1e-10


# ------------------------------------------------------------------ propagation

def _check_times(times) -> np.ndarray:
    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("need at least one time")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    return values


def _resolve_exact_method(method: str) -> str:
    if method == "auto":
        return "expm_multiply" if available("sparse_linalg") else "lanczos"
    if method not in _EXACT_METHODS:
        raise ValueError(f"exact propagation method must be 'auto' or one of "
                         f"{_EXACT_METHODS}, got {method!r}")
    return method


def _krylov_step(matvec, vector: np.ndarray, dt: float, dimension: int
                 ) -> tuple[np.ndarray, float]:
    """``exp(-i H dt) v`` in an ``m``-step Lanczos basis, with its error estimate.

    Full reorthogonalization keeps the basis orthonormal at these sizes.  The
    estimate is the standard ``beta_m |[exp(-i T dt)]_(m-1, 0)|`` for the
    residual the Krylov space omits.  It is exactly zero on happy breakdown,
    where the space is invariant and the step is exact.
    """
    beta = float(np.linalg.norm(vector))
    if beta == 0.0:
        return vector.copy(), 0.0
    basis = [vector / beta]
    alphas: list[float] = []
    betas: list[float] = []
    closed = False
    for j in range(dimension):
        w = matvec(basis[j])
        alpha = float(np.vdot(basis[j], w).real)
        alphas.append(alpha)
        stacked = np.column_stack(basis)
        for _ in range(2):
            w = w - stacked @ (stacked.conj().T @ w)
        b = float(np.linalg.norm(w))
        if b <= 1e-14 * max(1.0, abs(alpha)) or j == dimension - 1:
            closed = b <= 1e-14 * max(1.0, abs(alpha))
            tail = b
            break
        betas.append(b)
        basis.append(w / b)
    size = len(alphas)
    tridiagonal = (np.diag(alphas) + np.diag(betas[:size - 1], 1)
                   + np.diag(betas[:size - 1], -1))
    values, vectors = np.linalg.eigh(tridiagonal)
    coefficients = vectors @ (np.exp(-1j * values * dt) * vectors[0].conj())
    error = 0.0 if closed else beta * tail * abs(coefficients[-1])
    return beta * (np.column_stack(basis[:size]) @ coefficients), error


def _krylov_propagate(matvec, psi: np.ndarray, time: float, *, dimension: int,
                      tol: float) -> np.ndarray:
    """Adaptive Lanczos propagation to ``time``: halve the step until it is accurate."""
    state = psi.copy()
    remaining, dt = float(time), float(time)
    while abs(remaining) > 0.0:
        dt = math.copysign(min(abs(dt), abs(remaining)), remaining)
        for _ in range(60):
            candidate, error = _krylov_step(matvec, state, dt, dimension)
            if error <= tol:
                break
            dt *= 0.5
        else:  # pragma: no cover - needs a pathological operator
            raise RuntimeError("Lanczos propagation could not meet its tolerance")
        state = candidate
        remaining -= dt
        if abs(remaining) < 1e-15 * max(1.0, abs(time)):
            remaining = 0.0
    return state


def _hermitian(operator) -> None:
    if not operator.mv.is_hermitian(_HERMITICITY_TOL):
        raise ValueError("time evolution needs a Hermitian Hamiltonian")


def propagate(operator, psi, times, *, method: str = "auto", tol: float = 1e-12,
              krylov_dimension: int = 30) -> np.ndarray:
    """``exp(-iHt)|psi>`` for each ``t`` in ``times``, matrix-free.

    ``operator`` is a :class:`SectorOperator` (``psi`` holds sector amplitudes)
    or a :class:`PauliLinearOperator` (``psi`` holds ``2^n`` amplitudes).
    Returns an array of shape ``(len(times), dimension)``.  Each time is
    propagated from ``psi`` independently, so an error at one time does not
    carry into the next.  ``method`` is ``"expm_multiply"``, ``"lanczos"``,
    or ``"auto"`` (the first when SciPy is installed); ``tol`` is the
    Lanczos per-step error target.
    """
    if not isinstance(operator, (SectorOperator, PauliLinearOperator)):
        raise TypeError("operator must be a SectorOperator or PauliLinearOperator")
    _hermitian(operator)
    method = _resolve_exact_method(method)
    values = _check_times(times)
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size != operator.dimension:
        raise ValueError(f"state has {psi.size} amplitudes, operator acts on "
                         f"{operator.dimension}")
    if krylov_dimension < 2:
        raise ValueError("krylov_dimension must be at least 2")
    out = np.empty((values.size, psi.size), dtype=complex)
    if method == "expm_multiply":
        require("sparse_linalg", feature="clifford_qc.subspace.time_evolution")
        from scipy.sparse.linalg import expm_multiply

        linear = operator.as_linear_operator()
        trace = operator.trace()
        for k, t in enumerate(values):
            out[k] = (psi.copy() if t == 0.0 else
                      expm_multiply(-1j * t * linear, psi, traceA=-1j * t * trace))
    else:
        for k, t in enumerate(values):
            out[k] = _krylov_propagate(operator.matvec, psi, float(t),
                                       dimension=krylov_dimension, tol=tol)
    return out


# ------------------------------------------------------------------ circuits

def _hamiltonian_terms(hamiltonian) -> tuple[int, list[tuple[int, float]], float]:
    """``(n, [(code, real coefficient)], identity coefficient)`` in code order."""
    mv = hamiltonian.to_mv() if isinstance(hamiltonian, PauliSum) else hamiltonian
    if not isinstance(mv, MV):
        raise TypeError("hamiltonian must be a PauliSum or MV")
    scale = max((abs(c) for c in mv.terms.values()), default=1.0)
    terms = []
    identity = 0.0
    for code in sorted(mv.terms):
        coefficient = complex(mv.terms[code])
        if abs(coefficient.imag) > _HERMITICITY_TOL * max(scale, 1.0):
            raise ValueError("a Trotter circuit needs a Hermitian Hamiltonian "
                             "with real Pauli coefficients")
        if code == 0:
            identity = coefficient.real
        elif coefficient.real != 0.0:
            terms.append((code, coefficient.real))
    if not terms:
        raise ValueError("the Hamiltonian has no non-identity terms to evolve")
    return mv.n, terms, identity


def trotter_program(hamiltonian, time: float, *, steps: int,
                    order: int = 2) -> Program:
    """A product-formula circuit for ``exp(-iHt)`` as an IR ``Program``.

    Terms are taken in Pauli-code order, and the program's unitary is exactly
    the product the ``gates`` module forms.  ``order=1`` is
    :func:`clifford_qc.gates.trotter_unitary`, ``rotor(P, 2 c dt)`` per term
    and step.  ``order=2`` is the symmetric formula of
    :func:`clifford_qc.gates.trotter2_unitary`: a half sweep, then the same
    terms reversed.  Adjacent rotors on the same word are fused, which
    changes the rotor count and not the unitary.  The identity term is a global
    phase and is omitted; :func:`time_evolved_state` records its coefficient.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    n, terms, _ = _hamiltonian_terms(hamiltonian)
    dt = time / steps
    if order == 1:
        # gates.trotter_unitary forms the operator product R_1 R_2 ... R_K, in
        # which R_K acts on the state first; a circuit lists gates in the
        # order they act, so the sweep runs the terms backwards.
        sweep = [(code, 2.0 * c * dt) for code, c in reversed(terms)]
    else:
        half = [(code, c * dt) for code, c in terms]
        sweep = half + half[::-1]
    fused: list[list] = []
    for _ in range(steps):
        for code, angle in sweep:
            if fused and fused[-1][0] == code:
                fused[-1][1] += angle
            else:
                fused.append([code, angle])
    program = Program(n)
    for code, angle in fused:
        if angle != 0.0:
            program.append(Rotor(PauliWord(n, code), float(angle)))
    return program


def _apply_mv(mv: MV, psi: np.ndarray, index: np.ndarray) -> np.ndarray:
    out = np.zeros_like(psi)
    for code, coefficient in mv.terms.items():
        x_mask, z_mask, y_count = word_masks(mv.n, code)
        source = index ^ x_mask
        signs = 1.0 - 2.0 * parity(source, z_mask)
        out += complex(coefficient) * _PHASE4[y_count & 3] * signs * psi[source]
    return out


def apply_program(program: Program, psi, values=None) -> np.ndarray:
    """Run an IR ``Program`` on a full-space statevector, gate by gate.

    Each operation acts through its own few-term ``MV`` (a rotor has two
    words, a named Clifford at most four), so no ``2^n x 2^n`` matrix and no
    program-wide unitary is formed.  The latter matters for Trotter circuits,
    whose product unitary fills in Pauli support.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size != 2 ** program.n:
        raise ValueError(f"state has {psi.size} amplitudes, program acts on "
                         f"{2 ** program.n}")
    bindings = program.parameters.bind(values) if len(program.parameters) else {}
    index = np.arange(psi.size, dtype=np.int64)
    state = psi.copy()
    for op in program.ops:
        state = _apply_mv(program.op_to_mv(op, bindings), state, index)
    return state


# ------------------------------------------------------------------ inputs

def _reference_full(model) -> np.ndarray:
    try:
        return _determinant_amplitudes(model.reference, model.n)
    except ValueError:
        start = np.zeros(2 ** model.n, dtype=complex)
        start[0] = 1.0
        return apply_program(model.reference, start)


def _energy(operator, psi: np.ndarray) -> float:
    return float(operator.expectation(psi).real)


def _fidelity(first: np.ndarray, second: np.ndarray) -> float:
    overlap = abs(np.vdot(first, second)) ** 2
    return float(overlap / (np.vdot(first, first).real * np.vdot(second, second).real))


def time_evolved_state(backend, model, time: float, *, method: str = "auto",
                       trotter_steps: int | None = None, trotter_order: int = 2,
                       min_fidelity: float | None = None, tol: float = 1e-12,
                       krylov_dimension: int = 30) -> StateInput:
    """A QSCI sampling input ``exp(-iHt)|reference>``.

    ``backend`` is the model's sector backend, or ``None`` for a model with no
    particle-number sector (the spin arm).  With ``method`` ``"auto"``,
    ``"expm_multiply"`` or ``"lanczos"`` the state is propagated exactly and
    labelled ``oracle``.  With ``method="trotter"`` it is the state of
    :func:`trotter_program`'s circuit, labelled ``implementable``, and
    ``trotter_steps`` is required.  ``min_fidelity`` then refuses a circuit
    whose state falls below that fidelity with exact propagation.  The
    signature ``(backend, model)`` after binding ``time`` is the one
    :class:`~clifford_qc.subspace.fragment.QSCIFragmentSolver` accepts.
    """
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if method == "trotter":
        return _trotter_state(backend, model, time, steps=trotter_steps,
                              order=trotter_order, min_fidelity=min_fidelity,
                              tol=tol, krylov_dimension=krylov_dimension)
    if trotter_steps is not None or min_fidelity is not None:
        raise ValueError("trotter_steps and min_fidelity apply to method='trotter' "
                         "only; an exact propagator has no circuit to validate")
    used = _resolve_exact_method(method)
    if backend is None:
        operator = PauliLinearOperator(model.hamiltonian)
        reference = _reference_full(model)
    else:
        operator = backend.operator(model.hamiltonian)
        reference = backend.state_from_program(model.reference)
    state = propagate(operator, reference, [time], method=used, tol=tol,
                      krylov_dimension=krylov_dimension)[0]
    return StateInput(
        label=f"time_evolved(t={time:g})", category=ORACLE, amplitudes=state,
        basis=None if backend is None else backend.basis,
        metadata={
            "state_kind": "exact_time_evolution",
            "time": time,
            "propagator": used,
            "norm_defect": abs(float(np.linalg.norm(state)) - 1.0),
            "energy_drift": abs(_energy(operator, state) - _energy(operator, reference)),
            "time_units": "inverse model energy units (hbar = 1); exp(-iHt)",
        })


def _trotter_state(backend, model, time: float, *, steps, order: int,
                   min_fidelity: float | None, tol: float,
                   krylov_dimension: int) -> StateInput:
    if steps is None:
        raise ValueError("method='trotter' needs trotter_steps")
    if min_fidelity is not None and not 0.0 <= min_fidelity <= 1.0:
        raise ValueError("min_fidelity must lie in [0, 1]")
    program = trotter_program(model.hamiltonian, time, steps=steps, order=order)
    _, _, identity = _hamiltonian_terms(model.hamiltonian)
    reference = _reference_full(model)
    evolved = apply_program(program, reference)
    full = PauliLinearOperator(model.hamiltonian)
    exact_method = _resolve_exact_method("auto")
    exact = propagate(full, reference, [time], method=exact_method, tol=tol,
                      krylov_dimension=krylov_dimension)[0]
    fidelity = _fidelity(exact, evolved)
    if min_fidelity is not None and fidelity < min_fidelity:
        raise ValueError(
            f"the Trotter circuit reaches fidelity {fidelity:.6f} with exact "
            f"propagation, below the required {min_fidelity}; use more steps")
    amplitudes, basis, post_selection, leaked = _to_sampling_basis(backend, evolved)
    words = [op.word for op in program.ops]
    return StateInput(
        label=f"trotter{order}(t={time:g},steps={steps})", category=IMPLEMENTABLE,
        amplitudes=amplitudes, basis=basis, preparations=1,
        post_selection=post_selection,
        metadata={
            "state_kind": "trotter_circuit",
            "time": time,
            "trotter_order": order,
            "trotter_steps": steps,
            "rotor_count": len(program.ops),
            "distinct_rotor_words": len({word.code for word in words}),
            "max_rotor_weight": max(word.weight for word in words),
            "identity_phase_dropped": identity,
            "fidelity_to_exact": fidelity,
            "exact_propagator": exact_method,
            "sector_leakage": leaked,
            "energy_drift": abs(_energy(full, evolved) - _energy(full, reference)),
            "time_units": "inverse model energy units (hbar = 1); exp(-iHt)",
        })
