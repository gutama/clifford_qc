r"""Orbital bases: the free parameter every fermionic cost estimate depends on.

A single-particle rotation

.. math::

    b_p = \sum_i W_{pi} a_i, \qquad W W^{T} = 1,

leaves the spectrum of a fermionic Hamiltonian exactly invariant.  It does not
leave *any* of the quantities this package reports invariant.  On the
half-filled 8-site Hubbard ring at ``U = 4t``, moving from the site basis to
the momentum basis multiplies the Pauli-word count by 19 (41 -> 775) while
dividing the determinants needed for 1.6 mHa by 4 (3658 -> 906) and multiplying
the leading-determinant weight by 5 (0.06 -> 0.31); across all the bases here
the word count spans a factor of 93.  A word count quoted without naming its
basis is therefore not a reproducible number, which is the reason this module
exists: it turns the basis from an unstated default into a named argument.
``benchmarks/run_orbital_basis.py`` regenerates those numbers.

The rotation is applied to *spatial* orbitals and shared by both spin channels,
so :math:`S^2` and :math:`S_z` are preserved and a spin-sector reference stays
in its sector.

Three currencies, three entry points:

``rotate_model``
    the rotated ``Model`` -- Hamiltonian, reference determinant, metadata.
``orbital_rotation_program``
    the rotation as a circuit in this package's own rotor IR.  A Givens
    rotation on Jordan-Wigner modes is *exactly two commuting rotors*, so no
    new gate type is involved; see :func:`givens_rotation`.
``site_basis`` / ``momentum_basis`` / ``wavelet_basis`` / ``natural_orbital_basis``
    the candidate bases themselves.

None of them assume a particular basis is best.  On a translation-invariant
chain the momentum basis needs 4x fewer determinants than the site basis; add
enough diagonal disorder and that reverses, with non-interacting natural
orbitals ahead of both.  The point of the module is that the comparison is now
runnable rather than assumed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..fermion import c_op, cdag_op
from ..ir import PauliSum, Program
from ..multivector import MV
from .lattice import SPIN_DOWN, SPIN_UP, spin_orbital
from .spin import Model

__all__ = [
    "OrbitalBasis", "site_basis", "momentum_basis", "wavelet_basis",
    "natural_orbital_basis", "daubechies_filter", "quadrature_mirror",
    "rotate_one_body", "rotate_onsite_interaction", "rotate_model",
    "as_basis", "givens_rotation", "givens_network", "orbital_rotation_program",
    "single_particle_action",
]

_ORTHOGONALITY_TOL = 1e-9


# --------------------------------------------------------------- the basis
@dataclass(frozen=True)
class OrbitalBasis:
    """A named real orthogonal single-particle basis.

    ``matrix[p, i]`` is :math:`W_{pi}`, the amplitude of site/orbital ``i`` in
    rotated orbital ``p``.  Orthogonality is checked at construction because a
    non-orthogonal ``W`` silently changes the spectrum -- the one thing a basis
    change must never do -- and the resulting energy looks plausible.
    """

    name: str
    matrix: np.ndarray
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        W = np.asarray(self.matrix, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("orbital rotation must be a square matrix")
        residual = np.abs(W @ W.T - np.eye(W.shape[0])).max()
        if residual > _ORTHOGONALITY_TOL:
            raise ValueError(
                f"orbital rotation is not orthogonal (max |W Wt - 1| = {residual:.2e})")
        object.__setattr__(self, "matrix", W)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def n_orbitals(self) -> int:
        return int(self.matrix.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"OrbitalBasis({self.name!r}, n_orbitals={self.n_orbitals})"


def site_basis(n_orbitals: int) -> OrbitalBasis:
    """The identity: the basis every builder in this package uses by default."""
    return OrbitalBasis("site", np.eye(int(n_orbitals)))


def momentum_basis(n_orbitals: int) -> OrbitalBasis:
    """Real momentum basis of a periodic chain (constant, cos/sin pairs, alternating).

    Diagonalises a translation-invariant nearest-neighbour hopping matrix, so it
    is the strongest baseline on a clean ring and a deliberately unfair one on
    anything with broken translation symmetry.
    """
    N = int(n_orbitals)
    if N < 1:
        raise ValueError("need at least one orbital")
    j = np.arange(N)
    rows = [np.full(N, 1.0 / math.sqrt(N))]
    for k in range(1, (N + 1) // 2):
        rows.append(math.sqrt(2.0 / N) * np.cos(2 * np.pi * k * j / N))
        rows.append(math.sqrt(2.0 / N) * np.sin(2 * np.pi * k * j / N))
    if N % 2 == 0:
        rows.append(((-1.0) ** j) / math.sqrt(N))
    return OrbitalBasis("momentum", np.array(rows))


def daubechies_filter(vanishing_moments: int) -> np.ndarray:
    """Daubechies analysis lowpass ``h`` with ``P`` vanishing moments.

    Extremal-phase spectral factorisation of the half-band polynomial, normalised
    to ``sum(h) = sqrt(2)`` and ``||h|| = 1``.  ``P = 2`` reproduces the canonical
    D4 taps ``[0.48296291, 0.83651630, 0.22414387, -0.12940952]``.
    """
    p = int(vanishing_moments)
    if p < 1:
        raise ValueError("need at least one vanishing moment")
    if p == 1:
        return np.array([1.0, 1.0]) / math.sqrt(2.0)
    coefficients = [float(math.comb(p - 1 + k, k)) for k in range(p)]
    z_roots = []
    for y in np.roots(coefficients[::-1]):
        # y = (2 - z - 1/z)/4  =>  z^2 - (2 - 4y) z + 1 = 0; keep the root inside
        # the unit circle so the filter is minimum phase.
        b = 2.0 - 4.0 * y
        disc = np.sqrt(complex(b * b - 4.0))
        first, second = (b + disc) / 2.0, (b - disc) / 2.0
        z_roots.append(first if abs(first) < 1.0 else second)
    h = np.real(np.poly(np.array(z_roots + [-1.0] * p)))
    h = h / np.linalg.norm(h)
    if h.sum() < 0:
        h = -h
    return h


def quadrature_mirror(h: np.ndarray) -> np.ndarray:
    """Highpass partner ``g[n] = (-1)^n h[N-1-n]`` of a lowpass filter."""
    h = np.asarray(h, dtype=float)
    last = len(h) - 1
    return np.array([(-1.0) ** k * h[last - k] for k in range(len(h))])


def _dwt_level(h: np.ndarray, m: int) -> np.ndarray:
    """One periodic filter-bank level on ``m`` samples: [approximation; detail]."""
    g = quadrature_mirror(h)
    level = np.zeros((m, m))
    half = m // 2
    for k in range(half):
        for n in range(len(h)):
            level[k, (2 * k + n) % m] += h[n]
            level[half + k, (2 * k + n) % m] += g[n]
    return level


def wavelet_basis(n_orbitals: int, vanishing_moments: int = 2, *,
                  levels: int | None = None) -> OrbitalBasis:
    """Periodic multilevel Daubechies transform as an orbital basis.

    Output order is ``[cA_J, cD_J, ..., cD_1]``: the coarse band first, then
    detail bands from coarsest to finest.  ``levels`` defaults to the full
    decomposition ``log2(n_orbitals)``.

    This is the basis the free-fermion MERA correspondence suggests, and it is
    included so the suggestion can be *tested* rather than assumed: on clean and
    disordered Hubbard chains alike it loses to whichever of the site or
    momentum basis suits the model.
    """
    N = int(n_orbitals)
    if N < 2 or N & (N - 1):
        raise ValueError("wavelet basis needs a power-of-two orbital count >= 2")
    p = int(vanishing_moments)
    max_levels = int(round(math.log2(N)))
    depth = max_levels if levels is None else int(levels)
    if not (1 <= depth <= max_levels):
        raise ValueError(f"levels must be in [1, {max_levels}]")
    h = daubechies_filter(p)
    if len(h) > N:
        raise ValueError(f"db{p} needs at least {len(h)} orbitals")
    W = np.eye(N)
    m = N
    for _ in range(depth):
        if m < 2:
            break
        stage = np.eye(N)
        stage[:m, :m] = _dwt_level(h, m)
        W = stage @ W
        m //= 2
    return OrbitalBasis(f"db{p}", W,
                        {"vanishing_moments": p, "levels": depth,
                         "taps": int(len(h))})


def natural_orbital_basis(one_body: np.ndarray, *,
                          name: str = "natural") -> OrbitalBasis:
    """Eigenbasis of the one-body matrix -- the non-interacting natural orbitals.

    Free to compute, expressible in the rotor IR like any other rotation, and
    the basis that wins outright on a strongly disordered chain.  It is the
    baseline a reviewer will ask for, so it belongs in the package rather than
    in a reviewer's rebuttal.
    """
    h = np.asarray(one_body, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("one-body matrix must be square")
    if np.abs(h - h.T).max() > 1e-10:
        raise ValueError("one-body matrix must be symmetric")
    energies, vectors = np.linalg.eigh(h)
    # rows of W are the orbitals, so transpose the column-eigenvector convention
    return OrbitalBasis(name, vectors.T, {"orbital_energies": energies.tolist()})


def as_basis(basis, n_orbitals: int) -> OrbitalBasis:
    """Resolve ``"site"``, ``"momentum"``, ``"db<P>"``, a matrix, or a basis object.

    Every public entry point takes a basis in any of those forms, so a sweep can
    be written as ``for name in ("site", "momentum", "db2"):`` without the caller
    constructing anything.
    """
    if isinstance(basis, OrbitalBasis):
        if basis.n_orbitals != n_orbitals:
            raise ValueError(
                f"basis has {basis.n_orbitals} orbitals, model has {n_orbitals}")
        return basis
    if isinstance(basis, str):
        if basis == "site":
            return site_basis(n_orbitals)
        if basis == "momentum":
            return momentum_basis(n_orbitals)
        if basis.startswith("db"):
            return wavelet_basis(n_orbitals, int(basis[2:]))
        raise ValueError(f"unknown basis name {basis!r}")
    return OrbitalBasis("custom", np.asarray(basis, dtype=float))


# ----------------------------------------------------------- the rotation
def rotate_one_body(one_body: np.ndarray, basis) -> np.ndarray:
    """``h' = W h W^T``."""
    h = np.asarray(one_body, dtype=float)
    W = as_basis(basis, h.shape[0]).matrix
    return W @ h @ W.T


def rotate_onsite_interaction(onsite_u, basis, *,
                              n_orbitals: int | None = None) -> np.ndarray:
    r"""Onsite Hubbard interaction as the rotated tensor

    .. math::

        V_{pqrs} = \sum_i U_i W_{pi} W_{qi} W_{ri} W_{si},

    coupling ``b†_{p up} b_{q up} b†_{r dn} b_{s dn}``.  A single scalar ``U``
    broadcasts over sites.  In the site basis this collapses back to
    ``U_i delta_{pq} delta_{qr} delta_{rs}``; away from it the interaction is
    dense, which is exactly why the Pauli-word count explodes.
    """
    u = np.atleast_1d(np.asarray(onsite_u, dtype=float))
    if u.size == 1 and n_orbitals is not None:
        u = np.full(int(n_orbitals), float(u[0]))
    if n_orbitals is not None and u.size != int(n_orbitals):
        raise ValueError("onsite interaction length differs from orbital count")
    W = as_basis(basis, u.size).matrix
    return np.einsum("i,pi,qi,ri,si->pqrs", u, W, W, W, W, optimize=True)


def _hamiltonian_pauli_sum(one_body: np.ndarray, interaction: np.ndarray,
                           chemical_potential: float, tol: float) -> PauliSum:
    """JW Pauli sum of ``sum_pq h'_pq b†_p b_q + sum V_pqrs (up)(dn) - mu N``."""
    sites = one_body.shape[0]
    n_qubits = 2 * sites
    total = MV(n_qubits)
    for p in range(sites):
        for q in range(sites):
            coefficient = float(one_body[p, q])
            if abs(coefficient) <= tol:
                continue
            for spin in (SPIN_UP, SPIN_DOWN):
                total = total + coefficient * (
                    cdag_op(n_qubits, spin_orbital(p, spin))
                    * c_op(n_qubits, spin_orbital(q, spin)))
    for p, q, r, s in itertools.product(range(sites), repeat=4):
        coefficient = float(interaction[p, q, r, s])
        if abs(coefficient) <= tol:
            continue
        total = total + coefficient * (
            cdag_op(n_qubits, spin_orbital(p, SPIN_UP))
            * c_op(n_qubits, spin_orbital(q, SPIN_UP))
            * cdag_op(n_qubits, spin_orbital(r, SPIN_DOWN))
            * c_op(n_qubits, spin_orbital(s, SPIN_DOWN)))
    if chemical_potential:
        for orbital in range(n_qubits):
            total = total - float(chemical_potential) * (
                cdag_op(n_qubits, orbital) * c_op(n_qubits, orbital))
    if not total.is_hermitian(1e-10):  # pragma: no cover - structural
        raise ValueError("rotated Hamiltonian came out non-Hermitian")
    terms = {code: complex(coeff.real) for code, coeff in total.terms.items()
             if abs(coeff.real) > 1e-14}
    return PauliSum(n_qubits, terms)


def _lowest_orbital_reference(n_qubits: int, n_electrons: int) -> Program:
    """Aufbau determinant: fill rotated orbitals in order, alternating spin.

    A Neel pattern is meaningless once the orbitals are delocalised, so the
    reference is chosen by orbital index instead.  The electron count and
    ``S_z`` match :func:`~clifford_qc.models.lattice.hubbard`'s half-filled
    reference, which keeps the sector -- and therefore every A-CASE comparison
    against the site basis -- unchanged.
    """
    program = Program(n_qubits)
    for electron in range(int(n_electrons)):
        orbital, spin = divmod(electron, 2)
        program.clifford("X", spin_orbital(orbital, SPIN_UP if spin == 0
                                           else SPIN_DOWN))
    return program


def rotate_model(one_body: np.ndarray, onsite_u, basis, *,
                 chemical_potential: float | None = None,
                 n_electrons: int | None = None,
                 name: str | None = None,
                 tol: float = 1e-12) -> Model:
    """A single-band Hubbard cluster expressed in an arbitrary orbital basis.

    ``one_body`` is the real symmetric hopping/onsite-energy matrix over spatial
    orbitals, ``onsite_u`` a scalar or per-site interaction.  ``chemical_potential``
    defaults to ``U/2`` as in :func:`~clifford_qc.models.lattice.hubbard`; being
    proportional to the total number operator it is basis-invariant.

    The returned ``Model`` carries the basis name and the Pauli-word count in
    its metadata, so a ladder run records *which* basis produced a number
    instead of leaving it to be inferred.
    """
    h = np.asarray(one_body, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("one-body matrix must be square")
    if np.abs(h - h.T).max() > 1e-10:
        raise ValueError("one-body matrix must be symmetric")
    sites = int(h.shape[0])
    orbital_basis = as_basis(basis, sites)
    u = np.atleast_1d(np.asarray(onsite_u, dtype=float))
    if u.size == 1:
        u = np.full(sites, float(u[0]))
    if u.size != sites:
        raise ValueError("onsite interaction length differs from orbital count")
    mu = 0.5 * float(u.mean()) if chemical_potential is None \
        else float(chemical_potential)
    electrons = sites if n_electrons is None else int(n_electrons)

    rotated_h = rotate_one_body(h, orbital_basis)
    interaction = rotate_onsite_interaction(u, orbital_basis)
    hamiltonian = _hamiltonian_pauli_sum(rotated_h, interaction, mu, tol)
    reference = _lowest_orbital_reference(2 * sites, electrons)
    sz = sum(0.5 if index % 2 == 0 else -0.5 for index in range(electrons))
    label = name or f"hubbard-like({sites} orbitals, basis={orbital_basis.name})"
    return Model(
        name=label,
        n=2 * sites,
        hamiltonian=hamiltonian,
        reference=reference,
        hva_layers=(),
        metadata={
            "kind": "fermionic_orbital_basis",
            "sites": sites,
            "n_orbitals": 1,
            "spin_orbitals": 2 * sites,
            "spin_convention": "interleaved",
            "basis": orbital_basis.name,
            "basis_metadata": dict(orbital_basis.metadata),
            "pauli_words": len(hamiltonian.terms),
            "n_electrons": electrons,
            "sz": float(sz),
            "mu": mu,
            "U": u.tolist(),
        })


# --------------------------------------------------------------- circuits
def givens_rotation(n_qubits: int, p: int, q: int, theta: float) -> Program:
    r"""``exp(theta (b†_p b_q - b†_q b_p))`` as **two commuting rotors**.

    Under Jordan-Wigner the generator is

    .. math::

        K = \tfrac{i}{2}\bigl(X_p Z\cdots Z\, Y_q - Y_p Z\cdots Z\, X_q\bigr),

    and its two Pauli words commute, so the exponential factorises exactly with
    no Trotter error.  The rotor IR therefore already expresses the Givens/
    Vaidyanathan lattice factor -- adding an orbital basis needs no new gate.
    """
    n = int(n_qubits)
    if p == q:
        raise ValueError("Givens rotation needs two distinct modes")
    low, high = (p, q) if p < q else (q, p)
    sign = 1.0 if p < q else -1.0
    letters = ["I"] * n
    for k in range(low + 1, high):
        letters[k] = "Z"  # the Jordan-Wigner string
    program = Program(n)
    first = list(letters)
    first[low], first[high] = "X", "Y"
    program.rotor("".join(first), -sign * float(theta))
    second = list(letters)
    second[low], second[high] = "Y", "X"
    program.rotor("".join(second), sign * float(theta))
    return program


def givens_network(W: np.ndarray, tol: float = 1e-10):
    """Factor a real orthogonal ``W`` into adjacent-mode Givens rotations.

    Returns ``(steps, signs)`` where ``steps`` is a list of ``(mode, angle)``
    applied in order to triangularise ``W``, and ``signs`` is the residual
    diagonal of +-1.  Only adjacent modes are rotated, which is what a linear
    device -- and the Jordan-Wigner string length -- actually favours.

    Note for anyone quoting an asymptotic count: a wavelet transform is ``O(N)``
    rotations only if arbitrary mode reordering is free.  Restricted to adjacent
    modes the count grows like ``N^1.8`` against the momentum basis's ``N^2``,
    a constant-factor advantage rather than an asymptotic one.
    """
    A = np.array(W, dtype=float, copy=True)
    n = A.shape[0]
    steps: list[tuple[int, float]] = []
    for column in range(n - 1):
        for row in range(n - 1, column, -1):
            a, b = A[row - 1, column], A[row, column]
            if abs(b) <= tol:
                continue
            theta = math.atan2(b, a)
            c, s = math.cos(theta), math.sin(theta)
            A[row - 1:row + 1, :] = np.array([[c, s], [-s, c]]) @ A[row - 1:row + 1, :]
            steps.append((row - 1, theta))
    return steps, np.sign(np.diag(A))


def orbital_rotation_program(basis, n_orbitals: int | None = None, *,
                             spinful: bool = False) -> Program:
    """The basis change ``W`` as a rotor circuit.

    Triangularising gives ``G_k ... G_1 W = S``, hence ``W = G_1^T ... G_k^T S``.
    Program operations compose with later ones on the left, so ``S`` (as ``Z``
    Cliffords) is emitted first and the transposed rotations in reverse order
    after it.  The result satisfies ``U b†_p U† = sum_q W[q, p] b†_q`` exactly --
    see :func:`single_particle_action`.

    With ``spinful=True`` the same spatial rotation is emitted on both
    interleaved spin channels of a ``2 * n_orbitals``-qubit register, which is
    the form :func:`rotate_model` assumes.
    """
    W = as_basis(basis, n_orbitals) if n_orbitals is not None else \
        (basis if isinstance(basis, OrbitalBasis)
         else OrbitalBasis("custom", np.asarray(basis, dtype=float)))
    matrix = W.matrix
    modes = matrix.shape[0]
    steps, signs = givens_network(matrix)
    channels = ((SPIN_UP, SPIN_DOWN) if spinful else (None,))
    n_qubits = 2 * modes if spinful else modes

    def qubit(mode: int, spin) -> int:
        return spin_orbital(mode, spin) if spin is not None else mode

    program = Program(n_qubits)
    for spin in channels:
        for mode, sign in enumerate(signs):
            if sign < 0:
                program.clifford("Z", qubit(mode, spin))
    for spin in channels:
        for mode, theta in reversed(steps):
            factor = givens_rotation(n_qubits, qubit(mode, spin),
                                     qubit(mode + 1, spin), -theta)
            for operation in factor.ops:
                program.append(operation)
    return program


def single_particle_action(program: Program, n_modes: int) -> np.ndarray:
    """Recover ``R`` with ``U b†_p U† = sum_q R[q, p] b†_q`` (validation helper).

    Builds the dense unitary, so it is a test-scale tool: exponential in the
    mode count, and the reason the regression tests stop at eight modes.
    """
    from ..matrix import to_matrix

    unitary = to_matrix(program.unitary())
    action = np.zeros((n_modes, n_modes))
    scale = 2.0 ** (n_modes - 1)
    for p in range(n_modes):
        rotated = unitary @ to_matrix(cdag_op(n_modes, p)) @ unitary.conj().T
        for q in range(n_modes):
            action[q, p] = np.real(
                np.trace(to_matrix(c_op(n_modes, q)) @ rotated)) / scale
    return action
