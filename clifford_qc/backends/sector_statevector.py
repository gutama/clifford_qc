"""Sector-restricted statevector backend: the exact tier past dense reach.

The Witt/minimal-left-ideal representation of ``PLAN.md`` §3,
with a plain engineering surface. A pure state is stored as its amplitudes on
the occupation words of one particle-number (and optionally ``S_z``) sector --
``C(n,k)`` components rather than ``2^n`` -- and every operator acts on that
vector directly, so no ``2^n x 2^n`` matrix and no ``2^n`` index table is ever
built.

The action of a Pauli word is a bit-mask gather. Writing
``W = i^{n_Y} X^x Z^z``, ``W|b> = i^{n_Y} (-1)^{|z & b|} |b xor x>``: one XOR
per basis element, one phase, and a lookup of where ``b xor x`` sits in the
sector. Words are therefore **grouped by X-mask**: every word in a group shares
the same permutation ``b -> b xor x``, so the gather is computed once per group
and only the diagonal phases differ. Those phases do not depend on the state
either, so each group collapses to one precomputed coefficient vector and a
matvec becomes a handful of gather-multiply-scatter passes.

*Why term-wise projection is exact.* Individual words of a number-conserving
Hamiltonian do not conserve particle number -- that is the same leakage the
chemistry pool documents -- so most words map part of the sector outside it.
Dropping those components term by term is not an approximation: ``H`` commutes
with the sector projector ``P``, so for ``|psi>`` in the sector
``H|psi> = P H |psi> = sum_w h_w (P W_w |psi>)``, and ``P`` distributes over the
sum. The out-of-sector pieces cancel in the total; projecting each term is the
same arithmetic in a different order.

Reference tier, as the plan is explicit about: this serves baselines and
large-``n`` extension, not the compactness claim. What it buys is exactness
where ``dense_reference.exact_ground`` cannot go and ``sparse.to_sparse`` costs
``(#terms) 2^n`` nonzeros -- and, for a DMFT-style outer loop, an operator whose
expensive part is built once and reused across solves.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from ..ir import PauliSum, Program
from ..multivector import MV
from ..pauli_action import parity, word_masks

_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)


def _spin_sites(n: int, spin_ordering="interleaved") -> tuple[list[int], list[int]]:
    """Return up/down qubit indices for an explicit spin-orbital convention."""
    if spin_ordering == "interleaved":
        return ([j for j in range(n) if j % 2 == 0],
                [j for j in range(n) if j % 2 == 1])
    if spin_ordering == "blocked":
        if n % 2:
            raise ValueError("blocked spin ordering requires an even qubit count")
        return list(range(n // 2)), list(range(n // 2, n))
    if isinstance(spin_ordering, str):
        raise ValueError("spin_ordering must be 'interleaved', 'blocked', or a "
                         "length-n sequence of up/down labels")
    labels = tuple(spin_ordering)
    if len(labels) != n:
        raise ValueError(f"spin ordering has {len(labels)} entries, expected {n}")
    up_labels = {0, "up", "alpha", "+"}
    down_labels = {1, "down", "beta", "-"}
    unknown = [label for label in labels
               if label not in up_labels and label not in down_labels]
    if unknown:
        raise ValueError(f"unrecognised spin labels: {unknown!r}")
    return ([j for j, label in enumerate(labels) if label in up_labels],
            [j for j, label in enumerate(labels) if label in down_labels])


def sector_basis(n: int, n_electrons: int, sz: float | None = None, *,
                 spin_ordering="interleaved") -> np.ndarray:
    """Sorted occupation bitmasks of a ``(N, S_z)`` sector.

    Built by combination, never by filtering ``2^n`` candidates: for a fixed
    ``S_z`` the up and down sublattices are chosen independently, so the work is
    proportional to the sector size itself. (``sparse.sector_indices`` does
    enumerate the full space -- it exists to mask an already-dense matrix, and
    the contrast is the point of this module.)

    Qubit ``j`` is bit ``n-1-j``, matching ``pauli_action.word_masks`` and the dense
    bridge. ``spin_ordering`` may be ``"interleaved"`` (even qubits up),
    ``"blocked"`` (all up then all down), or an explicit sequence of labels.
    """
    if not (0 <= n_electrons <= n):
        raise ValueError(f"n_electrons must be in [0, {n}]")
    def masks(sites, count):
        for chosen in combinations(sites, count):
            mask = 0
            for j in chosen:
                mask |= 1 << (n - 1 - j)
            yield mask

    if sz is None:
        out = np.fromiter(masks(range(n), n_electrons), dtype=np.int64,
                          count=math.comb(n, n_electrons))
        return np.sort(out)
    up_sites, down_sites = _spin_sites(n, spin_ordering)
    two_sz = round(2.0 * float(sz))
    if abs(2.0 * float(sz) - two_sz) > 1e-12:
        raise ValueError("sz must be an integer or half-integer")
    if (n_electrons + two_sz) % 2:
        return np.zeros(0, dtype=np.int64)
    n_up = (n_electrons + two_sz) // 2
    n_down = n_electrons - n_up
    if not (0 <= n_up <= len(up_sites) and 0 <= n_down <= len(down_sites)):
        return np.zeros(0, dtype=np.int64)
    ups = np.fromiter(masks(up_sites, n_up), dtype=np.int64,
                      count=math.comb(len(up_sites), n_up))
    downs = np.fromiter(masks(down_sites, n_down), dtype=np.int64,
                        count=math.comb(len(down_sites), n_down))
    return np.sort((ups[:, None] | downs[None, :]).reshape(-1))


def sector_projector(n: int, n_electrons: int, sz: float | None = None, *,
                     spin_ordering="interleaved") -> MV:
    """The sector projector as a multivector -- validation and theory, small ``n``.

    ``P = sum_patterns prod_j (n_j or 1 - n_j)`` over the sector's occupation
    words. This is the object the ideal language names (``Cl(2n,C) P_0``
    restricted to fixed weight), and it is what the idempotence and commutation
    checks are run against. It has one term per pattern times ``2^n`` word
    products, so it is a small-``n`` object by construction; the backend never
    builds it.
    """
    from ..fermion import number_op_pauli
    from ..pauli import I

    basis = sector_basis(n, n_electrons, sz,
                         spin_ordering=spin_ordering)
    identity = I(n)
    projector = MV(n)
    for mask in basis.tolist():
        term = identity
        for j in range(n):
            occupied = (mask >> (n - 1 - j)) & 1
            factor = number_op_pauli(n, j)
            term = term * (factor if occupied else identity - factor)
        projector = projector + term
    return projector


class SectorOperator:
    """A ``PauliSum`` compiled to gather-scatter passes on one sector.

    Construction groups the words by X-mask, resolves each group's permutation
    once, and folds every word's phase and coefficient into a single vector, so
    an application costs one pass per group rather than one per word. That is
    what makes a repeated-solve workflow (a DMFT-style outer loop, a parameter
    sweep) pay the compilation once.
    """

    def __init__(self, backend: "SectorStatevectorBackend", operator, *,
                 precompute: bool = True, validate_sector: bool = True):
        mv = operator.to_mv() if isinstance(operator, PauliSum) else operator
        if mv.n != backend.n:
            raise ValueError("operator and sector act on different qubit counts")
        self.backend = backend
        self.n = mv.n
        self.precompute = bool(precompute)
        groups: dict[int, list[tuple[int, int, complex]]] = {}
        for code, coeff in mv.terms.items():
            x_mask, z_mask, y_count = word_masks(mv.n, code)
            groups.setdefault(x_mask, []).append((z_mask, y_count, complex(coeff)))
        self._groups = sorted(groups.items())
        self.words = len(mv.terms)
        self.groups = len(groups)
        if validate_sector:
            self._validate_sector_invariance()
        self._passes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        if self.precompute:
            self._passes = [pass_ for pass_ in
                            (self._build_pass(x_mask, entries)
                             for x_mask, entries in self._groups)
                            if pass_ is not None]

    def _build_pass(self, x_mask: int, entries):
        """``(source, target, coefficient)`` triple of one X-mask group.

        The permutation ``b -> b xor x`` is resolved by binary search on the
        sorted sector rather than through a ``2^n`` lookup table -- that table
        would reintroduce exactly the full-space memory this backend exists to
        avoid.
        """
        basis = self.backend.basis
        targets = basis ^ x_mask
        if x_mask == 0:
            positions = np.arange(basis.size, dtype=np.int64)
            live = np.ones(basis.size, dtype=bool)
        else:
            positions = np.searchsorted(basis, targets)
            live = positions < basis.size
            positions = np.where(live, positions, 0)
            live &= basis[positions] == targets
        coefficients = np.zeros(basis.size, dtype=complex)
        for z_mask, y_count, coeff in entries:
            phase = coeff * _PHASE4[y_count & 3]
            coefficients += phase * (1.0 - 2.0 * parity(basis, z_mask))
        keep = live & (coefficients != 0.0)
        if not keep.any():
            return None
        return np.flatnonzero(keep), positions[keep], coefficients[keep]

    def _validate_sector_invariance(self, tol: float = 1e-10) -> None:
        """Reject an operator whose combined action leaks out of this sector.

        Validation is performed after words with the same X mask are combined,
        so legitimate cancellations in a number-conserving fermionic term are
        retained.  Testing words independently would reject most hopping terms.
        """
        basis = self.backend.basis
        for x_mask, entries in self._groups:
            targets = basis ^ x_mask
            positions = np.searchsorted(basis, targets)
            live = positions < basis.size
            safe = np.where(live, positions, 0)
            live &= basis[safe] == targets
            if live.all():
                continue
            coefficients = np.zeros(basis.size, dtype=complex)
            scale = 0.0
            for z_mask, y_count, coeff in entries:
                scale += abs(coeff)
                coefficients += (coeff * _PHASE4[y_count & 3]
                                 * (1.0 - 2.0 * parity(basis, z_mask)))
            if np.any(np.abs(coefficients[~live]) > tol * max(scale, 1.0)):
                raise ValueError(
                    "operator does not conserve the requested particle/spin sector")

    @property
    def dimension(self) -> int:
        return self.backend.dimension

    def memory_estimate(self) -> dict[str, int]:
        """Bytes the compiled passes hold, against one state vector.

        The precomputed form costs about ``32 G`` bytes per amplitude for ``G``
        X-mask groups, which overtakes the state vector by a wide margin on a
        many-term Hamiltonian: a 24-qubit half-filled cluster has ~850k
        amplitudes and ~23 groups, so the passes want hundreds of megabytes
        while the state wants fourteen. That is the trade ``precompute=False``
        exists for -- one-shot large-``n`` references recompute per matvec, and
        repeated-solve workflows pay once and keep it.
        """
        per_pass = sum(source.nbytes + target.nbytes + coefficients.nbytes
                       for source, target, coefficients in (self._passes or ()))
        return {"compiled_bytes": per_pass,
                "state_bytes": 16 * self.dimension,
                "groups": self.groups, "words": self.words}

    def matvec(self, psi: np.ndarray) -> np.ndarray:
        """``P H P |psi>`` -- exact on the sector, by the argument in the module docstring."""
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        if psi.size != self.dimension:
            raise ValueError(f"state has {psi.size} amplitudes, sector has "
                             f"{self.dimension}")
        out = np.zeros(self.dimension, dtype=complex)
        passes = (self._passes if self._passes is not None
                  else (self._build_pass(x_mask, entries)
                        for x_mask, entries in self._groups))
        for pass_ in passes:
            if pass_ is None:
                continue
            source, target, coefficients = pass_
            # XOR is a permutation, hence every retained target is unique.
            out[target] += coefficients * psi[source]
        return out

    def expectation(self, psi: np.ndarray) -> complex:
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        norm = float(np.vdot(psi, psi).real)
        if norm <= 0.0:
            raise ValueError("cannot take the expectation of a zero state")
        return complex(np.vdot(psi, self.matvec(psi))) / norm

    def restrict(self, indices) -> np.ndarray:
        """Exact ``H[I, I]`` on a declared set of *sector* indices.

        The sampled-subspace projection of ``PLAN.md`` Phase 8B. It
        reuses the compiled gather-scatter passes rather than deriving
        Slater-Condon rules: the restricted matrix is a submatrix of the
        operator the sector tests already validate, so it inherits that
        validation instead of needing its own.

        ``indices`` index ``self.backend.basis``, not occupation words -- use
        ``backend.index_of`` to convert.  Selecting the whole sector returns the
        full sector matrix, and permuting ``indices`` permutes rows and columns
        together, so the spectrum is order-independent.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("cannot restrict to an empty index set")
        if indices.min() < 0 or indices.max() >= self.dimension:
            raise ValueError(f"indices must lie in [0, {self.dimension})")
        if np.unique(indices).size != indices.size:
            raise ValueError("restriction indices must be distinct")
        position = np.full(self.dimension, -1, dtype=np.int64)
        position[indices] = np.arange(indices.size, dtype=np.int64)
        out = np.zeros((indices.size, indices.size), dtype=complex)
        passes = (self._passes if self._passes is not None
                  else (self._build_pass(x_mask, entries)
                        for x_mask, entries in self._groups))
        for pass_ in passes:
            if pass_ is None:
                continue
            source, target, coefficients = pass_
            rows, columns = position[target], position[source]
            keep = (rows >= 0) & (columns >= 0)
            if not keep.any():
                continue
            # Targets are unique within a pass, but two passes can hit the same
            # entry, so the accumulation has to be unbuffered.
            np.add.at(out, (rows[keep], columns[keep]), coefficients[keep])
        return out

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator((self.dimension, self.dimension), matvec=self.matvec,
                              dtype=complex)


def lanczos_ground(matvec, dimension: int, *, k: int = 1, max_iter: int = 300,
                   tol: float = 1e-9, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenpairs by Lanczos with full reorthogonalization (numpy only).

    The optional fallback the plan allows for, kept because it needs nothing
    beyond numpy and because two independent implementations of the same
    eigenproblem are worth more than one: the tests check it against ``eigsh``
    and against the dense reference.

    Full reorthogonalization rather than the three-term recurrence alone -- the
    recurrence loses orthogonality after a few dozen steps and starts returning
    spurious copies of converged eigenvalues, which on a degenerate spectrum is
    indistinguishable from a real degeneracy.

    Convergence is tested on the *residual*, ``beta_k |s_k[i]|``, not on how much
    the eigenvalue moved. Lanczos eigenvalues converge quadratically faster than
    their eigenvectors, so a settled eigenvalue can still sit next to a vector
    with a residual orders of magnitude larger -- stopping on the eigenvalue
    delivers a ``tol`` the returned vectors do not meet.
    """
    rng = np.random.default_rng(seed)
    vectors = np.zeros((dimension, min(max_iter, dimension) + 1), dtype=complex)
    start = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    vectors[:, 0] = start / np.linalg.norm(start)
    alphas: list[float] = []
    betas: list[float] = []
    for step in range(min(max_iter, dimension)):
        w = matvec(vectors[:, step])
        alpha = float(np.vdot(vectors[:, step], w).real)
        alphas.append(alpha)
        w = w - alpha * vectors[:, step]
        if step:
            w = w - betas[-1] * vectors[:, step - 1]
        # full reorthogonalization against everything built so far
        overlaps = vectors[:, :step + 1].conj().T @ w
        w = w - vectors[:, :step + 1] @ overlaps
        beta = float(np.linalg.norm(w))
        tridiagonal = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        values, small = np.linalg.eigh(tridiagonal)
        if len(alphas) >= k:
            residuals = beta * np.abs(small[-1, :k])
            if float(np.max(residuals)) < tol:
                break
        if beta <= 1e-12 or step + 1 >= dimension:
            break
        betas.append(beta)
        vectors[:, step + 1] = w / beta
    tridiagonal = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    values, small = np.linalg.eigh(tridiagonal)
    k = min(k, values.size)
    return values[:k], vectors[:, :len(alphas)] @ small[:, :k]


class SectorStatevectorBackend:
    """Exact states and spectra on one ``(N, S_z)`` sector.

    Memory is ``C(n, N)`` amplitudes (or the ``S_z``-restricted product of two
    binomials), never ``2^n``: the sector is the storage unit, not a mask over
    the full space. Nothing here builds an operator matrix, so the cost of a
    ground-state solve is (number of X-mask groups) gather passes per matvec.
    """

    def __init__(self, n: int, n_electrons: int, sz: float | None = None, *,
                 spin_ordering="interleaved"):
        self.n = int(n)
        self.n_electrons = int(n_electrons)
        self.sz = sz
        self.spin_ordering = spin_ordering
        self.basis = sector_basis(self.n, self.n_electrons, sz,
                                  spin_ordering=spin_ordering)
        if self.basis.size == 0:
            raise ValueError("the requested sector is empty")

    @property
    def dimension(self) -> int:
        return int(self.basis.size)

    def memory_estimate(self) -> dict[str, int]:
        """Amplitudes actually stored, against what the full space would need."""
        return {"sector_amplitudes": self.dimension,
                "full_space_amplitudes": 2 ** self.n,
                "sector_bytes": 16 * self.dimension,
                "full_space_bytes": 16 * 2 ** self.n}

    def index_of(self, bits) -> int:
        """Sector index of an occupation string (``'0101'``) or bitmask."""
        mask = int(bits, 2) if isinstance(bits, str) else int(bits)
        position = int(np.searchsorted(self.basis, mask))
        if position >= self.basis.size or int(self.basis[position]) != mask:
            raise KeyError(f"occupation {mask:0{self.n}b} is outside the sector")
        return position

    def occupation_state(self, bits) -> np.ndarray:
        psi = np.zeros(self.dimension, dtype=complex)
        psi[self.index_of(bits)] = 1.0
        return psi

    def state_from_program(self, program: Program) -> np.ndarray:
        """The determinant an X-gate-only reference ``Program`` prepares.

        Only X gates are read: a reference that applies anything else is not a
        computational determinant and so is not a single sector basis word.
        """
        if program.n != self.n:
            raise ValueError("program and sector have different qubit counts")
        mask = 0
        for operation in program.ops:
            if getattr(operation, "name", None) != "X":
                raise ValueError("only X-gate determinant references map to one "
                                 "sector basis state")
            for qubit in operation.qubits:
                mask ^= 1 << (self.n - 1 - qubit)
        return self.occupation_state(mask)

    def operator(self, hamiltonian, *, precompute: bool = True,
                 validate_sector: bool = True) -> SectorOperator:
        return SectorOperator(self, hamiltonian, precompute=precompute,
                              validate_sector=validate_sector)

    def matvec(self, hamiltonian, psi: np.ndarray) -> np.ndarray:
        return self.operator(hamiltonian).matvec(psi)

    def expectation(self, hamiltonian, psi: np.ndarray) -> complex:
        # An observable need not conserve the sector: its expectation depends
        # only on P O P, which the projected matvec computes exactly.
        return self.operator(hamiltonian, validate_sector=False).expectation(psi)

    def ground_state(self, hamiltonian, k: int = 1, *, method: str = "auto",
                     precompute: bool = True, **kwargs
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Lowest ``k`` eigenpairs, matrix-free.

        ``method='eigsh'`` uses ARPACK through a ``LinearOperator`` (shifted, for
        the reason ``sparse.sparse_ground`` documents: ``which='SA'`` misreports
        the minimum on a degenerate spectrum). ``method='lanczos'`` uses the
        numpy-only fallback. ``'auto'`` prefers ARPACK when SciPy is installed
        and the sector is large enough for it, and otherwise falls back --
        including the small-sector case ARPACK refuses.
        """
        mv = hamiltonian.to_mv() if isinstance(hamiltonian, PauliSum) else hamiltonian
        if not mv.is_hermitian(1e-9):
            raise ValueError("ground_state requires a Hermitian Hamiltonian")
        operator = self.operator(hamiltonian, precompute=precompute,
                                 validate_sector=True)
        if method == "auto":
            try:
                import scipy.sparse.linalg  # noqa: F401
            except ImportError:  # pragma: no cover - scipy is a research extra
                method = "lanczos"
            else:
                method = "eigsh" if k < self.dimension - 1 else "dense"
        if method == "dense":
            columns = np.eye(self.dimension, dtype=complex)
            matrix = np.column_stack([operator.matvec(columns[:, i])
                                      for i in range(self.dimension)])
            values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
            return values[:k], vectors[:, :k]
        if method == "lanczos":
            return lanczos_ground(operator.matvec, self.dimension, k=k, **kwargs)
        if method == "eigsh":
            from scipy.sparse.linalg import LinearOperator, eigsh

            from ..sparse import spectral_bound

            shift = spectral_bound(hamiltonian)
            shifted = LinearOperator(
                (self.dimension, self.dimension), dtype=complex,
                matvec=lambda v: operator.matvec(v) - shift * v)
            values, vectors = eigsh(shifted, k=k, which="LM", **kwargs)
            order = np.argsort(values)
            return values[order].real + shift, vectors[:, order]
        raise ValueError("method must be 'auto', 'eigsh', 'lanczos', or 'dense'")

    def to_dense(self, psi: np.ndarray) -> np.ndarray:
        """Embed into the full ``2^n`` space -- cross-checks at small ``n`` only."""
        full = np.zeros(2 ** self.n, dtype=complex)
        full[self.basis] = np.asarray(psi, dtype=complex).reshape(-1)
        return full

    def from_dense(self, psi: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Restrict a full-space vector, refusing one with weight outside the sector."""
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        if psi.size != 2 ** self.n:
            raise ValueError("vector does not live in the full space")
        restricted = psi[self.basis]
        leaked = float(np.vdot(psi, psi).real - np.vdot(restricted, restricted).real)
        if leaked > tol:
            raise ValueError(f"state carries weight {leaked:.2e} outside the sector")
        return restricted
