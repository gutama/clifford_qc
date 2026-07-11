"""
ga_qc_operator.py — Operator-centric quantum computing in Cl(2n, C)
====================================================================

Conceptual layer
----------------
Algebra/signature : Cl(2n, C) ≅ M(2^n, C) — the complexified Clifford algebra
                    on 2n anticommuting generators gamma_0..gamma_{2n-1},
                    gamma_i^2 = +1, gamma_i gamma_j = -gamma_j gamma_i.
Object/operator   : EVERY n-qubit object lives in this ONE algebra:
                      states       -> density multivectors rho
                                      (rho† = rho, Tr rho = 2^n <rho>_0 = 1,
                                       rho² = rho iff pure)
                      gates        -> unitary multivectors U (U†U = 1);
                                      rotation gates are rotors
                                      exp(-i θ/2 P) = cos(θ/2) - i sin(θ/2) P
                      observables  -> Hermitian multivectors
                      channels     -> Kraus multivector sets, Σ K†K = 1
Domain meaning    : operator-centric picture (Hrdina–Návrat–Vašík 2022;
                    Silva 2025): evolution rho' = U rho U†, Born rule
                    p(m) = Tr(P_m rho) = 2^n <P_m rho>_0. No preferred spinor
                    ideal; mixed states and channels are native.

Implementation substrate
------------------------
The 4^n Pauli words {I,X,Y,Z}^{⊗n} are exactly the images of the 4^n basis
blades under the Jordan–Wigner isomorphism
    gamma_{2j}   = Z_0 ... Z_{j-1} X_j
    gamma_{2j+1} = Z_0 ... Z_{j-1} Y_j
so we store multivectors sparsely in the PAULI-WORD basis
{word_code -> complex}; the geometric product is qubit-local Pauli
multiplication with phase accumulation. Blade grade is recovered by decoding
a word back through JW. Storage is O(#active words), not O(4^n) dense;
Clifford (stabilizer) circuits stay maximally sparse — Clifford conjugation
maps one word/blade to exactly one word/blade (Gottesman–Knill in blade
language).

The fermionic (Witt) layer sits on top of the gammas:
    c_j = ½(gamma_{2j} + i gamma_{2j+1}),  c†_j = ½(gamma_{2j} - i gamma_{2j+1})
with CAR {c_i, c†_j} = δ_ij, c² = 0 — Jordan–Wigner is not a bolt-on
transformation but the change of basis inside the algebra.

Validation invariants (run `python ga_qc_operator.py`)
------------------------------------------------------
  * algebra:  Pauli table, associativity fuzz, (AB)† = B†A†
  * Clifford: gamma_i gamma_j + gamma_j gamma_i = 2 δ_ij, blade-grade decode
  * fermions: CAR relations, nilpotency, number operator = ½(1 - Z_j)
  * rep map:  multivector product == matrix product; Tr == 2^n <·>_0
  * gates:    U†U = 1 for X,Y,Z,H,S,T,R_P(θ), CNOT, CZ, SWAP, Toffoli
  * states:   Tr rho = 1, purity, Bell = ¼(1 + XX - YY + ZZ)
  * physics:  CHSH = 2√2; Grover-2 finds |11> with p = 1 exactly
  * channels: Σ K†K = 1; depolarizing fixed point purity -> 1/2^n
  * dynamics: Trotter + Taylor expm vs exact eigendecomposition (TFIM)
  * structure:CNOT conjugation of any Pauli word returns ONE word (G–K)

Transfer note
-------------
Preserved  : exact algebra isomorphism Cl(2n,C) ≅ M(2^n,C); every matrix-
             formalism theorem holds verbatim.
Approximate: only `expm_taylor` (truncated series) and `trotter_unitary`
             (product formula) — both benchmarked in the suite.
Not implied: no asymptotic speedup claim over matrices; the win is
             structural (one algebra for states/gates/observables/channels,
             word sparsity, geometric semantics: gates literally rotors).
"""

from __future__ import annotations
import cmath
import math
from functools import lru_cache
from numbers import Number

TOL = 1e-12


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")


def _validate_qubit(n: int, j: int, name: str = "j") -> None:
    _validate_n(n)
    if not isinstance(j, int) or not (0 <= j < n):
        raise ValueError(f"{name} must be an int in [0, {n}), got {j!r}")


def _validate_word_code(n: int, code: int) -> None:
    _validate_n(n)
    if not isinstance(code, int) or not (0 <= code < 4 ** n):
        raise ValueError(f"word code must be an int in [0, 4**n), got {code!r}")

# ----------------------------------------------------------------------
# 1. Kernel: sparse multivectors in the Pauli-word basis of Cl(2n, C)
# ----------------------------------------------------------------------
# Word code: 2 bits per qubit. 0=I, 1=X, 2=Y, 3=Z.

_L = {0: "I", 1: "X", 2: "Y", 3: "Z"}

# single-qubit product table: (a,b) -> (phase, result_letter)
_PTAB = {}
for a in range(4):
    _PTAB[(0, a)] = (1, a)
    _PTAB[(a, 0)] = (1, a)
    _PTAB[(a, a)] = (1, 0)
_PTAB[(1, 2)] = (1j, 3);  _PTAB[(2, 1)] = (-1j, 3)   # XY = iZ
_PTAB[(2, 3)] = (1j, 1);  _PTAB[(3, 2)] = (-1j, 1)   # YZ = iX
_PTAB[(3, 1)] = (1j, 2);  _PTAB[(1, 3)] = (-1j, 2)   # ZX = iY


@lru_cache(maxsize=1_000_000)
def _word_mul(n: int, a: int, b: int) -> tuple[complex, int]:
    phase, out = 1 + 0j, 0
    for j in range(n):
        la = (a >> (2 * j)) & 3
        lb = (b >> (2 * j)) & 3
        ph, lc = _PTAB[(la, lb)]
        phase *= ph
        out |= lc << (2 * j)
    return phase, out


class MV:
    """Sparse multivector of Cl(2n, C) in the Pauli-word basis."""

    __slots__ = ("n", "terms")

    def __init__(self, n: int, terms: dict[int, complex] | None = None):
        _validate_n(n)
        self.n = n
        self.terms: dict[int, complex] = {}
        if terms:
            for k, v in terms.items():
                kk = int(k)
                _validate_word_code(n, kk)
                cv = complex(v)
                if abs(cv) > TOL:
                    self.terms[kk] = self.terms.get(kk, 0.0) + cv
            self.terms = {k: v for k, v in self.terms.items() if abs(v) > TOL}

    # -- constructors -----------------------------------------------------
    @staticmethod
    def scalar(n: int, c: complex = 1.0) -> "MV":
        return MV(n, {0: c})

    @staticmethod
    def word(n: int, letters: str, c: complex = 1.0) -> "MV":
        """Create a Pauli word such as 'XIZ' (qubit 0 is the leftmost char)."""
        _validate_n(n)
        letters = letters.upper()
        if len(letters) != n:
            raise ValueError(f"expected {n} Pauli letters, got {len(letters)}")
        code = 0
        rev = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        for j, ch in enumerate(letters):
            if ch not in rev:
                raise ValueError(f"invalid Pauli letter {ch!r}; use I, X, Y, Z")
            code |= rev[ch] << (2 * j)
        return MV(n, {code: c})

    @staticmethod
    def from_terms(n: int, terms: dict[str, complex]) -> "MV":
        """Create from {'XIZ': coeff, ...}; useful for readable Hamiltonians."""
        out = MV(n)
        for letters, coeff in terms.items():
            out = out + MV.word(n, letters, coeff)
        return out

    # -- ring operations ----------------------------------------------------
    def __add__(self, o):
        o = self._coerce(o)
        t = dict(self.terms)
        for k, v in o.terms.items():
            t[k] = t.get(k, 0.0) + v
        return MV(self.n, t)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-1) * self._coerce(o)

    def __rsub__(self, o):
        return self._coerce(o) + (-1) * self

    def __neg__(self):
        return (-1) * self

    def __rmul__(self, c):
        if isinstance(c, Number):
            return MV(self.n, {k: complex(c) * v for k, v in self.terms.items()})
        return NotImplemented

    def __mul__(self, o):
        if isinstance(o, Number):
            return MV(self.n, {k: v * complex(o) for k, v in self.terms.items()})
        o = self._coerce(o)
        out: dict[int, complex] = {}
        for a, ca in self.terms.items():
            for b, cb in o.terms.items():
                ph, m = _word_mul(self.n, a, b)
                c = ca * cb * ph
                out[m] = out.get(m, 0.0) + c
        return MV(self.n, out)

    def __matmul__(self, o):
        """Use A @ B as an explicit synonym for the geometric/operator product."""
        return self * o

    def __truediv__(self, c):
        if not isinstance(c, Number):
            return NotImplemented
        if abs(complex(c)) <= TOL:
            raise ZeroDivisionError("division by zero scalar")
        return (1 / complex(c)) * self

    def _coerce(self, o):
        if isinstance(o, Number):
            return MV.scalar(self.n, complex(o))
        if isinstance(o, MV):
            if o.n != self.n:
                raise ValueError("mixed algebra sizes")
            return o
        raise TypeError(type(o))

    # -- involutions and functionals ------------------------------------------
    def dagger(self) -> "MV":
        """Hermitian adjoint: Pauli words are Hermitian, so just conjugate."""
        return MV(self.n, {k: v.conjugate() for k, v in self.terms.items()})

    def scalar_part(self) -> complex:
        return self.terms.get(0, 0.0)

    def trace(self) -> complex:
        return (2 ** self.n) * self.scalar_part()

    def norm_hs(self) -> float:
        """√Tr(A†A); Pauli words are HS-orthogonal with Tr(P²)=2^n."""
        return math.sqrt(2 ** self.n * sum(abs(v) ** 2
                                           for v in self.terms.values()))

    def is_close(self, o, tol=1e-9) -> bool:
        return (self - self._coerce(o)).norm_hs() < tol

    def nnz(self) -> int:
        return len(self.terms)

    def copy(self) -> "MV":
        return MV(self.n, dict(self.terms))

    def is_zero(self, tol: float = 1e-12) -> bool:
        return self.norm_hs() < tol

    def is_hermitian(self, tol: float = 1e-9) -> bool:
        return self.is_close(self.dagger(), tol)

    def is_unitary(self, tol: float = 1e-9) -> bool:
        return (self.dagger() * self).is_close(I(self.n), tol)

    def is_projector(self, tol: float = 1e-9) -> bool:
        return self.is_hermitian(tol) and (self * self).is_close(self, tol)

    def is_density(self, tol: float = 1e-9, check_psd: bool = True) -> bool:
        """Density-operator check. PSD uses the matrix backend, so keep it for small n."""
        ok = self.is_hermitian(tol) and abs(self.trace() - 1) < tol
        if ok and check_psd:
            import numpy as np
            ok = bool(np.min(np.linalg.eigvalsh(to_matrix(self))) >= -tol)
        return ok

    def word_letters(self, code: int) -> str:
        _validate_word_code(self.n, code)
        return "".join(_L[(code >> (2 * j)) & 3] for j in range(self.n))

    # -- Clifford blade structure (via inverse Jordan–Wigner) -------------------
    def _letters(self, code: int) -> list[int]:
        _validate_word_code(self.n, code)
        return [(code >> (2 * j)) & 3 for j in range(self.n)]

    def blade_mask(self, code: int) -> int:
        """Generator content S ⊆ {gamma_0..gamma_{2n-1}} of one Pauli word:
        invert JW from the top qubit down."""
        letters = self._letters(code)
        S = 0
        m = 0  # number of generators owned by qubits above current j
        for j in range(self.n - 1, -1, -1):
            L = letters[j]
            if m % 2 == 1:                # dressed by Z from above: undo it
                L = {0: 3, 3: 0, 1: 2, 2: 1}[L]
            own = {0: 0, 1: 1, 2: 2, 3: 3}[L]   # I,X,Y,Z -> gens {}/{x}/{y}/{x,y}
            if own in (1, 3):
                S |= 1 << (2 * j)
            if own in (2, 3):
                S |= 1 << (2 * j + 1)
            m += (1 if own in (1, 2) else (2 if own == 3 else 0))
        return S

    def grade(self, g: int) -> "MV":
        """Project onto Clifford grade g (generator count of each word)."""
        return MV(self.n, {k: v for k, v in self.terms.items()
                           if self.blade_mask(k).bit_count() == g})

    def grades(self) -> set[int]:
        return {self.blade_mask(k).bit_count() for k in self.terms}

    def __repr__(self):
        if not self.terms:
            return "0"
        bits = []
        for k in sorted(self.terms):
            w = "".join(_L[(k >> (2 * j)) & 3] for j in range(self.n))
            bits.append(f"({self.terms[k]:.4g})·{w}")
        return " + ".join(bits)


# ----------------------------------------------------------------------
# 2. Generators, qubit dictionary, fermionic (Witt) layer
# ----------------------------------------------------------------------

def I(n):
    _validate_n(n)
    return MV.scalar(n, 1.0)

def X(n, j):
    _validate_qubit(n, j)
    return MV.word(n, "I" * j + "X" + "I" * (n - j - 1))

def Y(n, j):
    _validate_qubit(n, j)
    return MV.word(n, "I" * j + "Y" + "I" * (n - j - 1))

def Z(n, j):
    _validate_qubit(n, j)
    return MV.word(n, "I" * j + "Z" + "I" * (n - j - 1))


def pauli_string(n: int, s: str) -> MV:
    return MV.word(n, s)


def comm(A: MV, B: MV) -> MV:
    """Commutator [A,B]."""
    return A * B - B * A


def anticomm(A: MV, B: MV) -> MV:
    """Anticommutator {A,B}."""
    return A * B + B * A


def pauli_basis(n: int) -> list[MV]:
    """All 4^n Pauli words as sparse basis elements."""
    _validate_n(n)
    return [MV(n, {code: 1.0}) for code in range(4 ** n)]


def tensor(A: MV, B: MV) -> MV:
    """Tensor product with B appended after A: matrix(tensor(A,B)) = kron(A,B)."""
    out: dict[int, complex] = {}
    for ca, va in A.terms.items():
        for cb, vb in B.terms.items():
            out[ca | (cb << (2 * A.n))] = out.get(ca | (cb << (2 * A.n)), 0.0) + va * vb
    return MV(A.n + B.n, out)


def gamma(n: int, i: int) -> MV:
    """Anticommuting Clifford generator via Jordan–Wigner:
    gamma_{2j} = Z…Z X_j,  gamma_{2j+1} = Z…Z Y_j."""
    _validate_n(n)
    if not isinstance(i, int) or not (0 <= i < 2 * n):
        raise ValueError(f"gamma index must be in [0, {2*n}), got {i!r}")
    j, kind = divmod(i, 2)
    letters = "Z" * j + ("X" if kind == 0 else "Y") + "I" * (n - j - 1)
    return MV.word(n, letters)


def c_op(n: int, j: int) -> MV:
    """Fermionic annihilation c_j = ½(gamma_{2j} + i gamma_{2j+1})."""
    _validate_qubit(n, j)
    return 0.5 * (gamma(n, 2 * j) + 1j * gamma(n, 2 * j + 1))


def cdag_op(n: int, j: int) -> MV:
    _validate_qubit(n, j)
    return 0.5 * (gamma(n, 2 * j) - 1j * gamma(n, 2 * j + 1))


# ----------------------------------------------------------------------
# 3. Gates as unitary multivectors (rotation gates are rotors)
# ----------------------------------------------------------------------

def H_gate(n, j):  return (1 / math.sqrt(2)) * (X(n, j) + Z(n, j))
def S_gate(n, j):  return 0.5 * ((1 + 1j) * I(n) + (1 - 1j) * Z(n, j))
def T_gate(n, j):
    w = cmath.exp(1j * math.pi / 4)
    return 0.5 * ((1 + w) * I(n) + (1 - w) * Z(n, j))


def rotor(P: MV, theta: float) -> MV:
    """Exact exp(-i θ/2 P) for any Hermitian word P with P² = 1."""
    return math.cos(theta / 2) * I(P.n) - 1j * math.sin(theta / 2) * P


def RX(n, j, th): return rotor(X(n, j), th)
def RY(n, j, th): return rotor(Y(n, j), th)
def RZ(n, j, th): return rotor(Z(n, j), th)


def controlled(U: MV, ctrl: int, check_identity_on_control: bool = False) -> MV:
    """C(U) = P0(ctrl) + P1(ctrl) U.

    U should act as identity on ctrl. Set check_identity_on_control=True to
    reject terms with X/Y/Z on the control qubit.
    """
    n = U.n
    _validate_qubit(n, ctrl, "ctrl")
    if check_identity_on_control:
        bad = [code for code in U.terms if ((code >> (2 * ctrl)) & 3) != 0]
        if bad:
            raise ValueError("controlled(U, ctrl) expects U to be identity on ctrl")
    return 0.5 * (I(n) + Z(n, ctrl)) + 0.5 * (I(n) - Z(n, ctrl)) * U


def CNOT(n, c, t): return controlled(X(n, t), c)
def CZ(n, c, t):   return controlled(Z(n, t), c)
def SWAP(n, a, b):
    return 0.5 * (I(n) + X(n, a) * X(n, b) + Y(n, a) * Y(n, b)
                  + Z(n, a) * Z(n, b))
def TOFFOLI(n, c1, c2, t):
    return controlled(controlled(X(n, t), c2), c1)


def expm_taylor(A: MV, order: int = 40) -> MV:
    """exp(A) by scaling-and-squaring + Taylor (general multivectors)."""
    s = max(0, int(math.ceil(math.log2(max(A.norm_hs(), 1e-30)))) + 1)
    B = (2.0 ** (-s)) * A
    term, out = I(A.n), I(A.n)
    for k in range(1, order + 1):
        term = (1.0 / k) * (term * B)
        out = out + term
    for _ in range(s):
        out = out * out
    return out


def _check_h_terms(H_terms: list[tuple[float, MV]]) -> int:
    if not H_terms:
        raise ValueError("H_terms must be non-empty")
    n = H_terms[0][1].n
    for _, P in H_terms:
        if P.n != n:
            raise ValueError("all Hamiltonian terms must live in the same algebra")
    return n


def trotter_unitary(H_terms: list[tuple[float, MV]], t: float,
                    steps: int) -> MV:
    """First-order Lie-Trotter for H = Σ c_k P_k, each P_k a Pauli word."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = t / steps
    n = _check_h_terms(H_terms)
    step = I(n)
    for c, P in H_terms:
        step = step * rotor(P, 2 * c * dt)      # exp(-i c P dt)
    U = I(n)
    for _ in range(steps):
        U = U * step
    return U


def trotter2_unitary(H_terms: list[tuple[float, MV]], t: float,
                     steps: int) -> MV:
    """Second-order symmetric Suzuki-Trotter unitary."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = t / steps
    n = _check_h_terms(H_terms)
    half_step = I(n)
    for c, P in H_terms:
        half_step = half_step * rotor(P, c * dt)
    step = half_step
    for c, P in reversed(H_terms):
        step = step * rotor(P, c * dt)
    U = I(n)
    for _ in range(steps):
        U = U * step
    return U


# ----------------------------------------------------------------------
# 4. States, evolution, measurement, channels
# ----------------------------------------------------------------------

def ket_density(n: int, bits: str) -> MV:
    """|b><b| = Π_j ½(1 ± Z_j) — a product of commuting idempotents."""
    _validate_n(n)
    if len(bits) != n or any(b not in "01" for b in bits):
        raise ValueError(f"bits must be a length-{n} string over '0'/'1'")
    rho = I(n)
    for j, b in enumerate(bits):
        rho = rho * (0.5 * (I(n) + (1 if b == "0" else -1) * Z(n, j)))
    return rho


def computational_projector(n: int, bits: str) -> MV:
    return ket_density(n, bits)


def evolve(rho: MV, U: MV) -> MV:
    return U * rho * U.dagger()


def expectation(rho: MV, O: MV) -> complex:
    return (O * rho).trace()


def probability(rho: MV, projector: MV, *, clip: bool = True) -> float:
    p = (projector * rho).trace().real
    if clip and abs(p) < 1e-12:
        p = 0.0
    if clip and abs(p - 1.0) < 1e-12:
        p = 1.0
    return p


def purity(rho: MV) -> float:
    return (rho * rho).trace().real


def computational_probabilities(rho: MV) -> dict[str, float]:
    probs: dict[str, float] = {}
    for k in range(2 ** rho.n):
        bits = format(k, f"0{rho.n}b")
        probs[bits] = probability(rho, ket_density(rho.n, bits))
    return probs


def measure(rho: MV, projectors: list[MV], *, check_projectors: bool = False):
    out = []
    for P in projectors:
        if P.n != rho.n:
            raise ValueError("projector and density operator have different n")
        if check_projectors and not P.is_projector():
            raise ValueError("measurement element is not an orthogonal projector")
        p = (P * rho).trace().real
        post = P * rho * P
        if p > TOL:
            post = (1.0 / p) * post
        out.append((p, post))
    return out


def z_projectors(n: int, j: int):
    return [0.5 * (I(n) + Z(n, j)), 0.5 * (I(n) - Z(n, j))]


def check_kraus(kraus: list[MV], tol: float = 1e-9) -> bool:
    if not kraus:
        raise ValueError("kraus list must be non-empty")
    n = kraus[0].n
    s = MV(n)
    for K in kraus:
        if K.n != n:
            raise ValueError("all Kraus operators must live in the same algebra")
        s = s + K.dagger() * K
    return s.is_close(I(n), tol)


def apply_channel(rho: MV, kraus: list[MV], *, check: bool = False) -> MV:
    if check and not check_kraus(kraus):
        raise ValueError("Kraus operators do not satisfy Σ K†K = 1")
    out = MV(rho.n)
    for K in kraus:
        if K.n != rho.n:
            raise ValueError("Kraus operator and rho have different n")
        out = out + K * rho * K.dagger()
    return out


def depolarizing(n, j, p):
    q = math.sqrt(p / 3)
    return [math.sqrt(1 - p) * I(n), q * X(n, j), q * Y(n, j), q * Z(n, j)]


def dephasing(n, j, p):
    return [math.sqrt(1 - p) * I(n), math.sqrt(p) * Z(n, j)]


def amplitude_damping(n, j, gamma_):
    P0 = 0.5 * (I(n) + Z(n, j))
    P1 = 0.5 * (I(n) - Z(n, j))
    sig = 0.5 * (X(n, j) + 1j * Y(n, j))            # |0><1|
    return [P0 + math.sqrt(1 - gamma_) * P1, math.sqrt(gamma_) * sig]


# ----------------------------------------------------------------------
# 5. Composite-system tools
# ----------------------------------------------------------------------

def partial_trace(rho: MV, traced: set[int]) -> MV:
    """Trace out `traced` qubits: keep words with I there (×2 per qubit)."""
    keep = [j for j in range(rho.n) if j not in traced]
    m = len(keep)
    out: dict[int, complex] = {}
    for code, c in rho.terms.items():
        if any(((code >> (2 * j)) & 3) for j in traced):
            continue                       # Tr of a non-identity Pauli is 0
        new = 0
        for pos, j in enumerate(keep):
            new |= ((code >> (2 * j)) & 3) << (2 * pos)
        out[new] = out.get(new, 0.0) + c * (2 ** len(traced))
    return MV(m, out)


def partial_transpose(rho: MV, qubits: set[int]) -> MV:
    """Xᵀ=X, Zᵀ=Z, Yᵀ=−Y ⇒ flip sign per Y on the transposed qubits."""
    for j in qubits:
        _validate_qubit(rho.n, j)
    t = {}
    for code, c in rho.terms.items():
        s = (-1) ** sum(1 for j in qubits if ((code >> (2 * j)) & 3) == 2)
        t[code] = s * c
    return MV(rho.n, t)


def negativity(rho: MV, transposed: set[int]) -> float:
    """Entanglement negativity = (||rho^T_B||_1 - 1)/2, via matrix backend."""
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(partial_transpose(rho, transposed)))
    return float((sum(abs(x) for x in ev) - 1.0) / 2.0)


def vn_entropy(rho: MV, *, base: float = 2.0, tol: float = 1e-12) -> float:
    """Von Neumann entropy of a density operator, via matrix backend."""
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(rho))
    vals = [max(float(x.real), 0.0) for x in ev if x > tol]
    if not vals:
        return 0.0
    log_base = math.log(base)
    return -sum(p * math.log(p) / log_base for p in vals)


# ----------------------------------------------------------------------
# 6. Matrix backend (validation only, small n)
# ----------------------------------------------------------------------

def _single_pauli_mats():
    import numpy as np
    return {0: np.eye(2, dtype=complex),
            1: np.array([[0, 1], [1, 0]], complex),
            2: np.array([[0, -1j], [1j, 0]], complex),
            3: np.array([[1, 0], [0, -1]], complex)}


def code_to_matrix(n: int, code: int):
    import numpy as np
    _validate_word_code(n, code)
    P = _single_pauli_mats()
    if n == 0:
        return np.array([[1]], dtype=complex)
    M = P[code & 3]
    for j in range(1, n):
        M = np.kron(M, P[(code >> (2 * j)) & 3])
    return M


def to_matrix(A: MV):
    import numpy as np
    dim = 2 ** A.n
    out = np.zeros((dim, dim), complex)
    for code, c in A.terms.items():
        out += c * code_to_matrix(A.n, code)
    return out


def from_matrix(M, n: int | None = None, tol: float = 1e-12) -> MV:
    """Expand a small dense matrix in the Pauli-word basis."""
    import numpy as np
    M = np.asarray(M, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    dim = M.shape[0]
    if n is None:
        if dim < 1 or dim & (dim - 1):
            raise ValueError("dimension must be a power of two")
        n = int(math.log2(dim))
    _validate_n(n)
    if dim != 2 ** n:
        raise ValueError(f"matrix shape {M.shape} incompatible with n={n}")
    coeffs: dict[int, complex] = {}
    denom = 2 ** n
    for code in range(4 ** n):
        P = code_to_matrix(n, code)
        c = np.trace(P.conj().T @ M) / denom
        if abs(c) > tol:
            coeffs[code] = complex(c)
    return MV(n, coeffs)


def density_from_statevector(psi, tol: float = 1e-12) -> MV:
    """Create |psi><psi| from a normalized or unnormalized statevector."""
    import numpy as np
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    dim = psi.shape[0]
    if dim < 1 or dim & (dim - 1):
        raise ValueError("statevector length must be a power of two")
    norm = np.linalg.norm(psi)
    if norm <= tol:
        raise ValueError("zero statevector")
    psi = psi / norm
    return from_matrix(np.outer(psi, psi.conj()), tol=tol)


def expm_matrix(A: MV, tol: float = 1e-12) -> MV:
    """Dense exact exp(A) through eigendecomposition. Validation/small-n helper."""
    import numpy as np
    M = to_matrix(A)
    w, V = np.linalg.eig(M)
    E = V @ np.diag(np.exp(w)) @ np.linalg.inv(V)
    return from_matrix(E, A.n, tol=tol)


# ----------------------------------------------------------------------
# 7. Verification suite
# ----------------------------------------------------------------------

def _check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def run_verification():
    import numpy as np
    print("== Operator-centric GA quantum framework: verification ==\n")

    n = 3
    print("-- algebra kernel --")
    rng = np.random.default_rng(7)
    def rand_mv():
        return MV(n, {int(rng.integers(0, 4 ** n)):
                      complex(rng.normal(), rng.normal()) for _ in range(6)})
    A, B, C = rand_mv(), rand_mv(), rand_mv()
    _check("associativity (AB)C = A(BC)", ((A * B) * C).is_close(A * (B * C), 1e-8))
    _check("(AB)† = B†A†", (A * B).dagger().is_close(B.dagger() * A.dagger(), 1e-8))
    _check("X Y = iZ (local)", (X(n, 1) * Y(n, 1)).is_close(1j * Z(n, 1)))
    _check("[X_0, X_1] = 0 (different qubits commute)",
           comm(X(n, 0), X(n, 1)).norm_hs() < 1e-12)
    _check("scalar left-subtraction and division work",
           (1 - X(n, 0)).is_close(I(n) - X(n, 0))
           and (2 * X(n, 0) / 2).is_close(X(n, 0)))

    print("-- Clifford generator layer (Jordan–Wigner) --")
    g = [gamma(n, i) for i in range(2 * n)]
    _check("gamma_i² = 1", all((g[i] * g[i]).is_close(I(n)) for i in range(2 * n)))
    _check("gamma_i gamma_j = -gamma_j gamma_i (i≠j, incl. across qubits)",
           all((g[i] * g[j] + g[j] * g[i]).norm_hs() < 1e-12
               for i in range(2 * n) for j in range(i + 1, 2 * n)))
    Ibig = 1
    full = I(n)
    for gi in g:
        full = full * gi
    _check("pseudoscalar gamma_0…gamma_5 is a phase × Z⊗…: grade 2n",
           full.grades() == {2 * n})
    _check("grade decode: gamma_i has grade 1", all(gi.grades() == {1} for gi in g))
    _check("grade decode: Z_j = -i gamma_2j gamma_2j+1 has grade 2",
           Z(n, 1).grades() == {2}
           and Z(n, 1).is_close(-1j * g[2] * g[3]))
    _check("grade decode: bare X_1 (JW-dressed) has grade 3",
           X(n, 1).grades() == {3})
    def blade_from_mask(mask: int) -> MV:
        out = I(n)
        for idx in range(2 * n):
            if (mask >> idx) & 1:
                out = out * gamma(n, idx)
        return out
    _check("inverse JW blade-mask decode for every gamma subset",
           all(blade_from_mask(mask).blade_mask(next(iter(blade_from_mask(mask).terms))) == mask
               for mask in range(1 << (2 * n))))

    print("-- fermionic (Witt) layer --")
    cs = [c_op(n, j) for j in range(n)]
    cds = [cdag_op(n, j) for j in range(n)]
    car = True
    for i_ in range(n):
        for j_ in range(n):
            anti = cs[i_] * cds[j_] + cds[j_] * cs[i_]
            car &= anti.is_close(I(n) if i_ == j_ else MV(n), 1e-12)
            car &= (cs[i_] * cs[j_] + cs[j_] * cs[i_]).norm_hs() < 1e-12
    _check("CAR: {c_i, c†_j} = δ_ij, {c_i, c_j} = 0", car)
    _check("nilpotency c² = 0 (Pauli exclusion)",
           all((c * c).norm_hs() < 1e-12 for c in cs))
    _check("number operator c†_j c_j = ½(1 − Z_j)",
           all((cds[j] * cs[j]).is_close(0.5 * (I(n) - Z(n, j)), 1e-12)
               for j in range(n)))

    print("-- representation map --")
    MA, MB = to_matrix(A), to_matrix(B)
    _check("matrix(A·B) == matrix(A) @ matrix(B)",
           np.allclose(to_matrix(A * B), MA @ MB, atol=1e-8))
    _check("Tr == 2^n <·>_0", abs(np.trace(MA) - A.trace()) < 1e-8)
    _check("dagger == conjugate transpose",
           np.allclose(to_matrix(A.dagger()), MA.conj().T, atol=1e-8))
    _check("HS norm matches", abs(A.norm_hs() -
           np.sqrt(np.trace(MA.conj().T @ MA).real)) < 1e-8)
    _check("from_matrix(to_matrix(A)) round-trip",
           from_matrix(MA, n).is_close(A, 1e-8))
    _check("tensor(A,B) matches numpy.kron",
           np.allclose(to_matrix(tensor(X(1, 0), Z(1, 0))),
                       np.kron(to_matrix(X(1, 0)), to_matrix(Z(1, 0))), atol=1e-12))

    print("-- gates --")
    gates = {"H": H_gate(n, 0), "S": S_gate(n, 1), "T": T_gate(n, 2),
             "RX": RX(n, 0, 0.7), "RY": RY(n, 1, 1.1), "RZ": RZ(n, 2, 2.3),
             "CNOT": CNOT(n, 0, 1), "CZ": CZ(n, 1, 2), "SWAP": SWAP(n, 0, 2),
             "TOFFOLI": TOFFOLI(n, 0, 1, 2)}
    _check("U†U = 1 for all gates",
           all(U.is_unitary(1e-9) for U in gates.values()))
    _check("rotor identity: RZ(θ) = cos(θ/2) − i sin(θ/2) Z",
           RZ(n, 0, 1.3).is_close(
               math.cos(0.65) * I(n) - 1j * math.sin(0.65) * Z(n, 0)))
    _check("H = (X+Z)/√2 squares to 1", (gates["H"] * gates["H"]).is_close(I(n)))
    _check("T² = S, S² = Z",
           (T_gate(n, 2) * T_gate(n, 2)).is_close(S_gate(n, 2), 1e-12)
           and (S_gate(n, 2) * S_gate(n, 2)).is_close(Z(n, 2), 1e-12))

    print("-- Gottesman–Knill blade structure --")
    U = CNOT(n, 0, 1)
    ok = all((U * pauli_string(n, s) * U.dagger()).nnz() == 1
             for s in ("XII", "YII", "ZII", "IXI", "IYI", "IZI"))
    _check("Clifford conjugation: one word -> one word", ok)
    _check("CNOT: X_c -> X_c X_t",
           (U * X(n, 0) * U.dagger()).is_close(X(n, 0) * X(n, 1)))
    _check("CNOT: Z_t -> Z_c Z_t",
           (U * Z(n, 1) * U.dagger()).is_close(Z(n, 0) * Z(n, 1)))
    Tg = T_gate(n, 0)
    _check("non-Clifford T spreads X into 2 words (universality boundary)",
           (Tg * X(n, 0) * Tg.dagger()).nnz() == 2)

    print("-- states, Bell, entanglement --")
    n2 = 2
    rho0 = ket_density(n2, "00")
    _check("Tr rho = 1, purity 1", rho0.is_density()
           and abs(purity(rho0) - 1) < 1e-12)
    Ubell = CNOT(n2, 0, 1) * H_gate(n2, 0)
    bell = evolve(rho0, Ubell)
    _check("Bell pure, trace 1",
           abs(purity(bell) - 1) < 1e-9 and abs(bell.trace() - 1) < 1e-9)
    target = 0.25 * (I(n2) + X(n2, 0) * X(n2, 1)
                     - Y(n2, 0) * Y(n2, 1) + Z(n2, 0) * Z(n2, 1))
    _check("Bell = ¼(1 + XX − YY + ZZ)", bell.is_close(target, 1e-9))
    red = partial_trace(bell, {1})
    _check("reduced Bell qubit maximally mixed (purity ½, trace 1)",
           abs(purity(red) - 0.5) < 1e-9 and abs(red.trace() - 1) < 1e-9)
    pt = partial_transpose(bell, {1})
    ev = np.linalg.eigvalsh(to_matrix(pt))
    _check("PPT witness: min eigenvalue of rho^{T_B} = −1/2",
           abs(min(ev) + 0.5) < 1e-9)
    _check("Bell negativity = 1/2 and reduced entropy = 1 bit",
           abs(negativity(bell, {1}) - 0.5) < 1e-9
           and abs(vn_entropy(red) - 1.0) < 1e-9)
    ghz = evolve(ket_density(3, "000"),
                 CNOT(3, 1, 2) * CNOT(3, 0, 1) * H_gate(3, 0))
    _check("GHZ pure; tracing ONE qubit gives purity ½ (classical corr.)",
           abs(purity(ghz) - 1) < 1e-9
           and abs(purity(partial_trace(ghz, {2})) - 0.5) < 1e-9)

    print("-- CHSH --")
    A0, A1 = Z(n2, 0), X(n2, 0)
    B0 = (1 / math.sqrt(2)) * (Z(n2, 1) + X(n2, 1))
    B1 = (1 / math.sqrt(2)) * (Z(n2, 1) - X(n2, 1))
    Sv = sum(s * expectation(bell, Aa * Bb).real
             for s, Aa, Bb in [(1, A0, B0), (1, A0, B1),
                               (1, A1, B0), (-1, A1, B1)])
    _check(f"CHSH = 2√2  (got {Sv:.6f})", abs(Sv - 2 * math.sqrt(2)) < 1e-9)

    print("-- measurement --")
    res = measure(bell, z_projectors(n2, 0))
    _check("Bell Z-meas: p = (½, ½)",
           abs(res[0][0] - 0.5) < 1e-9 and abs(res[1][0] - 0.5) < 1e-9)
    _check("collapse: outcome-0 post-state = |00><00|",
           res[0][1].is_close(ket_density(n2, "00"), 1e-9))

    print("-- Grover (2 qubits, marked |11>) --")
    Hs = H_gate(n2, 0) * H_gate(n2, 1)
    Gop = (Hs * (2 * ket_density(n2, "00") - I(n2)) * Hs) * CZ(n2, 0, 1)
    rho = evolve(evolve(ket_density(n2, "00"), Hs), Gop)
    p11 = probability(rho, ket_density(n2, "11"))
    _check(f"one iteration finds |11> with p = 1  (got {p11:.6f})",
           abs(p11 - 1) < 1e-9)
    probs = computational_probabilities(rho)
    _check("computational probabilities sum to 1",
           abs(sum(probs.values()) - 1.0) < 1e-9)

    print("-- channels --")
    for name, ks in [("depolarizing", depolarizing(n2, 0, 0.3)),
                     ("dephasing", dephasing(n2, 0, 0.2)),
                     ("amplitude damping", amplitude_damping(n2, 0, 0.4))]:
        _check(f"{name}: Σ K†K = 1", check_kraus(ks, 1e-9))
    noisy = bell
    for _ in range(60):
        noisy = apply_channel(noisy, depolarizing(n2, 0, 0.5))
        noisy = apply_channel(noisy, depolarizing(n2, 1, 0.5))
    _check("depolarizing fixed point: purity -> 1/4",
           abs(purity(noisy) - 0.25) < 1e-6)
    ad = ket_density(1, "1")
    for _ in range(200):
        ad = apply_channel(ad, amplitude_damping(1, 0, 0.1))
    _check("amplitude damping drives |1> -> |0>",
           ad.is_close(ket_density(1, "0"), 1e-6))

    print("-- Hamiltonian dynamics (2-site TFIM) --")
    Ht = [(1.0, X(n2, 0) * X(n2, 1)), (0.5, Z(n2, 0)), (0.5, Z(n2, 1))]
    Hmat = sum(c * to_matrix(P) for c, P in Ht)
    t = 0.9
    w, V = np.linalg.eigh(Hmat)
    Uex = V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T
    err_t = np.linalg.norm(to_matrix(trotter_unitary(Ht, t, 400)) - Uex)
    _check(f"Trotter(400) error {err_t:.2e} < 2e-2", err_t < 2e-2)
    err_t2 = np.linalg.norm(to_matrix(trotter2_unitary(Ht, t, 40)) - Uex)
    _check(f"Trotter2(40) error {err_t2:.2e} < 2e-3", err_t2 < 2e-3)
    Hmv = MV(n2)
    for c, P in Ht:
        Hmv = Hmv + c * P
    err_e = np.linalg.norm(to_matrix(expm_taylor((-1j * t) * Hmv)) - Uex)
    _check(f"Taylor expm error {err_e:.2e} < 1e-9", err_e < 1e-9)
    err_m = np.linalg.norm(to_matrix(expm_matrix((-1j * t) * Hmv)) - Uex)
    _check(f"matrix expm helper error {err_m:.2e} < 1e-9", err_m < 1e-9)

    print("\n-- structural report --")
    print(f"  Bell state: {bell.nnz()}/16 words | GHZ: {ghz.nnz()}/64 | "
          f"Toffoli: {TOFFOLI(3, 0, 1, 2).nnz()}/64 words")
    print(f"  CNOT Clifford grades: {sorted(CNOT(2, 0, 1).grades())} "
          f"(mixed-grade versor, not a pure rotor)")
    print(f"  RZ(θ) grades: {sorted(RZ(2, 0, 0.8).grades())} "
          f"(scalar + bivector: a genuine rotor)")
    print("\nAll checks passed.")


if __name__ == "__main__":
    run_verification()
