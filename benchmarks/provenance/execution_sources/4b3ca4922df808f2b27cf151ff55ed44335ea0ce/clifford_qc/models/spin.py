"""Spin-model builders for the ADAPT-VQE research program.

Every model packages the same three things:

- ``hamiltonian``: the exact ``PauliSum``;
- ``reference``: a Clifford ``Program`` preparing the reference product state
  (the ADAPT/HVA starting point);
- ``hva_layers``: named groups of Pauli words, one shared HVA parameter per
  group per depth layer.

All Hamiltonians here are real in the computational basis (X/Z/ZZ/XX/YY
terms), which is what the odd-Y pool-restriction theorem needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ir import PauliSum, PauliWord, Program
from .metadata import SPIN_LATTICE, model_metadata, validate_model_metadata


@dataclass(frozen=True)
class Model:
    """A Hamiltonian, its reference state, and the conventions both are read under.

    ``metadata`` is a contract, not a free-form dict: it is validated against
    :data:`~clifford_qc.models.metadata.MODEL_METADATA_SCHEMA` at construction.
    A model whose spin ordering or sector cannot be read is one whose energies
    cannot be compared with another model's, and the failure would otherwise
    arrive as a plausible number rather than as an exception --- see
    ``models/metadata.py`` for the 0.83 Ha worked example.  Build the dict with
    :func:`~clifford_qc.models.metadata.model_metadata`.
    """

    name: str
    n: int
    hamiltonian: PauliSum
    reference: Program
    hva_layers: tuple[tuple[str, tuple[PauliWord, ...]], ...]
    #: No default: under the contract the empty dict is exactly what
    #: ``__post_init__`` refuses, so a default would advertise an optional
    #: field that cannot be used. The signature states the requirement the
    #: validator enforces.
    metadata: dict

    def __post_init__(self) -> None:
        validate_model_metadata(self.metadata, n_qubits=self.n,
                                model_name=self.name,
                                pauli_words=len(self.hamiltonian.terms))

    def __post_init__(self) -> None:
        validate_model_metadata(self.metadata, n_qubits=self.n,
                                model_name=self.name,
                                pauli_words=len(self.hamiltonian.terms))


def _word(n: int, factors: dict[int, str]) -> PauliWord:
    letters = ["I"] * n
    for j, ch in factors.items():
        letters[j] = ch
    return PauliWord.from_label("".join(letters), n)


def _bond_pairs(n: int, periodic: bool) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:  # n=2 periodic would duplicate the single bond
        pairs.append((n - 1, 0))
    return pairs


def tfim(n: int, J: float = 1.0, h: float = 1.0, *, periodic: bool = False) -> Model:
    """1D transverse-field Ising model H = -J sum Z_i Z_{i+1} - h sum X_i.

    Reference state |+...+> (the h -> inf ground state), prepared by H gates.
    """
    if n < 1:
        raise ValueError("n must be positive")
    bonds = tuple(_word(n, {i: "Z", j: "Z"}) for i, j in _bond_pairs(n, periodic))
    fields = tuple(_word(n, {i: "X"}) for i in range(n))
    terms = {w.code: -float(J) for w in bonds}
    for w in fields:
        terms[w.code] = terms.get(w.code, 0.0) - float(h)
    ref = Program(n)
    for j in range(n):
        ref.clifford("H", j)
    return Model(
        name=f"tfim(n={n},J={J},h={h},{'pbc' if periodic else 'obc'})",
        n=n,
        hamiltonian=PauliSum(n, terms),
        reference=ref,
        hva_layers=(("zz", bonds), ("x", fields)),
        metadata=model_metadata(SPIN_LATTICE, J=float(J), h=float(h),
                                periodic=bool(periodic), sites=n,
                                lattice="chain"),
    )


def xxz(n: int, J: float = 1.0, delta: float = 1.0, *, periodic: bool = False) -> Model:
    """XXZ chain H = J sum (X_i X_j + Y_i Y_j + delta Z_i Z_j).

    Reference state is the Neel product state |0101...> (X gates on odd
    sites), the delta -> inf antiferromagnetic ground state for J > 0.
    """
    if n < 2:
        raise ValueError("XXZ needs at least two qubits")
    pairs = _bond_pairs(n, periodic)
    xx = tuple(_word(n, {i: "X", j: "X"}) for i, j in pairs)
    yy = tuple(_word(n, {i: "Y", j: "Y"}) for i, j in pairs)
    zz = tuple(_word(n, {i: "Z", j: "Z"}) for i, j in pairs)
    terms: dict[int, complex] = {}
    for w in xx + yy:
        terms[w.code] = terms.get(w.code, 0.0) + float(J)
    for w in zz:
        terms[w.code] = terms.get(w.code, 0.0) + float(J) * float(delta)
    ref = Program(n)
    for j in range(1, n, 2):
        ref.clifford("X", j)
    return Model(
        name=f"xxz(n={n},J={J},delta={delta},{'pbc' if periodic else 'obc'})",
        n=n,
        hamiltonian=PauliSum(n, terms),
        reference=ref,
        hva_layers=(("xx", xx), ("yy", yy), ("zz", zz)),
        metadata=model_metadata(SPIN_LATTICE, J=float(J), delta=float(delta),
                                periodic=bool(periodic), sites=n,
                                lattice="chain"),
    )


def random_ising(n: int, seed: int, J: float = 1.0, h: float = 1.0, *,
                 periodic: bool = False, spread: float = 0.5) -> Model:
    """Random-bond, random-transverse-field Ising chain.

    Bonds and fields are drawn uniformly from ``[(1-spread)*J, (1+spread)*J]``
    and ``[(1-spread)*h, (1+spread)*h]`` with a fixed seed, breaking
    translational symmetry while keeping the Hamiltonian real. Used to test
    ranking stability under disorder.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 <= spread <= 1.0):
        raise ValueError("spread must be in [0, 1]")
    rng = np.random.default_rng(seed)
    pairs = _bond_pairs(n, periodic)
    Js = rng.uniform((1 - spread) * J, (1 + spread) * J, size=len(pairs))
    hs = rng.uniform((1 - spread) * h, (1 + spread) * h, size=n)
    bonds = tuple(_word(n, {i: "Z", j: "Z"}) for i, j in pairs)
    fields = tuple(_word(n, {i: "X"}) for i in range(n))
    terms = {w.code: -float(c) for w, c in zip(bonds, Js)}
    for w, c in zip(fields, hs):
        terms[w.code] = terms.get(w.code, 0.0) - float(c)
    ref = Program(n)
    for j in range(n):
        ref.clifford("H", j)
    return Model(
        name=f"random_ising(n={n},seed={seed},{'pbc' if periodic else 'obc'})",
        n=n,
        hamiltonian=PauliSum(n, terms),
        reference=ref,
        hva_layers=(("zz", bonds), ("x", fields)),
        metadata=model_metadata(SPIN_LATTICE, J=float(J), h=float(h),
                                seed=int(seed), spread=float(spread),
                                periodic=bool(periodic), sites=n,
                                lattice="chain",
                                bond_couplings=[float(x) for x in Js],
                                field_couplings=[float(x) for x in hs]),
    )
