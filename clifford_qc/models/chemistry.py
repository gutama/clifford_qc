"""Molecular models via the OpenFermion bridge (``chemistry`` extra).

``molecule_model`` turns a PySCF-computed molecule (optionally reduced to
an active space) into the same ``Model`` the spin benchmarks use: the
Jordan-Wigner qubit Hamiltonian as a ``PauliSum`` and the Hartree-Fock
determinant as the Clifford reference Program (X gates on the occupied
spin-orbitals). ``excitation_pool`` provides the qubit-ADAPT candidate
pool: individual odd-Y Pauli words *derived from* JW-transformed,
particle-number- and S_z-conserving single/double generators.  The source
generators conserve those symmetries, but their individual Pauli-word
rotors generally do not; chemistry benchmarks therefore report explicit
sector leakage rather than calling the qubit pool symmetry preserving.
"""

from __future__ import annotations

import numpy as np
from ..capabilities import require

require('molecular_input', feature='clifford_qc.models.chemistry')

from openfermion.chem import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner

from ..ir import PauliSum, Program
from ..bridges.openfermion_bridge import qubit_operator_to_pauli_sum
from ..algorithms.pools import PoolOperator, is_odd_y
from .metadata import MOLECULAR, model_metadata
from .spin import Model


def _hf_reference(n_qubits: int, n_electrons: int, ms2: int = 0) -> Program:
    """Highest-``M_s`` Aufbau determinant in interleaved alpha/beta ordering."""
    if (n_electrons + ms2) % 2:
        raise ValueError("n_electrons and ms2 have incompatible parity")
    n_alpha = (n_electrons + ms2) // 2
    n_beta = n_electrons - n_alpha
    spatial = n_qubits // 2
    if min(n_alpha, n_beta) < 0 or max(n_alpha, n_beta) > spatial:
        raise ValueError("requested electron/spin sector does not fit the active space")
    prog = Program(n_qubits)
    for j in range(0, 2 * n_alpha, 2):
        prog.clifford("X", j)
    for j in range(1, 2 * n_beta, 2):
        prog.clifford("X", j)
    return prog


def molecule_model(geometry, basis: str = "sto-3g", multiplicity: int = 1,
                   charge: int = 0, *, name: str,
                   occupied_indices=None, active_indices=None,
                   run_fci: bool = True) -> Model:
    """Model for a molecule (PySCF SCF + optional FCI, JW mapping).

    ``occupied_indices``/``active_indices`` select an active space in
    spatial-orbital indices (OpenFermion convention); the frozen-core
    energy lands in the Hamiltonian's identity term, so ``exact_ground``
    of the returned Hamiltonian matches the active-space FCI energy.
    """
    from openfermionpyscf import run_pyscf

    molecule = run_pyscf(MolecularData(geometry, basis, multiplicity, charge,
                                       description=name),
                         run_scf=True, run_fci=run_fci)
    hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices)
    qubit_op = jordan_wigner(hamiltonian)
    pauli_sum = qubit_operator_to_pauli_sum(qubit_op)
    n_qubits = pauli_sum.n
    n_core = 2 * len(occupied_indices or ())
    n_active_electrons = molecule.n_electrons - n_core
    ms2 = int(multiplicity) - 1
    return Model(
        name=name,
        n=n_qubits,
        hamiltonian=pauli_sum,
        reference=_hf_reference(n_qubits, n_active_electrons, ms2),
        hva_layers=(),
        metadata=model_metadata(
            MOLECULAR,
            spin_orbitals=n_qubits,
            n_spatial_orbitals=n_qubits // 2,
            n_electrons=int(n_active_electrons),
            sz=0.5 * ms2,
            spin_convention="interleaved",
            source="pyscf",
            basis=basis,
            multiplicity=int(multiplicity),
            hf_energy=float(molecule.hf_energy),
            fci_energy=float(molecule.fci_energy) if run_fci else None,
            frozen_spatial_orbitals=list(occupied_indices or ()),
            active_spatial_orbitals=(list(active_indices) if active_indices
                                     else None),
        ),
    )


def fcidump_model(path, *, name: str | None = None,
                  n_electrons: int | None = None, ms2: int | None = None,
                  integral_tolerance: float = 1e-12) -> Model:
    """Compatibility route to the NumPy-only FCIDUMP interchange adapter.

    FCIDUMP no longer needs PySCF or OpenFermion to parse and map a record.
    This module still provides the historical import path for chemistry-extra
    users; the implementation lives in :mod:`clifford_qc.models.fcidump` so a
    downstream DFT or embedding workflow can use it in a core installation.
    """
    from .fcidump import fcidump_model as load

    return load(path, name=name, n_electrons=n_electrons, ms2=ms2,
                integral_tolerance=integral_tolerance)


def h2(bond_length: float = 0.7414) -> Model:
    return molecule_model([("H", (0, 0, 0)), ("H", (0, 0, bond_length))],
                          name=f"h2(r={bond_length})")


def h4_chain(spacing: float = 0.9) -> Model:
    geometry = [("H", (0, 0, i * spacing)) for i in range(4)]
    return molecule_model(geometry, name=f"h4_chain(r={spacing})")


def lih(bond_length: float = 1.5949) -> Model:
    """LiH with frozen core and the (2e, 2o) active space -> 4 qubits."""
    return molecule_model([("Li", (0, 0, 0)), ("H", (0, 0, bond_length))],
                          name=f"lih(r={bond_length})",
                          occupied_indices=[0], active_indices=[1, 2])


def beh2(bond_length: float = 1.3264) -> Model:
    """Linear BeH2 with frozen core and a (4e, 3o) active space -> 6 qubits."""
    geometry = [("Be", (0, 0, 0)), ("H", (0, 0, -bond_length)),
                ("H", (0, 0, bond_length))]
    return molecule_model(geometry, name=f"beh2(r={bond_length})",
                          occupied_indices=[0], active_indices=[1, 2, 3])


def h2o_qsci() -> Model:
    """QSCI-aligned H2O benchmark: STO-3G CAS(6e,5o) -> 10 qubits.

    The geometry and active-space size follow Kanno *et al.*,
    arXiv:2302.11320 / Phys. Rev. Research 8, 023268 (2026).  The paper
    specifies six active electrons in five active spatial orbitals but not the
    orbital indices.  With the canonical RHF/STO-3G orbitals used here, the
    two lowest occupied orbitals are frozen and the remaining five orbitals
    form the active space.  Keeping this convention explicit avoids claiming
    a stronger reproduction of the paper's orbital choice than is documented.
    """
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.2774, 0.8929, 0.2544)),
        ("H", (0.6068, -0.2383, -0.7169)),
    ]
    return molecule_model(
        geometry,
        name="h2o_qsci_sto3g_cas6e5o",
        occupied_indices=[0, 1],
        active_indices=[2, 3, 4, 5, 6],
    )


def beh2_frozen_core(bond_length: float = 3.0) -> Model:
    """Linear BeH2 STO-3G with only the Be 1s core frozen: CAS(4e,6o).

    ``bond_length=3.0`` Angstrom is the stretched, strongly-correlated BeH2
    geometry used by Feniou *et al.* (arXiv:2301.10196).  Their calculation
    uses the full STO-3G space; this benchmark freezes only the chemically
    inert Be 1s core to keep the 20-seed selector sweep tractable, and records
    that distinction rather than presenting it as an exact reproduction.
    """
    geometry = [
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, -bond_length)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    return molecule_model(
        geometry,
        name=f"beh2_stretched_sto3g_cas4e6o(r={bond_length})",
        occupied_indices=[0],
        active_indices=[1, 2, 3, 4, 5, 6],
    )


def _excitation_generators(n_qubits: int, n_electrons: int) -> list[FermionOperator]:
    """Anti-Hermitian singles/doubles generators that conserve particle
    number and total spin projection S_z.

    Spin-orbitals use OpenFermion's interleaved Jordan-Wigner ordering, so
    even indices carry spin up and odd indices spin down. A single ``i->a``
    conserves S_z iff ``i`` and ``a`` share spin, i.e. ``(a - i)`` is even.
    A double ``{i,j}->{a,b}`` conserves S_z iff the excited and de-excited
    sets carry the same number of spin-down (odd-index) orbitals, i.e.
    ``(i%2 + j%2) == (a%2 + b%2)``. Testing only ``(i+j)%2 == (a+b)%2``
    (the parity of that count difference) is necessary but not sufficient:
    it admits Delta S_z = +/-2 excitations such as two spin-up into two
    spin-down, which are not spin-conserving.
    """
    occupied = range(n_electrons)
    virtual = range(n_electrons, n_qubits)
    generators = []
    for i in occupied:
        for a in virtual:
            if (a - i) % 2 == 0:  # same spin under interleaved JW ordering
                generators.append(FermionOperator(((a, 1), (i, 0)))
                                  - FermionOperator(((i, 1), (a, 0))))
    for i in occupied:
        for j in occupied:
            if j <= i:
                continue
            for a in virtual:
                for b in virtual:
                    if b <= a:
                        continue
                    if (i % 2 + j % 2) == (a % 2 + b % 2):  # conserve total S_z
                        generators.append(
                            FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
                            - FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0))))
    return generators


def _excitation_label(generator: FermionOperator) -> str:
    """``E(a,b<-i,j)`` from the excitation half of an anti-Hermitian generator."""
    for term, coeff in sorted(generator.terms.items()):
        if coeff.real > 0:
            created = ",".join(str(idx) for idx, dag in term if dag)
            annihilated = ",".join(str(idx) for idx, dag in term if not dag)
            return f"E({created}<-{annihilated})"
    raise ValueError("generator has no positive-coefficient excitation term")


def excitation_multivectors(n_qubits: int, n_electrons: int) -> list[tuple[str, PauliSum]]:
    """A-CASE symmetry-preserving generators: whole JW images of excitations.

    The *complete* Jordan-Wigner image of each particle-number- and
    S_z-conserving generator, kept as one ``PauliSum`` rather than split into
    words. This is the distinction ``excitation_pool`` cannot make: an
    individual word of a conserving generator generally does not conserve
    ``N`` or ``S_z`` (the sector diagnostics in ``diagnostics.py`` exist to
    report exactly that), so a subspace built from split words can lower its
    Ritz value by leaking into unphysical sectors. Kept whole, the generator
    commutes with both symmetries and the subspace stays inside the sector the
    reference determinant occupies.
    """
    return [(_excitation_label(gen),
             qubit_operator_to_pauli_sum(jordan_wigner(gen), n_qubits))
            for gen in _excitation_generators(n_qubits, n_electrons)]


def excitation_pool(n_qubits: int, n_electrons: int) -> list[PoolOperator]:
    """Qubit-ADAPT pool: odd-Y Pauli words from JW excitations.

    Spin-conserving singles a†_a a_i - h.c. and doubles
    a†_a a†_b a_j a_i - h.c. over occupied {0..n_e-1} / virtual
    {n_e..n_q-1} spin-orbitals (see ``_excitation_generators`` for the
    particle- and S_z-conservation conditions); each anti-Hermitian
    generator's JW image splits into odd-Y words that enter the pool
    individually (deduplicated, ordered by first appearance).  This is the
    standard word-level qubit-ADAPT construction, not a symmetry-preserving
    fermionic-excitation ansatz: a single word need not commute with particle
    number or S_z even though the unsplit source generator does.
    """
    pool: dict[int, PoolOperator] = {}
    for gen in _excitation_generators(n_qubits, n_electrons):
        image = qubit_operator_to_pauli_sum(jordan_wigner(gen), n_qubits)
        for word, _ in image.items():
            if word.code != 0 and is_odd_y(word) and word.code not in pool:
                pool[word.code] = PoolOperator(word.label, word)
    return list(pool.values())
