"""The metadata contract, checked across builders rather than within one.

The suite already tests each model builder thoroughly and individually. What it
did not test is that the builders *agree with each other* -- which is how the
intersection of their metadata keys reached the empty set unnoticed, with not
even ``kind`` common to all of them. The central test here is therefore
parametrized over every public builder: whatever a builder is for, a consumer
must be able to read its conventions the same way.

The negative tests are anchored to the consequence rather than to the message.
Reading ``spin_convention`` wrong is not a cosmetic error: on the 2-site
Hubbard dimer the ``parity+2q`` reduction returns ``-4.0`` instead of
``-4.8284...``, with no exception and a plausible-looking one-word Hamiltonian.
``test_wrong_spin_ordering_silently_changes_the_reduced_energy`` pins that
number, so if a future refactor makes the two orderings agree, the justification
for requiring the key is re-examined rather than silently invalidated.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from clifford_qc.fermion_mapping import fermion_encoding
from clifford_qc.matrix import to_matrix
from clifford_qc.models import (FERMIONIC_KINDS, MODEL_KINDS,
                                MODEL_METADATA_SCHEMA, SPIN_KINDS)
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import (anderson_impurity, extended_hubbard,
                                        hubbard, kanamori, kitaev_honeycomb)
from clifford_qc.models.metadata import (model_metadata, spin_convention,
                                         validate_model_metadata)
from clifford_qc.models.orbital import rotate_model
from clifford_qc.models.spin import Model, random_ising, tfim, xxz

FCIDUMP = "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"


def ring_hopping(sites: int, t: float = 1.0) -> np.ndarray:
    matrix = np.zeros((sites, sites))
    bonds = [(i, i + 1) for i in range(sites - 1)]
    if sites > 2:
        bonds.append((sites - 1, 0))
    for i, j in bonds:
        matrix[i, j] -= t
        matrix[j, i] -= t
    return matrix


def every_builder():
    """One model per public builder, spin and fermionic alike."""
    return {
        "tfim": tfim(4),
        "xxz": xxz(4),
        "random_ising": random_ising(4, seed=0),
        "hubbard": hubbard(4, U=4.0),
        "extended_hubbard": extended_hubbard(4, U=4.0, V=1.0),
        "kanamori": kanamori(2, 2),
        "anderson_impurity": anderson_impurity(2),
        "kitaev_honeycomb": kitaev_honeycomb(1, 2),
        "fcidump": fcidump_model(FCIDUMP),
        "rotate_model": rotate_model(ring_hopping(4), 4.0, "db2"),
    }


_MODELS = every_builder()
BUILDERS = sorted(_MODELS)
# Split rather than parametrizing over everything and skipping half: an
# unconditional skip is indistinguishable in a report from a test that could
# not run, and it inflates the skip count REPRODUCING.md quotes.
FERMIONIC_BUILDERS = sorted(name for name, model in _MODELS.items()
                            if model.metadata["kind"] in FERMIONIC_KINDS)
SPIN_BUILDERS = sorted(name for name, model in _MODELS.items()
                       if model.metadata["kind"] in SPIN_KINDS)


@pytest.fixture(scope="module")
def models():
    return every_builder()


# ----------------------------------------------------------- cross-builder
@pytest.mark.parametrize("builder", BUILDERS)
def test_every_builder_declares_the_universal_keys(builder, models):
    metadata = models[builder].metadata
    assert metadata["metadata_schema"] == MODEL_METADATA_SCHEMA
    assert metadata["kind"] in MODEL_KINDS


def test_the_universal_keys_are_common_to_all_builders(models):
    """The regression this contract exists for: the intersection was empty."""
    shared = set.intersection(*(set(model.metadata) for model in models.values()))
    assert {"metadata_schema", "kind"} <= shared


@pytest.mark.parametrize("builder", FERMIONIC_BUILDERS)
def test_fermionic_models_declare_their_conventions_and_sector(builder, models):
    model = models[builder]
    metadata = model.metadata
    assert metadata["kind"] in FERMIONIC_KINDS
    assert spin_convention(model) == metadata["spin_convention"]
    assert metadata["spin_orbitals"] == model.n
    assert 2 * metadata["n_spatial_orbitals"] == model.n
    assert 0 <= metadata["n_electrons"] <= model.n
    # 2*S_z and N share parity for any determinant.
    assert abs(2.0 * metadata["sz"]) % 2 == metadata["n_electrons"] % 2


@pytest.mark.parametrize("builder", SPIN_BUILDERS)
def test_spin_models_declare_no_fermionic_metadata(builder, models):
    model = models[builder]
    for key in ("spin_convention", "spin_orbitals", "n_electrons", "sz"):
        assert key not in model.metadata
    with pytest.raises(ValueError, match="spin model"):
        spin_convention(model)


@pytest.mark.parametrize("builder", FERMIONIC_BUILDERS)
def test_declared_sector_matches_the_reference_program(builder, models):
    """The sector in metadata is the one the reference determinant occupies.

    Checked against ``reference_sector``, which reads the gates rather than the
    dict, so this compares two independent statements of the same fact.
    """
    from clifford_qc.models.lattice import reference_sector

    model = models[builder]
    electrons, sz = reference_sector(model.reference)
    assert electrons == model.metadata["n_electrons"]
    assert sz == pytest.approx(model.metadata["sz"])


# ------------------------------------------------------ construction refusal
def test_a_model_cannot_be_built_without_the_contract():
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="missing"):
        dataclasses.replace(reference, metadata={})


def test_an_unknown_kind_is_refused():
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="kind must be one of"):
        dataclasses.replace(reference, metadata={
            **reference.metadata, "kind": "lattice_gauge_theory"})


def test_a_stale_schema_string_is_refused():
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="metadata_schema must be"):
        dataclasses.replace(reference, metadata={
            **reference.metadata,
            "metadata_schema": "clifford_qc.model_metadata.v0"})


def test_an_unknown_spin_ordering_is_refused():
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="spin_convention must be one of"):
        dataclasses.replace(reference, metadata={
            **reference.metadata, "spin_convention": "sorted"})


def test_a_fermionic_model_missing_its_sector_is_refused():
    reference = hubbard(2, U=4.0)
    metadata = {k: v for k, v in reference.metadata.items() if k != "n_electrons"}
    with pytest.raises(ValueError, match="n_electrons"):
        dataclasses.replace(reference, metadata=metadata)


def test_spin_orbitals_disagreeing_with_the_qubit_count_is_refused():
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="but the model has n="):
        dataclasses.replace(reference, metadata={
            **reference.metadata, "spin_orbitals": reference.n + 2})


def test_a_declared_word_count_that_disagrees_is_refused():
    """A derivable quantity may be omitted; a wrong one reads as authoritative."""
    reference = rotate_model(ring_hopping(4), 4.0, "site")
    assert reference.metadata["pauli_words"] == len(reference.hamiltonian.terms)
    with pytest.raises(ValueError, match="pauli_words"):
        dataclasses.replace(reference, metadata={
            **reference.metadata, "pauli_words": 1})


def test_an_empty_sector_is_refused_by_parity():
    """``N`` odd with ``S_z`` integral names a sector with no determinants."""
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="mismatched parity"):
        dataclasses.replace(reference, metadata={
            **reference.metadata, "n_electrons": 3, "sz": 0.0})


# ------------------------------------------------------------- the assembler
def test_the_assembler_refuses_fermionic_metadata_on_a_spin_kind():
    with pytest.raises(ValueError, match="cannot declare fermionic metadata"):
        model_metadata("spin_lattice", n_electrons=2, sz=0.0)


def test_the_assembler_requires_the_full_sector_for_a_fermionic_kind():
    with pytest.raises(ValueError, match="needs"):
        model_metadata("molecular", spin_orbitals=4, n_electrons=2)


def test_the_validator_is_usable_before_a_model_exists():
    metadata = model_metadata("molecular", spin_orbitals=4, n_spatial_orbitals=2,
                              n_electrons=2, sz=0.0)
    validate_model_metadata(metadata, n_qubits=4, model_name="payload")
    with pytest.raises(ValueError, match="but the model has n="):
        validate_model_metadata(metadata, n_qubits=6, model_name="payload")


# --------------------------------------------------- why the key has no default
def test_wrong_spin_ordering_silently_changes_the_reduced_energy():
    """The measured cost of the default this contract removed.

    No exception is raised in either branch and both Hamiltonians look
    plausible; only the energy distinguishes them. This is why
    ``spin_convention`` raises instead of defaulting.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("stim")
    from clifford_qc.sparse import sparse_ground_in_sector

    model = hubbard(2, t=1.0, U=4.0, periodic=False)
    values, _ = sparse_ground_in_sector(model.hamiltonian, n_electrons=2, sz=0.0)
    exact = float(np.real(values[0]))
    assert exact == pytest.approx(-4.828427124746, abs=1e-9)

    hamiltonian = model.hamiltonian.to_mv()
    energies = {}
    for ordering in ("interleaved", "blocked"):
        encoding = fermion_encoding("parity+2q", n=4, spin_ordering=ordering,
                                    n_electrons=2, sz=0.0)
        restricted = encoding.restriction().operator(hamiltonian)
        energies[ordering] = float(np.linalg.eigvalsh(to_matrix(restricted))[0])

    assert energies["interleaved"] == pytest.approx(exact, abs=1e-9)
    assert energies["blocked"] == pytest.approx(-4.0, abs=1e-9)
    assert abs(energies["blocked"] - exact) > 0.8


def test_the_model_declares_the_ordering_the_reduction_needs():
    """The contract closes the loop: the consumer reads, never guesses."""
    model = hubbard(2, t=1.0, U=4.0, periodic=False)
    assert spin_convention(model) == "interleaved"
    stripped = {k: v for k, v in model.metadata.items()
                if k != "spin_convention"}
    with pytest.raises(ValueError, match="spin_convention"):
        dataclasses.replace(model, metadata=stripped)


def test_a_hand_built_model_must_state_its_conventions():
    """``Model`` is constructible directly, so the contract binds there too."""
    reference = hubbard(2, U=4.0)
    with pytest.raises(ValueError, match="missing"):
        Model(name="hand-built", n=reference.n,
              hamiltonian=reference.hamiltonian, reference=reference.reference,
              hva_layers=(), metadata={"kind": "fermionic_lattice"})
