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
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.fermion_mapping import fermion_encoding
from clifford_qc.matrix import to_matrix
from clifford_qc.models import (FERMIONIC_KINDS, MODEL_KINDS,
                                MODEL_METADATA_SCHEMA, SPIN_KINDS,
                                SPIN_ORDERINGS)
from clifford_qc.models.effective import (EFFECTIVE_HAMILTONIAN_SCHEMA,
                                          effective_hamiltonian)
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import (anderson_impurity, extended_hubbard,
                                        hubbard, kanamori, kitaev_honeycomb)
from clifford_qc.models.metadata import (_FERMIONIC_KEYS, model_metadata,
                                         sector_parity_matches,
                                         spin_convention,
                                         validate_model_metadata)
from clifford_qc.models.orbital import rotate_model
from clifford_qc.models.spin import Model, random_ising, tfim, xxz

FCIDUMP = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "h4_sto3g_r0.9.FCIDUMP"


def ring_hopping(sites: int, t: float = 1.0) -> np.ndarray:
    matrix = np.zeros((sites, sites))
    bonds = [(i, i + 1) for i in range(sites - 1)]
    if sites > 2:
        bonds.append((sites - 1, 0))
    for i, j in bonds:
        matrix[i, j] -= t
        matrix[j, i] -= t
    return matrix


def downfolded_model():
    """``effective_hamiltonian`` from a minimal payload -- numpy only.

    It is a public builder like any other, and it declares
    ``n_spatial_orbitals``, which makes the validator assert
    ``2 * sites == n_qubits`` on every downfolded payload. Left out of the
    cross-builder check, that assertion would go unexercised.
    """
    return effective_hamiltonian({
        "schema": EFFECTIVE_HAMILTONIAN_SCHEMA,
        "one_body": [[0.0, -1.0], [-1.0, 0.0]],
        "onsite_u": 2.0,
        "reference_occupied_spin_orbitals": [0, 3],
        "sector": {"n_electrons": 2, "sz": 0.0},
    })


def every_builder():
    """One model per public builder, spin and fermionic alike."""
    return {
        "effective_hamiltonian": downfolded_model(),
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


# Split rather than parametrizing over everything and skipping half: an
# unconditional skip is indistinguishable in a report from a test that could
# not run, and it inflates the skip count REPRODUCING.md quotes.
#
# The split is stated here rather than derived by building every model at
# import: `--collect-only` would otherwise parse the H4 FCIDUMP and assemble
# four Hamiltonians before a single test runs, and `benchmarks/check_docs.py`
# runs exactly that command. `test_the_split_covers_every_builder` below checks
# these names against `every_builder()`, so the two cannot drift.
SPIN_BUILDERS = ["kitaev_honeycomb", "random_ising", "tfim", "xxz"]
FERMIONIC_BUILDERS = ["anderson_impurity", "effective_hamiltonian",
                      "extended_hubbard", "fcidump", "hubbard", "kanamori",
                      "rotate_model"]
BUILDERS = sorted(SPIN_BUILDERS + FERMIONIC_BUILDERS)


@pytest.fixture(scope="module")
def models():
    return every_builder()


def test_the_split_covers_every_builder(models):
    """The static lists above are the ones `every_builder` actually returns."""
    assert sorted(models) == BUILDERS
    by_kind = {name: model.metadata["kind"] for name, model in models.items()}
    assert sorted(n for n, k in by_kind.items() if k in SPIN_KINDS) \
        == sorted(SPIN_BUILDERS)
    assert sorted(n for n, k in by_kind.items() if k in FERMIONIC_KINDS) \
        == sorted(FERMIONIC_BUILDERS)


# ----------------------------------------------------------- cross-builder
@pytest.mark.parametrize("builder", BUILDERS)
def test_every_builder_declares_the_universal_keys(builder, models):
    metadata = models[builder].metadata
    assert metadata["metadata_schema"] == MODEL_METADATA_SCHEMA
    assert metadata["kind"] in MODEL_KINDS


def test_the_chemistry_builders_agree_with_the_rest(models):
    """The pyscf-gated family, checked against the same contract.

    One gate for the whole family rather than a parametrization: pyscf is
    absent from the numpy-only job, and a per-builder skip would multiply one
    unavailable dependency into several lines of a skip count REPRODUCING.md
    quotes.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    from clifford_qc.models.chemistry import h2

    model = h2(0.735)
    assert model.metadata["metadata_schema"] == MODEL_METADATA_SCHEMA
    assert model.metadata["kind"] in FERMIONIC_KINDS
    assert spin_convention(model) in SPIN_ORDERINGS
    assert model.metadata["spin_orbitals"] == model.n
    assert sector_parity_matches(model.metadata["sz"],
                                 model.metadata["n_electrons"])
    shared = set.intersection(set(model.metadata),
                              *(set(m.metadata) for m in models.values()))
    assert {"metadata_schema", "kind"} <= shared


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
    # 2*S_z and N share parity for any determinant. Asked via the validator's
    # own predicate rather than restated here: a second spelling of a float
    # comparison is a second chance to get the wrap-around wrong.
    assert sector_parity_matches(metadata["sz"], metadata["n_electrons"])


@pytest.mark.parametrize("builder", SPIN_BUILDERS)
def test_spin_models_declare_no_fermionic_metadata(builder, models):
    model = models[builder]
    for key in _FERMIONIC_KEYS:
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
    metadata = model_metadata("molecular", spin_convention="interleaved",
                              spin_orbitals=4, n_spatial_orbitals=2,
                              n_electrons=2, sz=0.0)
    validate_model_metadata(metadata, n_qubits=4, model_name="payload")
    with pytest.raises(ValueError, match="but the model has n="):
        validate_model_metadata(metadata, n_qubits=6, model_name="payload")


def test_the_assembler_refuses_to_default_the_spin_ordering():
    """A default here would only move the silent default from the readers.

    ``validate_model_metadata`` cannot catch the omission -- a defaulted value
    is present and legal by the time it looks -- so the assembler is the only
    place the omission is still visible.
    """
    with pytest.raises(ValueError, match=r"needs \['spin_convention'\]"):
        model_metadata("molecular", spin_orbitals=4, n_spatial_orbitals=2,
                       n_electrons=2, sz=0.0)
    with pytest.raises(ValueError, match="spin_convention must be one of"):
        model_metadata("molecular", spin_convention="sideways", spin_orbitals=4,
                       n_spatial_orbitals=2, n_electrons=2, sz=0.0)


def test_a_spin_model_cannot_declare_an_ordering_it_does_not_have():
    """Refused, not silently dropped: the caller asked for something wrong."""
    with pytest.raises(ValueError, match="cannot declare fermionic metadata"):
        model_metadata("spin_lattice", spin_convention="blocked", sites=4)


def test_the_sector_parity_check_wraps():
    """A float a hair below an even 2*S_z is the same sector, not an empty one.

    Compared without wrapping, ``(2*sz) % 2`` lands just below 2.0 rather than
    just above 0.0 and the sector is refused -- and only on one side of zero,
    so the same physical sector passes or fails depending on the sign.
    """
    assert sector_parity_matches(1.0, 2)
    assert sector_parity_matches(0.9999999999999999, 2)
    assert sector_parity_matches(-0.9999999999999999, 2)
    assert sector_parity_matches(0.5, 1) and sector_parity_matches(-0.5, 1)
    assert not sector_parity_matches(0.5, 2)
    assert not sector_parity_matches(0.0, 3)


def test_the_validator_accepts_numpy_integers_in_a_payload():
    """``np.int64`` is not an ``int`` subclass the way ``np.float64`` is a float.

    A payload that has not been through a builder carries whatever numpy handed
    it, and the validator is documented for exactly that use.
    """
    metadata = model_metadata("molecular", spin_convention="interleaved",
                              spin_orbitals=4, n_spatial_orbitals=2,
                              n_electrons=2, sz=0.0)
    numpyish = dict(metadata, spin_orbitals=np.int64(4),
                    n_spatial_orbitals=np.int64(2), n_electrons=np.int64(2),
                    sz=np.float64(0.0))
    validate_model_metadata(numpyish, n_qubits=4, model_name="payload")
    for bad in (True, 4.5, "4"):
        with pytest.raises(TypeError, match="must be an int"):
            validate_model_metadata(dict(metadata, spin_orbitals=bad),
                                    n_qubits=4, model_name="payload")


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
