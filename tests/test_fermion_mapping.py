"""R2b contracts for linear fermion encodings and two-qubit reduction."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import c_op, cdag_op, total_number_op, total_sz_op
from clifford_qc.fermion_mapping import FermionEncoding, fermion_encoding
from clifford_qc.models.lattice import hubbard
from clifford_qc.pauli import I, Z
from clifford_qc.subspace import (
    Generator,
    assert_mapping_invariants,
    identity_generator,
    restricted_sector_operators,
)


def _simulate_cnot_program(program, occupation):
    bits = list(occupation)
    for operation in program.ops:
        assert operation.name == "CX"
        control, target = operation.qubits
        bits[target] ^= bits[control]
    return tuple(bits)


def _two_site_problem():
    model = hubbard((1, 2), t=1.0, U=4.0)
    reference = ExactMVBackend().state(model.reference, ())
    # A same-spin hopping direction preserves N, S_z, spin-up parity, and
    # total parity.  It exercises a non-diagonal whole-fermion generator.
    hopping = (
        cdag_op(model.n, 2) * c_op(model.n, 0)
        + cdag_op(model.n, 0) * c_op(model.n, 2)
    )
    generators = [
        identity_generator(model.n),
        Generator("N", total_number_op(model.n)),
        Generator("hop_up", hopping),
    ]
    return model, reference, generators


def test_declared_encoding_matrices_match_the_standard_bit_conventions():
    assert fermion_encoding("jw", 6).rows == (1, 2, 4, 8, 16, 32)
    assert fermion_encoding("parity", 6).rows == (1, 3, 7, 15, 31, 63)
    # Fenwick intervals: [0], [0:2], [2], [0:4], [4], [4:6].
    assert fermion_encoding("bk", 6).rows == (1, 3, 4, 15, 16, 48)


@pytest.mark.parametrize("name", ["jw", "parity", "bk", "parity+2q", "bk+2q"])
def test_cnot_network_realizes_the_declared_binary_matrix(name):
    kwargs = {} if "+2q" not in name else {"n_electrons": 2, "sz": 0.0}
    encoding = fermion_encoding(name, 6, **kwargs)
    for occupation in itertools.product((0, 1), repeat=6):
        assert _simulate_cnot_program(
            encoding.program(), occupation
        ) == encoding.encode_bits(occupation)


def test_unreduced_networks_use_the_standard_linear_size_constructions():
    assert len(fermion_encoding("jw", 8).program().ops) == 0
    assert len(fermion_encoding("parity", 8).program().ops) == 7
    assert len(fermion_encoding("bk", 8).program().ops) == 7


@pytest.mark.parametrize("name", ["parity+2q", "bk+2q"])
def test_reduced_networks_use_a_linear_base_plus_fixup_construction(name):
    base_name = name.split("+", 1)[0]
    for n in range(4, 34, 2):
        base_size = len(fermion_encoding(base_name, n).program().ops)
        reduced_size = len(
            fermion_encoding(name, n, n_electrons=2, sz=0.0).program().ops
        )
        assert reduced_size <= base_size + 5 * n


def test_program_rows_remain_authoritative_for_custom_or_mismatched_names():
    rows = fermion_encoding("parity", 4).rows
    for encoding in (
        FermionEncoding("jw", 4, rows),
        FermionEncoding("custom-linear", 4, rows),
    ):
        for occupation in itertools.product((0, 1), repeat=4):
            assert _simulate_cnot_program(
                encoding.program(), occupation
            ) == encoding.encode_bits(occupation)


@pytest.mark.parametrize("name", ["parity+2q", "bk+2q"])
def test_reduced_encodings_put_declared_sector_parities_on_fixed_qubits(name):
    encoding = fermion_encoding(name, 6, n_electrons=2, sz=0.0)
    assert encoding.fixed_qubits == (4, 5)
    assert encoding.signs == (-1, 1)  # odd N_alpha, even N
    # Every determinant in N=2, Sz=0 has the same final two encoded bits,
    # independent of how its one alpha and one beta electron are distributed.
    for alpha in (0, 2, 4):
        for beta in (1, 3, 5):
            occupation = [0] * 6
            occupation[alpha] = occupation[beta] = 1
            encoded = encoding.encode_bits(occupation)
            assert encoded[-2:] == (1, 0)


@pytest.mark.parametrize("name", ["parity+2q", "bk+2q"])
@pytest.mark.parametrize(
    ("n", "n_electrons", "sz"),
    [(4, 2, 0.0), (6, 3, 0.5), (8, 4, 1.0)],
)
def test_reduced_sector_bits_are_constant_across_sizes_and_sectors(
    name, n, n_electrons, sz
):
    encoding = fermion_encoding(
        name, n, n_electrons=n_electrons, sz=sz
    )
    n_alpha = int(n_electrons / 2 + sz)
    expected_bits = (n_alpha % 2, n_electrons % 2)
    expected_signs = tuple(1 if bit == 0 else -1 for bit in expected_bits)
    assert encoding.signs == expected_signs

    matching_determinants = 0
    for occupation in itertools.product((0, 1), repeat=n):
        if sum(occupation) != n_electrons:
            continue
        if sum(occupation[::2]) - sum(occupation[1::2]) != 2 * sz:
            continue
        matching_determinants += 1
        assert encoding.encode_bits(occupation)[-2:] == expected_bits
    assert matching_determinants > 0


def test_encoding_factory_rejects_ambiguous_or_impossible_reductions():
    with pytest.raises(ValueError, match="require n_electrons and sz"):
        fermion_encoding("bk+2q", 6)
    with pytest.raises(ValueError, match="at least four"):
        fermion_encoding("parity+2q", 2, n_electrons=1, sz=0.5)
    with pytest.raises(ValueError, match="even spin-orbital"):
        fermion_encoding("parity+2q", 5, n_electrons=2, sz=0.0)
    with pytest.raises(ValueError, match="integral spin sector"):
        fermion_encoding("bk+2q", 6, n_electrons=2, sz=0.25)
    with pytest.raises(ValueError, match="integral spin sector"):
        fermion_encoding("parity+2q", 8, n_electrons=4, sz=1e-9)
    with pytest.raises(ValueError, match="cannot exceed"):
        fermion_encoding("bk+2q", 6, n_electrons=8, sz=0.0)
    with pytest.raises(ValueError, match="larger than its orbital count"):
        fermion_encoding("bk+2q", 6, n_electrons=4, sz=2.0)
    with pytest.raises(ValueError, match="unknown fermion encoding"):
        fermion_encoding("ternary-tree", 6)
    with pytest.raises(ValueError, match="invertible"):
        FermionEncoding("broken", 3, (1, 1, 4))
    with pytest.raises(ValueError, match="binary entries"):
        fermion_encoding("bk", 3).encode_bits((0, 0.5, 1))
    with pytest.raises(ValueError, match="spin_ordering"):
        fermion_encoding("parity", 6, spin_ordering="alternating")


@pytest.mark.parametrize("name", ["parity+2q", "bk+2q"])
def test_reduction_rotation_maps_both_physical_symmetries_to_fixed_z(name):
    pytest.importorskip("stim")
    n = 6
    restriction = fermion_encoding(
        name, n, n_electrons=2, sz=0.0
    ).restriction()
    spin_up_parity = I(n)
    total_parity = I(n)
    for qubit in range(n):
        total_parity = total_parity * Z(n, qubit)
        if qubit % 2 == 0:
            spin_up_parity = spin_up_parity * Z(n, qubit)
    assert restriction.rotate(spin_up_parity).is_close(Z(n, n - 2), 1e-12)
    assert restriction.rotate(total_parity).is_close(Z(n, n - 1), 1e-12)


@pytest.mark.parametrize("name", ["jw", "parity", "bk"])
def test_pure_mapping_arms_pass_the_complete_small_system_invariant_gate(name):
    pytest.importorskip("stim")
    model, reference, generators = _two_site_problem()
    report = assert_mapping_invariants(
        reference,
        model.hamiltonian,
        generators,
        fermion_encoding(name, model.n).restriction(),
    )
    assert report.qubits_removed == 0
    assert report.mapped_qubits == model.n
    assert report.word_bijection_checked
    assert report.dense_spectrum_checked
    assert report.word_universe_after == report.word_universe_before
    assert report.max_spectrum_error == pytest.approx(0.0, abs=1e-12)
    assert report.max_operator_transport_error == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ["parity+2q", "bk+2q"])
def test_reduced_mapping_arms_match_an_independent_fixed_parity_block(name):
    pytest.importorskip("stim")
    model, reference, generators = _two_site_problem()
    encoding = fermion_encoding(
        name,
        model.n,
        n_electrons=model.metadata["n_electrons"],
        sz=model.metadata["sz"],
    )
    restriction = encoding.restriction()
    report = assert_mapping_invariants(
        reference, model.hamiltonian, generators, restriction
    )
    assert report.qubits_removed == 2
    assert report.mapped_qubits == model.n - 2
    assert not report.word_bijection_checked
    assert report.dense_spectrum_checked
    assert report.max_spectrum_error == pytest.approx(0.0, abs=1e-12)
    assert report.max_operator_transport_error == pytest.approx(0.0, abs=1e-12)

    number, sz = restricted_sector_operators(restriction)
    assert number.n == sz.n == model.n - 2
    # Transported sector operators remain Hermitian and have the eigenvalue
    # sets of their independently transformed fixed-parity blocks.
    assert np.all(np.isreal(np.linalg.eigvalsh(to_matrix(number))))
    assert np.all(np.isreal(np.linalg.eigvalsh(to_matrix(sz))))


def test_blocked_spin_ordering_survives_the_restriction_boundary():
    pytest.importorskip("stim")
    encoding = fermion_encoding(
        "parity+2q",
        6,
        n_electrons=3,
        sz=0.5,
        spin_ordering="blocked",
    )
    restriction = encoding.restriction()
    _, transported_sz = restricted_sector_operators(restriction)
    expected = restriction.operator(
        total_sz_op(6, spin_ordering="blocked"), require_commuting=True
    )
    wrong = restriction.operator(
        total_sz_op(6, spin_ordering="interleaved"), require_commuting=True
    )
    assert restriction.spin_ordering == "blocked"
    assert transported_sz.is_close(expected, 1e-12)
    assert not transported_sz.is_close(wrong, 1e-12)


def test_wrong_reduction_sector_is_rejected_by_reference_transport():
    pytest.importorskip("stim")
    model, reference, generators = _two_site_problem()
    wrong = fermion_encoding(
        "parity+2q", model.n, n_electrons=0, sz=0.0
    ).restriction()
    with pytest.raises(ValueError, match="reference is not in the fixed sector"):
        assert_mapping_invariants(reference, model.hamiltonian, generators, wrong)


def test_sector_changing_generator_blocks_the_mapping_arm():
    pytest.importorskip("stim")
    model, reference, generators = _two_site_problem()
    changes_total_parity = c_op(model.n, 0) + cdag_op(model.n, 0)
    generators.append(Generator("changes_total_parity", changes_total_parity))
    restriction = fermion_encoding(
        "bk+2q",
        model.n,
        n_electrons=model.metadata["n_electrons"],
        sz=model.metadata["sz"],
    ).restriction()
    with pytest.raises(AssertionError, match="physical generator domain"):
        assert_mapping_invariants(
            reference, model.hamiltonian, generators, restriction
        )


def test_reduction_names_generator_collisions_instead_of_crashing():
    pytest.importorskip("stim")
    model, reference, _ = _two_site_problem()
    total_parity = I(model.n)
    for qubit in range(model.n):
        total_parity = total_parity * Z(model.n, qubit)
    generators = [
        identity_generator(model.n),
        Generator("total_parity", total_parity),
    ]
    restriction = fermion_encoding(
        "parity+2q",
        model.n,
        n_electrons=model.metadata["n_electrons"],
        sz=model.metadata["sz"],
    ).restriction()
    with pytest.raises(
        AssertionError, match="collapsed distinct generators.*'I'.*total_parity"
    ):
        assert_mapping_invariants(
            reference, model.hamiltonian, generators, restriction
        )


def test_comparison_tolerance_does_not_change_zero_or_leakage_gates():
    pytest.importorskip("stim")
    model, reference, _ = _two_site_problem()
    generators = [
        identity_generator(model.n),
        Generator("small_number", 1e-6 * total_number_op(model.n)),
    ]
    report = assert_mapping_invariants(
        reference,
        model.hamiltonian,
        generators,
        fermion_encoding("parity", model.n).restriction(),
        relative_tolerance=1e-3,
        zero_tolerance=1e-12,
        leakage_tolerance=1e-12,
    )
    assert report.mapping_name == "parity"
    assert report.relative_tolerance == 1e-3
    assert report.zero_tolerance == 1e-12


def test_mapping_gate_rejects_invalid_tolerance_contracts():
    pytest.importorskip("stim")
    model, reference, generators = _two_site_problem()
    restriction = fermion_encoding("jw", model.n).restriction()
    with pytest.raises(ValueError, match="non-negative"):
        assert_mapping_invariants(
            reference,
            model.hamiltonian,
            generators,
            restriction,
            leakage_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="cannot both be zero"):
        assert_mapping_invariants(
            reference,
            model.hamiltonian,
            generators,
            restriction,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
