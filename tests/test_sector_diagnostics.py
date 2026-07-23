import pytest

from clifford_qc import Parameter, PauliWord, Program, Rotor, comm, number_op_pauli
from clifford_qc.backends import ExactMVBackend
from clifford_qc.diagnostics import fermionic_sector_diagnostics
from clifford_qc.states import ket_density


def test_sector_diagnostics_on_closed_shell_basis_state():
    diag = fermionic_sector_diagnostics(ket_density(4, "1100"), 2)
    assert diag == pytest.approx({
        "particle_number_mean": 2.0,
        "particle_number_variance": 0.0,
        "spin_z_mean": 0.0,
        "spin_z_variance": 0.0,
        "target_sector_weight": 1.0,
    })


def test_published_h4_word_trajectory_reports_small_spin_leakage():
    """Regression for the corrected 160-word H4 exact trajectory.

    The words are derived from conserving excitation generators, but the
    independently parameterized Pauli rotors are not themselves symmetry
    preserving.  The benchmark must report this small nonzero leakage.
    """
    labels = (
        "IIYXXXII", "XZZYXZZX", "IXYIIXXI", "YXIIIIXX",
        "YYIIYXII", "IIYXIIXX", "YZZYIYXI", "IYXIXZZX",
        "XZXIYZXI", "IXZYIXZX", "YXIIXXII", "IIXXIIYX",
    )
    theta = (
        0.3167582582159472, 0.12610534283368602, 0.13052567311729607,
        0.08541101304140586, 0.06285445104430547, 0.03906789805989886,
        -0.08258553858045035, -0.08511756138118202,
        -0.04130560999486077, 0.04998129752732237,
        0.05990270889382398, -0.03486358570358782,
    )
    prog = Program(8)
    for q in range(4):
        prog.clifford("X", q)
    for k, label in enumerate(labels):
        prog.append(Rotor(PauliWord.from_label(label), Parameter(f"t{k}")))
    rho = ExactMVBackend().state(prog, theta)
    diag = fermionic_sector_diagnostics(rho, 4)

    assert diag["particle_number_mean"] == pytest.approx(4.0, abs=2e-8)
    assert diag["particle_number_variance"] == pytest.approx(3.964041e-8, rel=2e-4)
    assert diag["spin_z_variance"] == pytest.approx(8.066411e-4, rel=2e-5)
    assert diag["target_sector_weight"] == pytest.approx(0.9997983372, abs=2e-10)
    assert diag["target_sector_weight"] < 1.0


def test_individual_excitation_word_is_not_number_conserving_operator():
    word = PauliWord.from_label("IIYXXXII").to_mv()
    number = sum((number_op_pauli(8, j) for j in range(8)), start=0 * word)
    assert comm(number, word).terms
