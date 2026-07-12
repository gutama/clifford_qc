"""clifford_qc: operator-centric quantum computing in Cl(2n,C)."""

from .multivector import MV, TOL, label_to_code, code_to_label, word_mul
from .pauli import I, X, Y, Z, P, pauli_string, comm, anticomm, pauli_basis, tensor
from .clifford import gamma, pseudoscalar, blade_from_mask, grade, grades
from .fermion import c_op, cdag_op, number_op, number_op_pauli
from .gates import (
    H, S, T, H_gate, S_gate, T_gate, rotor, RX, RY, RZ,
    controlled, CNOT, CZ, SWAP, TOFFOLI, expm_taylor,
    trotter_unitary, trotter2_unitary,
)
from .states import (
    ket_density, computational_projector, plus_density, evolve, expectation,
    probability, purity, computational_probabilities, measure, z_projectors,
    partial_trace, partial_transpose, bell_density, ghz_density,
)
from .channels import check_kraus, apply_channel, depolarizing, dephasing, amplitude_damping
from .matrix import code_to_matrix, to_matrix, from_matrix, density_from_statevector, expm_matrix, exact_ground
from .diagnostics import negativity, vn_entropy, fidelity_pure, trace_cyclicity_error
from .ir import (
    PauliWord, PauliSum, Parameter, ParameterGroup, Rotor, NamedClifford,
    MeasurementTask, Program, conjugate_pauli_word, clifford_angle_index,
    expectation_value, parameter_shift_gradient, adjoint_gradient,
)
from .qasm3 import to_qasm3

__all__ = [name for name in globals() if not name.startswith("_")]
__version__ = "0.3.0"
