"""Phase 15 convergence reporting on a bounded two-qubit toy.

The one-addition basis misses the second qubit's eigenstate. Reporting the
true residual exposes that without extending the pool or changing growth.
The analytic full-space spectrum supplies ground-state evidence separately.
"""

import json
import math

from clifford_qc.pauli import X, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import ConvergenceConfig, GroundStateReference, run_acase


def main():
    hamiltonian = Z(2, 0) + X(2, 0) + .2 * X(2, 1)
    reference = GroundStateReference(
        -math.sqrt(2) - .2, -math.sqrt(2) + .2,
        "analytic full-space spectrum of Z0 + X0 + 0.2 X1")
    result = run_acase(
        ket_density(2, "00"), hamiltonian, [X(2, 0), X(2, 1)], max_size=1,
        convergence=ConvergenceConfig(residual_tolerance=.01, ground_reference=reference))
    print(json.dumps(result.convergence_reports[0].as_dict(), indent=2))


if __name__ == "__main__":
    main()
