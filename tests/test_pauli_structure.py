import json
from pathlib import Path

import pytest

from clifford_qc import (MV, X, Z, c_op, cdag_op, gf2_rank,
                         spin_conserving_x_rank_ceiling,
                         validate_spin_conserving_x_rank, x_mask_rank)
from clifford_qc.models import (anderson_impurity, extended_hubbard,
                                fcidump_model, hubbard, kanamori,
                                load_effective_hamiltonian)


ROOT = Path(__file__).resolve().parents[1]
FCIDUMPS = (
    ("h4_sto3g_r0.9.FCIDUMP", 5),
    ("beh2_sto3g_r1.3264.FCIDUMP", 3),
    ("lih_sto3g_r1.5949_cas4e4o.FCIDUMP", 5),
    ("h2o_sto3g_cas8e6o.FCIDUMP", 8),
)


def test_gf2_rank_eliminates_packed_rows_and_validates_width():
    assert gf2_rank([0b1010, 0b0110, 0b1100, 0, 0b1010], 4) == 2
    assert gf2_rank([], 0) == 0

    with pytest.raises(ValueError, match="exceeds width"):
        gf2_rank([0b10000], 4)
    with pytest.raises(ValueError, match="non-negative"):
        gf2_rank([-1], 4)
    with pytest.raises(TypeError, match="integers"):
        gf2_rank([1.0], 4)


def test_x_mask_rank_ignores_z_only_and_dependent_masks():
    operator = X(4, 0) + X(4, 1) + X(4, 0) * X(4, 1) + Z(4, 3)
    assert x_mask_rank(operator) == 2


def test_native_spin_conserving_jordan_wigner_hamiltonian_obeys_ceiling():
    n = 6
    hamiltonian = MV(n)
    for spin in (0, 1):
        p, q = spin, 2 + spin
        hop = cdag_op(n, p) * c_op(n, q)
        hamiltonian = hamiltonian + hop + hop.dagger()
    hamiltonian = hamiltonian + (
        cdag_op(n, 0) * c_op(n, 0)
        * cdag_op(n, 1) * c_op(n, 1)
    )

    rank = validate_spin_conserving_x_rank(hamiltonian)
    assert rank == 2
    assert rank <= spin_conserving_x_rank_ceiling(n)


@pytest.mark.parametrize("filename, expected_rank", FCIDUMPS)
def test_committed_fcidump_models_obey_spin_conserving_ceiling(filename, expected_rank):
    model = fcidump_model(ROOT / "benchmarks" / "data" / filename)
    rank = validate_spin_conserving_x_rank(model.hamiltonian)
    assert rank == expected_rank
    assert rank <= model.n - 2


@pytest.mark.parametrize(
    "factory, expected_rank",
    [
        (lambda: hubbard((2, 2)), 6),
        (lambda: extended_hubbard((2, 2)), 6),
        (lambda: kanamori(sites=2, n_orbitals=2), 5),
        (lambda: anderson_impurity(n_bath=2), 4),
    ],
    ids=("hubbard", "extended-hubbard", "kanamori", "anderson-impurity"),
)
def test_fermionic_lattice_models_obey_spin_conserving_ceiling(factory, expected_rank):
    model = factory()
    rank = validate_spin_conserving_x_rank(model.hamiltonian)
    assert rank == expected_rank
    assert rank <= model.n - 2


def test_effective_hamiltonian_ingestion_obeys_spin_conserving_ceiling(tmp_path):
    payload = {
        "schema": "clifford_qc.effective_hamiltonian.v1",
        "name": "rank-check-dimer",
        "energy_unit": "eV",
        "one_body": [[0.0, -1.0], [-1.0, 0.0]],
        "onsite_u": [4.0, 4.0],
        "reference_occupied_spin_orbitals": [0, 3],
        "sector": {"n_electrons": 2, "sz": 0.0},
    }
    path = tmp_path / "effective.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = load_effective_hamiltonian(path)
    rank = validate_spin_conserving_x_rank(model.hamiltonian)
    assert rank == 2
    assert rank <= model.n - 2


def test_validator_rejects_total_parity_only_counterexample():
    operator = (X(4, 0) * X(4, 1)
                + X(4, 0) * X(4, 2)
                + X(4, 0) * X(4, 3))
    assert x_mask_rank(operator) == 3
    assert spin_conserving_x_rank_ceiling(4) == 2
    with pytest.raises(ValueError, match=r"r_X=3 exceeds 2"):
        validate_spin_conserving_x_rank(operator)
