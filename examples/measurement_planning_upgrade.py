"""Demonstrate variance-first, Ritz-weighted, and noise-gated measurement planning.

Run from the repository root:
    python examples/measurement_planning_upgrade.py

Uses only NumPy (no Stim or SciPy required): defaults to ``method="graph_state"``
readout compilation, reports Crawford's optimal shot ratio ``R_hat`` alongside
two-qubit gate cost across ``k``-commutativity rungs, enforces a GALIC-style
per-group noise gate on the superconducting device card, and demonstrates
density-free Ritz-weighted replanning on ``SampledProjectedProvider``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks.run_sampled_ratio import JointShotSimulator  # noqa: E402
from clifford_qc.ir import PauliWord  # noqa: E402
from clifford_qc.measurement import (  # noqa: E402
    DeviceCard,
    NoiseGatePolicy,
    compare_block_rungs,
    evaluate_plan_variance,
    plan_measurement,
)
from clifford_qc.measurement.functionals import ritz_functional  # noqa: E402
from clifford_qc.models import tfim  # noqa: E402
from clifford_qc.models.fcidump import fcidump_model  # noqa: E402
from clifford_qc.sampled_reference import PreparedSampledReference  # noqa: E402
from clifford_qc.subspace import identity_generator, pauli_orbit  # noqa: E402
from clifford_qc.subspace.sampled import SampledProjectedProvider  # noqa: E402

H4_FCIDUMP = ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"
BEH2_FCIDUMP = ROOT / "benchmarks/data/beh2_sto3g_r1.3264.FCIDUMP"
SUPERCONDUCTING_CARD = ROOT / "benchmarks/configs/device_cards/superconducting-like.json"


def _row_summary(label: str, metrics) -> dict[str, object]:
    return {
        "label": label,
        "block_size": metrics.block_size,
        "strategy": metrics.strategy,
        "method": metrics.method,
        "settings": metrics.n_settings,
        "r_hat": round(metrics.r_hat, 4),
        "grouped_variance": round(metrics.grouped_variance, 4),
        "cz_total": metrics.two_qubit_gates_total,
        "cz_max": metrics.two_qubit_gates_max,
        "depth_2q_max": metrics.two_qubit_depth_max,
        "admissible_settings": metrics.noise_gate_admissible_settings,
    }


def main() -> None:
    # 1. H4 Hamiltonian across k-commutativity rungs: LDF vs weighted SI.
    h4 = fcidump_model(H4_FCIDUMP)
    h4_ham = h4.hamiltonian.to_mv()
    h4_codes = sorted(c for c in h4_ham.terms if c != 0)
    h4_ldf = compare_block_rungs(h4.n, h4_codes, h4_ham, strategy="largest_degree")
    h4_si = compare_block_rungs(h4.n, h4_codes, h4_ham, strategy="sorted_insertion")

    # 2. BeH2 Hamiltonian under GALIC-style superconducting noise gate.
    beh2 = fcidump_model(BEH2_FCIDUMP)
    beh2_ham = beh2.hamiltonian.to_mv()
    beh2_codes = sorted(c for c in beh2_ham.terms if c != 0)
    card = DeviceCard.load(SUPERCONDUCTING_CARD)
    gate = NoiseGatePolicy.from_device_card(card, max_relative_bias=0.10)

    beh2_qwc = evaluate_plan_variance(
        plan_measurement(beh2.n, beh2_codes, 1, weights=beh2_ham),
        beh2_ham,
        strategy="sorted_insertion",
        noise_gate=gate,
    )
    beh2_fc = evaluate_plan_variance(
        plan_measurement(beh2.n, beh2_codes, beh2.n, weights=beh2_ham),
        beh2_ham,
        strategy="sorted_insertion",
        noise_gate=gate,
    )
    beh2_gated = evaluate_plan_variance(
        plan_measurement(beh2.n, beh2_codes, beh2.n, weights=beh2_ham, noise_gate=gate),
        beh2_ham,
        strategy="sorted_insertion",
        noise_gate=gate,
    )

    # 3. Density-free Ritz-weighted replanning on SampledProjectedProvider.
    spin = tfim(4, J=1.0, h=0.8)
    generators = [
        identity_generator(spin.n),
        *pauli_orbit([PauliWord.from_label(lbl) for lbl in ("XIII", "IXII", "IIXI", "IIIX")]),
    ]
    state = np.zeros(1 << spin.n, dtype=complex)
    state[0] = 1.0
    reference = PreparedSampledReference(
        spin.n, "tfim4-all-zero", JointShotSimulator(state, spin.n).draw
    )
    pilot_provider = SampledProjectedProvider(reference, spin.hamiltonian, generators, block_size=2)
    pilot_obs = reference.sample(
        pilot_provider.plan.settings, [200] * len(pilot_provider.plan.settings), seed=11
    )
    pilot_solved = pilot_provider.reconstruct(pilot_obs).problem.solve(tau_s=0.05)
    ritz_vec = pilot_solved.ritz_vector(0)
    ritz_energy = float(pilot_solved.energies[0])
    target_func = ritz_functional(
        pilot_provider.bank, pilot_provider.session.indices, ritz_vec, ritz_energy
    )
    ritz_provider = pilot_provider.with_ritz_plan(ritz_vec, ritz_energy, noise_gate=gate)

    payload = {
        "h4_rungs_largest_degree": [_row_summary(f"k={m.block_size}", m) for m in h4_ldf],
        "h4_rungs_sorted_insertion": [_row_summary(f"k={m.block_size}", m) for m in h4_si],
        "beh2_noise_gate_policy": gate.as_dict(),
        "beh2_comparison": [
            _row_summary("QWC (k=1)", beh2_qwc),
            _row_summary("Unconstrained FC (k=8)", beh2_fc),
            _row_summary("GALIC noise-gated FC (k=8)", beh2_gated),
        ],
        "tfim4_ritz_replanning": {
            "pilot_energy": round(ritz_energy, 6),
            "before_replan": _row_summary(
                "pilot LDF (k=2)", pilot_provider.plan_metrics(target_func)
            ),
            "after_ritz_replan": _row_summary(
                "Ritz-weighted noise-gated SI (k=2)", ritz_provider.plan_metrics(target_func)
            ),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
