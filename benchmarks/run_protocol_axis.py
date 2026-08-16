"""R3 -- the protocol axis: the ``mapping x k`` grid.

R2b measured five fermion encodings at one fixed measurement protocol (QWC),
and its record carries ``deferred_to_r3`` naming the two quantities it did not
measure: ``G(k)`` and coverage across protocol rungs. This producer supplies
them, by sweeping the dyadic block-commuting rung ``k`` inside each mapping arm
instead of holding it at ``k = 1``.

**Why the k=1 column must reproduce R2b.** The grid extends a frozen record, so
one grouping rule has to span it. ``block_commuting_partition`` at ``k = 1`` is
qubit-wise commutation, and on every arm of both systems it returns exactly the
setting count R2b froze -- H4 ``913/533/351/615/403`` and BeH2
``353/41/27/41/27``. That agreement is asserted here, not assumed: a rung that
drifts from the frozen column means the grid is measuring a different partition
than the record it claims to extend, and the run stops.

**What this record may claim.** Group counts, coverage, and the synthesis
resources a declared device card prices at uniform shots. It is labelled
``structural``: no accuracy-matched ``C(epsilon)`` appears, because that needs
R1's exact-tier shot search, which only BeH2 clears on its bias floor. P5's
second clause -- that the mapping's ``C_time`` gap closes monotonically -- is
therefore *not* settled here; only its first clause, about ``G(k = n)``, is
measurable from this record.

    python benchmarks/run_protocol_axis.py
    python benchmarks/check_protocol_axis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.fermion_mapping import fermion_encoding
from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.measurement.block_synthesis import (
    SINGLE_QUBIT_GATES,
    synthesize_block_settings,
)
from clifford_qc.measurement.cost import estimator_information
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import Generator
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.elements import MatrixElementBank

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_clifford_hierarchy import _resource_metrics
    from benchmarks.run_mapping_axis import (
        _build_model,
        _device_costs,
        _selected_generators,
        load_config as load_mapping_config,
        load_device_cards,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_clifford_hierarchy import _resource_metrics
    from run_mapping_axis import (
        _build_model,
        _device_costs,
        _selected_generators,
        load_config as load_mapping_config,
        load_device_cards,
    )

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "protocol_axis.json"
REFERENCE = HERE / "reference_results" / "protocol_axis.json"
MAPPING_REFERENCE = HERE / "reference_results" / "mapping_axis.json"
SCHEMA = "clifford_qc.protocol_axis.v1"


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "clifford_qc.protocol_axis_config.v1":
        raise ValueError("unexpected protocol-axis config schema")
    return config


def frozen_qwc_settings() -> dict[tuple[str, str], int]:
    """The ``k = 1`` setting count R2b froze, per ``(system, arm)``."""
    record = json.loads(MAPPING_REFERENCE.read_text(encoding="utf-8"))
    return {
        (system["system"], arm["mapping"]): int(arm["measurement"]["settings"])
        for system in record["systems"]
        for arm in system["arms"]
    }


def _mapped_bank(name: str, model, reference):
    """Transport the frozen selection through one encoding, as R2b does."""
    reduced = name.endswith("+2q")
    encoding = fermion_encoding(
        name,
        model.n,
        n_electrons=int(model.metadata["n_electrons"]) if reduced else None,
        sz=float(model.metadata["sz"]) if reduced else None,
        spin_ordering=model.metadata.get("spin_convention", "interleaved"),
    )
    return encoding, encoding.restriction()


def _rung(
    n: int,
    codes: Sequence[int],
    block_size: int,
    cards,
    shots: int,
) -> dict:
    groups = block_commuting_partition(n, codes, block_size)
    synthesis = synthesize_block_settings(n, codes, groups, block_size)
    settings = synthesis.settings
    estimators = {
        estimator: estimator_information(
            synthesis.compatibility, synthesis.assignment, shots,
            estimator=estimator,
        )
        for estimator in ("single_assignment", "pooled")
    }
    coverage = synthesis.coverage
    return {
        "block_size": block_size,
        "settings": synthesis.n_settings,
        "group_sizes": {
            "min": min(len(g) for g in groups),
            "max": max(len(g) for g in groups),
            "mean": sum(len(g) for g in groups) / len(groups),
        },
        "words_per_setting": len(codes) / synthesis.n_settings,
        "gate_counts_per_sweep": dict(sorted(synthesis.gate_counts.items())),
        "one_qubit_gate_set": list(SINGLE_QUBIT_GATES),
        "resource_metrics": _resource_metrics(settings),
        "coverage": {
            "mean_fraction_of_words_read": sum(coverage) / len(coverage),
            "max_fraction_of_words_read": max(coverage),
            "min_fraction_of_words_read": min(coverage),
        },
        "estimators": estimators,
        "device_costs": _device_costs(
            cards, settings, shots_per_setting=shots, n_qubits=n
        ),
        "z_only_restrictions_checked": synthesis.z_only_restrictions_checked,
    }


def build_system_record(
    spec: dict,
    *,
    arms: Sequence[str],
    block_sizes: Sequence[int],
    cards,
    shots: int,
    frozen: dict[tuple[str, str], int],
) -> dict:
    model, construction = _build_model(spec)
    _, selected = _selected_generators(model, spec)
    reference = ExactMVBackend().state(model.reference, ())
    backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact_energy = float(backend.ground_state(model.hamiltonian, k=1)[0][0])

    arm_records = []
    for name in arms:
        encoding, restriction = _mapped_bank(name, model, reference)
        transported = restriction.transport(
            hamiltonian=as_multivector(model.hamiltonian),
            reference=reference,
            generators=(generator.mv for generator in selected),
        )
        mapped = [
            Generator(generator.label, image)
            for generator, image in zip(selected, transported.generators)
        ]
        bank = MatrixElementBank(
            transported.reference, transported.hamiltonian, mapped
        )
        result = bank.solve()
        codes = sorted(code for code in bank.word_set() if code != 0)
        n = transported.n
        # k is a block width, so it cannot exceed the arm's measured register.
        rungs = sorted({min(int(k), n) for k in block_sizes})
        rows = [_rung(n, codes, k, cards, shots) for k in rungs]

        expected = frozen.get((spec["key"], name))
        observed = next(row["settings"] for row in rows if row["block_size"] == 1)
        if expected is not None and observed != expected:
            raise ValueError(
                f"{spec['key']}/{name}: k=1 gives {observed} settings but the "
                f"frozen R2b record froze {expected}; the grid would not be "
                f"extending the record it claims to extend"
            )
        arm_records.append({
            "mapping": name,
            "measured_qubits": n,
            "word_universe": len(codes),
            "basis_size": len(mapped),
            "retained_rank": int(result.effective_rank),
            "ground_energy": float(result.ground_energy),
            "error_millihartree": float(result.ground_energy - exact_energy) * 1e3,
            "frozen_qwc_settings": expected,
            "rungs": rows,
        })

    return {
        "system": spec["key"],
        "label": spec["label"],
        "n_qubits": model.n,
        "exact_sector_energy": exact_energy,
        "construction": construction,
        "arms": arm_records,
    }


def _mapping_spread(system: dict) -> dict:
    """P5's first clause: does the mapping gap in ``G(k)`` close as ``k -> n``?

    Only arms sharing a measured qubit count are compared. A ``+2q`` arm has
    two fewer qubits, so its ``k = n`` rung is a narrower full-commuting
    problem; folding it into the same spread would compare different registers
    and manufacture a gap that is a register-size effect.
    """
    by_width: dict[int, list[dict]] = {}
    for arm in system["arms"]:
        by_width.setdefault(arm["measured_qubits"], []).append(arm)
    out = {}
    for width, arms in sorted(by_width.items()):
        if len(arms) < 2:
            continue
        widths = {}
        for block_size in sorted(
            {row["block_size"] for arm in arms for row in arm["rungs"]}
        ):
            counts = [
                row["settings"]
                for arm in arms
                for row in arm["rungs"]
                if row["block_size"] == block_size
            ]
            if len(counts) != len(arms):
                continue
            widths[str(block_size)] = {
                "settings": counts,
                "spread": max(counts) / min(counts),
            }
        out[str(width)] = {
            "arms": [arm["mapping"] for arm in arms],
            "by_block_size": widths,
        }
    return out


def build_record(config: dict | None = None, cards=None) -> dict:
    config = load_config() if config is None else config
    cards = load_device_cards() if cards is None else list(cards)
    mapping_config = load_mapping_config()
    specs = {spec["key"]: spec for spec in mapping_config["systems"]}
    frozen = frozen_qwc_settings()
    protocol = config["protocol"]
    systems = [
        build_system_record(
            specs[key],
            arms=config["mapping_arms"],
            block_sizes=protocol["block_sizes"],
            cards=cards,
            shots=int(protocol["uniform_raw_shots_per_setting"]),
            frozen=frozen,
        )
        for key in config["systems"]
    ]
    record = {
        "schema": SCHEMA,
        "phase": "R3_protocol_axis",
        "discharges": ["G(k) across protocol rungs", "coverage across protocol rungs"],
        "protocol": protocol,
        "claim_boundary": config["claim_boundary"],
        "mapping_axis_reference": {
            "path": "benchmarks/reference_results/mapping_axis.json",
            "k1_settings_reproduced": True,
        },
        "device_cards": [
            {"name": card.name, "sha256": card.sha256} for card in cards
        ],
        "systems": systems,
        "mapping_spread": {
            system["system"]: _mapping_spread(system) for system in systems
        },
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record()
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for system in record["systems"]:
        print(f"{system['system']}:")
        for arm in system["arms"]:
            counts = " -> ".join(
                f"k={row['block_size']}:{row['settings']}" for row in arm["rungs"]
            )
            print(f"  {arm['mapping']:10s} n={arm['measured_qubits']} {counts}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
