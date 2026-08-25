#!/usr/bin/env python3
"""R2b raw-pool fermion-mapping measurement experiment.

The selected physical generators are frozen by earlier raw-pool A-CASE runs.
This producer changes only their qubit representation across JW, parity,
parity+2q, BK, and BK+2q.  Every arm must pass the R2 mapping-invariant gate
before any word-weight, grouping, or device-card number is emitted.

The committed record uses one fixed-QWC policy: the established greedy on the
three affordable systems and a separately labelled scalable cover on H2O.  The
later R3 phase owns the ``mapping x k`` block-commuting grid; keeping that
interaction out of this file prevents this result from silently consuming its
successor phase.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.fermion_mapping import fermion_encoding
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import qwc_basis_cover, qwc_groups, shared_basis
from clifford_qc.measurement.cost import (
    DeviceCard,
    SettingResources,
    cost_schedule,
    inflate_shots_for_fidelity,
)
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.models import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    Generator,
    assert_mapping_invariants,
    determinant_excitations,
    identity_generator,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.elements import MatrixElementBank


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "mapping_axis.json"
REFERENCE = HERE / "reference_results" / "mapping_axis.json"
DEVICE_CARDS = HERE / "configs" / "device_cards"
SCHEMA = "clifford_qc.mapping_axis.v1"
STRUCTURAL_QR3_EXCLUSIONS = {
    "h4_converged": (
        "h4 and h4_converged are one physical H4 instance at two subspace "
        "budgets; h4_converged is retained for subspace-robustness and P5, "
        "not counted as another instance in structural QR3"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _selection_payload(record: dict) -> dict:
    """Return the scientific selection identity, excluding run provenance."""
    labels = record.get("labels") or record.get("basis_labels")
    if not labels:
        raise ValueError("JSON selection source declares no selected labels")
    payload = {"labels": list(labels)}
    for key in ("schema", "system", "ground_energy", "bank_provenance"):
        if key in record:
            payload[key] = record[key]
    return payload


def _selection_payload_sha256(record: dict) -> str:
    return _canonical_sha256(_selection_payload(record))


def _mv_payload(operator: MV) -> list[list[float | int]]:
    return [
        [int(code), float(complex(value).real), float(complex(value).imag)]
        for code, value in sorted(operator.terms.items())
    ]


def _mv_sha256(operator: MV) -> str:
    return _canonical_sha256(_mv_payload(operator))


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "clifford_qc.mapping_axis_config.v1":
        raise ValueError("unsupported mapping-axis config schema")
    arms = config.get("mapping_arms")
    if arms != ["jw", "parity", "parity+2q", "bk", "bk+2q"]:
        raise ValueError("mapping-axis arms or their predeclared order drifted")
    keys = [system.get("key") for system in config.get("systems", [])]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("mapping-axis system keys must be non-empty and unique")
    grouping = config.get("measurement", {}).get("grouping_protocol_by_system")
    if not isinstance(grouping, dict) or set(grouping) != set(keys):
        raise ValueError("every mapping-axis system must pin one grouping protocol")
    if any(
        value not in ("qwc_groups", "qwc_basis_cover")
        for value in grouping.values()
    ):
        raise ValueError("unsupported mapping-axis grouping protocol")
    return config


def load_device_cards(paths: Sequence[Path] | None = None) -> list[DeviceCard]:
    selected = sorted(DEVICE_CARDS.glob("*.json")) if paths is None else list(paths)
    cards = sorted(
        (DeviceCard.load(path) for path in selected), key=lambda card: card.name
    )
    if not cards or len({card.name for card in cards}) != len(cards):
        raise ValueError("device-card set must be non-empty and uniquely named")
    return cards


def _selection_row(spec: dict) -> dict | None:
    source_name = spec.get("selection_source")
    if not source_name:
        return None
    source = HERE.parent / source_name
    if not source.exists():
        raise FileNotFoundError(f"selection source is missing: {source_name}")
    if source.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matches = [
            row
            for row in rows
            if row.get("rung_name") == spec.get("selection_rung", spec["key"])
            and row.get("method") == spec.get("selection_method", "acase_exact")
        ]
        if len(matches) != 1:
            raise ValueError(f"selection source does not identify one row for {spec['key']}")
        row = matches[0]
        expected = spec.get("selection_row_sha256")
        if expected and _canonical_sha256(row) != expected:
            raise ValueError(f"selection row hash drifted for {spec['key']}")
        return row
    if source.suffix == ".json":
        row = json.loads(source.read_text(encoding="utf-8"))
        expected = spec.get("selection_payload_sha256")
        if not expected:
            raise ValueError(
                f"JSON selection source has no canonical payload hash for {spec['key']}"
            )
        if _selection_payload_sha256(row) != expected:
            raise ValueError(f"selection payload hash drifted for {spec['key']}")
        return row
    raise ValueError(f"unsupported selection source {source_name!r}")


def _build_model(spec: dict):
    if spec["builder"] == "fcidump":
        source = HERE.parent / spec["source"]
        provenance_path = HERE.parent / spec["provenance"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if _file_sha256(source) != provenance["fcidump_sha256"]:
            raise ValueError(f"FCIDUMP digest drifted for {spec['key']}")
        model = fcidump_model(source, name=provenance["name"])
        construction = {
            "kind": "committed_fcidump",
            "source": spec["source"],
            "source_sha256": model.metadata["source_sha256"],
            "provenance": spec["provenance"],
            "provenance_sha256": _file_sha256(provenance_path),
            "external_sector_energy": float(provenance["reference_energies"]["fci"]),
        }
        return model, construction
    if spec["builder"] == "hubbard":
        shape = tuple(int(value) for value in spec["shape"])
        t = float(spec.get("t", 1.0))
        interaction = float(spec.get("U", 4.0))
        model = hubbard(shape, t=t, U=interaction, periodic=bool(spec.get("periodic", False)))
        return model, {
            "kind": "native_hubbard",
            "shape": list(shape),
            "t": t,
            "U": interaction,
            "periodic": bool(spec.get("periodic", False)),
        }
    raise ValueError(f"unsupported mapping-axis builder {spec['builder']!r}")


def _raw_pool(model) -> list[Generator]:
    return determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2
    )


def _selected_generators(model, spec: dict) -> tuple[list[Generator], list[Generator]]:
    raw = _raw_pool(model)
    by_label = {generator.label: generator for generator in raw}
    if len(by_label) != len(raw):
        raise AssertionError("raw generator enumeration contains duplicate labels")
    labels = list(spec["selected_labels"])
    if not labels or labels[0] != "I" or len(labels) != len(set(labels)):
        raise ValueError("selected labels must be unique and begin with identity")
    missing = sorted(set(labels[1:]) - set(by_label))
    if missing:
        raise ValueError(f"selected labels are absent from the raw pool: {missing}")
    selected = [identity_generator(model.n), *(by_label[label] for label in labels[1:])]
    return raw, selected


def _generator_domain_sha256(generators: Sequence[Generator]) -> str:
    return _canonical_sha256(
        [
            {"label": generator.label, "operator": _mv_payload(generator.mv)}
            for generator in generators
        ]
    )


def _weight_distribution(code_occurrences: Iterable[int], n: int) -> dict:
    codes = [int(code) for code in code_occurrences]
    histogram = Counter(
        sum(1 for qubit in range(n) if ((code >> (2 * qubit)) & 3) != 0)
        for code in codes
    )
    total = sum(weight * count for weight, count in histogram.items())
    return {
        "term_occurrences": len(codes),
        "distinct_words": len(set(codes)),
        "mean_weight": (total / len(codes)) if codes else 0.0,
        "max_weight": max(histogram, default=0),
        "histogram": {str(weight): histogram[weight] for weight in sorted(histogram)},
    }


def _element_occurrences(bank: MatrixElementBank, basis_size: int) -> list[int]:
    codes: list[int] = []
    for column in range(basis_size):
        for row in range(column + 1):
            codes.extend(bank.overlap_operator(row, column).terms)
            codes.extend(bank.element_operator(row, column).terms)
    return codes


def _weight_ledger(
    hamiltonian: MV, generators: Sequence[Generator], bank: MatrixElementBank
) -> dict:
    return {
        "hamiltonian_multiset": _weight_distribution(hamiltonian.terms, hamiltonian.n),
        "generator_multiset": _weight_distribution(
            (code for generator in generators for code in generator.mv.terms),
            hamiltonian.n,
        ),
        "element_operator_multiset": _weight_distribution(
            _element_occurrences(bank, len(generators)), hamiltonian.n
        ),
        "deduplicated_word_universe": _weight_distribution(
            bank.word_set(), hamiltonian.n
        ),
    }


def _measurement_resources(groups: Sequence[Sequence[PauliWord]]) -> list[SettingResources]:
    settings = []
    for group in groups:
        basis = shared_basis(group)
        x_count = sum(letter == "X" for letter in basis.values())
        y_count = sum(letter == "Y" for letter in basis.values())
        n_1q = x_count + 2 * y_count
        d_1q = 2 if y_count else (1 if x_count else 0)
        settings.append(SettingResources(n_1q=n_1q, n_2q=0, d_1q=d_1q, d_2q=0))
    if not settings:
        raise AssertionError("a non-empty word universe produced no measurement setting")
    return settings


def _summary(values: Sequence[int]) -> dict:
    return {
        "sum": int(sum(values)),
        "mean": float(sum(values) / len(values)),
        "max": int(max(values)),
    }


def _measurement_ledger(
    groups, settings: Sequence[SettingResources], *, protocol: str
) -> dict:
    sizes = [len(group) for group in groups]
    return {
        "protocol": protocol,
        "settings": len(groups),
        "group_size": _summary(sizes),
        "resource_metrics": {
            "N_1q": _summary([setting.n_1q for setting in settings]),
            "N_2q": _summary([setting.n_2q for setting in settings]),
            "D_1q": _summary([setting.d_1q for setting in settings]),
            "D_2q": _summary([setting.d_2q for setting in settings]),
        },
        "validity": "PASS: every assigned word is QWC with its shared basis",
    }


def _device_costs(
    cards: Sequence[DeviceCard],
    settings: Sequence[SettingResources],
    *,
    shots_per_setting: int,
    n_qubits: int,
) -> dict:
    output = {}
    for card in cards:
        equal_effective = inflate_shots_for_fidelity(
            card, settings, shots_per_setting, n_qubits
        )
        output[card.name] = {
            "device_card_sha256": card.sha256,
            "fixed_uniform_raw_shots": cost_schedule(
                card,
                settings,
                shots_per_setting,
                n_qubits=n_qubits,
                evidence_tier="exact",
            ),
            "equal_effective_shots": cost_schedule(
                card,
                settings,
                equal_effective,
                n_qubits=n_qubits,
                evidence_tier="exact",
            ),
        }
    return output


def _functional_variance(reference: MV, coefficients: dict[int, float]) -> float:
    if not coefficients:
        return 0.0
    functional = MV(reference.n, coefficients)
    scale = float(2**reference.n)
    mean = scale * functional.trace_pairing(reference)
    second = scale * (functional * functional).trace_pairing(reference)
    if abs(complex(mean).imag) > 1e-10 or abs(complex(second).imag) > 1e-10:
        raise AssertionError("Hermitian Ritz functional acquired a complex moment")
    variance = float(complex(second).real - complex(mean).real**2)
    guard = max(abs(complex(second).real), complex(mean).real**2, 1.0)
    if variance < -1e-10 * guard:
        raise AssertionError(f"negative exact Ritz-functional variance {variance}")
    return max(0.0, variance)


def _accuracy_matched_cost(
    cards: Sequence[DeviceCard],
    settings: Sequence[SettingResources],
    groups: Sequence[Sequence[PauliWord]],
    bank: MatrixElementBank,
    result,
    exact_energy: float,
    target_millihartree: float,
) -> dict:
    bias = abs(float(result.ground_energy) - exact_energy) * 1e3
    base = {
        "epsilon_millihartree": target_millihartree,
        "exact_subspace_bias_millihartree": bias,
        "estimator": "single_assignment",
        "evidence_tier": "asymptotic",
        "claim_boundary": (
            "exact bias floor plus first-order Ritz-functional variance; not the "
            "R1 nonlinear exact-oracle shot search or a finite-sample certificate"
        ),
    }
    if bias >= target_millihartree:
        return {
            **base,
            "status": "bias_floor_exceeds_target",
            "linearized_variance_ha2_at_one_shot_per_setting": None,
            "effective_shots_per_setting": None,
            "device_costs": {},
        }
    functional = ritz_functional(
        bank, result.indices, result.ritz_vector(), result.ground_energy
    )
    remaining = set(functional.coefficients)
    variance = 0.0
    for group in groups:
        live = {
            word.code: float(functional.coefficients[word.code])
            for word in group
            if word.code in functional.coefficients
        }
        remaining.difference_update(live)
        variance += _functional_variance(bank.reference, live)
    if remaining:
        raise AssertionError(f"Ritz functional has {len(remaining)} ungrouped words")
    stochastic_budget = (target_millihartree**2 - bias**2) * 1e-6
    effective_shots = max(1, math.ceil(variance / stochastic_budget))
    device_costs = {}
    for card in cards:
        raw = inflate_shots_for_fidelity(card, settings, effective_shots, bank.n)
        device_costs[card.name] = cost_schedule(
            card,
            settings,
            raw,
            n_qubits=bank.n,
            evidence_tier="asymptotic",
            epsilon=target_millihartree * 1e-3,
        )
    return {
        **base,
        "status": "priced",
        "linearized_variance_ha2_at_one_shot_per_setting": variance,
        "effective_shots_per_setting": effective_shots,
        "device_costs": device_costs,
    }


def _cnot_network(program) -> dict:
    ready: dict[int, int] = {}
    depth = 0
    for operation in program.ops:
        if operation.name != "CX":
            raise AssertionError("fermion mapping witness is not CNOT-only")
        control, target = operation.qubits
        layer = 1 + max(ready.get(control, 0), ready.get(target, 0))
        ready[control] = ready[target] = layer
        depth = max(depth, layer)
    return {
        "logical_cnot_count": len(program.ops),
        "logical_cnot_depth": depth,
        "charged_per_preparation": False,
        "role": "representation-equivalence witness; encoded determinants are prepared directly",
    }


def _build_arm(
    name: str,
    model,
    reference: MV,
    selected: Sequence[Generator],
    exact_energy: float,
    cards: Sequence[DeviceCard],
    measurement: dict,
    grouping_protocol: str,
) -> dict:
    reduced = name.endswith("+2q")
    encoding = fermion_encoding(
        name,
        model.n,
        n_electrons=int(model.metadata["n_electrons"]) if reduced else None,
        sz=float(model.metadata["sz"]) if reduced else None,
        spin_ordering=model.metadata.get("spin_convention", "interleaved"),
    )
    restriction = encoding.restriction()
    invariant = assert_mapping_invariants(
        reference, model.hamiltonian, selected, restriction
    )
    transported = restriction.transport(
        hamiltonian=as_multivector(model.hamiltonian),
        reference=reference,
        generators=(generator.mv for generator in selected),
    )
    mapped = [
        Generator(generator.label, image)
        for generator, image in zip(selected, transported.generators)
    ]
    bank = MatrixElementBank(transported.reference, transported.hamiltonian, mapped)
    result = bank.solve()
    words = [
        PauliWord(transported.n, code)
        for code in sorted(bank.word_set())
        if code != 0
    ]
    if grouping_protocol == "qwc_groups":
        groups = qwc_groups(words)
    elif grouping_protocol == "qwc_basis_cover":
        groups = qwc_basis_cover(words)
    else:
        raise ValueError(f"unsupported QWC grouping protocol {grouping_protocol!r}")
    settings = _measurement_resources(groups)
    return {
        "mapping": name,
        "encoding": {
            "rows": list(encoding.rows),
            "fixed_qubits": list(encoding.fixed_qubits),
            "fixed_signs": list(encoding.signs),
            "source_qubits": model.n,
            "measured_qubits": transported.n,
            "spin_ordering": encoding.spin_ordering,
            "network": _cnot_network(encoding.program()),
        },
        "invariants": invariant.to_dict(),
        "basis": {
            "size": len(mapped),
            "labels": [generator.label for generator in mapped],
            "retained_rank": int(result.effective_rank),
            "condition_number": float(result.condition_number),
            "ground_energy": float(result.ground_energy),
            "error_millihartree": float(result.ground_energy - exact_energy) * 1e3,
        },
        "weights": _weight_ledger(transported.hamiltonian, mapped, bank),
        "measurement": _measurement_ledger(
            groups, settings, protocol=grouping_protocol
        ),
        "device_costs": _device_costs(
            cards,
            settings,
            shots_per_setting=int(measurement["uniform_raw_shots_per_setting"]),
            n_qubits=transported.n,
        ),
        "accuracy_matched_cost": _accuracy_matched_cost(
            cards,
            settings,
            groups,
            bank,
            result,
            exact_energy,
            float(measurement["accuracy_target_millihartree"]),
        ),
    }


def build_system_record(
    spec: dict,
    *,
    arms: Sequence[str],
    cards: Sequence[DeviceCard],
    measurement: dict,
) -> dict:
    grouping_protocols = measurement.get("grouping_protocol_by_system", {})
    grouping_protocol = grouping_protocols.get(spec["key"])
    if grouping_protocol not in ("qwc_groups", "qwc_basis_cover"):
        raise ValueError(f"missing grouping protocol for {spec['key']}")
    model, construction = _build_model(spec)
    if not spec.get("selection_source") and not spec.get("selection_rule"):
        raise ValueError(
            f"{spec['key']} declares neither a selection source nor a selection rule"
        )
    selection = _selection_row(spec)
    if selection is not None:
        recorded_labels = selection.get("labels") or selection.get("basis_labels")
        if recorded_labels is not None and list(recorded_labels) != list(spec["selected_labels"]):
            raise ValueError(f"frozen selection labels drifted for {spec['key']}")
    raw, selected = _selected_generators(model, spec)
    reference = ExactMVBackend().state(model.reference, ())
    source_bank = MatrixElementBank(reference, model.hamiltonian, selected)
    source_result = source_bank.solve()
    expected = float(spec["expected_selected_energy"])
    tolerance = float(spec["selected_energy_tolerance"])
    if abs(float(source_result.ground_energy) - expected) > tolerance:
        raise ValueError(
            f"frozen selected energy drifted for {spec['key']}: "
            f"{source_result.ground_energy} != {expected}"
        )
    backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    # ``method='dense'`` rather than the default ``'auto'``. Every energy this
    # record reports is a difference against this reference in millihartree --
    # a residue of order 1 against energies of order 1e4 -- so ARPACK's
    # process-dependent last bit (a measured 5e-14 Ha spread) lands as a
    # ~1e-10 mHa shift on the whole record at once. ``run_protocol_cost.py``
    # already takes the dense path for this reason; this closes the same hole
    # here, where the record is being rebuilt anyway.
    exact_energy = float(
        backend.ground_state(model.hamiltonian, k=1, method="dense")[0][0]
    )
    external = construction.get("external_sector_energy")
    if external is not None and abs(exact_energy - float(external)) > 5e-10:
        raise ValueError(f"sector solve disagrees with external reference for {spec['key']}")
    rows = [
        _build_arm(
            name,
            model,
            reference,
            selected,
            exact_energy,
            cards,
            measurement,
            grouping_protocol,
        )
        for name in arms
    ]
    return {
        "system": spec["key"],
        "label": spec["label"],
        "construction": construction,
        "n_qubits": model.n,
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "spin_ordering": model.metadata.get("spin_convention", "interleaved"),
        "grouping_protocol": grouping_protocol,
        "hamiltonian_terms": len(model.hamiltonian.terms),
        "hamiltonian_sha256": _mv_sha256(as_multivector(model.hamiltonian)),
        "reference_sha256": _mv_sha256(reference),
        "exact_sector_energy": exact_energy,
        "raw_pool": {
            "size": len(raw),
            "sha256": _generator_domain_sha256(raw),
            "max_excitation_rank": 2,
        },
        "selected_domain": {
            "size": len(selected),
            "labels": [generator.label for generator in selected],
            "sha256": _generator_domain_sha256(selected),
            "source": spec.get("selection_source"),
            "source_row_sha256": spec.get("selection_row_sha256"),
            "source_payload_sha256": spec.get("selection_payload_sha256"),
            # A system whose labels come from a frozen record cites the record;
            # one whose labels come from running the growth rule to its own
            # stopping threshold cites the rule, and a test re-derives it. Both
            # are freezes -- what must never happen is a selection that cites
            # neither, which is why the builder rejects that below.
            "selection_rule": spec.get("selection_rule"),
        },
        "arms": rows,
    }


def _metric_value(arm: dict, metric: str) -> float:
    if metric == "qwc_settings":
        return float(arm["measurement"]["settings"])
    if metric == "mean_word_weight":
        return float(arm["weights"]["deduplicated_word_universe"]["mean_weight"])
    raise ValueError(f"unknown QR3 metric {metric!r}")


def _spread_summary(systems: Sequence[dict], metric: str) -> dict:
    within = {}
    for system in systems:
        values = [_metric_value(arm, metric) for arm in system["arms"]]
        minimum = min(values)
        maximum = max(values)
        within[system["system"]] = {
            "minimum": minimum,
            "maximum": maximum,
            "mapping_spread_factor": maximum / minimum,
        }
    by_mapping = {}
    for index, mapping in enumerate(systems[0]["arms"]):
        values = [
            _metric_value(system["arms"][index], metric)
            for system in systems
        ]
        minimum = min(values)
        maximum = max(values)
        by_mapping[mapping["mapping"]] = {
            "minimum": minimum,
            "maximum": maximum,
            "instance_spread_factor": maximum / minimum,
        }
    max_mapping = max(item["mapping_spread_factor"] for item in within.values())
    min_instance = min(item["instance_spread_factor"] for item in by_mapping.values())
    return {
        "systems_included": [system["system"] for system in systems],
        "within_system": within,
        "across_instances": by_mapping,
        "max_mapping_spread_factor": max_mapping,
        "min_instance_spread_factor": min_instance,
        "margin_factor": min_instance / max_mapping,
        "verdict": (
            "mapping_spread_smaller_than_instance_spread"
            if max_mapping < min_instance
            else "mapping_spread_not_smaller_than_instance_spread"
        ),
    }



def _qr3_summary(systems: Sequence[dict], cards: Sequence[DeviceCard]) -> dict:
    # A second bank on the same Hamiltonian is a subspace-robustness row, not a
    # second physical instance. Keep it available to the accuracy eligibility
    # ledger below, where the budget-8 bank fails the bias gate and the converged
    # bank is the H4 representative, but do not double-count H4 structurally.
    structural_systems = [
        system
        for system in systems
        if system["system"] not in STRUCTURAL_QR3_EXCLUSIONS
    ]
    matched_qwc = [
        system
        for system in structural_systems
        if system["grouping_protocol"] == "qwc_groups"
    ]
    structural = {
        "qwc_settings_matched_greedy": _spread_summary(
            matched_qwc, "qwc_settings"
        ),
        "mean_word_weight": _spread_summary(
            structural_systems, "mean_word_weight"
        ),
    }
    eligible = [
        system["system"]
        for system in systems
        if all(
            arm["accuracy_matched_cost"]["status"] == "priced"
            for arm in system["arms"]
        )
    ]
    return {
        "structural": structural,
        "structural_verdict": (
            "mapping_spread_smaller_than_instance_spread_on_both_independent_metrics"
            if all(
                summary["verdict"]
                == "mapping_spread_smaller_than_instance_spread"
                for summary in structural.values()
            )
            else "mapping_spread_not_smaller_on_every_independent_metric"
        ),
        "subspace_robustness_exclusion": {
            "systems": list(STRUCTURAL_QR3_EXCLUSIONS),
            "reason": " ".join(STRUCTURAL_QR3_EXCLUSIONS.values()),
        },
        "qwc_exclusion": {
            "systems": [
                system["system"]
                for system in structural_systems
                if system["grouping_protocol"] != "qwc_groups"
            ],
            "reason": (
                "scalable-cover counts are constructive upper bounds from a different "
                "heuristic and are excluded from the cross-instance QWC verdict"
            ),
        },
        "device_card_projections": {
            "cards": [card.name for card in cards],
            "independent_evidence": False,
            "reason": (
                "QWC basis rotations have N_2q=D_2q=0, so fixed-shot card times "
                "are derived projections of setting count and one-qubit rotations, "
                "not independent confirmations"
            ),
        },
        "accuracy_matched": {
            "eligible_systems": eligible,
            "verdict": (
                "insufficient_eligible_instances"
                if len(eligible) < 2
                else "eligible_for_cross_instance_comparison"
            ),
        },
        "claim_boundary": (
            "like-for-like ratio comparison on algorithm-independent word weight and "
            "matched largest-degree-greedy QWC physical instances; alternate "
            "subspace budgets, H2O scalable-cover counts, and device-card projections "
            "are descriptive only; accuracy-matched QR3 abstains unless at least two "
            "complete physical instances clear the exact bias floor"
        ),
    }

def build_record(
    *,
    config: dict | None = None,
    system_keys: Sequence[str] | None = None,
    device_cards: Sequence[DeviceCard] | None = None,
) -> dict:
    config = load_config() if config is None else config
    cards = load_device_cards() if device_cards is None else list(device_cards)
    wanted = set(system_keys or (system["key"] for system in config["systems"]))
    specs = [system for system in config["systems"] if system["key"] in wanted]
    if {system["key"] for system in specs} != wanted:
        raise ValueError(f"unknown mapping-axis systems: {sorted(wanted - {s['key'] for s in specs})}")
    systems = [
        build_system_record(
            spec,
            arms=config["mapping_arms"],
            cards=cards,
            measurement=config["measurement"],
        )
        for spec in specs
    ]
    complete = len(specs) == len(config["systems"])
    record = {
        "schema": SCHEMA,
        "phase": "R2b_raw_pool_mapping_axis",
        "scope": "complete" if complete else "partial",
        "config_sha256": _canonical_sha256(config),
        "mapping_arms": list(config["mapping_arms"]),
        "pool_contract": config["pool"],
        "measurement_contract": {
            **config["measurement"],
            "grouping_claim": (
                "H4, BeH2, and Hubbard use the established largest-degree greedy; "
                "H2O uses a full-basis-seeded first-fit scalable cover and is excluded "
                "from the cross-instance QWC verdict; neither protocol claims a "
                "minimum coloring"
            ),
            "deferred_to_r3": [
                "G(k) across protocol rungs",
                "coverage across protocol rungs",
            ],
            "mapping_network_charge": (
                "not charged per shot: each encoded reference determinant is prepared "
                "directly; the CNOT network is retained as an equivalence witness"
            ),
        },
        "device_cards": [
            {"sha256": card.sha256, **card.to_dict()} for card in cards
        ],
        "systems": systems,
        "qr2": "PASS: every emitted arm passed the mapping-invariance gate",
        "qr3": _qr3_summary(systems, cards) if complete else {
            "verdict": "not_evaluated_on_partial_record"
        },
        "claim_boundary": (
            "exact representation and resource counts plus an asymptotic "
            "single-assignment shot model; no nonlinear finite-shot mapping search, "
            "hardware calibration, "
            "optimal-coloring claim, H2O cross-heuristic QWC verdict, independent "
            "device-card corroboration, or mapping-by-protocol interaction"
        ),
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        default="",
        help="comma-separated subset; omitted means the complete committed scope",
    )
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    systems = tuple(value for value in args.systems.split(",") if value) or None
    record = build_record(system_keys=systems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for system in record["systems"]:
        summary = ", ".join(
            f"{arm['mapping']}:{arm['measurement']['settings']}"
            for arm in system["arms"]
        )
        print(f"{system['system']}: {summary}")
    print(args.out)


if __name__ == "__main__":
    main()
