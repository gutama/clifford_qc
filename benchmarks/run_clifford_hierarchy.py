"""Reproduce the dyadic block-commuting measurement hierarchy.

Only the measurement compatibility rule varies.  For each dyadic block size
``k`` we greedily group the words of a retained DA-CASE bank so that grouped
words commute inside every contiguous ``k``-qubit block, synthesize an exact
block-local Clifford diagonalizer with stim, and count logical CX gates and
two-qubit depth.

Two eight-qubit systems are run at the same block sizes so the trade is
measured on more than one Hamiltonian:

``h4``
    The retained determinant-resolution bank of the matched H4 contract,
    reconstructed from the labels stored in ``matched_h4.json`` so that the
    manuscript's ledger and this experiment share one bank by construction.
``beh2``
    An independent DA-CASE run on the frozen BeH2 CAS(4e,4o) FCIDUMP, grown
    under the same budget and candidate rule.  BeH2 stops on its own lowering
    threshold before the budget is spent, so its bank is smaller; that is
    reported rather than padded.

    python benchmarks/run_clifford_hierarchy.py --system h4
    python benchmarks/run_clifford_hierarchy.py --system beh2
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.bridges.stim_bridge import CliffordMap
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.measurement.compiled import CompiledSetting
from clifford_qc.measurement.cost import (
    DeviceCard,
    SettingResources,
    break_even_surface,
    cost_schedule,
    estimator_information,
    inflate_shots_for_fidelity,
    setting_fidelity,
)
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.models import fcidump_model
from clifford_qc.pauli_kernel import pauli_lane_mask
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    determinant_excitations,
    identity_generator,
    occupied_spin_orbitals,
    run_acase,
)
from clifford_qc.subspace.elements import MatrixElementBank

try:
    import stim
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise SystemExit(
        "run_clifford_hierarchy.py requires the stim extra: "
        "pip install -e '.[stim]'"
    ) from exc


HERE = Path(__file__).resolve().parent
MATCHED = HERE / "reference_results" / "matched_h4.json"
H4_FCIDUMP = HERE / "data" / "h4_sto3g_r0.9.FCIDUMP"
BEH2_FCIDUMP = HERE / "data" / "beh2_sto3g_r1.3264.FCIDUMP"
BEH2_PROVENANCE = HERE / "data" / "beh2_sto3g_r1.3264.provenance.json"
DEVICE_CARDS = HERE / "configs" / "device_cards"
SHOTS_PER_SETTING = 8_000
BLOCK_SIZES = (1, 2, 4, 8)
BUDGET = 8
ACCURACY_TARGET_MILLIHARTREE = 1.6
BREAK_EVEN_T2Q_RATIOS = (0.0, 0.1, 0.5, 1.0, 2.0)
BREAK_EVEN_EPS_2Q = (0.0, 0.005, 0.01, 0.02, 0.05)


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized uint64 population count for the declared NumPy >=1.23."""
    work = np.asarray(values, dtype=np.uint64).copy()
    work -= (work >> np.uint64(1)) & np.uint64(0x5555555555555555)
    work = (
        (work & np.uint64(0x3333333333333333))
        + ((work >> np.uint64(2)) & np.uint64(0x3333333333333333))
    )
    work = (work + (work >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    work *= np.uint64(0x0101010101010101)
    return (work >> np.uint64(56)).astype(np.int16)


def _local_code(code: int, start: int, size: int) -> int:
    out = 0
    for q in range(size):
        out |= ((code >> (2 * (start + q))) & 3) << (2 * q)
    return out


def _local_xz(code: int, size: int) -> int:
    x = z = 0
    for q in range(size):
        letter = (code >> (2 * q)) & 3
        if letter in (1, 2):
            x |= 1 << q
        if letter in (2, 3):
            z |= 1 << q
    return x | (z << size)


def _independent_codes(codes: list[int], size: int) -> list[int]:
    """Keep an independent GF(2) basis without changing its Pauli elements."""
    pivots: dict[int, int] = {}
    kept: list[int] = []
    for code in sorted(set(codes)):
        if code == 0:
            continue
        reduced = _local_xz(code, size)
        for pivot in sorted(pivots, reverse=True):
            if (reduced >> pivot) & 1:
                reduced ^= pivots[pivot]
        if reduced:
            pivots[reduced.bit_length() - 1] = reduced
            kept.append(code)
    return kept


def _stim_label(code: int, size: int) -> str:
    return "".join("IXYZ"[(code >> (2 * q)) & 3] for q in range(size))


def _circuit_stats(circuit) -> tuple[Counter, int, int, int]:
    """Gate counts, the CX-only depth, and a schedulable one-/two-qubit pair.

    ``cx_depth`` is the two-qubit critical path on its own, which is the
    ``logical_cx_depth_*`` column the frozen v2 ledger reports.

    The cost model instead charges ``t_1q * D_1q + t_2q * D_2q`` as wall-clock
    time, so that pair has to come from one schedule: tracking the two gate
    kinds on independent clocks drops every dependency that runs through a CX
    and undercounts exactly the deep, 1q/2q-interleaved rungs.  Here one shared
    per-qubit clock schedules gates as-soon-as-possible into type-homogeneous
    layers, so ``D_1q + D_2q`` is a real critical path and the priced time is a
    duration some schedule attains.
    """
    counts: Counter = Counter()
    last_cx_layer: dict[int, int] = {}
    cx_depth = 0
    ready: dict[int, int] = {}
    layer_kind: dict[int, str] = {}

    def schedule(kind: str, qubits: tuple[int, ...]) -> None:
        layer = 1 + max((ready.get(qubit, 0) for qubit in qubits), default=0)
        while layer_kind.get(layer, kind) != kind:
            layer += 1
        layer_kind[layer] = kind
        for qubit in qubits:
            ready[qubit] = layer

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        if name == "CX":
            if len(targets) % 2:
                raise AssertionError("CX instruction has an odd target count")
            for a, b in zip(targets[0::2], targets[1::2]):
                qa, qb = a.value, b.value
                counts["CX"] += 1
                layer = 1 + max(
                    last_cx_layer.get(qa, 0), last_cx_layer.get(qb, 0)
                )
                last_cx_layer[qa] = last_cx_layer[qb] = layer
                cx_depth = max(cx_depth, layer)
                schedule("2q", (qa, qb))
        elif name in {"H", "S"}:
            counts[name] += len(targets)
            for target in targets:
                schedule("1q", (target.value,))
        else:
            raise AssertionError(f"unexpected tableau-elimination gate {name}")
    depth_1q = sum(1 for kind in layer_kind.values() if kind == "1q")
    depth_2q = sum(1 for kind in layer_kind.values() if kind == "2q")
    return counts, cx_depth, depth_1q, depth_2q


def _x_masks(diagonalizer, size: int, codes: np.ndarray) -> np.ndarray:
    """X mask after Clifford conjugation, for the banked local codes only.

    Tabulating all ``4**size`` codes costs 65536 entries at ``size=8`` and is
    rebuilt per setting, while only ``len(codes)`` of them are ever read.  The
    per-qubit ``X``/``Z`` image masks XOR-reduce directly over the code array
    instead, which drops the ``4**size`` factor.
    """
    codes = np.asarray(codes, dtype=np.int64)
    masks = np.zeros(len(codes), dtype=np.int64)
    for qubit in range(size):
        x_bits, _ = diagonalizer.x_output(qubit).to_numpy()
        x_from_x = sum(int(bit) << q for q, bit in enumerate(x_bits))
        x_bits, _ = diagonalizer.z_output(qubit).to_numpy()
        x_from_z = sum(int(bit) << q for q, bit in enumerate(x_bits))
        # Letter Y and Z carry Z; letters X and Y carry X (low bit xor high).
        z_bit = (codes >> (2 * qubit + 1)) & 1
        x_bit = ((codes >> (2 * qubit)) & 1) ^ z_bit
        masks ^= x_bit * x_from_x
        masks ^= z_bit * x_from_z
    return masks


def _diagonalizer(local_members: list[int], size: int):
    """Return the one canonical local Clifford used for costing and sampling."""
    basis = _independent_codes(local_members, size)
    if not basis:
        return stim.Tableau(size)
    tableau = stim.Tableau.from_stabilizers(
        [stim.PauliString(_stim_label(code, size)) for code in basis],
        allow_redundant=False,
        allow_underconstrained=True,
    )
    return tableau.inverse()


def _synthesize(n: int, codes: list[int], groups: list[list[int]],
                block_size: int) -> tuple[dict, list[SettingResources], np.ndarray, list[int]]:
    total: Counter = Counter()
    cx_per_setting: list[int] = []
    cx_depth_per_setting: list[int] = []
    depth_1q_per_setting: list[int] = []
    depth_2q_per_setting: list[int] = []
    setting_resources: list[SettingResources] = []
    compatibility = np.zeros((len(groups), len(codes)), dtype=bool)
    assignment = [-1] * len(codes)
    local_codes = {
        (start, min(block_size, n - start)): np.asarray(
            [_local_code(code, start, min(block_size, n - start)) for code in codes],
            dtype=np.uint16,
        )
        for start in range(0, n, block_size)
    }
    checked = 0
    for setting_index, members in enumerate(groups):
        setting_counts: Counter = Counter()
        block_cx_depths: list[int] = []
        block_depths_1q: list[int] = []
        block_depths_2q: list[int] = []
        setting_reads = np.ones(len(codes), dtype=bool)
        for member in members:
            if assignment[member] != -1:
                raise AssertionError("a word was assigned to more than one group")
            assignment[member] = setting_index
        for start in range(0, n, block_size):
            size = min(block_size, n - start)
            local = [_local_code(codes[i], start, size) for i in members]
            diagonalizer = _diagonalizer(local, size)

            # Strong circuit invariant: every member, not only the basis,
            # must become computational-basis diagonal.
            for code in set(local):
                transformed = diagonalizer(
                    stim.PauliString(_stim_label(code, size)))
                x_bits, _ = transformed.to_numpy()
                if bool(x_bits.any()):
                    raise AssertionError(
                        f"k={block_size} block={start}: {_stim_label(code, size)} "
                        f"did not map to Z-only ({transformed})")
                checked += 1

            counts, cx_depth, depth_1q, depth_2q = _circuit_stats(
                diagonalizer.to_circuit("elimination"))
            setting_counts.update(counts)
            block_cx_depths.append(cx_depth)
            block_depths_1q.append(depth_1q)
            block_depths_2q.append(depth_2q)
            setting_reads &= (
                _x_masks(diagonalizer, size, local_codes[(start, size)]) == 0
            )

        total.update(setting_counts)
        cx_per_setting.append(setting_counts["CX"])
        # Blocks occupy disjoint qubits, so a setting runs them in parallel.
        cx_depth_per_setting.append(max(block_cx_depths, default=0))
        depth_1q = max(block_depths_1q, default=0)
        depth_2q = max(block_depths_2q, default=0)
        depth_1q_per_setting.append(depth_1q)
        depth_2q_per_setting.append(depth_2q)
        setting_resources.append(SettingResources(
            n_1q=setting_counts["H"] + setting_counts["S"],
            n_2q=setting_counts["CX"],
            d_1q=depth_1q,
            d_2q=depth_2q,
        ))
        if not bool(setting_reads[members].all()):
            raise AssertionError("an assigned word is not read by its setting")
        compatibility[setting_index] = setting_reads

    if any(index < 0 for index in assignment):
        raise AssertionError("the grouping did not assign every word")

    settings = len(groups)
    cx = sum(cx_per_setting)
    depth_sum = sum(cx_depth_per_setting)
    row = {
        "block_size": block_size,
        "settings": settings,
        "word_samples_per_preparation": len(codes) / settings,
        "logical_cx_per_sweep": cx,
        "mean_logical_cx_per_setting": cx / settings,
        "max_logical_cx_per_setting": max(cx_per_setting),
        "logical_cx_depth_sum": depth_sum,
        "mean_logical_cx_depth": depth_sum / settings,
        "max_logical_cx_depth": max(cx_depth_per_setting),
        "state_preparations_at_uniform_shots": settings * SHOTS_PER_SETTING,
        "logical_cx_applications_at_uniform_shots": cx * SHOTS_PER_SETTING,
        "gate_counts_per_sweep": dict(sorted(total.items())),
        "z_only_restrictions_checked": checked,
    }
    return row, setting_resources, compatibility, assignment


def _compiled_settings(n: int, codes: list[int], groups: list[list[int]],
                       block_size: int) -> tuple[CompiledSetting, ...]:
    """Compile the hierarchy grouping into signed joint-readout settings.

    The structural ledger only needed a Boolean compatibility matrix.  The
    exact-tier shot search needs the stronger object behind that matrix: the
    actual Clifford applied before measurement and the signed Z parity of
    every bank word it reads.  This routine independently rechecks those
    readouts against the global Stim tableau so a block-offset or phase error
    cannot enter the nonlinear estimator silently.
    """
    compiled = []
    for setting_index, members in enumerate(groups):
        circuit = stim.Circuit()
        readouts = {code: [1, []] for code in codes}
        readable = {code: True for code in codes}
        for start in range(0, n, block_size):
            size = min(block_size, n - start)
            local_members = [_local_code(codes[index], start, size) for index in members]
            diagonalizer = _diagonalizer(local_members, size)

            for instruction in diagonalizer.to_circuit("elimination"):
                targets = [start + target.value for target in instruction.targets_copy()]
                circuit.append(instruction.name, targets)

            for code in codes:
                local = _local_code(code, start, size)
                transformed = diagonalizer(stim.PauliString(_stim_label(local, size)))
                x_bits, z_bits = transformed.to_numpy()
                if bool(x_bits.any()):
                    readable[code] = False
                    continue
                phase = complex(transformed.sign)
                if abs(phase.imag) > 1e-12 or round(phase.real) not in (-1, 1):
                    raise AssertionError("Hermitian Pauli acquired a non-real readout phase")
                readouts[code][0] *= int(round(phase.real))
                readouts[code][1].extend(
                    start + qubit for qubit, present in enumerate(z_bits) if present
                )

        # Force the tableau to retain idle trailing qubits without changing it.
        circuit.append("I", range(n))
        clifford = CliffordMap(n, circuit.to_tableau())
        explicit = {
            code: (int(readouts[code][0]), tuple(readouts[code][1]))
            for code in codes if readable[code]
        }
        assigned_codes = tuple(codes[index] for index in members)
        if any(code not in explicit for code in assigned_codes):
            raise AssertionError("an assigned word lacks a compiled readout")

        for code, expected in explicit.items():
            phase, image = clifford.conjugate(PauliWord(n, code))
            if any(image.letter(qubit) not in ("I", "Z") for qubit in range(n)):
                raise AssertionError("global compiled readout is not Z-only")
            actual_sign = int(round(complex(phase).real))
            actual_positions = tuple(image.support())
            if abs(complex(phase).imag) > 1e-12 or (actual_sign, actual_positions) != expected:
                raise AssertionError("block readout disagrees with the global Clifford tableau")

        compiled.append(CompiledSetting(
            key=("dyadic-block", block_size, setting_index),
            clifford=clifford,
            assigned_word_codes=assigned_codes,
            readouts=explicit,
        ))
    return tuple(compiled)


def _h4_bank() -> dict:
    """The retained determinant bank of the matched H4 contract."""
    matched = json.loads(MATCHED.read_text(encoding="utf-8"))
    selected = next(
        row for row in matched["rows"] if row["arm"] == "A-CASE (determinant)")
    model = fcidump_model(H4_FCIDUMP)
    rho = ExactMVBackend().state(model.reference, ())
    determinants = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    by_label = {generator.label: generator for generator in determinants}
    family = [identity_generator(model.n)] + [
        by_label[label] for label in selected["labels"] if label != "I"
    ]
    bank = MatrixElementBank(rho, model.hamiltonian, family)
    result = bank.solve()
    codes = sorted(bank.word_set(range(len(family))))
    if len(codes) != selected["final_words"]:
        raise AssertionError(
            f"matched bank drift: {len(codes)} != {selected['final_words']}")
    return {
        "system": "h4",
        "label": "H$_4$",
        "source": str(MATCHED.relative_to(HERE.parent)),
        "bank_provenance": (
            "retained determinant-resolution bank of the matched H4 contract, "
            f"arm {selected['arm']!r}"),
        "n_qubits": model.n,
        "basis_size": len(family),
        "basis_labels": list(selected["labels"]),
        "ground_energy": selected["ground_energy"],
        "_exact_ground_energy": (
            selected["ground_energy"] - selected["error_millihartree"] * 1e-3
        ),
        "error_millihartree": selected["error_millihartree"],
        "condition_number": selected["condition_number"],
        "codes": codes,
        "committed_qwc_groups": selected["final_qwc_groups"],
        "_matrix_bank": bank,
        "_result": result,
    }


def _beh2_bank() -> dict:
    """An independent DA-CASE run on the frozen BeH2 CAS(4e,4o) FCIDUMP."""
    provenance = json.loads(BEH2_PROVENANCE.read_text(encoding="utf-8"))
    model = fcidump_model(BEH2_FCIDUMP, name=provenance["name"])
    if model.metadata["source_sha256"] != provenance["fcidump_sha256"]:
        raise ValueError("FCIDUMP digest disagrees with its provenance record")
    sector = SectorStatevectorBackend(
        model.n, model.metadata["n_electrons"], model.metadata["sz"])
    exact_energy = float(
        sector.ground_state(model.hamiltonian, k=4, method="dense")[0][0])
    if abs(exact_energy - provenance["reference_energies"]["fci"]) > 1e-10:
        raise ValueError("sector-exact energy disagrees with external provenance")

    rho = ExactMVBackend().state(model.reference, ())
    determinants = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    run = run_acase(rho, model.hamiltonian, determinants, max_size=BUDGET,
                    exact_ground_energy=exact_energy)
    result = run.result
    energy = float(result.energies[0])
    codes = sorted(run.bank.word_set(result.indices))
    return {
        "system": "beh2",
        "label": "BeH$_2$",
        "source": str(BEH2_FCIDUMP.relative_to(HERE.parent)),
        "bank_provenance": (
            f"{provenance['name']}, PySCF {provenance['generator']['version']}; "
            f"DA-CASE grown at budget {BUDGET}, stopped by "
            f"{run.stopped_reason!r}"),
        "n_qubits": model.n,
        "basis_size": len(result.basis_labels),
        "basis_labels": list(result.basis_labels),
        "ground_energy": energy,
        "_exact_ground_energy": exact_energy,
        "error_millihartree": (energy - exact_energy) * 1e3,
        "condition_number": result.condition_number,
        "codes": codes,
        "committed_qwc_groups": None,
        "_matrix_bank": run.bank,
        "_result": result,
    }


SYSTEMS = {"h4": _h4_bank, "beh2": _beh2_bank}

V2_ROW_FIELDS = (
    "block_size", "settings", "word_samples_per_preparation",
    "logical_cx_per_sweep", "mean_logical_cx_per_setting",
    "max_logical_cx_per_setting", "logical_cx_depth_sum",
    "mean_logical_cx_depth", "max_logical_cx_depth",
    "state_preparations_at_uniform_shots",
    "logical_cx_applications_at_uniform_shots", "gate_counts_per_sweep",
    "z_only_restrictions_checked", "cx_to_preparation_cost_ratio_vs_qwc",
)
V2_TOP_FIELDS = (
    "word_universe", "shots_per_setting", "grouping", "synthesis",
    "logical_model", "system", "label", "source", "bank_provenance",
    "n_qubits", "basis_size", "basis_labels", "ground_energy",
    "error_millihartree", "condition_number", "qwc_to_full",
    "most_permissive_level", "invariant",
)


def load_device_cards(paths: list[Path] | None = None) -> list[DeviceCard]:
    """Load a deterministic, uniquely named set of declared device scenarios.

    Cards are ordered by name, so the record's card order — which the checker
    gates on — never depends on ``--device-card`` argument order.
    """
    selected = sorted(DEVICE_CARDS.glob("*.json")) if paths is None else paths
    if not selected:
        raise ValueError("at least one device card is required")
    cards = sorted(
        (DeviceCard.load(path) for path in selected), key=lambda card: card.name
    )
    names = [card.name for card in cards]
    if len(names) != len(set(names)):
        raise ValueError("device-card names must be unique")
    return cards


def legacy_v2_projection(record: dict) -> dict:
    """Project a v3 record onto the exact field contract of the frozen v2 row."""
    if record.get("schema") != "clifford_qc.clifford_measurement_hierarchy.v3":
        raise ValueError("legacy projection requires a hierarchy v3 record")
    projected = {
        "schema": "clifford_qc.clifford_measurement_hierarchy.v2",
        **{field: record[field] for field in V2_TOP_FIELDS},
        "rows": [
            {field: row[field] for field in V2_ROW_FIELDS}
            for row in record["rows"]
        ],
    }
    # The frozen v2 records serialized an integral condition number as a JSON
    # integer.  That is a legacy serialization detail of the compat contract,
    # so it is restored here rather than typed into the live v3 record, where
    # it would make the field's JSON type depend on the bank's value.
    condition = projected["condition_number"]
    if isinstance(condition, float) and condition.is_integer():
        projected["condition_number"] = int(condition)
    return projected


def _resource_metrics(settings: list[SettingResources]) -> dict:
    def summary(field: str) -> dict[str, float | int]:
        values = [getattr(setting, field) for setting in settings]
        return {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "N_1q": summary("n_1q"),
        "N_2q": summary("n_2q"),
        "D_1q": summary("d_1q"),
        "D_2q": summary("d_2q"),
    }


def _device_costs(
    cards: list[DeviceCard],
    settings: list[SettingResources],
    compatibility: np.ndarray,
    assignment: list[int],
    n_qubits: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for card in cards:
        fidelities = [setting_fidelity(card, setting, n_qubits) for setting in settings]
        equal_effective_raw = inflate_shots_for_fidelity(
            card, settings, SHOTS_PER_SETTING, n_qubits
        )
        results[card.name] = {
            "device_card_sha256": card.sha256,
            "fixed_uniform_raw_shots": cost_schedule(
                card, settings, SHOTS_PER_SETTING, n_qubits=n_qubits,
                evidence_tier="exact",
            ),
            "equal_effective_shots": cost_schedule(
                card, settings, equal_effective_raw, n_qubits=n_qubits,
                evidence_tier="exact",
            ),
            "estimator_information_at_uniform_raw_shots": {
                estimator: estimator_information(
                    compatibility, assignment, SHOTS_PER_SETTING,
                    estimator=estimator, fidelities=fidelities,
                )
                for estimator in ("single_assignment", "pooled")
            },
        }
    return results


def _pauli_quadratic_variance(
    n: int,
    rho,
    codes: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Exact per-shot variance of a commuting real Pauli functional.

    The cross terms below are the packed product of ``pauli_kernel``, applied
    to whole arrays at once: ``lo`` is ``pauli_lane_mask`` and the exponent is
    ``_word_mul_unchecked``'s, with ``_PHASE4`` restricted to the real part
    that a commuting pair always has.  ``tests/test_clifford_hierarchy_cost.py``
    pins this array form against ``word_mul`` so the two cannot drift.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 32:
        raise ValueError("n must be an integer in [0, 32] for uint64 Pauli codes")
    if len(codes) == 0:
        return 0.0
    codes = np.asarray(codes, dtype=np.uint64)
    coefficients = np.asarray(coefficients, dtype=float)
    scale = float(2**n)
    means = {
        int(code): scale * complex(value).real
        for code, value in rho.terms.items()
    }

    lo = np.uint64(pauli_lane_mask(n))
    z = (codes >> np.uint64(1)) & lo
    x = (codes & lo) ^ z
    xz_count = _popcount_u64(x & z)
    second = float(np.dot(coefficients, coefficients))
    phase_lookup = np.asarray([1.0, 0.0, -1.0, 0.0])
    for i in range(len(codes) - 1):
        other = slice(i + 1, None)
        product = codes[i] ^ codes[other]
        z_product = (product >> np.uint64(1)) & lo
        x_product = (product & lo) ^ z_product
        exponent = (
            int(xz_count[i])
            + xz_count[other]
            - _popcount_u64(x_product & z_product)
            + 2 * _popcount_u64(z[i] & x[other])
        ) % 4
        if bool((exponent % 2).any()):
            raise AssertionError("one synthesized setting contains anticommuting words")
        product_means = phase_lookup[exponent] * np.fromiter(
            (means.get(int(code), 0.0) for code in product),
            dtype=float,
            count=len(product),
        )
        second += 2.0 * coefficients[i] * float(
            np.dot(coefficients[other], product_means)
        )
    word_means = np.fromiter(
        (means.get(int(code), 0.0) for code in codes),
        dtype=float,
        count=len(codes),
    )
    mean = float(np.dot(coefficients, word_means))
    variance = second - mean * mean
    scale_guard = max(abs(second), mean * mean, 1.0)
    if variance < -1e-10 * scale_guard:
        raise AssertionError(f"negative Pauli-functional variance {variance}")
    return max(0.0, variance)


def _ritz_variance_per_shot_sum(
    n: int,
    rho,
    codes: list[int],
    coefficients: dict[int, float],
    compatibility: np.ndarray,
    assignment: list[int],
    estimator: str,
) -> float:
    """Sum of per-setting Ritz-Jacobian variances for uniform shots.

    The pooled branch splits each word's coefficient uniformly over its reading
    settings.  Those are the inverse-variance (Neyman) weights that
    ``GroupedWordCache.group_weights`` documents as ``N_g / sum N_g'`` only
    when every reading setting delivers the same number of effective shots.
    The caller is ``_accuracy_matched_costs``, which prices exactly that
    schedule: ``inflate_shots_for_fidelity`` raises each setting's raw shots to
    a common effective count, so the weights are minimum-variance there up to
    integer shot rounding.  Do not reuse this at a non-uniform effective
    schedule -- that is the regime the shot- and fidelity-weighted pooling in
    ``estimator_information`` covers, and it reports a different estimator.
    """
    code_positions = {code: index for index, code in enumerate(codes)}
    unknown = sorted(set(coefficients) - set(code_positions))
    if unknown:
        raise AssertionError(f"Ritz functional contains {len(unknown)} unbanked words")
    coefficient_vector = np.zeros(len(codes), dtype=float)
    for code, value in coefficients.items():
        coefficient_vector[code_positions[code]] = value
    assignment_array = np.asarray(assignment, dtype=np.int64)
    reader_counts = compatibility.sum(axis=0)
    if bool((reader_counts <= 0).any()):
        raise AssertionError("some words have no compatible measurement setting")
    total = 0.0
    for setting_index, compatible in enumerate(compatibility):
        if estimator == "single_assignment":
            live = (assignment_array == setting_index) & (coefficient_vector != 0.0)
            weights = coefficient_vector[live]
        elif estimator == "pooled":
            live = compatible & (coefficient_vector != 0.0)
            weights = coefficient_vector[live] / reader_counts[live]
        else:  # pragma: no cover - internal callers are fixed above
            raise ValueError("unknown estimator")
        total += _pauli_quadratic_variance(
            n, rho, np.asarray(codes, dtype=np.uint64)[live], weights
        )
    return total


def _accuracy_matched_costs(
    cards: list[DeviceCard],
    settings: list[SettingResources],
    compatibility: np.ndarray,
    assignment: list[int],
    *,
    n_qubits: int,
    rho,
    codes: list[int],
    ritz_coefficients: dict[int, float],
    bias_millihartree: float,
) -> dict[str, dict]:
    """Asymptotic delta-method ``C(epsilon)`` with an exact bias-floor gate."""
    target_mha = ACCURACY_TARGET_MILLIHARTREE
    bias_mha = abs(float(bias_millihartree))
    output: dict[str, dict] = {}
    for estimator in ("single_assignment", "pooled"):
        base: dict[str, object] = {
            "epsilon_millihartree": target_mha,
            "exact_subspace_bias_millihartree": bias_mha,
            "evidence_tier": "asymptotic",
            "C_time_epsilon_us": None,
            "C_time_epsilon_device_card": None,
            "pooled_weighting": (
                "uniform over the settings that read each word, which is the "
                "inverse-variance combination at the equal-effective-shot "
                "schedule priced here"
            ),
            "claim_boundary": (
                "first-order Ritz-functional propagation about the exact frozen "
                "bank; not a finite-sample energy certificate or nonlinear solver study"
            ),
        }
        if bias_mha >= target_mha:
            base.update({
                "status": "bias_floor_exceeds_target",
                "effective_shots_per_setting": None,
                "linearized_variance_ha2_at_one_shot_per_setting": None,
                "inadmissible_device_cards": [card.name for card in cards],
                "device_costs": {},
            })
            output[estimator] = base
            continue
        variance = _ritz_variance_per_shot_sum(
            n_qubits, rho, codes, ritz_coefficients,
            compatibility, assignment, estimator,
        )
        stochastic_budget_ha2 = (
            target_mha**2 - bias_mha**2
        ) * 1e-6
        effective_shots = max(1, math.ceil(variance / stochastic_budget_ha2))
        device_costs = {}
        for card in cards:
            raw = inflate_shots_for_fidelity(
                card, settings, effective_shots, n_qubits
            )
            device_costs[card.name] = cost_schedule(
                card, settings, raw, n_qubits=n_qubits,
                evidence_tier="asymptotic", epsilon=target_mha * 1e-3,
            )
        # A scalar runtime has to name the card that attains it: pricing a rung
        # at the cheapest admissible card while another declared card cannot
        # run it at all would report a time no single device delivers.
        finite_costs = {
            name: cost["accuracy"]["C_time_epsilon_us"]
            for name, cost in device_costs.items()
            if cost["accuracy"]["C_time_epsilon_us"] is not None
        }
        inadmissible = sorted(set(device_costs) - set(finite_costs))
        if not finite_costs:
            status = "inadmissible"
        elif inadmissible:
            status = "partially_priced"
        else:
            status = "priced"
        best = min(finite_costs, key=lambda name: (finite_costs[name], name), default=None)
        base.update({
            "status": status,
            "effective_shots_per_setting": effective_shots,
            "linearized_variance_ha2_at_one_shot_per_setting": variance,
            "C_time_epsilon_us": finite_costs[best] if best is not None else None,
            "C_time_epsilon_device_card": best,
            "inadmissible_device_cards": inadmissible,
            "device_costs": device_costs,
        })
        output[estimator] = base
    return output


def _ordering_verdict(by_estimator: dict[str, list[dict]]) -> dict[str, object]:
    """Compare estimator rankings only on their common admissible rungs."""
    assigned = [item["block_size"] for item in by_estimator["single_assignment"]]
    pooled = [item["block_size"] for item in by_estimator["pooled"]]
    common = set(assigned) & set(pooled)
    return {
        "compared_block_sizes": sorted(common),
        "pooling_reorders_protocols": (
            None if not assigned or not pooled else
            [k for k in assigned if k in common]
            != [k for k in pooled if k in common]
        ),
    }


def _accuracy_ordering(rows: list[dict], cards: list[DeviceCard]) -> dict[str, dict]:
    """Rank admissible protocol rungs by asymptotic ``C(epsilon)``."""
    output = {}
    for card in cards:
        by_estimator = {}
        for estimator in ("single_assignment", "pooled"):
            live = []
            for row in rows:
                cost = row["accuracy_matched_costs"][estimator]
                ledger = cost.get("device_costs", {}).get(card.name)
                if ledger is None:
                    continue
                value = ledger["accuracy"]["C_time_epsilon_us"]
                if value is not None:
                    live.append({
                        "block_size": row["block_size"],
                        "C_time_epsilon_us": value,
                    })
            by_estimator[estimator] = sorted(
                live, key=lambda item: (item["C_time_epsilon_us"], item["block_size"])
            )
        # Only rungs both estimators admit can be reordered by the choice of
        # estimator; a differing admissible set is a different question.  With
        # nothing admissible there is no reading at all, so abstain rather than
        # report a measured "pooling changes nothing".
        by_estimator.update(_ordering_verdict(by_estimator))
        output[card.name] = by_estimator
    return output


def record_path(system: str) -> Path:
    return HERE / "reference_results" / f"clifford_hierarchy_{system}.json"


def build_record(system: str = "h4", device_cards: list[DeviceCard] | None = None) -> dict:
    cards = load_device_cards() if device_cards is None else list(device_cards)
    if not cards:
        raise ValueError("at least one device card is required")
    bank = SYSTEMS[system]()
    codes = bank.pop("codes")
    committed_groups = bank.pop("committed_qwc_groups")
    bank.pop("_exact_ground_energy")
    matrix_bank = bank.pop("_matrix_bank")
    result = bank.pop("_result")
    ritz = ritz_functional(
        matrix_bank, result.indices, result.ritz_vector(), result.ground_energy
    )

    rows = []
    compiled: list[tuple[list[SettingResources], np.ndarray, list[int]]] = []
    for block_size in BLOCK_SIZES:
        groups = block_commuting_partition(bank["n_qubits"], codes, block_size)
        row, settings, compatibility, assignment = _synthesize(
            bank["n_qubits"], codes, groups, block_size
        )
        row["resource_metrics"] = _resource_metrics(settings)
        row["estimators"] = {
            estimator: estimator_information(
                compatibility, assignment, SHOTS_PER_SETTING, estimator=estimator
            )
            for estimator in ("single_assignment", "pooled")
        }
        row["device_costs"] = _device_costs(
            cards, settings, compatibility, assignment, bank["n_qubits"]
        )
        row["accuracy_matched_costs"] = _accuracy_matched_costs(
            cards, settings, compatibility, assignment,
            n_qubits=bank["n_qubits"], rho=matrix_bank.reference, codes=codes,
            ritz_coefficients=ritz.coefficients,
            bias_millihartree=bank["error_millihartree"],
        )
        rows.append(row)
        compiled.append((settings, compatibility, assignment))

    qwc, full = rows[0], rows[-1]
    if committed_groups is not None and qwc["settings"] != committed_groups:
        raise AssertionError(
            f"k=1 grouping is not the committed QWC ledger: "
            f"{qwc['settings']} != {committed_groups}")

    # Break-even against the QWC endpoint under C_k = R (c_prep G_k + c_CX N_k):
    # level k is cheaper than k=1 iff c_CX/c_prep is below this ratio.  Under
    # the fixed greedy grouping and stim-elimination synthesis below, which
    # level is most permissive is instance- and compilation-dependent.
    for row in rows[1:]:
        row["cx_to_preparation_cost_ratio_vs_qwc"] = (
            (qwc["settings"] - row["settings"]) / row["logical_cx_per_sweep"])
    rows[0]["cx_to_preparation_cost_ratio_vs_qwc"] = None
    best = max(rows[1:], key=lambda row: row["cx_to_preparation_cost_ratio_vs_qwc"])
    settings_saved = qwc["settings"] - full["settings"]
    qwc_settings = compiled[0][0]
    for row, (settings, _, _) in zip(rows, compiled):
        if row is qwc:
            row["break_even_vs_qwc"] = None
            continue
        row["break_even_vs_qwc"] = {
            card.name: break_even_surface(
                card, qwc_settings, settings, SHOTS_PER_SETTING,
                n_qubits=bank["n_qubits"],
                t_2q_ratios=BREAK_EVEN_T2Q_RATIOS,
                eps_2q_values=BREAK_EVEN_EPS_2Q,
                evidence_tier="exact",
            )
            for card in cards
        }
    return {
        "schema": "clifford_qc.clifford_measurement_hierarchy.v3",
        "word_universe": len(codes),
        "shots_per_setting": SHOTS_PER_SETTING,
        "grouping": (
            "largest-conflict-degree greedy coloring; two words are compatible "
            "iff their restrictions commute inside every contiguous k-qubit block"
        ),
        "synthesis": (
            "independent binary-symplectic stabilizer basis per block; inverse "
            "stim tableau elimination; exact Z-only conjugation check on all members"
        ),
        "logical_model": (
            "all-to-all logical Clifford circuits; no routing, noise, or mitigation"
        ),
        "cost_model": (
            "declared device-card timing plus independent-error fidelity surrogate; "
            "routing is a declared scalar sensitivity multiplier, not compilation"
        ),
        "device_cards": [
            {"sha256": card.sha256, **card.to_dict()} for card in cards
        ],
        "accuracy_matching": {
            "epsilon_millihartree": ACCURACY_TARGET_MILLIHARTREE,
            "exact_subspace_bias_millihartree": abs(bank["error_millihartree"]),
            "default_evidence_tier": "exact",
            "default_tier_status": "unavailable",
            "reported_evidence_tier": "asymptotic",
            "reported_tier_status": (
                "bias_floor_exceeds_target"
                if abs(bank["error_millihartree"]) >= ACCURACY_TARGET_MILLIHARTREE
                else "priced"
            ),
            "reason": (
                "no finite-sample Ritz-energy certificate exists; reported C(epsilon) "
                "uses the exact bias floor plus first-order Ritz-functional variance"
            ),
        },
        "accuracy_matched_ordering": _accuracy_ordering(rows, cards),
        **bank,
        "rows": rows,
        "qwc_to_full": {
            "settings_saved": settings_saved,
            "preparation_reduction_fraction": settings_saved / qwc["settings"],
            "logical_cx_per_preparation_avoided": (
                full["logical_cx_per_sweep"] / settings_saved),
            "max_cx_to_preparation_cost_ratio": (
                settings_saved / full["logical_cx_per_sweep"]),
        },
        "most_permissive_level": {
            "block_size": best["block_size"],
            "cx_to_preparation_cost_ratio_vs_qwc": (
                best["cx_to_preparation_cost_ratio_vs_qwc"]),
        },
        "invariant": "PASS: every grouped Pauli restriction maps to Z-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="h4")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--device-card", type=Path, action="append", default=None,
        help="device-card JSON (repeatable; defaults to every declared R1 card)",
    )
    args = parser.parse_args()
    cards = load_device_cards(args.device_card)
    record = stamp_record(build_record(args.system, cards))
    out = args.out or record_path(args.system)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"{args.system}: M={record['basis_size']} W={record['word_universe']}")
    for row in record["rows"]:
        print(
            f"k={row['block_size']}: G={row['settings']} "
            f"CX={row['logical_cx_per_sweep']} "
            f"D2={row['mean_logical_cx_depth']:.2f}/"
            f"{row['max_logical_cx_depth']} "
            f"coverage={row['estimators']['pooled']['coverage_fraction']['mean']:.4f}"
        )
    matching = record["accuracy_matching"]
    print(
        "accuracy matching: "
        f"{matching['reported_tier_status']} at {matching['reported_evidence_tier']} "
        f"tier (exact tier {matching['default_tier_status']})"
    )
    print(out)


if __name__ == "__main__":
    main()
