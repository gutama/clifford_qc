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
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models import fcidump_model
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
SHOTS_PER_SETTING = 8_000
BLOCK_SIZES = (1, 2, 4, 8)
BUDGET = 8


def _xz(n: int, code: int) -> tuple[int, int]:
    x = z = 0
    for q in range(n):
        letter = (code >> (2 * q)) & 3
        if letter in (1, 2):
            x |= 1 << q
        if letter in (2, 3):
            z |= 1 << q
    return x, z


def _parity_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized uint64 parity, compatible with the declared NumPy >=1.23."""
    work = np.array(values, dtype=np.uint64, copy=True)
    work ^= work >> 32
    work ^= work >> 16
    work ^= work >> 8
    work ^= work >> 4
    work ^= work >> 2
    work ^= work >> 1
    return (work & np.uint64(1)).astype(bool)


def _partition(n: int, codes: list[int], block_size: int) -> list[list[int]]:
    """Largest-conflict-degree greedy partition under block commutativity."""
    width = len(codes)
    xs = np.asarray([_xz(n, code)[0] for code in codes], dtype=np.uint64)
    zs = np.asarray([_xz(n, code)[1] for code in codes], dtype=np.uint64)
    block_masks = [
        np.uint64(((1 << min(block_size, n - start)) - 1) << start)
        for start in range(0, n, block_size)
    ]

    degrees = np.empty(width, dtype=np.int32)
    conflicts: list[int] = []
    for i in range(width):
        # One bit per qubit marks a local anticommutation contribution.
        cross = (xs[i] & zs) ^ (zs[i] & xs)
        bad = np.zeros(width, dtype=bool)
        for mask in block_masks:
            bad |= _parity_u64(cross & mask)
        degrees[i] = int(bad.sum())
        packed = np.packbits(bad, bitorder="little")
        conflicts.append(int.from_bytes(packed.tobytes(), "little"))

    order = sorted(range(width), key=lambda i: (-int(degrees[i]), codes[i]))
    groups: list[list[int]] = []
    group_masks: list[int] = []
    for i in order:
        conflict = conflicts[i]
        for group_index, member_mask in enumerate(group_masks):
            if conflict & member_mask == 0:
                groups[group_index].append(i)
                group_masks[group_index] = member_mask | (1 << i)
                break
        else:
            groups.append([i])
            group_masks.append(1 << i)

    # Independent grouping invariant: no member conflicts with its group mask.
    for members, member_mask in zip(groups, group_masks):
        for i in members:
            if conflicts[i] & member_mask:
                raise AssertionError("block-commuting partition invariant failed")
    return groups


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


def _circuit_stats(circuit) -> tuple[Counter, int]:
    """Literal logical gate counts and CX-only depth for a stim elimination."""
    counts: Counter = Counter()
    last_layer: dict[int, int] = {}
    depth = 0
    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        if name == "CX":
            if len(targets) % 2:
                raise AssertionError("CX instruction has an odd target count")
            for a, b in zip(targets[0::2], targets[1::2]):
                qa, qb = a.value, b.value
                counts["CX"] += 1
                layer = 1 + max(last_layer.get(qa, 0), last_layer.get(qb, 0))
                last_layer[qa] = last_layer[qb] = layer
                depth = max(depth, layer)
        elif name in {"H", "S"}:
            counts[name] += len(targets)
        else:
            raise AssertionError(f"unexpected tableau-elimination gate {name}")
    return counts, depth


def _synthesize(n: int, codes: list[int], groups: list[list[int]],
                block_size: int) -> dict:
    total: Counter = Counter()
    cx_per_setting: list[int] = []
    depth_per_setting: list[int] = []
    checked = 0
    for members in groups:
        setting_counts: Counter = Counter()
        block_depths: list[int] = []
        for start in range(0, n, block_size):
            size = min(block_size, n - start)
            local = [_local_code(codes[i], start, size) for i in members]
            basis = _independent_codes(local, size)
            if not basis:
                block_depths.append(0)
                continue
            stabilizers = [stim.PauliString(_stim_label(code, size))
                           for code in basis]
            tableau = stim.Tableau.from_stabilizers(
                stabilizers, allow_redundant=False, allow_underconstrained=True)
            diagonalizer = tableau.inverse()

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

            counts, depth = _circuit_stats(
                diagonalizer.to_circuit("elimination"))
            setting_counts.update(counts)
            block_depths.append(depth)

        total.update(setting_counts)
        cx_per_setting.append(setting_counts["CX"])
        depth_per_setting.append(max(block_depths, default=0))

    settings = len(groups)
    cx = sum(cx_per_setting)
    depth_sum = sum(depth_per_setting)
    return {
        "block_size": block_size,
        "settings": settings,
        "word_samples_per_preparation": len(codes) / settings,
        "logical_cx_per_sweep": cx,
        "mean_logical_cx_per_setting": cx / settings,
        "max_logical_cx_per_setting": max(cx_per_setting),
        "logical_cx_depth_sum": depth_sum,
        "mean_logical_cx_depth": depth_sum / settings,
        "max_logical_cx_depth": max(depth_per_setting),
        "state_preparations_at_uniform_shots": settings * SHOTS_PER_SETTING,
        "logical_cx_applications_at_uniform_shots": cx * SHOTS_PER_SETTING,
        "gate_counts_per_sweep": dict(sorted(total.items())),
        "z_only_restrictions_checked": checked,
    }


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
    bank.solve()
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
        "error_millihartree": selected["error_millihartree"],
        "condition_number": selected["condition_number"],
        "codes": codes,
        "committed_qwc_groups": selected["final_qwc_groups"],
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
        "error_millihartree": (energy - exact_energy) * 1e3,
        "condition_number": result.condition_number,
        "codes": codes,
        "committed_qwc_groups": None,
    }


SYSTEMS = {"h4": _h4_bank, "beh2": _beh2_bank}


def record_path(system: str) -> Path:
    return HERE / "reference_results" / f"clifford_hierarchy_{system}.json"


def build_record(system: str = "h4") -> dict:
    bank = SYSTEMS[system]()
    codes = bank.pop("codes")
    committed_groups = bank.pop("committed_qwc_groups")

    rows = []
    for block_size in BLOCK_SIZES:
        groups = _partition(bank["n_qubits"], codes, block_size)
        rows.append(_synthesize(bank["n_qubits"], codes, groups, block_size))

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
    return {
        "schema": "clifford_qc.clifford_measurement_hierarchy.v2",
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
    args = parser.parse_args()
    record = stamp_record(build_record(args.system))
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
            f"{row['max_logical_cx_depth']}")
    print(out)


if __name__ == "__main__":
    main()
