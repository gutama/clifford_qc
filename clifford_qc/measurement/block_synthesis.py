"""Block-local Clifford diagonalizers for the dyadic measurement hierarchy.

:mod:`clifford_qc.measurement.block_commuting` decides *which* words may share
a setting. This module builds the circuit that setting actually runs: one exact
Clifford diagonalizer per contiguous ``k``-qubit block, synthesized with stim,
plus the gate counts, depths, and readout coverage the cost model prices.

Blocks occupy disjoint qubits, so a setting runs them in parallel -- its depth
is the widest block's, not their sum -- and no two-qubit gate ever crosses a
block boundary. That is what makes ``k`` a cost axis rather than a relabelling.

This module imports stim at module level and is deliberately **not** re-exported
from ``clifford_qc.measurement``: the package must stay importable without the
``stim`` extra, exactly as ``clifford_qc.bridges.stim_bridge`` does.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import stim

from .block_commuting import block_ranges
from .cost import SettingResources


def _local_code_dtype(size: int):
    """The narrowest unsigned dtype holding a ``size``-qubit packed code.

    A packed code spends two bits per qubit, so ``size = 8`` lands exactly on
    ``uint16``'s ceiling (``4**8 - 1 == 65535``) and one qubit more would wrap.
    The dyadic ladder currently stops at ``k = 8``, i.e. precisely on that
    boundary, so the truncation is invisible today and would corrupt
    compatibility and coverage silently the first time a wider rung is run.
    Widen with the block, and refuse a width no integer dtype can hold.
    """
    bits = 2 * size
    for dtype in (np.uint16, np.uint32, np.uint64):
        if bits <= np.iinfo(dtype).bits:
            return dtype
    raise ValueError(
        f"a {size}-qubit block needs {bits} bits per packed code, beyond uint64"
    )


def local_code(code: int, start: int, size: int) -> int:
    """The sub-word ``code`` restricts to, on ``size`` qubits from ``start``."""
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


def independent_codes(codes: Sequence[int], size: int) -> list[int]:
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


def stim_label(code: int, size: int) -> str:
    return "".join("IXYZ"[(code >> (2 * q)) & 3] for q in range(size))


# The declared one-qubit gate set the cost model charges ``t_1q`` per element
# of. S_DAG is included because hardware realises it as one pulse, exactly like
# S; synthesising it as three S gates would price a rotation at three times its
# physical cost.
SINGLE_QUBIT_GATES = ("H", "S", "S_DAG")


def circuit_stats(circuit) -> tuple[Counter, int, int, int]:
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
        elif name in SINGLE_QUBIT_GATES:
            counts[name] += len(targets)
            for target in targets:
                schedule("1q", (target.value,))
        else:
            raise AssertionError(f"unexpected tableau-elimination gate {name}")
    depth_1q = sum(1 for kind in layer_kind.values() if kind == "1q")
    depth_2q = sum(1 for kind in layer_kind.values() if kind == "2q")
    return counts, cx_depth, depth_1q, depth_2q


def x_masks(diagonalizer, size: int, codes: np.ndarray) -> np.ndarray:
    """X mask after Clifford conjugation, for the banked local codes only.

    Tabulating all ``4**size`` codes costs 65536 entries at ``size=8`` and is
    rebuilt per setting, while only ``len(codes)`` of them are ever read.  The
    per-qubit ``X``/``Z`` image masks XOR-reduce directly over the code array
    instead, which drops the ``4**size`` factor.
    """
    # Unsigned throughout: these are bit patterns, and a signed right shift
    # would sign-extend a code whose top bit is set once blocks get wide.
    codes = np.asarray(codes, dtype=np.uint64)
    masks = np.zeros(len(codes), dtype=np.uint64)
    for qubit in range(size):
        x_bits, _ = diagonalizer.x_output(qubit).to_numpy()
        x_from_x = sum(int(bit) << q for q, bit in enumerate(x_bits))
        x_bits, _ = diagonalizer.z_output(qubit).to_numpy()
        x_from_z = sum(int(bit) << q for q, bit in enumerate(x_bits))
        # Letter Y and Z carry Z; letters X and Y carry X (low bit xor high).
        z_bit = (codes >> np.uint64(2 * qubit + 1)) & np.uint64(1)
        x_bit = ((codes >> np.uint64(2 * qubit)) & np.uint64(1)) ^ z_bit
        masks ^= x_bit * np.uint64(x_from_x)
        masks ^= z_bit * np.uint64(x_from_z)
    return masks


@lru_cache(maxsize=1)
def _minimal_single_qubit_words() -> dict[tuple[str, str], tuple[str, ...]]:
    """Minimal ``SINGLE_QUBIT_GATES`` word for each of the 24 one-qubit Cliffords.

    Breadth-first from the identity, so the first word reaching an element is a
    shortest one; ties break on the generator order above, which makes the
    choice deterministic and therefore reproducible in a committed record.
    """
    gates = {name: stim.Tableau.from_named_gate(name)
             for name in SINGLE_QUBIT_GATES}
    start = stim.Tableau(1)
    words = {_tableau_key(start): ()}
    frontier = deque([start])
    while frontier:
        current = frontier.popleft()
        word = words[_tableau_key(current)]
        for name in SINGLE_QUBIT_GATES:
            nxt = current.then(gates[name])
            key = _tableau_key(nxt)
            if key not in words:
                words[key] = word + (name,)
                frontier.append(nxt)
    return words


def _tableau_key(tableau) -> tuple[str, str]:
    return (str(tableau.x_output(0)), str(tableau.z_output(0)))


@lru_cache(maxsize=1)
def _z_preserving_corrections() -> tuple:
    """One-qubit Cliffords mapping ``Z`` to ``+-Z``.

    Post-multiplying a diagonalizer by one of these on any output qubit leaves
    it a diagonalizer -- the image of every member stays a product of ``Z``\\ s,
    with at most a sign change, which the readout bookkeeping already tracks.
    That freedom is a coset, and stim's synthesis picks an arbitrary member of
    it; choosing the cheapest member instead is what makes the emitted circuit
    a *minimal* basis rotation rather than an arbitrary one.
    """
    gates = {name: stim.Tableau.from_named_gate(name)
             for name in SINGLE_QUBIT_GATES}
    out = []
    for key, word in sorted(_minimal_single_qubit_words().items()):
        if key[1] not in ("+Z", "-Z"):
            continue
        tableau = stim.Tableau(1)
        for name in word:
            tableau = tableau.then(gates[name])
        out.append((len(word), word, tableau))
    return tuple(out)


def _layer_cost(circuit, size: int) -> tuple[int, int]:
    """``(two-qubit gates, one-qubit gates)`` of the reduced circuit.

    Ordered so a caller can minimise lexicographically. The two-qubit count
    leads because it must: every device card prices a CX far above a rotation
    (200us against 10us on the ion-like card), so trading entangling gates for
    one-qubit savings is a cost regression however much it shortens the
    rotation layer.
    """
    two_qubit = 0
    one_qubit = 0
    for word, qubits in _reduced_instructions(circuit, size):
        if len(qubits) == 2:
            two_qubit += 1
        else:
            one_qubit += len(word)
    return two_qubit, one_qubit


def _reduced_instructions(circuit, size: int):
    """``(word, qubits)`` pairs after collapsing consecutive one-qubit runs.

    Merging is exact: only *adjacent* one-qubit gates on the same qubit are
    combined, and their product is re-emitted as a shortest word for the same
    Clifford. Nothing is commuted past a CX.
    """
    gates = {name: stim.Tableau.from_named_gate(name)
             for name in SINGLE_QUBIT_GATES}
    words = _minimal_single_qubit_words()
    pending = {q: stim.Tableau(1) for q in range(size)}
    out: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def flush(qubit: int) -> None:
        word = words[_tableau_key(pending[qubit])]
        if word:
            out.append((word, (qubit,)))
        pending[qubit] = stim.Tableau(1)

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        if name == "CX":
            for a, b in zip(targets[0::2], targets[1::2]):
                flush(a.value)
                flush(b.value)
                out.append((("CX",), (a.value, b.value)))
        elif name in gates:
            for target in targets:
                pending[target.value] = pending[target.value].then(gates[name])
        else:
            raise AssertionError(f"unexpected tableau-elimination gate {name}")
    for qubit in range(size):
        flush(qubit)
    return out


def diagonalizer_circuit(tableau, size: int):
    """A minimal-one-qubit-layer circuit implementing ``tableau``.

    stim's ``to_circuit("elimination")`` is correct but unoptimized -- it emits
    ``S H S H S S H S S`` for a one-qubit Y rotation that two gates realise.
    Its two-qubit structure is kept verbatim; only maximal one-qubit runs are
    re-expressed, so the circuit's action is unchanged by construction.
    """
    circuit = stim.Circuit()
    for word, qubits in _reduced_instructions(
            tableau.to_circuit("elimination"), size):
        for name in word:
            circuit.append(name, list(qubits))
    return circuit


def block_diagonalizer(local_members: Sequence[int], size: int):
    """The one canonical local Clifford used for costing and sampling.

    Any Clifford sending the group's generators to ``Z``-type words will do, so
    the choice is fixed by cost: stim's representative is post-multiplied, one
    output qubit at a time, by the cheapest ``Z``-preserving correction. At
    ``k = 1`` this recovers the minimal basis rotation exactly -- one gate for
    an ``X`` block, two for ``Y``, none for ``Z`` -- which is the analytic
    convention the fixed-QWC mapping-axis record is priced under.
    """
    basis = independent_codes(local_members, size)
    if not basis:
        return stim.Tableau(size)
    tableau = stim.Tableau.from_stabilizers(
        [stim.PauliString(stim_label(code, size)) for code in basis],
        allow_redundant=False,
        allow_underconstrained=True,
    ).inverse()

    for qubit in range(size):
        best = None
        for length, word, correction in _z_preserving_corrections():
            candidate = stim.Tableau(size)
            candidate.append(correction, [qubit])
            candidate = tableau.then(candidate)
            cost = _layer_cost(candidate.to_circuit("elimination"), size)
            # Ties break on the correction's own word length and then its
            # letters, so the representative is a function of the group alone.
            marker = (cost, length, word)
            if best is None or marker < best[0]:
                best = (marker, candidate)
        tableau = best[1]
    return tableau


@dataclass
class BlockSynthesis:
    """Protocol-independent synthesis ledger for one ``k`` rung.

    Everything here is a function of the bank, the partition, and ``k`` alone,
    so a producer can price it under any device card or shot budget without
    re-synthesizing.
    """

    block_size: int
    settings: list[SettingResources]
    gate_counts: Counter
    cx_per_setting: list[int]
    cx_depth_per_setting: list[int]
    compatibility: np.ndarray
    assignment: list[int]
    z_only_restrictions_checked: int = 0
    coverage: list[float] = field(default_factory=list)

    @property
    def n_settings(self) -> int:
        return len(self.settings)

    @property
    def logical_cx_per_sweep(self) -> int:
        return sum(self.cx_per_setting)


def synthesize_block_settings(
    n: int,
    codes: Sequence[int],
    groups: Sequence[Sequence[int]],
    block_size: int,
) -> BlockSynthesis:
    """Synthesize one block-local diagonalizer per block per setting.

    ``groups`` holds indices into ``codes``, as
    :func:`~clifford_qc.measurement.block_commuting.block_commuting_partition`
    returns them. Two invariants are enforced rather than assumed, because both
    failure modes return plausible numbers instead of raising: every member of a
    group -- not merely the independent basis the tableau was built from -- must
    become computational-basis diagonal, and every assigned word must actually
    be read by the setting it was assigned to.
    """
    codes = list(codes)
    total: Counter = Counter()
    cx_per_setting: list[int] = []
    cx_depth_per_setting: list[int] = []
    setting_resources: list[SettingResources] = []
    compatibility = np.zeros((len(groups), len(codes)), dtype=bool)
    assignment = [-1] * len(codes)
    ranges = block_ranges(n, block_size)
    local_codes = {
        (start, size): np.asarray(
            [local_code(code, start, size) for code in codes],
            dtype=_local_code_dtype(size),
        )
        for start, size in ranges
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
        for start, size in ranges:
            local = [local_code(codes[i], start, size) for i in members]
            diagonalizer = block_diagonalizer(local, size)

            # Strong circuit invariant: every member, not only the basis,
            # must become computational-basis diagonal.
            for code in set(local):
                transformed = diagonalizer(
                    stim.PauliString(stim_label(code, size)))
                x_bits, _ = transformed.to_numpy()
                if bool(x_bits.any()):
                    raise AssertionError(
                        f"k={block_size} block={start}: {stim_label(code, size)} "
                        f"did not map to Z-only ({transformed})")
                checked += 1

            counts, cx_depth, depth_1q, depth_2q = circuit_stats(
                diagonalizer_circuit(diagonalizer, size))
            setting_counts.update(counts)
            block_cx_depths.append(cx_depth)
            block_depths_1q.append(depth_1q)
            block_depths_2q.append(depth_2q)
            setting_reads &= (
                x_masks(diagonalizer, size, local_codes[(start, size)]) == 0
            )

        total.update(setting_counts)
        cx_per_setting.append(setting_counts["CX"])
        # Blocks occupy disjoint qubits, so a setting runs them in parallel.
        cx_depth_per_setting.append(max(block_cx_depths, default=0))
        depth_1q = max(block_depths_1q, default=0)
        depth_2q = max(block_depths_2q, default=0)
        setting_resources.append(SettingResources(
            n_1q=sum(setting_counts[name] for name in SINGLE_QUBIT_GATES),
            n_2q=setting_counts["CX"],
            d_1q=depth_1q,
            d_2q=depth_2q,
        ))
        # ``members`` is any Sequence by contract.  NumPy interprets a tuple
        # as multi-axis indexing, so normalize it for one-dimensional fancy
        # indexing instead of requiring callers to provide mutable lists.
        if not bool(setting_reads[list(members)].all()):
            raise AssertionError("an assigned word is not read by its setting")
        compatibility[setting_index] = setting_reads

    if any(index < 0 for index in assignment):
        raise AssertionError("the grouping did not assign every word")

    return BlockSynthesis(
        block_size=block_size,
        settings=setting_resources,
        gate_counts=total,
        cx_per_setting=cx_per_setting,
        cx_depth_per_setting=cx_depth_per_setting,
        compatibility=compatibility,
        assignment=assignment,
        z_only_restrictions_checked=checked,
        coverage=[float(row.sum()) / len(codes) for row in compatibility]
        if len(codes) else [],
    )
