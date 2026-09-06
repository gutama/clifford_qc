"""Executable plans for block-commuting Pauli measurement.

The lower-level grouping, synthesis, and sampling modules deliberately expose
separate concerns.  This module is their public boundary: it freezes one word
universe and block size into a partition, verified Clifford settings, signed
readout maps, and the synthesis ledger used for resource accounting.

``stim`` remains an optional dependency.  Importing :mod:`clifford_qc.measurement`
does not load it; only :func:`compile_block_measurement_plan` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

from ..ir import PauliSum, PauliWord
from ..multivector import MV
from ..pauli_structure import validate_spin_conserving_x_rank
from .block_commuting import block_commuting_partition, block_wise_commute
from .compiled import CompiledSetting

if TYPE_CHECKING:  # pragma: no cover - types only; block_synthesis imports stim
    from .block_synthesis import BlockSynthesis
    from .cost import SettingResources


@dataclass(frozen=True)
class CompiledMeasurementPlan:
    """One reproducible block-commuting measurement protocol.

    ``codes`` is the fixed, distinct word universe. ``groups`` partitions its
    indices exactly once. ``settings`` are directly consumable by
    :class:`CompiledMeasurementSampler`; ``synthesis`` carries the matched gate,
    depth, assignment, and compatibility accounting.

    ``spin_conserving_x_rank`` is populated only when the caller supplied a
    Hamiltonian under the explicit spin-conserving Jordan--Wigner claim.
    """

    n: int
    block_size: int
    codes: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    settings: tuple[CompiledSetting, ...]
    synthesis: "BlockSynthesis"
    spin_conserving_x_rank: int | None = None

    @property
    def assignment(self) -> tuple[int, ...]:
        return tuple(self.synthesis.assignment)

    @property
    def compatibility(self):
        """Boolean ``settings x words`` readout matrix (read-only view)."""
        view = self.synthesis.compatibility.view()
        view.setflags(write=False)
        return view

    @property
    def resources(self) -> tuple["SettingResources", ...]:
        return tuple(self.synthesis.settings)


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _word_codes(n: int, codes: Sequence[int]) -> tuple[int, ...]:
    out = []
    limit = 1 << (2 * n)
    for code in codes:
        if not isinstance(code, Integral) or isinstance(code, bool):
            raise TypeError("word codes must be integers")
        code = int(code)
        if not 0 <= code < limit:
            raise ValueError(f"word code {code} lies outside the {n}-qubit register")
        out.append(code)
    if len(set(out)) != len(out):
        raise ValueError("compiled measurement word codes must be distinct")
    return tuple(out)


def _measurement_groups(
    n: int,
    codes: tuple[int, ...],
    block_size: int,
    groups: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    raw = (block_commuting_partition(n, codes, block_size)
           if groups is None else groups)
    out = []
    assigned = []
    words = tuple(PauliWord(n, code) for code in codes)
    for group in raw:
        members = []
        for index in group:
            if not isinstance(index, Integral) or isinstance(index, bool):
                raise TypeError("measurement group indices must be integers")
            index = int(index)
            if not 0 <= index < len(codes):
                raise ValueError("measurement group index is out of range")
            members.append(index)
        if not members:
            raise ValueError("measurement groups must be non-empty")
        if len(set(members)) != len(members):
            raise ValueError("a measurement group contains a duplicate word index")
        for left, right in combinations(members, 2):
            if not block_wise_commute(words[left], words[right], block_size):
                raise ValueError("a supplied measurement group is not block-wise commuting")
        out.append(tuple(members))
        assigned.extend(members)
    if sorted(assigned) != list(range(len(codes))):
        raise ValueError("measurement groups must assign every word exactly once")
    return tuple(out)


def _compile_settings(
    n: int,
    codes: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
    block_size: int,
) -> tuple[CompiledSetting, ...]:
    try:
        import stim
        from ..bridges.stim_bridge import CliffordMap
        from .block_commuting import block_ranges
        from .block_synthesis import (
            block_diagonalizer,
            diagonalizer_circuit,
            local_code,
            stim_label,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        if exc.name == "stim":
            raise ImportError(
                "compiled block measurement requires the stim extra: "
                "pip install -e '.[stim]'"
            ) from exc
        raise

    compiled = []
    for setting_index, members in enumerate(groups):
        circuit = stim.Circuit()
        readouts = {code: [1, []] for code in codes}
        readable = {code: True for code in codes}
        for start, size in block_ranges(n, block_size):
            local_members = [local_code(codes[index], start, size)
                             for index in members]
            diagonalizer = block_diagonalizer(local_members, size)

            # Use the same reduced circuit whose resources are priced by the
            # synthesis ledger, so the sampled and costed Clifford are identical.
            for instruction in diagonalizer_circuit(diagonalizer, size):
                targets = [start + target.value
                           for target in instruction.targets_copy()]
                circuit.append(instruction.name, targets)

            for code in codes:
                local = local_code(code, start, size)
                transformed = diagonalizer(stim.PauliString(stim_label(local, size)))
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

        # Preserve idle trailing qubits when converting the circuit to a tableau.
        circuit.append("I", range(n))
        clifford = CliffordMap(n, circuit.to_tableau())
        explicit = {
            code: (int(readouts[code][0]), tuple(readouts[code][1]))
            for code in codes if readable[code]
        }
        assigned_codes = tuple(codes[index] for index in members)
        if any(code not in explicit for code in assigned_codes):
            raise AssertionError("an assigned word lacks a compiled readout")

        # Independent global check of every signed block-composed readout.
        for code, expected in explicit.items():
            phase, image = clifford.conjugate(PauliWord(n, code))
            if any(image.letter(qubit) not in ("I", "Z") for qubit in range(n)):
                raise AssertionError("global compiled readout is not Z-only")
            actual_sign = int(round(complex(phase).real))
            actual_positions = tuple(image.support())
            if (abs(complex(phase).imag) > 1e-12
                    or (actual_sign, actual_positions) != expected):
                raise AssertionError(
                    "block readout disagrees with the global Clifford tableau"
                )

        compiled.append(CompiledSetting(
            key=("block-commuting", n, block_size, assigned_codes),
            clifford=clifford,
            assigned_word_codes=assigned_codes,
            readouts=explicit,
        ))
    return tuple(compiled)


def compile_block_measurement_plan(
    n: int,
    codes: Sequence[int],
    block_size: int,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    spin_conserving_jw_hamiltonian: MV | PauliSum | None = None,
) -> CompiledMeasurementPlan:
    """Compile a fixed word universe into executable block-commuting settings.

    ``groups=None`` selects the deterministic largest-conflict-degree greedy.
    Supplying groups supports frozen or externally chosen partitions, which are
    still checked for complete exact-once assignment and block compatibility.

    The low-level compiler remains Hamiltonian-agnostic.  Passing
    ``spin_conserving_jw_hamiltonian`` is the explicit declaration that the
    source Hamiltonian is a spin-conserving Jordan--Wigner construction; only
    that opt-in path invokes the Phase 13 X-rank gate.
    """
    n = _positive_int(n, "qubit count")
    block_size = _positive_int(block_size, "block size")
    codes = _word_codes(n, codes)

    rank = None
    if spin_conserving_jw_hamiltonian is not None:
        if not isinstance(spin_conserving_jw_hamiltonian, (MV, PauliSum)):
            raise TypeError("spin_conserving_jw_hamiltonian must be an MV or PauliSum")
        if spin_conserving_jw_hamiltonian.n != n:
            raise ValueError("source Hamiltonian and measurement plan use different widths")
        rank = validate_spin_conserving_x_rank(spin_conserving_jw_hamiltonian)

    groups = _measurement_groups(n, codes, block_size, groups)

    # Importing the package remains Stim-free; compilation is the opt-in point.
    try:
        from .block_synthesis import synthesize_block_settings
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        if exc.name == "stim":
            raise ImportError(
                "compiled block measurement requires the stim extra: "
                "pip install -e '.[stim]'"
            ) from exc
        raise

    synthesis = synthesize_block_settings(n, codes, groups, block_size)
    settings = _compile_settings(n, codes, groups, block_size)
    if len(settings) != synthesis.n_settings:
        raise AssertionError("compiled settings and synthesis ledger disagree")
    for index, setting in enumerate(settings):
        expected = {code for word_index, code in enumerate(codes)
                    if synthesis.compatibility[index, word_index]}
        if set(setting.readouts) != expected:
            raise AssertionError(
                "compiled readouts disagree with synthesis compatibility"
            )

    return CompiledMeasurementPlan(
        n=n,
        block_size=block_size,
        codes=codes,
        groups=groups,
        settings=settings,
        synthesis=synthesis,
        spin_conserving_x_rank=rank,
    )


__all__ = ["CompiledMeasurementPlan", "compile_block_measurement_plan"]
