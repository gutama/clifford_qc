"""Core value object and coercion helpers shared by generator domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ir import PauliSum, PauliWord
from ..multivector import MV


@dataclass(frozen=True)
class Generator:
    """One labelled basis direction ``A_i``, whose state stays virtual."""

    label: str
    mv: MV

    @property
    def n(self) -> int:
        return self.mv.n

    def support(self) -> int:
        """Number of Pauli words in ``A_i`` (the resource term ``S_A``)."""
        return self.mv.nnz()


def _to_mv(obj, n: int | None = None) -> MV:
    if isinstance(obj, MV):
        return obj
    if isinstance(obj, (PauliWord, PauliSum)):
        return obj.to_mv()
    if hasattr(obj, "word"):
        return obj.word.to_mv()
    raise TypeError(f"cannot read a generator multivector from {type(obj).__name__}")


def _label_of(obj, index: int) -> str:
    for attr in ("label", "name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    return f"A{index}"


def as_generators(items: Iterable) -> list[Generator]:
    """Coerce supported operator-like values into labelled generators."""
    out: list[Generator] = []
    for index, item in enumerate(items):
        if isinstance(item, Generator):
            out.append(item)
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            label, payload = item
            out.append(Generator(label, _to_mv(payload)))
            continue
        out.append(Generator(_label_of(item, index), _to_mv(item)))
    if not out:
        raise ValueError("no generators given")
    n = out[0].n
    if any(generator.n != n for generator in out):
        raise ValueError("generators live in different algebras")
    return out


def _scalar_free_key(mv: MV):
    """Hash a multivector up to an overall nonzero complex factor."""
    terms = [(code, value) for code, value in sorted(mv.terms.items())
             if abs(value) > 1e-15]
    if not terms:
        return None
    pivot = terms[0][1]
    return tuple((code, complex(round((value / pivot).real, 12),
                                round((value / pivot).imag, 12)))
                 for code, value in terms)


__all__ = ["Generator", "as_generators"]
