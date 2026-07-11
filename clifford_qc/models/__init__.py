"""Benchmark model builders (spin systems; chemistry arrives via bridges)."""

from .spin import Model, tfim, xxz, random_ising

__all__ = ["Model", "tfim", "xxz", "random_ising"]
