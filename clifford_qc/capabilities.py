r"""Which optional capabilities this installation has, without importing them.

NumPy is the only core runtime dependency; nine extras gate everything else.
The problem this module solves is that the way to *discover* an extra was to
trip over it, and the trip could happen a long way from the call:
``FermionEncoding.restriction()`` raises ``ModuleNotFoundError: stim`` from
``bridges/stim_bridge``, four frames down, and nothing in the signature
suggests a stabilizer backend is involved.

Eight modules import an optional dependency at module scope, so importing them
raises rather than degrading; nine more import one lazily inside a function,
so the failure arrives mid-call.  Of those seventeen sites only two named the
extra to install.  ``measurement/planning.py`` set the good precedent --

    ImportError: compiled block measurement requires the stim extra:
                 pip install -e '.[stim]'

-- and :func:`require` generalises it, so every site can raise that shape
without restating the pip incantation.

Probing is done with ``importlib.util.find_spec``, which locates a module
without executing it.  That matters for two reasons: importing PySCF to find
out whether PySCF is installed costs seconds, and a capability report must not
have side effects on the interpreter it is reporting about.

The capability table is checked against ``pyproject.toml`` by
``tests/test_capabilities.py``, in both directions: every extra a capability
claims must exist, and every optional dependency imported anywhere under
``clifford_qc/`` must be claimed by some capability.  A new optional import
therefore fails the suite until it is declared, which is what keeps this file
from becoming the stale list it replaces.
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Iterable, NamedTuple


class Capability(NamedTuple):
    """One optional capability: what it needs, what it unlocks, where it lives."""

    name: str
    modules: tuple[str, ...]
    extra: str
    unlocks: str
    entry_points: tuple[str, ...]


#: Every optional capability, keyed by name. ``modules`` are import names;
#: ``extra`` is the ``pyproject.toml`` extra that supplies them.
CAPABILITIES: dict[str, Capability] = {
    capability.name: capability for capability in (
        Capability(
            "sparse_linalg", ("scipy",), "research",
            "sparse eigensolvers, sector ground states, SciPy minimisers",
            ("sparse.sparse_ground_in_sector", "algorithms.optimize",
             "backends.sector_statevector", "pauli_action", "dense_reference"),
        ),
        Capability(
            "stabilizer_backend", ("stim",), "stim",
            "stabilizer-state execution of Clifford programs",
            ("backends.stabilizer",),
        ),
        Capability(
            "compiled_clifford_measurement", ("stim",), "stim",
            "compiled Clifford readouts for block-commuting measurement plans",
            ("measurement.planning", "measurement.block_synthesis"),
        ),
        Capability(
            "contextual_restriction", ("stim",), "stim",
            "stabilizer-compiled contextual and tapering restrictions",
            ("subspace.contextual", "fermion_mapping.FermionEncoding.restriction",
             "algorithms.initialization"),
        ),
        Capability(
            "fermionic_operators", ("openfermion",), "openfermion",
            "JW excitation generators and pools (FCIDUMP needs none of it)",
            ("models.chemistry", "subspace.fermionic_generators"),
        ),
        Capability(
            "molecular_input", ("openfermionpyscf", "pyscf"), "chemistry",
            "PySCF molecular structure input -- the SCF run behind a Model",
            ("models.chemistry.molecule_model",),
        ),
        Capability(
            "openfermion_bridge", ("openfermion",), "openfermion",
            "OpenFermion operator interchange",
            ("bridges.openfermion_bridge",),
        ),
        Capability(
            "pytket_bridge", ("pytket",), "pytket",
            "pytket circuit interchange",
            ("bridges.pytket_bridge",),
        ),
        Capability(
            "pennylane_bridge", ("pennylane",), "pennylane",
            "PennyLane circuit interchange",
            ("bridges.pennylane_bridge",),
        ),
        Capability(
            "pyzx_bridge", ("pyzx",), "pyzx",
            "PyZX diagram interchange and circuit optimisation",
            ("bridges.pyzx_bridge",),
        ),
    )
}


@functools.lru_cache(maxsize=None)
def _importable(module: str) -> bool:
    """Is ``module`` importable, without importing it.

    ``find_spec`` raises rather than returning ``None`` when an ancestor
    package is itself missing or broken, and a broken install should read as
    unavailable rather than as a crash in the capability report.

    A spec with no loader is a *namespace* package -- what Python synthesises
    for any directory on ``sys.path`` that happens to carry the right name. A
    stray ``pyzx/`` in the working directory would otherwise be reported as the
    real thing, and the caller would meet ``No module named 'pyzx.circuit'``
    further in, which is the failure shape this module exists to remove. A real
    optional dependency always has a loader, so requiring one costs nothing and
    is still no import.

    Cached because three capabilities claim ``stim`` and two claim
    ``openfermion``: one report would otherwise walk ``sys.path`` through every
    meta-path finder several times for the same answer.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.loader is not None


def missing_modules(capability: str) -> tuple[str, ...]:
    """Which of a capability's modules are not importable."""
    try:
        record = CAPABILITIES[capability]
    except KeyError:
        raise ValueError(
            f"unknown capability {capability!r}; expected one of "
            f"{sorted(CAPABILITIES)}") from None
    return tuple(module for module in record.modules if not _importable(module))


def available(capability: str) -> bool:
    """Is every module this capability needs importable."""
    return not missing_modules(capability)


def install_hint(capability: str) -> str:
    """The pip command that supplies a capability's extra.

    Phrased against the published distribution rather than an editable
    checkout: this string is the only guidance emitted at every ``require()``
    site and in the CLI, and someone who ran ``pip install clifford-qc`` has no
    ``.`` to install from. ``clifford_qc/bridges/__init__.py`` already documents
    this form.
    """
    try:
        record = CAPABILITIES[capability]
    except KeyError:
        raise ValueError(
            f"unknown capability {capability!r}; expected one of "
            f"{sorted(CAPABILITIES)}") from None
    return f"pip install 'clifford-qc[{record.extra}]'"


def capabilities() -> dict[str, dict]:
    """Report every optional capability and whether this install has it.

    Returns a mapping from capability name to
    ``{available, extra, modules, missing, unlocks, install}``.  Nothing is
    imported, so this is safe to call in a report, a test, or a CI job that
    wants to assert which capabilities an environment is expected to lack.
    """
    report = {}
    for name, record in CAPABILITIES.items():
        missing = missing_modules(name)
        report[name] = {
            "available": not missing,
            "extra": record.extra,
            "modules": list(record.modules),
            "missing": list(missing),
            "unlocks": record.unlocks,
            "install": install_hint(name),
        }
    return report


def require(capability: str, *, feature: str | None = None) -> None:
    """Raise a pointed ``ModuleNotFoundError`` unless ``capability`` is available.

    ``feature`` names the caller, since the capability name is an internal
    label and the user knows which function they called.  The guidance follows
    the shape ``measurement/planning.py`` established: what needs it, which
    extra supplies it, and the command that installs it.

    **The message deliberately opens with the interpreter's own phrasing.**
    Tooling already reads that wording -- ``benchmarks/check_docs.py`` greps
    stderr for ``ModuleNotFoundError: No module named 'X'`` to decide that a
    script's flags are unverifiable rather than broken, and
    ``measurement/planning.py`` branches on ``exc.name``.  Raising a bare
    ``ImportError`` with a friendlier message turned that skip into a failure,
    so this keeps the canonical prefix and the ``name`` attribute and appends
    the guidance.  A better error message is not worth breaking a consumer
    that was reading the old one.
    """
    missing = missing_modules(capability)
    if not missing:
        return
    record = CAPABILITIES[capability]
    what = feature or record.unlocks
    error = ModuleNotFoundError(
        f"No module named {missing[0]!r} -- {what} requires the "
        f"{record.extra!r} extra"
        + (f" ({', '.join(missing)} missing)" if len(missing) > 1 else "")
        + f": {install_hint(capability)}",
        name=missing[0])
    raise error


def format_report(report: dict[str, dict] | None = None) -> str:
    """The capability report as aligned text, for ``verify`` and the CLI."""
    report = capabilities() if report is None else report
    if not report:
        return ""
    width = max(len(name) for name in report)
    lines = []
    for name in sorted(report):
        entry = report[name]
        mark = "yes" if entry["available"] else "no "
        suffix = "" if entry["available"] else f"  ({entry['install']})"
        lines.append(f"  [{mark}] {name:<{width}}  extra={entry['extra']}{suffix}")
    return "\n".join(lines)


def missing_capabilities(report: dict[str, dict] | None = None) -> list[str]:
    """Names of the capabilities this install lacks, sorted."""
    report = capabilities() if report is None else report
    return sorted(name for name, entry in report.items() if not entry["available"])


def optional_modules() -> frozenset[str]:
    """Every import name any capability claims, for the drift check."""
    return frozenset(module for record in CAPABILITIES.values()
                     for module in record.modules)


def extras_claimed() -> frozenset[str]:
    """Every ``pyproject.toml`` extra any capability claims."""
    return frozenset(record.extra for record in CAPABILITIES.values())


def _main(argv: Iterable[str] | None = None) -> int:
    """``python -m clifford_qc.capabilities [--require NAME ...]``.

    ``--require`` exits non-zero when a named capability is absent, which is
    what lets a CI job state the environment it believes it is running in
    rather than discovering the answer through a test failure.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require", action="append", default=[],
                        metavar="CAPABILITY",
                        help="exit non-zero unless this capability is available; "
                             "repeatable")
    arguments = parser.parse_args(None if argv is None else list(argv))

    unknown = sorted(set(arguments.require) - set(CAPABILITIES))
    if unknown:
        parser.error(f"unknown capability {unknown}; expected one of "
                     f"{sorted(CAPABILITIES)}")

    report = capabilities()
    print("clifford_qc optional capabilities\n")
    print(format_report(report))
    absent = missing_capabilities(report)
    print()
    if absent:
        print(f"{len(absent)} of {len(CAPABILITIES)} unavailable: "
              f"{', '.join(absent)}")
    else:
        print(f"all {len(CAPABILITIES)} capabilities available")

    unmet = [name for name in arguments.require if not report[name]["available"]]
    if unmet:
        print()
        for name in unmet:
            print(f"REQUIRED but absent: {name} -- {install_hint(name)}")
        return 1
    return 0


__all__ = [
    "CAPABILITIES",
    "Capability",
    "available",
    "capabilities",
    "extras_claimed",
    "format_report",
    "install_hint",
    "missing_capabilities",
    "missing_modules",
    "optional_modules",
    "require",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
