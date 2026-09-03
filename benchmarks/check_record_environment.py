"""Check the interpreter and libraries against the records they must reproduce.

Every record under ``benchmarks/reference_results`` carries a provenance stamp
naming the Python and library versions that produced it, and the record gates
compare fresh values against the committed ones.  Those versions are therefore
part of the contract rather than metadata: a finite-shot record is only
value-comparable under the NumPy that drew its shots, and an exact record is
only last-bit comparable under the LAPACK that NumPy bundles.

This gate answers two questions, in this order:

* do the committed records agree with each other on a version?  A record
  rebuilt against a newer NumPy than its siblings is a failure this repository
  has actually hit, and no single-record comparison can see it -- each record
  is self-consistent, and only the set is contradictory.
* does the running environment match what they agree on?

It runs before the value gates, and before the package is even installed, so an
unreproducible environment is named in seconds instead of being diagnosed as
numeric drift after a twelve-minute rebuild.  ``--python`` and ``--constraints``
emit the agreed interpreter and the agreed library versions, which is what CI
selects and installs against, so no pin is written down a second time where it
could drift away from the records that justify it.

Standard library only, by necessity: both emitting modes have to run before
``pip install``.  The per-record equivalent for use *inside* a gate, once the
package is importable, is
:func:`clifford_qc.reproducibility.sampling_stream_mismatch`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

DATA = Path(__file__).resolve().parent / "reference_results"

# Python patch releases do not move floating-point results; NumPy and SciPy
# releases do, through the sampling stream and the bundled BLAS/LAPACK.  So the
# interpreter is matched on its minor version and the libraries exactly.
PYTHON_PARTS = 2


def _minor(version: object) -> object:
    """Truncate a Python version to the granularity that affects results."""
    if not isinstance(version, str):
        return version
    return ".".join(version.split(".")[:PYTHON_PARTS])


def survey(
    data: Path = DATA,
    record_names: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Map package -> version -> the records declaring it, plus any read errors.

    ``python`` is folded in as one more package, truncated first, so agreement
    and comparison need only one code path.  A record with no provenance block
    makes no claim about its environment and is passed over; a record that
    declares ``null`` for a package was built without it.

    ``record_names`` deliberately narrows the survey to named basenames. It is
    for running one value gate under the environment stamped on its own record
    while a separately reported cross-record inconsistency awaits regeneration;
    it is not a replacement for the default all-record consistency check.
    """
    declarations: dict[str, dict[str, list[str]]] = {}
    problems: list[str] = []
    if record_names:
        paths = []
        for name in record_names:
            if Path(name).name != name:
                problems.append(f"{name}: record selectors must be basenames")
                continue
            path = data / name
            if not path.is_file():
                problems.append(f"{name}: selected record does not exist")
                continue
            paths.append(path)
    else:
        paths = sorted(data.glob("*.json"))
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        provenance = record.get("provenance") if isinstance(record, dict) else None
        if not isinstance(provenance, dict):
            continue
        stamped: dict[str, object] = {"python": _minor(provenance.get("python"))}
        dependencies = provenance.get("dependencies")
        if isinstance(dependencies, dict):
            stamped.update(dependencies)
        for name, version in stamped.items():
            if not isinstance(version, str):
                continue
            declarations.setdefault(name, {}).setdefault(version, []).append(path.name)
    return declarations, problems


def agreed(declarations: dict[str, dict[str, list[str]]],
           problems: list[str]) -> dict[str, str]:
    """Return the one version per package the records agree on.

    A package the records disagree about has no reproducible pin, so it is
    reported and left out rather than resolved by a majority vote.
    """
    versions: dict[str, str] = {}
    for name, by_version in sorted(declarations.items()):
        if len(by_version) == 1:
            versions[name] = next(iter(by_version))
            continue
        detail = "; ".join(f"{version} ({', '.join(sorted(records))})"
                           for version, records in sorted(by_version.items()))
        problems.append(
            f"{name}: the committed records disagree -- {detail}. No single "
            "environment reproduces all of them, so rebuild the odd ones out "
            "rather than pinning to one side of the split")
    return versions


def _installed(name: str) -> str | None:
    if name == "python":
        return str(_minor(platform.python_version()))
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify(versions: dict[str, str], problems: list[str]) -> None:
    """Compare the running environment against the agreed versions.

    An absent package is not a mismatch: the chemistry and bridge extras are
    optional, and the gates that need one say so themselves.  A *different*
    version is, because it silently answers a different question.
    """
    for name, expected in sorted(versions.items()):
        actual = _installed(name)
        if actual is not None and actual != expected:
            problems.append(
                f"{name} {actual} differs from the {expected} the committed "
                "records were produced under; a rebuild here verifies nothing")


def constraints(versions: dict[str, str]) -> str:
    """Render the agreed versions as a pip constraints file."""
    lines = [
        "# Generated by benchmarks/check_record_environment.py --constraints.",
        "# Versions declared by the selected stamped record set.",
        "# Rebuild a record to change a pin, not this file.",
        f"# python=={versions.get('python', 'unknown')}",
    ]
    lines += [f"{name}=={version}"
              for name, version in sorted(versions.items()) if name != "python"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    emit = parser.add_mutually_exclusive_group()
    emit.add_argument(
        "--constraints", action="store_true",
        help="write a pip constraints file to stdout instead of checking the "
             "running environment")
    emit.add_argument(
        "--python", action="store_true",
        help="write the agreed interpreter version to stdout instead of "
             "checking the running environment")
    parser.add_argument(
        "--record",
        action="append",
        dest="records",
        metavar="BASENAME",
        help="restrict the survey to a named record; repeat for a set. This "
             "supports record-local value gates and does not certify agreement "
             "with records outside the set",
    )
    args = parser.parse_args(argv)
    emitting = args.constraints or args.python

    declarations, problems = survey(DATA, args.records)
    versions = agreed(declarations, problems)
    if not emitting:
        verify(versions, problems)
    if args.python and "python" not in versions and not problems:
        problems.append(
            "no committed record names the interpreter it was produced under")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1

    if args.constraints:
        sys.stdout.write(constraints(versions))
        return 0
    if args.python:
        print(versions["python"])
        return 0

    summary = ", ".join(f"{name}=={version}"
                        for name, version in sorted(versions.items()))
    print(f"environment matches every stamped record ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
