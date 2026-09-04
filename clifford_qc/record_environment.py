"""The environment contract shared by the record gates and the producers.

Every record under ``benchmarks/reference_results`` carries a provenance stamp
naming the Python and library versions that produced it, and the record gates
compare fresh values against the committed ones.  Those versions are therefore
part of the contract rather than metadata: a finite-shot record is only
value-comparable under the NumPy that drew its shots, and an exact record is
only last-bit comparable under the LAPACK that NumPy bundles.

Two callers read this module, from opposite ends of a record's life:

* ``benchmarks/check_record_environment.py``, the gate CI runs before anything
  is installed.  It loads this file by path rather than importing the package,
  which is why nothing here may import NumPy, the rest of ``clifford_qc``, or
  anything else outside the standard library.
* :func:`clifford_qc.reproducibility.stamp_record`, at the producer boundary,
  through :func:`guard`.

The second caller closes the gap the first cannot.  A pin derived *from* the
records can only speak once a record exists, so a producer run under some other
interpreter writes a record that declares its own environment by fiat; the
committed set is then split, and nothing says so until a cross-record check
runs -- which is how ``priceability_screen.json`` and
``r3b_margin_stop_probe.json`` came to be committed under Python 3.11 while
their ten siblings were built under 3.12.  Refusing at the stamp is the only
place that costs a rebuild rather than a retraction.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Mapping, Sequence

# The committed record set, as seen from an editable install or a source
# checkout.  Outside one -- an installed wheel, say -- there are no committed
# records to disagree with and the guard stands down.
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "benchmarks" / "reference_results"

# Set to 1 to stamp under an environment the committed records do not declare.
# It authorizes a migration of the whole set, not one record: see REPRODUCING.md.
ALLOW_MIGRATION_ENV = "CLIFFORD_QC_ALLOW_ENVIRONMENT_MIGRATION"

# Records whose migration to the current environment is outstanding, named one
# by one with a reason.  A record listed here is passed over by the default
# survey, so it neither sets the pin nor widens what the producer guard will
# accept, but it stays readable through ``include_pending`` and the gate reports
# it on every run.  Naming is the point: the alternative this replaces was a
# survey that could not see the record at all.
PENDING = (Path(__file__).resolve().parent.parent / "benchmarks" / "migrations"
           / "pending_environment_migration.json")

# Python patch releases do not move floating-point results; NumPy and SciPy
# releases do, through the sampling stream and the bundled BLAS/LAPACK.  So the
# interpreter is matched on its minor version and the libraries exactly.
PYTHON_PARTS = 2

# How many record names to name per version before falling back to a count.
NAMED_RECORDS = 3


class UndeclaredEnvironment(RuntimeError):
    """Raised when a producer's environment matches no committed record."""


def _minor(version: object) -> object:
    """Truncate a Python version to the granularity that affects results."""
    if not isinstance(version, str):
        return version
    return ".".join(version.split(".")[:PYTHON_PARTS])


def pending(path: Path | None = None) -> dict[str, dict]:
    """Read the outstanding-migration manifest, keyed by record basename.

    Absent or unreadable, it lists nothing: the manifest records a decision
    about specific records, and losing it must not silently exempt them.
    """
    manifest = PENDING if path is None else path
    try:
        listed = json.loads(manifest.read_text(encoding="utf-8")).get("records")
    except (OSError, ValueError, AttributeError):
        return {}
    return listed if isinstance(listed, dict) else {}


def exemption_problems(data: Path | None = None) -> list[str]:
    """Check that every outstanding record is still the one that was exempted.

    A pending entry is a decision about a specific file, so it is anchored to
    that file's digest.  Exempting a *name* would mean the record could be
    deleted, corrupted, or re-stamped to any environment at all and the gate
    would keep passing, because the survey never opens a file it is skipping.
    Absence, unreadability and drift are therefore failures, and an entry with
    no digest cannot exempt anything.
    """
    directory = DEFAULT_DATA if data is None else data
    problems = []
    for name, entry in sorted(pending().items()):
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(expected, str):
            problems.append(
                f"{name}: listed as awaiting migration with no sha256, so the "
                "exemption is anchored to nothing; add the record's digest")
            continue
        try:
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError as exc:
            problems.append(
                f"{name}: listed as awaiting migration but cannot be read "
                f"({exc}); an entry cannot outlive its record")
            continue
        if actual != expected:
            problems.append(
                f"{name}: changed since it was listed as awaiting migration "
                f"({actual[:12]}... is not the exempted {expected[:12]}...). A "
                "record is exempt as reviewed, not as a name: rebuild it and "
                "delete the entry, or re-anchor the entry to what it now is")
    return problems


def _provenance_blocks(path: Path) -> tuple[list[dict], str | None]:
    """Every provenance block in one record, or the error that stopped reading.

    A JSONL record is a sequence of rows rather than one document, and each row
    carries its own stamp, so every row is read: a bench that writes rows as it
    runs can straddle a version change, and one row is not the record.
    """
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            documents = [json.loads(text)]
    except (OSError, ValueError) as exc:
        return [], f"{path.name}: unreadable ({exc})"
    return [document["provenance"] for document in documents
            if isinstance(document, dict)
            and isinstance(document.get("provenance"), dict)], None


def survey(
    data: Path = DEFAULT_DATA,
    record_names: Sequence[str] | None = None,
    include_pending: bool = False,
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Map package -> version -> the records declaring it, plus any read errors.

    ``python`` is folded in as one more package, truncated first, so agreement
    and comparison need only one code path.  A record with no provenance block
    makes no claim about its environment and is passed over; a record that
    declares ``null`` for a package was built without it.

    Both record shapes are surveyed.  Reading only ``*.json`` was a silent
    under-count rather than a policy: a JSONL record stamps its rows exactly as
    a JSON record stamps its document, and one that had drifted off the agreed
    environment could not be seen by the gate or by the producer guard.

    Records named in the outstanding-migration manifest are passed over unless
    ``include_pending``, which is how the gate reads back what an outstanding
    record declares in order to report it.  ``record_names``
    deliberately narrows the survey to named basenames. It is for running one
    value gate under the environment stamped on its own record while a
    separately reported cross-record inconsistency awaits regeneration; it is
    not a replacement for the default all-record consistency check.
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
        paths = sorted(list(data.glob("*.json")) + list(data.glob("*.jsonl")))
    outstanding = () if include_pending else tuple(pending())
    for path in paths:
        if path.name in outstanding:
            continue
        blocks, error = _provenance_blocks(path)
        if error is not None:
            problems.append(error)
            continue
        for provenance in blocks:
            stamped: dict[str, object] = {"python": _minor(provenance.get("python"))}
            dependencies = provenance.get("dependencies")
            if isinstance(dependencies, dict):
                stamped.update(dependencies)
            for name, version in stamped.items():
                if not isinstance(version, str):
                    continue
                # Rows of one JSONL record are one declaration, not hundreds.
                records = declarations.setdefault(name, {}).setdefault(version, [])
                if path.name not in records:
                    records.append(path.name)
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


def _declared_by(by_version: dict[str, list[str]]) -> str:
    """Render the versions the records declare, naming a few of the records."""
    parts = []
    for version, records in sorted(by_version.items()):
        names = sorted(records)
        shown = ", ".join(names[:NAMED_RECORDS])
        if len(names) > NAMED_RECORDS:
            shown += f", and {len(names) - NAMED_RECORDS} more"
        parts.append(f"{version} ({shown})")
    return "; ".join(parts)


def undeclared(declarations: dict[str, dict[str, list[str]]]) -> list[str]:
    """Report each package whose live version no committed record declares.

    Membership, not agreement, is the test.  A repository whose records already
    disagree is repaired by rebuilding the odd ones out, and a rebuild runs
    under one of the versions in the split -- so demanding agreement here would
    refuse the only run that ends the split.  What it still refuses is a *third*
    environment, which is how a split starts.
    """
    reports = []
    for name, by_version in sorted(declarations.items()):
        actual = _installed(name)
        if actual is None or actual in by_version:
            continue
        reports.append(
            f"{name} {actual} is declared by no committed record; they were "
            f"built under {_declared_by(by_version)}")
    return reports


def guard(data: Path | None = None) -> None:
    """Refuse an environment the committed records cannot reproduce.

    Read errors are not raised here.  A record this run is not producing may be
    unreadable for reasons of its own, and reporting that is the gate's job;
    refusing to stamp over it would turn one broken file into a stalled bench.
    """
    data = DEFAULT_DATA if data is None else data
    if not data.is_dir():
        return
    declarations, _ = survey(data)
    reports = undeclared(declarations)
    if not reports:
        return
    detail = "\n  ".join(reports)
    raise UndeclaredEnvironment(
        "this environment produces records nothing else in the repository can "
        f"reproduce:\n  {detail}\n"
        "Rebuild under a declared environment -- `python "
        "benchmarks/check_record_environment.py --constraints` writes the pins "
        "and `--python` names the interpreter -- or, if this run is a "
        "deliberate migration of the whole record set, re-run it with "
        f"{ALLOW_MIGRATION_ENV}=1 and migrate every record.")


def migration_allowed(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the environment authorizes stamping outside the declared set."""
    source = os.environ if environ is None else environ
    return source.get(ALLOW_MIGRATION_ENV, "").strip().lower() in {
        "1", "true", "yes", "on"}


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
