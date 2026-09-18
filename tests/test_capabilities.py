"""The capability table, checked against the tree it describes.

A hand-maintained list of optional dependencies is exactly the kind of thing
that is correct on the day it is written and wrong six months later. So the
load-bearing tests here are the two drift checks, and they run in both
directions:

- every extra a capability claims must exist in ``pyproject.toml``
  (:func:`test_every_claimed_extra_exists`), and
- every optional dependency imported anywhere under ``clifford_qc/`` must be
  claimed by some capability
  (:func:`test_every_optional_import_in_the_tree_is_claimed`).

The second is the one that matters. It walks the package with ``ast``, collects
every third-party import that is not a core dependency, and fails if the table
does not account for it -- so adding an optional import without declaring it
breaks the suite rather than silently recreating the stale list this module
replaced.

Availability itself is deliberately not asserted. The core install is
numpy-only by design; which extras are present is a property of the
environment, and a test that demanded any of them would fail in the numpy-only
CI job for the wrong reason.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

from clifford_qc import capabilities as caps

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "clifford_qc"

#: Modules that are core (numpy), part of the package, or in the standard
#: library. Anything else imported under ``clifford_qc/`` is an optional
#: dependency and must be declared in the capability table.
CORE_MODULES = frozenset({"numpy", "clifford_qc"})


def _toml():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        pytest.skip("tomllib needs Python 3.11+")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _third_party_imports() -> dict[str, set[str]]:
    """``{top-level import name: {files that import it}}`` across the package."""
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: package-internal by definition
                names = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if not name or name in CORE_MODULES:
                    continue
                if name in sys.stdlib_module_names:
                    continue
                found.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return found


# ------------------------------------------------------------- drift checks
def test_every_claimed_extra_exists():
    optional = _toml()["project"]["optional-dependencies"]
    unknown = sorted(caps.extras_claimed() - set(optional))
    assert unknown == [], f"capability table claims extras that do not exist: {unknown}"


def test_every_optional_import_in_the_tree_is_claimed():
    """The check that keeps the table from going stale as the tree grows."""
    imported = _third_party_imports()
    undeclared = {name: sorted(files) for name, files in imported.items()
                  if name not in caps.optional_modules()}
    assert undeclared == {}, (
        "these third-party imports are not declared in "
        f"clifford_qc/capabilities.py: {undeclared}")


def test_every_claimed_module_is_actually_imported_somewhere():
    """The reverse: a capability claiming a module nobody imports is dead."""
    imported = set(_third_party_imports())
    # pyscf reaches the tree through openfermionpyscf rather than directly.
    indirect = {"pyscf"}
    unused = sorted(caps.optional_modules() - imported - indirect)
    assert unused == [], f"capability table claims unimported modules: {unused}"


def test_declared_extras_cover_every_optional_dependency_in_pyproject():
    """Every extra that ships a runtime dependency is owned by a capability.

    Tooling extras are excluded: they gate no importable capability.
    """
    optional = _toml()["project"]["optional-dependencies"]
    tooling = {"test", "release", "bridges"}  # bridges is an aggregate of others
    uncovered = sorted(set(optional) - tooling - caps.extras_claimed())
    assert uncovered == [], f"extras with no capability: {uncovered}"


# ------------------------------------------------------------ report shape
def test_the_report_covers_every_capability_without_importing_anything():
    before = set(sys.modules)
    report = caps.capabilities()
    assert set(report) == set(caps.CAPABILITIES)
    # find_spec locates without executing, so no capability module is now loaded
    # that was not loaded before. (Import machinery may add its own entries.)
    newly_loaded = {name for name in set(sys.modules) - before
                    if name in caps.optional_modules()}
    assert newly_loaded == set()


@pytest.mark.parametrize("capability", sorted(caps.CAPABILITIES))
def test_each_entry_is_self_consistent(capability):
    entry = caps.capabilities()[capability]
    record = caps.CAPABILITIES[capability]
    assert entry["extra"] == record.extra
    assert entry["modules"] == list(record.modules)
    assert entry["available"] == caps.available(capability)
    assert entry["available"] == (entry["missing"] == [])
    assert set(entry["missing"]) <= set(record.modules)
    assert record.extra in entry["install"]
    assert record.unlocks and record.entry_points


def test_an_unknown_capability_is_refused():
    with pytest.raises(ValueError, match="unknown capability"):
        caps.available("quantum_advantage")


def test_missing_capabilities_agrees_with_the_report():
    report = caps.capabilities()
    assert caps.missing_capabilities(report) == sorted(
        name for name, entry in report.items() if not entry["available"])


def test_format_report_lists_every_capability_and_hints_only_absent_ones():
    report = caps.capabilities()
    text = caps.format_report(report)
    lines = text.splitlines()
    assert len(lines) == len(report)
    for name, entry in report.items():
        line = next(line for line in lines if name in line)
        assert (entry["install"] in line) is not entry["available"]


# ------------------------------------------------------------------ require
def test_require_is_silent_when_the_capability_is_present():
    present = [name for name in caps.CAPABILITIES if caps.available(name)]
    if not present:  # pragma: no cover - a bare numpy-only environment
        pytest.skip("no optional capability installed here")
    caps.require(present[0])


def test_require_names_the_extra_the_feature_and_the_command():
    absent = caps.missing_capabilities()
    if not absent:  # pragma: no cover - a full-extras environment
        pytest.skip("every capability installed here")
    capability = absent[0]
    record = caps.CAPABILITIES[capability]
    with pytest.raises(ImportError) as raised:
        caps.require(capability, feature="the thing the user called")
    message = str(raised.value)
    assert "the thing the user called" in message
    assert record.extra in message
    assert caps.install_hint(capability) in message
    # the missing module is named, so the message survives being pasted alone
    assert any(module in message for module in caps.missing_modules(capability))


def test_require_keeps_the_phrasing_existing_tooling_reads():
    """A friendlier message must not break consumers reading the old one.

    Raising a bare ``ImportError`` here turned ``check_docs.py``'s "flags
    unverified, optional dependency absent" *skip* into a *failure*, because
    that checker greps stderr for the interpreter's own wording. So the message
    keeps the canonical prefix and the ``name`` attribute that
    ``measurement/planning.py`` branches on.
    """
    import re

    absent = caps.missing_capabilities()
    if not absent:  # pragma: no cover - a full-extras environment
        pytest.skip("every capability installed here")
    capability = absent[0]
    expected = caps.missing_modules(capability)[0]
    with pytest.raises(ModuleNotFoundError) as raised:
        caps.require(capability)
    assert raised.value.name == expected
    # what check_docs.py greps for, as the interpreter would print it
    rendered = f"ModuleNotFoundError: {raised.value}"
    match = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", rendered)
    assert match is not None and match.group(1) == expected


def test_a_deep_entry_point_states_its_own_requirement():
    """``restriction()`` used to raise ``ModuleNotFoundError`` four frames down.

    Whether it raises or succeeds depends on the environment; what must hold in
    both is that the failure names the extra rather than the transitive module.
    """
    from clifford_qc.fermion_mapping import fermion_encoding

    encoding = fermion_encoding("parity+2q", n=4, n_electrons=2, sz=0.0)
    if caps.available("contextual_restriction"):
        assert encoding.restriction() is not None
        return
    with pytest.raises(ImportError, match="stim"):  # pragma: no cover
        encoding.restriction()
