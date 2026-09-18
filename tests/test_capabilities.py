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
def test_the_report_covers_every_capability_without_importing_anything(monkeypatch):
    """No capability module is executed to answer whether it is installed.

    Snapshotting ``sys.modules`` is not enough on its own: by the time this
    file runs, the modules that *are* installed have usually been imported by
    an earlier test, and the ones that are not cannot be loaded by any
    implementation -- so the test passes whatever ``_importable`` does. The
    probe is therefore pointed at a module that is importable, definitely not
    yet imported, and cheap: a standard-library one, unloaded for the duration.
    """
    report = caps.capabilities()
    assert set(report) == set(caps.CAPABILITIES)

    canary = "wave"  # stdlib, importable, not pulled in by this package
    monkeypatch.delitem(sys.modules, canary, raising=False)
    monkeypatch.setattr(caps, "CAPABILITIES", dict(
        caps.CAPABILITIES,
        _canary=caps.Capability("_canary", (canary,), "research", "probe", ("x",)),
    ))
    caps._importable.cache_clear()
    try:
        assert caps.capabilities()["_canary"]["available"] is True
        assert canary not in sys.modules, (
            f"capabilities() imported {canary!r} to find out whether it exists; "
            "the report must locate modules without executing them")
    finally:
        caps._importable.cache_clear()


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


def _module_scope_optional_imports(tree, optional) -> set[str]:
    """Optional dependencies imported at a module's top level, not in a function."""
    names: set[str] = set()
    for node in tree.body:  # direct children only: module scope by definition
        if isinstance(node, ast.Import):
            found = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found = [] if node.level else [(node.module or "").split(".")[0]]
        else:
            continue
        names.update(name for name in found if name in optional)
    return names


def _calls_require_at_module_scope(tree) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "require":
            return True
    return False


def test_every_module_scope_optional_import_is_guarded():
    """The invariant the table exists to enforce, not just to describe.

    ``test_every_optional_import_in_the_tree_is_claimed`` checks that a
    dependency is *named* in the table. It does not check that the module
    importing it says so before the interpreter does -- so deleting any
    ``require()`` call, or adding a ninth unguarded module-scope ``import
    stim``, passed the whole suite and quietly restored the bare
    four-frames-down ``ModuleNotFoundError`` this module removes.
    """
    optional = set(caps.optional_modules())
    unguarded = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _module_scope_optional_imports(tree, optional)
        if imports and not _calls_require_at_module_scope(tree):
            unguarded[str(path.relative_to(ROOT))] = sorted(imports)
    assert unguarded == {}, (
        "these modules import an optional dependency at module scope without a "
        f"module-scope require(): {unguarded}")


def test_the_guard_check_is_not_vacuous(tmp_path):
    """A negative control: the check above must actually fire."""
    optional = set(caps.optional_modules())
    guarded = ast.parse("from .capabilities import require\n"
                        "require('stabilizer_backend')\n"
                        "import stim\n")
    bare = ast.parse("import stim\n")
    assert _module_scope_optional_imports(bare, optional) == {"stim"}
    assert not _calls_require_at_module_scope(bare)
    assert _calls_require_at_module_scope(guarded)
    # An import inside a function is not module scope and needs no guard.
    lazy = ast.parse("def f():\n    import stim\n    return stim\n")
    assert _module_scope_optional_imports(lazy, optional) == set()


def test_every_entry_point_resolves_to_something_in_the_tree():
    """`entry_points` is documentation, and undocumented drift is how it rots.

    The field tells a reader where a capability is actually used; nothing
    checked it, so a renamed or deleted entry point stayed listed. Each entry
    is a dotted path under ``clifford_qc``; its longest module prefix must be a
    real file, and any remaining attributes must exist in that file's AST.
    """
    unresolved = {}
    for name, record in caps.CAPABILITIES.items():
        for entry in record.entry_points:
            parts = entry.split(".")
            for cut in range(len(parts), 0, -1):
                module = PACKAGE.joinpath(*parts[:cut])
                candidate = module.with_suffix(".py")
                if candidate.exists():
                    break
                if (module / "__init__.py").exists():
                    candidate = module / "__init__.py"
                    break
            else:
                unresolved[f"{name}:{entry}"] = "no module prefix exists"
                continue
            tree = ast.parse(candidate.read_text(encoding="utf-8"))
            defined = {node.name for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                            ast.ClassDef))}
            for attribute in parts[cut:]:
                if attribute not in defined:
                    unresolved[f"{name}:{entry}"] = f"{attribute!r} not in {candidate.name}"
    assert unresolved == {}, f"capability entry points that do not resolve: {unresolved}"


# ------------------------------------------------------------------- the CLI
def test_the_cli_reports_every_capability(capsys):
    assert caps._main([]) == 0
    printed = capsys.readouterr().out
    for name in caps.CAPABILITIES:
        assert name in printed
    assert "clifford_qc optional capabilities" in printed


def test_require_flag_decides_the_exit_code(capsys):
    """The contract a CI job leans on, which nothing exercised.

    `--require` is the reason this module has a command line at all: a job says
    which environment it believes it is in and fails loudly when it is wrong.
    An unconditional `return 0` would have shipped silently.
    """
    present = [n for n in caps.CAPABILITIES if caps.available(n)]
    absent = [n for n in caps.CAPABILITIES if not caps.available(n)]
    if present:
        assert caps._main(["--require", present[0]]) == 0
        assert "REQUIRED but absent" not in capsys.readouterr().out
    if absent:
        assert caps._main(["--require", absent[0]]) == 1
        out = capsys.readouterr().out
        assert f"REQUIRED but absent: {absent[0]}" in out
        assert caps.install_hint(absent[0]) in out
    if present and absent:
        # One unmet requirement is enough to fail the whole invocation.
        assert caps._main(["--require", present[0], "--require", absent[0]]) == 1


def test_an_unknown_required_capability_is_a_usage_error():
    with pytest.raises(SystemExit) as raised:
        caps._main(["--require", "teleportation"])
    assert raised.value.code == 2  # argparse usage error, not a silent pass


# --------------------------------------------------------- exported surface
def test_the_public_surface_rejects_unknown_names_consistently():
    """One catchable error type across the exported functions."""
    for function in (caps.available, caps.missing_modules, caps.install_hint):
        with pytest.raises(ValueError, match="unknown capability"):
            function("bogus")


def test_format_report_handles_an_empty_report():
    """`report` is a parameter, so a caller may filter it down to nothing."""
    assert caps.format_report({}) == ""


def test_a_namespace_package_does_not_count_as_available(tmp_path, monkeypatch):
    """A bare directory on sys.path is not the dependency it is named after.

    `find_spec` synthesises a namespace-package spec with no loader for any
    directory carrying the right name, so without the loader check a stray
    `pyzx/` in the working directory reports the bridge as available and the
    caller meets `No module named 'pyzx.circuit'` further in.
    """
    victim = next(iter(caps.CAPABILITIES["pyzx_bridge"].modules))
    (tmp_path / victim).mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    caps._importable.cache_clear()
    try:
        import importlib.util
        assert importlib.util.find_spec(victim) is not None  # the trap
        assert not caps._importable(victim)  # ... which we do not fall into
    finally:
        caps._importable.cache_clear()
