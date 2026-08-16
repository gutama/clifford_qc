"""Verify REPRODUCING.md still describes the code it claims to reproduce.

Three separate documentation drifts reached a submission-candidate manuscript
on this branch: the per-matrix `summarize.py` commands regenerated only the
CSV (which is *how* the Markdown summaries went stale), the predeclared-
parameter section quoted one global delta while five certification experiments
used their own, and the environment check understated the test count. None
broke a published number; all three broke the reproduction contract, and
nothing detected them -- the record, figure, and table checkers all look at
artifacts, not at the prose telling you how to rebuild them.

This closes that gap. It checks that:

  1. every ``python benchmarks/X.py`` command names a script that exists;
  2. every long flag in those commands is one the script actually accepts;
  3. the predeclared delta/eps table matches the constants in each script;
  4. every ``run_*.py`` benchmark is documented somewhere;
  5. every committed record is named by the document;
  6. the quoted ``pytest`` test count matches what the suite collects.

Constants are read with ``ast`` rather than by importing, so a check never
executes benchmark code.

    python benchmarks/check_docs.py

Exits nonzero on any drift.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "REPRODUCING.md"
BENCH = ROOT / "benchmarks"
DATA = BENCH / "reference_results"

# Records that are outputs of another record's script rather than of their own
# documented command, or that the document names indirectly.
RECORD_ALIASES = {"h4_exact_trajectory.json": "reproduce_exact_h4.py"}


def bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def commands(text: str) -> list[str]:
    """Shell commands from fenced bash blocks, line continuations joined."""
    out = []
    for block in bash_blocks(text):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def script_flags(script: Path) -> tuple[set[str] | None, str | None]:
    """Long flags the script accepts, plus a reason if they cannot be read.

    A script whose module-scope imports need an optional extra (``stim``,
    ``pyscf``, ...) cannot answer ``--help`` in a bare environment. That is a
    missing dependency, not documentation drift, so it is reported as a skip
    rather than a failure -- CI lints the docs without installing every extra.
    Any other nonzero exit is a genuine defect and fails.
    """
    done = subprocess.run([sys.executable, str(script), "--help"],
                          capture_output=True, text=True)
    if done.returncode == 0:
        return set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", done.stdout)), None
    missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", done.stderr)
    if missing:
        return None, f"optional dependency {missing.group(1)!r} not installed"
    detail = (done.stderr or done.stdout).strip().splitlines()
    return None, detail[-1] if detail else f"exit {done.returncode}"


def module_constants(script: Path) -> dict[str, object]:
    """Module-level literal assignments, read statically (no import)."""
    tree = ast.parse(script.read_text())
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def check_commands(text: str, problems: list[str], skipped: list[str]) -> None:
    for cmd in commands(text):
        m = re.match(r"python3? (benchmarks/[\w.]+\.py)", cmd)
        if not m:
            continue
        script = ROOT / m.group(1)
        if not script.exists():
            problems.append(f"documented command names a missing script: {m.group(1)}")
            continue
        used = set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", cmd))
        if not used:
            continue
        accepted, why = script_flags(script)
        if accepted is None:
            if why and why.startswith("optional dependency"):
                skipped.append(f"{m.group(1)}: flags unverified ({why})")
            else:
                problems.append(f"{m.group(1)} --help fails: {why}")
            continue
        for flag in sorted(used - accepted):
            problems.append(f"{m.group(1)} does not accept {flag} "
                            f"(documented in: {cmd[:70]})")


def check_parameter_table(text: str, problems: list[str]) -> None:
    """The predeclared delta/eps table must match each script's constants."""
    rows = re.findall(r"^\| `(run_[\w]+\.py)` \| ([^|]+) \| ([^|]+) \|", text, re.M)
    if not rows:
        problems.append("predeclared-parameter table not found in REPRODUCING.md")
        return
    for name, delta_cell, eps_cell in rows:
        script = BENCH / name
        if not script.exists():
            problems.append(f"parameter table names a missing script: {name}")
            continue
        const = module_constants(script)
        documented = set(re.findall(r"\d+\.\d+", delta_cell))
        actual = set()
        for key in ("DELTA", "DELTAS", "TRAJ_DELTA"):
            if key in const:
                v = const[key]
                actual |= {f"{x:g}" if isinstance(x, float) else str(x)
                           for x in (v if isinstance(v, tuple) else (v,))}
        # compare as floats so 0.10 and 0.1 agree
        if actual and {float(x) for x in documented} != {float(x) for x in actual}:
            problems.append(f"{name}: table says delta {sorted(documented)}, "
                            f"code has {sorted(actual)}")
        documented_eps = {float(x) for x in re.findall(r"\d+\.\d+", eps_cell)}
        actual_eps = set()
        for key in ("EPS", "EPS_GRID"):
            if key in const:
                v = const[key]
                actual_eps |= set(v if isinstance(v, tuple) else (v,))
        if actual_eps and documented_eps != actual_eps:
            problems.append(f"{name}: table says eps {sorted(documented_eps)}, "
                            f"code has {sorted(actual_eps)}")


def check_coverage(text: str, problems: list[str]) -> None:
    for script in sorted(BENCH.glob("run_*.py")):
        if script.name not in text:
            problems.append(f"{script.name} is not documented in REPRODUCING.md")
    for record in sorted(DATA.glob("*.json*")):
        name = record.name
        if name in text or RECORD_ALIASES.get(name, "\0") in text:
            continue
        stem = name.split(".")[0]
        if stem in text:
            continue
        problems.append(f"committed record {name} is not named in REPRODUCING.md")


def _importorskip_modules(tree: ast.AST) -> list[tuple[str, bool]]:
    """``(module, is_module_level)`` for every ``pytest.importorskip`` call.

    Whether the call sits at module level is the whole distinction this
    function exists to draw: a module-level guard makes pytest drop the file
    at *collection*, so its tests never reach the collected count and the file
    contributes exactly one skip. A guard inside a test body does neither --
    the file is collected normally and only the guarded tests skip, one skip
    each. Conflating the two is what let the environment section drift.
    """
    def calls(node: ast.AST):
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not (isinstance(func, ast.Attribute)
                    and func.attr == "importorskip"):
                continue
            if child.args and isinstance(child.args[0], ast.Constant):
                if isinstance(child.args[0].value, str):
                    yield child

    # Module level means "reached when the file is imported", which covers a
    # bare call and the ``mod = pytest.importorskip(...)`` binding alike -- any
    # top-level statement that is not a deferred function or class body.
    module_level = {
        id(call)
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef))
        for call in calls(stmt)
    }
    return [(call.args[0].value, id(call) in module_level)
            for call in calls(tree)]


def guarded_modules() -> set[str]:
    """Every module name any test guards with ``pytest.importorskip``."""
    names: set[str] = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        names.update(name for name, _ in _importorskip_modules(tree))
    return names


def dropped_test_files() -> list[tuple[str, str]]:
    """(file, module) for each test file pytest drops at collection.

    Only *module-level* guards count. Read from the tests rather than
    hardcoded, so the list cannot go stale.
    """
    out = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for mod, at_module_level in _importorskip_modules(tree):
            if at_module_level and importlib.util.find_spec(mod) is None:
                out.append((path.name, mod))
                break
    return out


def _documented_extras(text: str) -> set[str] | None:
    """The extras the environment section tells the reader to install."""
    m = re.search(r"pip install -e \.?\[([^\]]+)\]", text)
    if not m:
        return None
    return {part.strip() for part in m.group(1).split(",") if part.strip()}


def _modules_provided_by(extras: set[str]) -> set[str]:
    """Top-level module names the given extras (plus base deps) install.

    Read out of ``pyproject.toml`` so adding an extra cannot silently change
    what this check believes the documented environment contains.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        return set()
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data.get("project", {})
    requirements = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        requirements.extend(optional.get(extra, []))
    modules = set()
    for requirement in requirements:
        # "openfermion>=1.6" / "numpy" / "clifford-qc[stim,...]" -> base name
        name = re.split(r"[<>=!\[;\s]", requirement, maxsplit=1)[0].strip()
        if name:
            modules.add(name.replace("-", "_").lower())
    return modules


def check_test_count(text: str, problems: list[str], skipped: list[str],
                     required: bool = False) -> None:
    """The quoted ``pytest`` line must match what the suite actually collects.

    This drifted by 147 tests before anything noticed, because every other
    check here looks at benchmark scripts and records -- not at the environment
    section a reader runs first. It then drifted by another 75, because the
    check itself compared the quoted *skip* count against the number of files
    carrying a guard. Those two agreed only while every guard was module level;
    once tests began guarding individual cases, the comparison matched neither
    the files pytest drops nor the skips it reports, so the check excused
    itself as "unverified" in every environment and enforced nothing.

    What is checkable without running the suite is the arithmetic relating the
    two quoted numbers. With ``D`` files dropped at collection, each
    contributing one skip, the remaining ``S - D`` skips are per-test and *are*
    collected, so

        collected == passed + (skipped - D)

    holds whenever the suite is green. Both ``collected`` and ``D`` are
    computed here, so a drift in either quoted number breaks the identity.

    The identity is environment-specific, so it is only enforced when the
    installed extras are the ones the document tells the reader to install --
    determined by importability of the guarded modules, not by counting files.
    Collection only, so no test is ever executed.
    """
    m = re.search(r"^\s*pytest\s+#\s*(\d+) passed, (\d+) skipped", text, re.M)
    if not m:
        problems.append("environment section does not quote a pytest count")
        return
    want_passed, want_skipped = int(m.group(1)), int(m.group(2))

    extras = _documented_extras(text)
    provided = _modules_provided_by(extras) if extras is not None else set()
    guarded = guarded_modules()
    # A guarded module counts as expected-present when the documented extras
    # supply it, or when it ships inside this package (``clifford_qc.*``).
    expected_absent = {
        name for name in guarded
        if not name.startswith("clifford_qc")
        and name.split(".")[0].lower() not in provided
    }
    actual_absent = {
        name for name in guarded if importlib.util.find_spec(name) is None
    }
    if not provided or actual_absent != expected_absent:
        unexpected = sorted(actual_absent - expected_absent)
        missing = sorted(expected_absent - actual_absent)
        message = (
            f"installed extras differ from the documented "
            f"{sorted(extras) if extras else '(unparsed)'} install: "
            f"absent but expected present {unexpected or 'none'}; "
            f"present but expected absent {missing or 'none'}"
        )
        # Under --require-test-count the caller has promised the documented
        # environment, so a mismatch is the failure -- otherwise the check
        # that owns this contract would quietly excuse itself in the very
        # job meant to enforce it.
        (problems if required else skipped).append(
            f"test count {'not enforceable' if required else 'unverified'} "
            f"({message})")
        return

    dropped = dropped_test_files()
    if want_skipped < len(dropped):
        problems.append(
            f"REPRODUCING.md quotes {want_skipped} skipped, but {len(dropped)} "
            f"test file(s) are dropped at collection: "
            + ", ".join(f"{f} ({mod})" for f, mod in dropped))
        return

    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                          cwd=ROOT, capture_output=True, text=True)
    found = re.search(r"^(\d+) tests? collected", done.stdout, re.M)
    if not found:
        detail = (done.stderr or done.stdout).strip().splitlines()
        skipped.append("test count unverified "
                       f"({detail[-1] if detail else f'exit {done.returncode}'})")
        return
    collected = int(found.group(1))
    expected = want_passed + (want_skipped - len(dropped))
    if collected != expected:
        problems.append(
            f"REPRODUCING.md quotes {want_passed} passed, {want_skipped} "
            f"skipped with {len(dropped)} file(s) dropped at collection, "
            f"implying {expected} collected; the suite collects {collected}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-test-count", action="store_true",
                        help="fail, rather than skip, when the installed "
                             "extras do not match the environment "
                             "REPRODUCING.md quotes its test count for")
    args = parser.parse_args(argv)

    text = DOC.read_text()
    problems: list[str] = []
    skipped: list[str] = []
    check_commands(text, problems, skipped)
    check_parameter_table(text, problems)
    check_coverage(text, problems)
    check_test_count(text, problems, skipped, required=args.require_test_count)
    for note in skipped:
        print(f"SKIP {note}")
    for p in problems:
        print(f"FAIL {p}")
    print("OK (REPRODUCING.md matches the code)" if not problems
          else f"\n{len(problems)} drift(s) between REPRODUCING.md and the code")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
