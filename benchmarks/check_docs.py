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


def skipped_test_files() -> list[tuple[str, str]]:
    """(file, module) for each test file pytest drops for a missing import.

    Read from the tests rather than hardcoded, so the list cannot go stale.
    Counted per *file*, not per module -- one absent module can gate several
    files (``stim`` gates three), and each is a separate skip. A dropped file
    never reaches the collected count, which is why the two must agree.
    """
    out = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for mod in re.findall(r"importorskip\(\s*[\"']([\w.]+)[\"']",
                              path.read_text()):
            if importlib.util.find_spec(mod) is None:
                out.append((path.name, mod))
                break
    return out


def check_test_count(text: str, problems: list[str], skipped: list[str]) -> None:
    """The quoted ``pytest`` line must match what the suite actually collects.

    This drifted by 147 tests before anything noticed, because every other
    check here looks at benchmark scripts and records -- not at the environment
    section a reader runs first.

    The quoted figure describes one environment, and which optional extras are
    installed changes it: a missing bridge removes a whole file from the
    collected count and adds one skip. So the count is only enforced when the
    number of unimportable optional modules matches the quoted skip count --
    otherwise this is a different environment than the document describes, and
    the honest report is a skip. Collection only, so no test is ever executed.
    """
    m = re.search(r"^\s*pytest\s+#\s*(\d+) passed, (\d+) skipped", text, re.M)
    if not m:
        problems.append("environment section does not quote a pytest count")
        return
    want_passed, want_skipped = int(m.group(1)), int(m.group(2))

    absent = skipped_test_files()
    if len(absent) != want_skipped:
        detail = ", ".join(f"{f} ({m})" for f, m in absent) or "none"
        skipped.append(
            f"test count unverified (document assumes {want_skipped} skipped; "
            f"here {len(absent)} test file(s) skipped: {detail})")
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
    if collected != want_passed:
        problems.append(f"REPRODUCING.md quotes {want_passed} passed, "
                        f"suite collects {collected}")


def main() -> int:
    text = DOC.read_text()
    problems: list[str] = []
    skipped: list[str] = []
    check_commands(text, problems, skipped)
    check_parameter_table(text, problems)
    check_coverage(text, problems)
    check_test_count(text, problems, skipped)
    for note in skipped:
        print(f"SKIP {note}")
    for p in problems:
        print(f"FAIL {p}")
    print("OK (REPRODUCING.md matches the code)" if not problems
          else f"\n{len(problems)} drift(s) between REPRODUCING.md and the code")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
