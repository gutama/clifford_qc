"""Small helpers for reproducibility gates on versioned JSON records."""

from __future__ import annotations

import functools
import math
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Mapping


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True,
            timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def execution_provenance() -> dict[str, Any]:
    """JSON-safe execution identity for benchmark and manuscript records.

    The block records the source revision, package/runtime versions, platform,
    numerical-library configuration, and a digest of the complete installed
    distribution set.  Volatile fields are intentionally metadata: record
    comparisons ignore ``provenance`` while still comparing every scientific
    and resource field.
    """
    import numpy as np

    from . import __version__

    sha = (os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
           or _git_output("rev-parse", "HEAD") or "unknown")
    dirty_text = _git_output("status", "--porcelain")
    distributions = sorted(
        f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
        for dist in importlib.metadata.distributions())
    freeze_digest = hashlib.sha256(
        "\n".join(distributions).encode("utf-8")).hexdigest()

    try:
        blas = np.__config__.show(mode="dicts")
    except TypeError:  # NumPy 1.x has only the printing form of show().
        stream = StringIO()
        with redirect_stdout(stream):
            np.__config__.show()
        blas = {"summary": stream.getvalue().strip()}

    return {
        "schema": "clifford_qc.execution_provenance.v1",
        "git_sha": sha,
        "git_dirty": None if dirty_text is None else bool(dirty_text),
        "clifford_qc": __version__,
        "python": platform.python_version(),
        "dependencies": {
            name: _version(name)
            for name in ("numpy", "scipy", "pyscf", "openfermion",
                         "openfermionpyscf", "stim")
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "blas_lapack": blas,
        "environment_sha256": freeze_digest,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


_environment_guarded = False


def _guard_environment(allow_environment_migration: bool) -> None:
    """Refuse to stamp under an environment the committed records disown.

    Surveyed once per process: a producer that stamps ten thousand JSONL rows
    reads the committed set once, and neither that set nor the installed
    versions move while it runs.  A run that is authorized to migrate is not
    memoized, so an unauthorized stamp later in the same process is still
    checked.
    """
    global _environment_guarded
    if _environment_guarded or allow_environment_migration:
        return
    from . import record_environment
    if record_environment.migration_allowed():
        return
    record_environment.guard()
    _environment_guarded = True


def stamp_record(record: dict[str, Any],
                 provenance: dict[str, Any] | None = None,
                 *,
                 allow_environment_migration: bool = False) -> dict[str, Any]:
    """Return a shallow copy of one record with execution provenance attached.

    The stamp is where the environment contract is enforced, because it is the
    one boundary every producer crosses.  A record built under a Python or
    library version no committed record declares cannot be reproduced beside
    its siblings, so stamping it raises
    :class:`clifford_qc.record_environment.UndeclaredEnvironment` here -- a
    rebuild -- rather than being discovered by a cross-record gate once it is
    committed -- a retraction.  Pass ``allow_environment_migration``, or set
    ``CLIFFORD_QC_ALLOW_ENVIRONMENT_MIGRATION=1``, when the run is a deliberate
    migration of the whole record set.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be a mapping object")
    _guard_environment(allow_environment_migration)
    stamped = dict(record)
    stamped["provenance"] = (execution_provenance() if provenance is None
                             else provenance)
    # Fail at the producer boundary, not after a long benchmark has written an
    # unreadable artifact.
    json.dumps(stamped["provenance"])
    return stamped


def compare_json_records(
        expected: Any,
        actual: Any,
        path: str = "$",
        *,
        atol: float = 1e-11,
        rtol: float = 1e-11,
        ignored_keys: frozenset[str] = frozenset({"provenance"}),
        key_tolerances: Mapping[str, tuple[float, float]] | None = None,
        path_tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Compare JSON-like values with exact discrete and tolerant float fields.

    Keys, list lengths, strings, booleans, integers, and nulls are exact.
    Floating fields alone use :func:`math.isclose`, which isolates harmless
    BLAS/LAPACK last-bit changes without weakening resource or replica counts.

    ``key_tolerances`` maps a field name to its own ``(rtol, atol)``.  It exists
    for quantities whose *own* magnitude is not the right error scale: a small
    difference of two large energies loses most of its significant digits to
    cancellation, so a relative tolerance on the difference is far stricter than
    the arithmetic that produced it can honour.  Such a field needs an absolute
    tolerance set by the energies it came from, not by the residue.  Naming the
    field explicitly keeps every other float on the tight default.

    ``path_tolerances`` maps a complete comparison path to its own
    ``(rtol, atol)`` and takes precedence over ``key_tolerances``. It is
    for mixed records in which regenerated numerical outputs need a
    cross-machine floor while fixed numerical contracts with the same terminal
    key must remain exact.
    """
    problems: list[str] = []
    overrides = dict(key_tolerances or {})
    path_overrides = dict(path_tolerances or {})

    def tolerances(here: str) -> tuple[float, float]:
        if here in path_overrides:
            return path_overrides[here]
        key = here.rsplit(".", 1)[-1]
        return overrides.get(key, (rtol, atol))

    def compare(left: Any, right: Any, here: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict):
                problems.append(
                    f"{here}: expected object, got {type(right).__name__}")
                return
            left_keys = set(left) - ignored_keys
            right_keys = set(right) - ignored_keys
            missing = sorted(left_keys - right_keys)
            extra = sorted(right_keys - left_keys)
            if missing:
                problems.append(f"{here}: missing keys {missing}")
            if extra:
                problems.append(f"{here}: unexpected keys {extra}")
            for key in sorted(left_keys & right_keys):
                compare(left[key], right[key], f"{here}.{key}")
            return

        if isinstance(left, list):
            if not isinstance(right, list):
                problems.append(
                    f"{here}: expected array, got {type(right).__name__}")
                return
            if len(left) != len(right):
                problems.append(
                    f"{here}: expected {len(left)} entries, got {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{here}[{index}]")
            return

        # bool is a subclass of int, so exact discrete types come first.
        if left is None or isinstance(left, (bool, str, int)):
            if type(right) is not type(left) or right != left:
                problems.append(f"{here}: expected {left!r}, got {right!r}")
            return

        if isinstance(left, float):
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                problems.append(
                    f"{here}: expected floating value, got {right!r}")
                return
            field_rtol, field_atol = tolerances(here)
            if not math.isclose(
                    left, float(right), rel_tol=field_rtol, abs_tol=field_atol):
                problems.append(
                    f"{here}: expected {left:.17g}, got {float(right):.17g}")
            return

        problems.append(
            f"{here}: unsupported value type {type(left).__name__}")

    compare(expected, actual, path)
    return problems


def sampling_stream_mismatch(
        record: Any, *, packages: tuple[str, ...] = ("numpy",),
) -> list[str]:
    """Report named packages this environment cannot reproduce ``record`` under.

    A record whose values come from finite-shot sampling followed by a
    projected eigensolve is only value-comparable under the library versions
    that produced it.  A named package that is *missing* counts as well as one
    at a different version: naming it is the caller's statement that its
    absence moves the values rather than only the speed, as SciPy's does when
    the optimizer falls back to the pure-Python one.  A record that declares
    ``null`` for a package is a different case -- it was produced without it,
    and claims nothing.

    Two distinct mechanisms put a differing version out of reach, and a version
    stamp is the cheapest thing that detects either:

    * a NumPy release may change the bundled BLAS/LAPACK, so an ill-conditioned
      generalized eigenproblem lands on a different solution -- the shots are
      identical, the answer is not, and the difference can be amplified far
      above last-bit noise;
    * ``numpy.random.Generator`` carries no cross-version bit-stream guarantee
      (NEP 19 froze ``RandomState`` for that purpose), so a release is also
      free to change the draws themselves.

    Either way the result is deterministic within a version and unequal between
    them, which is invisible in a value diff: it reports a shifted mean or
    quantile and invites the reader to widen a tolerance.  Widening is the
    wrong response, because nothing here is noisy in the run-to-run sense.
    Callers should surface this list *before* any numeric comparison so the
    diagnosis names the environment rather than the arithmetic.
    """
    if not isinstance(record, dict):
        return []
    stamped = record.get("provenance")
    if not isinstance(stamped, dict):
        return []
    declared = stamped.get("dependencies")
    if not isinstance(declared, dict):
        return []
    problems: list[str] = []
    for package in packages:
        expected = declared.get(package)
        if expected is None:
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual is None:
            problems.append(
                f"{package} is not installed, but {expected} produced this "
                "record; a package the caller names here is one whose absence "
                "changes the values rather than only the speed, so a rebuild "
                "answers a different question rather than verifying this record"
            )
        elif actual != expected:
            problems.append(
                f"{package} {actual} differs from the {expected} that produced "
                "this record; finite-shot values are not comparable across "
                "versions, which may change the bundled BLAS/LAPACK an "
                "ill-conditioned eigensolve depends on or the sampling stream "
                "itself, so a rebuild here answers a different question rather "
                "than verifying this record"
            )
    return problems


# The floor a committed-record comparison actually has to clear.
#
# These records reproduce bit for bit on one machine. Rebuilding
# ``finite_shot_rethink.json`` and diffing every compared field against the
# committed one drifts by exactly zero, at ``OMP_NUM_THREADS`` 1 and 4 alike.
# So the standing position in these checkers -- that widening is the wrong
# response because nothing here is noisy in the run-to-run sense -- is correct,
# and this constant does not contradict it.
#
# What it adds is that run-to-run determinism is not machine-to-machine
# determinism, and a CI gate rebuilding a record committed from a different
# machine is making the second comparison rather than the first. OpenBLAS
# selects kernels by instruction set as well as by thread count, so a differing
# runner sums a reduction in a different order. Measured on ``main`` across five
# merges, the surviving drift is 3.4e-12 to 1.0e-11 absolute and up to 9.8e-11
# relative, on millihartree quantities of order 0.07 to 2.6. A gate set at
# 1e-12 therefore sits inside its own noise floor and resolves on which runner
# it drew, which is what made three of them red for five merges and one of them
# flip red to green on unchanged inputs.
#
# 1e-9 clears the worst observed drift by an order of magnitude. On the
# measured 0.07--2.6 mHa energy-difference fields it still rejects a 1e-8 mHa
# change. That statement is deliberately local, not a blanket guarantee:
# math.isclose's relative leg scales with large-valued fields. Fixed numerical
# contracts therefore use exact key/path overrides at the checker call sites.
#
# It is for committed-versus-rebuilt comparisons only. A record re-derived
# against its own contents in one process crosses no machine boundary and stays
# exact -- ``check_protocol_cost``'s verdict re-derivation and
# ``check_mapping_axis``'s QR3 summary both keep their tight comparison.
CROSS_MACHINE_ATOL = 1e-9
CROSS_MACHINE_RTOL = 1e-9


def guarded_contract_problems(inner):
    """Wrap a contract function so a malformed record is reported, not raised.

    A checker is handed the one object that might be broken, so it has to
    survive being handed a broken one: a traceback names a line, while a
    returned problem names the record. Every checker here had reached the same
    conclusion separately and written its own copy, which is how a third one
    could ship without the guard at all -- the convention existed but nothing
    held it. One decorated definition is what makes it a convention.

    The caught set is deliberately the shapes a wrong-typed record produces --
    a missing attribute, an absent key, a comparison or membership test across
    types, an unparseable value, an overflowing float. Anything else is a bug
    in the contract itself and should still raise.
    """
    @functools.wraps(inner)
    def guarded(record: Any) -> list[str]:
        try:
            return inner(record)
        except (AttributeError, KeyError, TypeError,
                ValueError, OverflowError) as exc:
            return [f"malformed record reached a guarded checker path: {exc}"]

    guarded.unguarded = inner
    return guarded


__all__ = ["CROSS_MACHINE_ATOL", "CROSS_MACHINE_RTOL",
           "compare_json_records", "execution_provenance",
           "guarded_contract_problems", "sampling_stream_mismatch",
           "stamp_record"]
