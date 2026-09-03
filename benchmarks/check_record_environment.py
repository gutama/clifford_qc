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
``pip install``.  That is also why the contract itself --
``clifford_qc/record_environment.py``, which this file is the command line for
-- is loaded from its path rather than imported: importing the package would
pull in NumPy, which is not installed yet.  The same contract is enforced at
the other end of a record's life by
:func:`clifford_qc.reproducibility.stamp_record`, which refuses to stamp a
record under an environment no committed record declares, so a producer cannot
quietly start a split that only this gate would notice.  The per-record
equivalent for use *inside* a gate, once the package is importable, is
:func:`clifford_qc.reproducibility.sampling_stream_mismatch`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CONTRACT = (Path(__file__).resolve().parent.parent
            / "clifford_qc" / "record_environment.py")


def _load_contract():
    """Return the shared contract, importing it normally where that works.

    Path-loading exists only because the two emitting modes run before ``pip
    install``, when ``import clifford_qc`` cannot succeed -- its package
    imports NumPy.  Wherever the package *is* importable, importing it is what
    keeps one module object in play: a second copy loaded by path would carry
    its own module state, so the gate and the producer guard could read
    different manifests and disagree about the same repository.
    """
    try:
        from clifford_qc import record_environment
        return record_environment
    except Exception:  # pragma: no cover - the pre-install path CI relies on
        pass
    spec = importlib.util.spec_from_file_location(
        "clifford_qc_record_environment", CONTRACT)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise SystemExit(f"cannot load the record environment contract: {CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_contract = _load_contract()

agreed = _contract.agreed
constraints = _contract.constraints
pending = _contract.pending
survey = _contract.survey
verify = _contract.verify

DATA = Path(__file__).resolve().parent / "reference_results"


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

    # Always, and on stderr: stdout is redirected into the files the install
    # reads, and an outstanding migration that only printed on failure would go
    # unmentioned for exactly as long as nothing else was wrong.
    for name, entry in sorted(pending().items()):
        declares = entry.get("declares", {}) if isinstance(entry, dict) else {}
        summary = ", ".join(f"{package} {version}"
                            for package, version in sorted(declares.items()))
        print(f"NOTE {name} still declares {summary or 'a superseded environment'}"
              " and is not counted; its migration is outstanding, see "
              "benchmarks/migrations/pending_environment_migration.json",
              file=sys.stderr)

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
