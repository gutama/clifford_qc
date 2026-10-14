"""Validate versioned evidence metadata and historical record digest bindings."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evidence_metadata import check_records


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        problems = check_records(ROOT)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems = [str(exc)]
    for problem in problems:
        print(f"FAIL evidence: {problem}")
    if not problems:
        print("OK: versioned evidence declarations and historical digest bindings")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
