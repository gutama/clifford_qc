"""The process-wide kernel memos are observable, clearable and configurable.

Word products and blade masks are memoized for the whole process. A full table
is hundreds of megabytes that no bank's resource ledger owns, and nothing used
to report it or let a memory-bound run shrink it. The default size is part of
committed storage measurements and must not move; what is pinned here is that
the tables can be read, emptied and resized without changing a single result.
"""

import os
import subprocess
import sys

import pytest

import clifford_qc
from clifford_qc import X, Y, Z, clear_kernel_caches, kernel_cache_info
from clifford_qc.pauli_kernel import (
    DEFAULT_KERNEL_CACHE_SIZE, KERNEL_CACHE_SIZE_ENV, word_mul_reference,
)

SCRIPT = (
    "import json, clifford_qc as cq\n"
    "from clifford_qc import X, Y, Z\n"
    "a = (X(3, 0) * Y(3, 1) + 0.5 * Z(3, 2)) * (Y(3, 0) + 1j * X(3, 2))\n"
    "print(json.dumps({'terms': sorted((k, v.real, v.imag) for k, v in a.terms.items()),\n"
    "                  'grades': sorted(a.grades()),\n"
    "                  'info': cq.kernel_cache_info()}))\n"
)


def run_with(size):
    env = dict(os.environ)
    env.pop(KERNEL_CACHE_SIZE_ENV, None)
    if size is not None:
        env[KERNEL_CACHE_SIZE_ENV] = size
    return subprocess.run([sys.executable, "-c", SCRIPT], env=env, text=True,
                          capture_output=True,
                          cwd=os.path.dirname(os.path.dirname(clifford_qc.__file__)))


def test_both_memos_are_reported_and_fill_as_the_kernel_is_used():
    clear_kernel_caches()
    assert {name: row["currsize"] for name, row in kernel_cache_info().items()} == {
        "blade_masks": 0, "word_products": 0}
    product = X(2, 0) * Y(2, 1)
    product.grades()
    info = kernel_cache_info()
    assert info["word_products"]["currsize"] >= 1
    assert info["blade_masks"]["currsize"] >= 1
    assert set(info["word_products"]) == {"hits", "misses", "maxsize", "currsize"}


def test_clearing_changes_no_result():
    before = (X(3, 0) * Y(3, 1)) * (Z(3, 1) + Y(3, 2))
    clear_kernel_caches()
    after = (X(3, 0) * Y(3, 1)) * (Z(3, 1) + Y(3, 2))
    assert after.terms == before.terms
    assert list(after.terms) == list(before.terms)  # same generation order too
    for code in after.terms:
        assert clifford_qc.word_mul(3, code, code) == word_mul_reference(3, code, code)


def test_the_default_size_is_the_one_committed_ledgers_were_measured_with():
    """Storage records count boxed integers this table shares; do not move it."""
    assert DEFAULT_KERNEL_CACHE_SIZE == 1_000_000
    if os.environ.get(KERNEL_CACHE_SIZE_ENV) is None:
        assert {row["maxsize"] for row in kernel_cache_info().values()} == {1_000_000}


@pytest.mark.parametrize("size", ["0", "16"])
def test_a_smaller_or_disabled_table_gives_identical_results(size):
    import json

    default, resized = run_with(None), run_with(size)
    assert default.returncode == 0 and resized.returncode == 0, resized.stderr
    default, resized = json.loads(default.stdout), json.loads(resized.stdout)
    assert resized["terms"] == default["terms"]
    assert resized["grades"] == default["grades"]
    assert {row["maxsize"] for row in resized["info"].values()} == {int(size)}
    assert all(row["currsize"] <= int(size) for row in resized["info"].values())


@pytest.mark.parametrize("size", ["-1", "many", "1.5"])
def test_an_unusable_size_is_refused_at_import(size):
    done = run_with(size)
    assert done.returncode != 0
    assert KERNEL_CACHE_SIZE_ENV in done.stderr
