"""The environment-section count check in ``benchmarks/check_docs.py``.

The quoted ``pytest`` pair drifted twice: once by 147 tests with nothing
watching, then by another 75 *while* a check existed, because that check
compared the quoted skip count against the number of files carrying a
``pytest.importorskip``. Those two agree only while every guard is module
level. Once a test guards an individual case, its file is still collected and
contributes one skip per guarded test rather than one per file, and the
comparison matches neither quantity -- so the check reported "unverified" in
every environment and enforced nothing.

The distinction it now draws is therefore the thing to pin.
"""

import ast

from benchmarks.check_docs import (
    _importorskip_modules,
    dropped_test_files,
    guarded_modules,
)


def _modules(source: str):
    return _importorskip_modules(ast.parse(source))


def test_bare_module_level_guard_drops_the_file():
    assert _modules('import pytest\npytest.importorskip("stim")\n') == [
        ("stim", True)
    ]


def test_bound_module_level_guard_also_drops_the_file():
    """``mod = pytest.importorskip(...)`` runs at import, same as a bare call.

    Missing this form is what made the first repair undercount: it found 4 of
    the 10 files the documented environment actually drops.
    """
    assert _modules('import pytest\nstim = pytest.importorskip("stim")\n') == [
        ("stim", True)
    ]


def test_in_test_guard_does_not_drop_the_file():
    source = (
        "import pytest\n"
        "def test_thing():\n"
        "    pytest.importorskip('stim')\n"
    )
    assert _modules(source) == [("stim", False)]


def test_guard_inside_a_class_body_does_not_drop_the_file():
    source = (
        "import pytest\n"
        "class TestGroup:\n"
        "    def test_thing(self):\n"
        "        pytest.importorskip('stim')\n"
    )
    assert _modules(source) == [("stim", False)]


def test_module_level_conditional_guard_drops_the_file():
    """A guard under a top-level ``if`` still runs at import time."""
    source = (
        "import pytest\n"
        "import sys\n"
        "if sys.version_info >= (3, 10):\n"
        "    pytest.importorskip('stim')\n"
    )
    assert _modules(source) == [("stim", True)]


def test_both_kinds_in_one_file_are_distinguished():
    source = (
        "import pytest\n"
        "pytest.importorskip('pyzx')\n"
        "def test_thing():\n"
        "    pytest.importorskip('stim')\n"
    )
    assert sorted(_modules(source)) == [("pyzx", True), ("stim", False)]


def test_non_literal_and_unrelated_calls_are_ignored():
    source = (
        "import pytest\n"
        "name = 'stim'\n"
        "pytest.importorskip(name)\n"
        "pytest.mark.skipif('stim')\n"
    )
    assert _modules(source) == []


# ---------------------------------------------------------------------------
# Against the real tree


def test_dropped_files_are_a_subset_of_guarded_files():
    """Every dropped file is dropped for a module the environment lacks."""
    guarded = guarded_modules()
    for name, module in dropped_test_files():
        assert module in guarded, name


def test_this_file_is_never_dropped():
    """It guards nothing, so it must be collected in every environment."""
    assert __file__.split("/")[-1] not in {
        name for name, _ in dropped_test_files()
    }


def test_in_test_guards_keep_their_files_collected():
    """``test_fermion_mapping`` guards individual tests, so it is collected.

    It is the file whose per-test guards broke the old comparison, so it is
    the one worth naming: whatever the environment, it must never appear as
    dropped at collection.
    """
    assert "test_fermion_mapping.py" not in {
        name for name, _ in dropped_test_files()
    }
