"""Shared candidate values preserve legacy types without upward imports."""
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from clifford_qc.ir import PauliWord
from clifford_qc.pool_operator import PoolOperator, is_odd_y

ROOT = Path(__file__).resolve().parents[1]


def checkout_subprocess(code):
    identity = '''
from pathlib import Path
import clifford_qc
assert Path(clifford_qc.__file__).resolve() == Path('clifford_qc/__init__.py').resolve()
'''
    return subprocess.run([sys.executable, "-c", code + identity], cwd=ROOT,
                          capture_output=True, text=True)


def test_old_pool_names_reexport_the_same_type_and_predicate():
    from clifford_qc.algorithms import pools
    assert pools.PoolOperator is PoolOperator
    assert pools.is_odd_y is is_odd_y
    value = PoolOperator("candidate", PauliWord.from_label("XYZ"))
    assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(FrozenInstanceError):
        value.label = "changed"
    # Old pickles reference the original module; that lookup stays available.
    legacy = b"cclifford_qc.algorithms.pools\nPoolOperator\n."
    assert pickle.loads(legacy) is PoolOperator


def test_pool_content_order_and_odd_y_filter_remain_compatible():
    from clifford_qc.algorithms.pools import all_words_pool, local_pool, odd_y_filter
    assert [(op.label, op.word.label) for op in local_pool(2)] == [
        ("Y0", "YI"), ("Z1Y0", "YZ"), ("Y1", "IY"), ("Z0Y1", "ZY")]
    assert [op.word.label for op in odd_y_filter(all_words_pool(2))] == [
        "YI", "IY", "XY", "YX", "YZ", "ZY"]
    assert all(type(op) is PoolOperator for op in local_pool(3))


def test_shared_pool_module_imports_without_solvers_or_chemistry():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('clifford_qc.algorithms', 'clifford_qc.subspace',
                                'openfermion', 'pyscf')):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from clifford_qc.pool_operator import PoolOperator, is_odd_y
from clifford_qc.ir import PauliWord
assert is_odd_y(PauliWord.from_label('Y'))
'''
    done = checkout_subprocess(code)
    assert done.returncode == 0, done.stdout + done.stderr


def test_chemistry_has_no_module_scope_algorithm_import():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "clifford_qc/models/chemistry.py").read_text(encoding="utf-8"))
    imports = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert "pool_operator" in imports
    assert not any("algorithms" in module for module in imports if module)


def test_optional_chemistry_import_and_pool_order_without_solvers():
    pytest.importorskip("openfermion")
    code = '''
import hashlib, importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('clifford_qc.algorithms', 'clifford_qc.subspace')):
            raise AssertionError('upward import: ' + fullname)
sys.meta_path.insert(0, Block())
from clifford_qc.models.chemistry import excitation_pool
from clifford_qc.pool_operator import PoolOperator
expected = {
    (4, 2): (12, '2bc5a8500a9c80a70e1614c34c878f9cd0513b9e95156296befa62376776fdac'),
    (8, 4): (160, 'a659e5f533307c986d2384cf0a51c40ccbedcde71aec85708ccd703288d2c091'),
}
for (n, e), (count, digest) in expected.items():
    pool = excitation_pool(n, e)
    assert len(pool) == count and all(type(op) is PoolOperator for op in pool)
    pairs = [(op.label, op.word.code) for op in pool]
    assert hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest() == digest
'''
    done = checkout_subprocess(code)
    assert done.returncode == 0, done.stdout + done.stderr
