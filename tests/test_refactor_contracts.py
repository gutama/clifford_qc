"""Compatibility and dependency contracts for the R1--R4 refactor.

The scientific regression tests exercise the numerical behaviour.  This file
protects the architectural part of the refactor: legacy import paths remain
valid while shared kernels have one canonical implementation and production
modules do not reach through another module's private namespace.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "clifford_qc"


def _imports_adapt_workflow(node: ast.AST) -> bool:
    """Recognize direct imports of the ADAPT workflow in every common form."""
    if isinstance(node, ast.Import):
        return any(alias.name.endswith("algorithms.adapt")
                   for alias in node.names)
    if not isinstance(node, ast.ImportFrom) or not node.module:
        return False
    if node.module.endswith("algorithms.adapt"):
        return True
    return (node.module.endswith("algorithms")
            and any(alias.name in {"adapt", "*"} for alias in node.names))


def test_legacy_modules_are_compatibility_facades():
    from clifford_qc import multivector
    from clifford_qc import pauli_kernel
    from clifford_qc import selection
    from clifford_qc.algorithms import adapt
    from clifford_qc.measurement import functionals, session
    from clifford_qc.subspace import certification, elements, linalg, measured
    from clifford_qc.subspace import projection, uncertainty

    assert multivector.word_mul is pauli_kernel.word_mul
    assert multivector.label_to_code is pauli_kernel.label_to_code
    assert multivector.code_to_label is pauli_kernel.code_to_label
    assert adapt.canonical_argmax is selection.canonical_argmax
    assert adapt.TIE_RTOL == selection.TIE_RTOL
    assert adapt.TIE_ATOL == selection.TIE_ATOL
    assert elements.MatrixElementBank is projection.MatrixElementBank
    assert measured.WordFunctional is functionals.WordFunctional
    assert measured.SharedMeasurement is session.SharedMeasurement
    assert measured.ritz_uncertainty is uncertainty.ritz_uncertainty
    assert measured.run_certified_acase is certification.run_certified_acase
    assert measured.solve_projected is linalg.solve_projected


def test_public_subspace_imports_stay_available():
    from clifford_qc.subspace import (
        ASYMPTOTIC,
        EXACT,
        FINITE_SAMPLE,
        HEURISTIC,
        MatrixElementBank,
        SharedMeasurement,
        WordFunctional,
        canonical_eigh,
        projected_matrices,
        run_certified_acase,
        solve_projected,
        solve_subspace,
    )

    assert {EXACT, ASYMPTOTIC, FINITE_SAMPLE, HEURISTIC} == {
        "exact", "asymptotic", "finite_sample", "heuristic"
    }
    assert all(callable(obj) for obj in (
        MatrixElementBank,
        SharedMeasurement,
        WordFunctional,
        canonical_eigh,
        projected_matrices,
        run_certified_acase,
        solve_projected,
        solve_subspace,
    ))


def test_target_layers_do_not_import_private_symbols_across_modules():
    """Private helpers must not become implicit cross-module contracts."""
    targets = [
        ROOT / "measurement" / "bank.py",
        ROOT / "subspace" / "adaptive.py",
        ROOT / "subspace" / "elements.py",
        ROOT / "subspace" / "generators.py",
        ROOT / "subspace" / "measured.py",
        ROOT / "subspace" / "measured_response.py",
        ROOT / "subspace" / "solver.py",
    ]
    violations = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")
    assert violations == []


def test_subspace_kernel_does_not_depend_on_adapt_workflow():
    offenders = []
    for path in (ROOT / "subspace").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _imports_adapt_workflow(node):
                offenders.append(path.name)
    assert offenders == []


def test_adapt_dependency_guard_covers_equivalent_import_forms():
    statements = (
        "from ..algorithms.adapt import run_adapt",
        "from ..algorithms import adapt",
        "import clifford_qc.algorithms.adapt",
        "import clifford_qc.algorithms.adapt as adapt",
    )
    for statement in statements:
        tree = ast.parse(statement)
        assert any(_imports_adapt_workflow(node) for node in ast.walk(tree)), statement
