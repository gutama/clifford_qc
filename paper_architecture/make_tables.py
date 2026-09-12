"""Generate every number the architecture paper typesets.

Two kinds of source feed this module, and the difference is the paper's
subject:

1. **Committed records** under ``benchmarks/reference_results/`` and
   ``PHASE_STATUS.json``.  These are read exactly as the benchmark gates
   committed them; nothing here recomputes a result.
2. **A live census of the repository itself** -- module and line counts per
   layer, the producer/gate pairing, and how each committed record declares its
   evidence tier.  The census is written to ``data/source_census.json`` and
   ``check_manuscript.py`` re-derives it from the tree, so a module added
   without a layer assignment or a checker added without a declared class
   fails the paper gate rather than silently ageing a table.  Records without
   an evidence declaration are counted and exposed in the coverage table; the
   current architecture does not reject them.

Run ``python paper_architecture/make_tables.py`` after any change to either.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
DATA = HERE / "data"

PACKAGE = ROOT / "clifford_qc"
TESTS = ROOT / "tests"
BENCHMARKS = ROOT / "benchmarks"
RECORDS = BENCHMARKS / "reference_results"

CENSUS = DATA / "source_census.json"
DEFAULT_CENSUS = CENSUS
PHASE_STATUS = ROOT / "PHASE_STATUS.json"
HIER_H4 = RECORDS / "clifford_hierarchy_h4_v2.json"
HIER_BEH2 = RECORDS / "clifford_hierarchy_beh2_v2.json"
PHASE14B = RECORDS / "phase14b_qwc_vs_fc.json"
PROTOCOL_COST = RECORDS / "protocol_cost.json"
MAPPING_AXIS = RECORDS / "mapping_axis.json"
R3C = RECORDS / "r3c_lih_full_cost.json"
R3D = RECORDS / "r3d_qr3_refinement.json"
R4A = RECORDS / "r4a_contextual_screen.json"
G1 = RECORDS / "g1_structural_preconditioner.json"

TABLE_SOURCES: dict[str, tuple[Path, ...]] = {
    "layers.tex": (CENSUS,),
    "evidence_coverage.tex": (CENSUS,),
    "gate_classes.tex": (CENSUS,),
    "hierarchy.tex": (HIER_H4, HIER_BEH2),
    "phase14b.tex": (PHASE14B,),
    "cost.tex": (PROTOCOL_COST,),
    "qr3.tex": (MAPPING_AXIS, R3C, R3D),
    "r4a.tex": (R4A,),
    "ledger.tex": (PHASE_STATUS,),
    "numbers.tex": (
        CENSUS, PHASE_STATUS, HIER_H4, HIER_BEH2, PHASE14B, PROTOCOL_COST,
        MAPPING_AXIS, R3C, R3D, R4A, G1,
    ),
}

# ---------------------------------------------------------------------------
# The declared architecture: which module belongs to which layer.
#
# This is the one hand-written mapping in the paper, and it is the claim of
# Sec. II: the layer stack is a partition of the package, not a reading of it.
# ``layer_census`` below refuses to emit a table unless every module in
# clifford_qc/ lands in exactly one layer, so a new module forces an explicit
# decision here instead of quietly falling outside the architecture the paper
# describes.
# ---------------------------------------------------------------------------
KERNEL_MODULES = (
    "multivector.py", "pauli_kernel.py", "pauli.py", "clifford.py", "gates.py",
    "states.py", "fermion.py", "channels.py", "dense_reference.py",
    "sparse.py", "pauli_action.py", "pauli_structure.py", "diagnostics.py",
    "matrix.py", "selection.py",
)
IR_MODULES = ("ir.py", "qasm3.py", "fermion_mapping.py")
EVIDENCE_MODULES = ("verify.py", "reproducibility.py", "record_environment.py")
ORCHESTRATION_MODULES = ("workflows.py",)
SURFACE_MODULES = ("__init__.py",)

LAYERS: tuple[tuple[str, str], ...] = (
    ("Algebra kernel", "kernel"),
    ("Program IR", "ir"),
    ("Execution", "backends"),
    ("Measurement", "measurement"),
    ("Problem definition", "models"),
    ("Algorithms", "algorithms"),
    ("Subspace solvers", "subspace"),
    ("Cross-layer orchestration", "orchestration"),
    ("Evidence and reproduction", "evidence"),
    ("Optional bridges", "bridges"),
    ("Package surface", "surface"),
)


def _layer_of(relative: Path) -> str:
    """The declared layer of one module path relative to ``clifford_qc/``."""
    if len(relative.parts) > 1:
        return relative.parts[0]
    name = relative.name
    if name in KERNEL_MODULES:
        return "kernel"
    if name in IR_MODULES:
        return "ir"
    if name in EVIDENCE_MODULES:
        return "evidence"
    if name in ORCHESTRATION_MODULES:
        return "orchestration"
    if name in SURFACE_MODULES:
        return "surface"
    return "unassigned"


# ---------------------------------------------------------------------------
# The declared evidence contract: what each gate in benchmarks/ asserts.
#
# ARCHITECTURE.md states that the 42 producers and the check_*.py gates are not
# two views of one list.  These classes are that statement in machine-readable
# form; ``gate_census`` refuses to emit a table unless the classes cover every
# check_*.py on disk exactly once.
# ---------------------------------------------------------------------------
GATE_CLASSES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Value-rebuilt",
        "recomputes its same-stem producer's record and fails on numeric drift",
        (
            "check_clifford_hierarchy", "check_exact_shot_search",
            "check_finite_shot_optimization", "check_finite_shot_rethink",
            "check_g1_structural_preconditioner", "check_krylov_width",
            "check_mapping_axis", "check_matched_h4",
            "check_phase14b_qwc_vs_fc", "check_priceability_screen",
            "check_protocol_axis", "check_protocol_cost",
            "check_qr3b_instance_preflight", "check_r3b_margin_stop_probe",
            "check_r3c_lih_full_cost", "check_r3d_qr3_refinement",
            "check_r4a_contextual_screen", "check_warm_start",
        ),
    ),
    (
        "Preregistration",
        "compares a predeclared plan against committed constants; no rebuild",
        (
            "check_phase14b_preregistration", "check_r3b_preregistration",
            "check_r3c_preregistration", "check_r3d_preregistration",
            "check_r4a_preregistration",
        ),
    ),
    (
        "Lineage and environment",
        "provenance stamping, environment migration, regeneration identity",
        (
            "check_record_environment", "check_regenerated_record",
            "check_r3_environment_migration",
        ),
    ),
    (
        "Plan ledger",
        "the phase status file against the plan document and the source tree",
        ("check_phase_status",),
    ),
    (
        "Documentation",
        "every documented command, flag, constant, and record still exists",
        ("check_docs",),
    ),
    (
        "Cross-artifact",
        "compares separately produced repository artifacts rather than a same-stem rebuild",
        ("check_molecular", "check_summaries"),
    ),
)

CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# Which jobs run without being asked for.  Everything else in the workflow sits
# behind `workflow_dispatch`, and Sec. IV distinguishes the two rather than
# calling both "in CI".
DISPATCH_GUARD = "github.event_name == 'workflow_dispatch'"


def ci_gate_sets() -> tuple[set[str], set[str]]:
    """Gates the workflow names, split into automatic and dispatch-only.

    Parsed from the workflow text rather than declared here: a hard-coded pair
    subtracted from the total keeps reporting the same number after a gate is
    added to the repository or dropped from the workflow, which is exactly the
    drift this paper claims to catch.  Jobs are found by their two-space
    indent, and a job carrying the dispatch guard contributes to the manual
    set.
    """
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    starts = [(m.start(), m.group(1))
              for m in re.finditer(r"(?m)^  ([a-z][\w-]*):$", text)]
    automatic: set[str] = set()
    dispatch: set[str] = set()
    for index, (offset, _name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        block = text[offset:end]
        gates = set(re.findall(r"benchmarks/(check_\w+)\.py", block))
        gates |= set(re.findall(r"gate:\s*(check_\w+)", block))
        if not gates:
            continue
        (dispatch if DISPATCH_GUARD in block else automatic).update(gates)
    return automatic, dispatch - automatic


def _git_blob_sha(path: Path) -> str:
    """Git blob identity, so a regenerated source forces a regenerated table."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _write(name: str, rows: list[str]) -> None:
    """Write one table fragment with its source and generator bindings."""
    TABLES.mkdir(parents=True, exist_ok=True)  # noqa: F821 (rebound by main)
    # TABLE_SOURCES is declared at import time, while main() may rebind CENSUS
    # to a scratch destination.  Bind census-backed fragments to the census
    # actually used for this generation, not to the default committed path.
    sources = tuple(CENSUS if path == DEFAULT_CENSUS else path
                    for path in TABLE_SOURCES[name])
    header = [f"% source-git-blob-sha: {_git_blob_sha(path)}" for path in sources]
    header.append(f"% generator-git-blob-sha: {_git_blob_sha(Path(__file__).resolve())}")
    (TABLES / name).write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _math(body: str) -> str:
    """Wrap a generated quantity so it typesets in text and in math mode.

    The same macro is used in a table cell, in running prose, and inside a
    displayed relation such as $k = \\cqcKStarIon$.  A ``$``-delimited value
    would close the surrounding math there, so every generated number is
    emitted through ``\\ensuremath``.
    """
    return rf"\ensuremath{{{body}}}"


def _int(value: float | int) -> str:
    """Thousands-separated integer, safe inside a REVTeX cell."""
    return _math(f"{int(round(value)):,}".replace(",", r"{,}"))


def _num(value: float, digits: int = 2) -> str:
    return _math(f"{value:.{digits}f}")


def _pct(fraction: float, digits: int = 1) -> str:
    return _math(rf"{100.0 * fraction:.{digits}f}\%")


def _pow2(value: float) -> str:
    """Exact powers of two read better as powers of two; this paper has many."""
    n = int(round(value))
    if n > 0 and (n & (n - 1)) == 0:
        return _math(rf"2^{{{n.bit_length() - 1}}}")
    return _int(n)


def _sec(microseconds: float, digits: int = 3) -> str:
    """Microseconds as seconds at fixed significant figures.

    The three device cards span five orders of magnitude in gate time, so a
    fixed number of decimals prints either noise or zero.
    """
    seconds = microseconds / 1e6
    if seconds == 0.0:
        return _math("0")
    exponent = math.floor(math.log10(abs(seconds)))
    decimals = max(0, digits - 1 - int(exponent))
    return _math(f"{seconds:,.{decimals}f}".replace(",", r"{,}"))


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
def layer_census() -> dict:
    """Modules and physical lines per declared layer, plus the test suite."""
    per_layer: dict[str, dict[str, int]] = {}
    unassigned: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        layer = _layer_of(relative)
        if layer == "unassigned":
            unassigned.append(str(relative))
            continue
        bucket = per_layer.setdefault(layer, {"modules": 0, "lines": 0})
        bucket["modules"] += 1
        bucket["lines"] += _lines(path)
    declared = {key for _, key in LAYERS}
    if unassigned:
        raise SystemExit(
            "modules with no declared layer in make_tables.LAYERS: "
            + ", ".join(unassigned))
    if set(per_layer) - declared:
        raise SystemExit(
            "census produced layers absent from LAYERS: "
            + ", ".join(sorted(set(per_layer) - declared)))
    tests = sorted(TESTS.glob("test_*.py"))
    return {
        "layers": per_layer,
        "package_modules": sum(v["modules"] for v in per_layer.values()),
        "package_lines": sum(v["lines"] for v in per_layer.values()),
        "test_modules": len(tests),
        "test_lines": sum(_lines(path) for path in tests),
    }


def gate_census() -> dict:
    """Producers, gates, and the declared class of every gate."""
    producers = sorted(path.stem for path in BENCHMARKS.glob("run_*.py"))
    gates = sorted(path.stem for path in BENCHMARKS.glob("check_*.py"))
    declared: list[str] = []
    for _, _, stems in GATE_CLASSES:
        declared.extend(stems)
    missing = sorted(set(gates) - set(declared))
    extra = sorted(set(declared) - set(gates))
    if missing:
        raise SystemExit(
            "gates with no declared class in make_tables.GATE_CLASSES: "
            + ", ".join(missing))
    if extra:
        raise SystemExit(
            "GATE_CLASSES names gates that no longer exist: " + ", ".join(extra))
    duplicates = sorted({stem for stem in declared if declared.count(stem) > 1})
    if duplicates:
        raise SystemExit("gate classified twice: " + ", ".join(duplicates))
    automatic, dispatch = ci_gate_sets()
    named = automatic | dispatch
    unknown = sorted(named - set(gates))
    if unknown:
        raise SystemExit(
            "the CI workflow names gates that do not exist: " + ", ".join(unknown))
    # A producer is value-rebuilt only if a same-stem checker exists.
    paired = sorted(
        stem for stem in producers
        if stem.replace("run_", "check_", 1) in set(gates))
    return {
        "producers": len(producers),
        "gates": len(gates),
        "same_stem_pairs": len(paired),
        "producers_without_gate": len(producers) - len(paired),
        "classes": {
            name: len(stems) for name, _, stems in GATE_CLASSES
        },
        "named_in_workflow": len(named),
        "run_on_every_pull_request": len(automatic),
        "automatic_gates": sorted(automatic),
        "manual_dispatch_only": len(dispatch),
        "dispatch_gates": sorted(dispatch),
        "absent_from_workflow": sorted(set(gates) - named),
    }


EVIDENCE_KEYS = (
    "evidence_tier", "evidence", "evidence_role", "evidence_tier_basis",
)


def _declaration_form(payload: object) -> str:
    """How one committed record declares the evidence its numbers carry.

    ``evidence_role`` is not a tier: it says what a record is for, not what
    kind of number it holds.  ``evidence_tier_basis`` explains a declaration
    but is likewise not a tier by itself.  Both are digested; neither inflates
    the tier coverage this paper reports on itself.
    """
    if not isinstance(payload, dict):
        return "none"
    for key in ("evidence_tier", "evidence"):
        value = payload.get(key)
        if isinstance(value, str):
            return "top_level_tier"
        if isinstance(value, dict):
            return "per_quantity_mapping"

    def contains_declaration(node: object, keys: tuple[str, ...]) -> bool:
        if isinstance(node, dict):
            return any(key in keys for key in node) or any(
                contains_declaration(value, keys) for value in node.values())
        if isinstance(node, list):
            return any(contains_declaration(value, keys) for value in node)
        return False

    nested_values = list(payload.values())
    nested_tier = any(contains_declaration(
        value, ("evidence_tier", "evidence")) for value in nested_values)
    nested_declaration = any(contains_declaration(
        value, EVIDENCE_KEYS) for value in nested_values)
    if isinstance(payload.get("evidence_role"), str):
        return "role_with_nested_tier" if nested_tier else "role_only"
    if nested_declaration:
        return "nested_in_subobject"
    if isinstance(payload.get("evidence_tier_basis"), str):
        return "basis_only"
    return "none"


def _declaration_digest(payload: object) -> str:
    """Digest of every evidence declaration in a record, at any depth.

    The form alone is too coarse to back the guarantee this paper makes.  A
    record can keep its form while a nested tier or one entry of a per-quantity
    map changes, and the census would not move.  Walking the whole record and
    digesting every evidence-keyed value, with its path, closes that.
    """
    found: dict[str, object] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in EVIDENCE_KEYS:
                    # A per-quantity map is a declaration too, and recording
                    # only scalars would miss one of its entries changing.
                    found[here] = value
                walk(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "")
    canonical = json.dumps(found, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def record_census() -> dict:
    """Classify JSON evidence declarations and count other series files."""
    forms: dict[str, list[str]] = {
        "top_level_tier": [],
        "per_quantity_mapping": [],
        "role_with_nested_tier": [],
        "role_only": [],
        "basis_only": [],
        "nested_in_subobject": [],
        "none": [],
    }
    series: list[str] = []
    tiers: dict[str, int] = {}
    declarations: dict[str, str] = {}
    for path in sorted(RECORDS.iterdir()):
        if path.suffix != ".json":
            series.append(path.name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        form = _declaration_form(payload)
        forms[form].append(path.name)
        declarations[path.name] = _declaration_digest(payload)
        if form == "top_level_tier" and isinstance(payload, dict):
            for key in ("evidence_tier", "evidence"):
                value = payload.get(key)
                if isinstance(value, str):
                    tiers[value] = tiers.get(value, 0) + 1
                    break
    return {
        "json_records": sum(len(names) for names in forms.values()),
        "series_files": len(series),
        "forms": {form: len(names) for form, names in forms.items()},
        "examples": {
            form: (names[0] if names else None) for form, names in forms.items()
        },
        "distinct_top_level_tiers": len(tiers),
        "top_level_tiers": dict(sorted(tiers.items())),
        "declarations": declarations,
    }


def write_census() -> dict:
    """Compute the census and commit it, so the checker can re-derive it."""
    census = {
        "schema": "clifford_qc.architecture_paper_census.v1",
        "source": layer_census(),
        "gates": gate_census(),
        "records": record_census(),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    CENSUS.write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return census


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def layers_table(census: dict) -> None:
    source = census["source"]
    total_lines = source["package_lines"]
    rows = []
    for label, key in LAYERS:
        bucket = source["layers"][key]
        rows.append(
            f"{label} & {_int(bucket['modules'])} & {_int(bucket['lines'])} & "
            f"{_pct(bucket['lines'] / total_lines)} \\\\")
    rows.append(r"\colrule")
    rows.append(
        f"Package total & {_int(source['package_modules'])} & "
        f"{_int(total_lines)} & {_pct(1.0)} \\\\")
    # The test suite is not a layer and does not take a share of the package;
    # it is tabulated here because its size against the package is the claim.
    rows.append(
        f"Test suite & {_int(source['test_modules'])} & "
        f"{_int(source['test_lines'])} & --- \\\\")
    _write("layers.tex", rows)


def evidence_coverage_table(census: dict) -> None:
    records = census["records"]
    labels = {
        "top_level_tier": "Top-level tier",
        "per_quantity_mapping": "Per-quantity mapping",
        "role_with_nested_tier": "Role plus nested tier",
        "role_only": "Role, not a tier",
        "basis_only": "Tier basis, not a tier",
        "nested_in_subobject": "Nested in sub-object",
        "none": "No evidence",
    }
    rows = []
    for key, label in labels.items():
        example = records["examples"][key]
        stem = Path(example).stem.replace("_", "-") if example else None
        shown = rf"\texttt{{{stem}}}" if stem else "---"
        rows.append(
            f"{label} & {_int(records['forms'][key])} & {shown} \\\\")
    rows.append(r"\colrule")
    rows.append(
        f"JSON records & {_int(records['json_records'])} & "
        f"{_int(records['distinct_top_level_tiers'])} tier strings \\\\")
    rows.append(
        f"Series files (JSONL, CSV, MD) & "
        f"{_int(records['series_files'])} & not inspected here \\\\")
    _write("evidence_coverage.tex", rows)


def gate_classes_table(census: dict) -> None:
    gates = census["gates"]
    rows = []
    for name, asserts, _ in GATE_CLASSES:
        rows.append(f"{name} & {_int(gates['classes'][name])} & {asserts} \\\\")
    rows.append(r"\colrule")
    rows.append(
        f"Gates, all classes & {_int(gates['gates'])} & "
        f"{_int(gates['named_in_workflow'])} named in the CI workflow \\\\")
    rows.append(
        f"Run on every pull request & "
        f"{_int(gates['run_on_every_pull_request'])} & "
        f"{_int(gates['manual_dispatch_only'])} more by manual dispatch \\\\")
    rows.append(
        f"Producers & {_int(gates['producers'])} & "
        f"{_int(gates['same_stem_pairs'])} have a same-stem gate \\\\")
    rows.append(
        f"Producers with no value gate & "
        f"{_int(gates['producers_without_gate'])} & "
        r"manuscript evidence, drivers, exploratory output \\")
    _write("gate_classes.tex", rows)


def hierarchy_table() -> None:
    rows = []
    for label, path in (("H$_4$", HIER_H4), ("BeH$_2$", HIER_BEH2)):
        record = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(record["rows"]):
            ratio = row["cx_to_preparation_cost_ratio_vs_qwc"]
            ratio_cell = "---" if ratio is None else _num(ratio)
            rows.append(
                f"{label if index == 0 else ''} & {_int(row['block_size'])} & "
                f"{_int(row['settings'])} & "
                f"{_num(row['mean_logical_cx_per_setting'], 1)} & "
                f"{_int(row['max_logical_cx_depth'])} & "
                f"{_int(row['state_preparations_at_uniform_shots'])} & "
                f"{ratio_cell} \\\\")
        if label.startswith("H$_4$"):
            rows.append(r"\colrule")
    _write("hierarchy.tex", rows)


def phase14b_table() -> None:
    record = json.loads(PHASE14B.read_text(encoding="utf-8"))
    qwc = record["protocols"]["qwc"]
    full = record["protocols"]["fully_commuting"]
    decision = record["decision"]

    def plan(arm: dict, field: str) -> float:
        return arm["compiled_plan"][field]

    def resource(arm: dict, field: str) -> float:
        return arm["compiled_plan"]["resource_sums"][field]

    def ratio(new: float, old: float) -> str:
        return "---" if not old else _num(new / old, 3)

    rows = [
        f"Compiled settings & {_int(plan(qwc, 'settings'))} & "
        f"{_int(plan(full, 'settings'))} & "
        f"{ratio(plan(full, 'settings'), plan(qwc, 'settings'))} \\\\",
        f"Settings the Ritz functional touches & "
        f"{_int(plan(qwc, 'ritz_touched_assigned_settings'))} & "
        f"{_int(plan(full, 'ritz_touched_assigned_settings'))} & "
        f"{ratio(plan(full, 'ritz_touched_assigned_settings'), plan(qwc, 'ritz_touched_assigned_settings'))} \\\\",
        f"Readable word--setting pairs & "
        f"{_int(plan(qwc, 'readable_word_setting_pairs'))} & "
        f"{_int(plan(full, 'readable_word_setting_pairs'))} & "
        f"{ratio(plan(full, 'readable_word_setting_pairs'), plan(qwc, 'readable_word_setting_pairs'))} \\\\",
        f"One-qubit gates, summed over settings & "
        f"{_int(resource(qwc, 'N_1q'))} & {_int(resource(full, 'N_1q'))} & "
        f"{ratio(resource(full, 'N_1q'), resource(qwc, 'N_1q'))} \\\\",
        f"Two-qubit gates, summed over settings & "
        f"{_int(resource(qwc, 'N_2q'))} & {_int(resource(full, 'N_2q'))} & "
        f"{ratio(resource(full, 'N_2q'), resource(qwc, 'N_2q'))} \\\\",
        f"Certified total physical shots & "
        f"{_pow2(qwc['certification']['certified_total_physical_shots'])} & "
        f"{_pow2(full['certification']['certified_total_physical_shots'])} & "
        f"{_num(decision['fully_commuting_to_qwc_certified_shot_ratio'], 5)} \\\\",
        r"\colrule",
    ]
    for card in ("ion-like", "logical-alltoall", "superconducting-like"):
        qwc_led = qwc["device_costs"][card]["ledger"]
        full_led = full["device_costs"][card]["ledger"]
        card_ratio = decision["card_specific"][card][
            "fully_commuting_to_qwc_runtime_ratio"]
        qwc_time = qwc_led["accuracy"]["C_time_epsilon_us"]
        full_time = full_led["accuracy"]["C_time_epsilon_us"]
        full_cell = (
            r"inadmissible" if full_time is None else _sec(full_time))
        ratio_cell = "---" if card_ratio is None else _num(card_ratio, 3)
        rows.append(
            rf"$C_{{\mathrm{{time}}}}(\epsilon)$, {card} (s) & "
            f"{_sec(qwc_time)} & {full_cell} & {ratio_cell} \\\\")
    rows.append(
        f"Settings below the fidelity floor, superconducting-like & "
        f"{_int(qwc['device_costs']['superconducting-like']['ledger']['inadmissible_settings'])} & "
        f"{_int(full['device_costs']['superconducting-like']['ledger']['inadmissible_settings'])} & "
        "--- \\\\")
    _write("phase14b.tex", rows)


def cost_table() -> None:
    """Accuracy-matched cost of the priced instance, per card and estimator.

    One mapping arm (JW) is reported throughout so the table varies only in the
    card and the estimator: that is the claim, which is that $k^*$ is not a
    property of the Hamiltonian alone.  The mapping axis enters as the spread
    over all five arms at the same rung.
    """
    record = json.loads(PROTOCOL_COST.read_text(encoding="utf-8"))
    beh2 = record["systems"]["beh2"]
    rows = []
    for card in ("ion-like", "logical-alltoall", "superconducting-like"):
        for estimator in ("single_assignment", "pooled"):
            arm = beh2["k_star"][card][estimator]["by_arm"]["jw"]
            point = arm["k_star_point"]
            region = ", ".join(str(k) for k in sorted(set(arm["k_star_region"])))
            cost = next(entry for entry in arm["costs"]
                        if entry["block_size"] == point)
            spread = beh2["k_star"][card]["mapping_cost_spread"][estimator][
                "all_arms"]["by_block_size"][str(point)]
            rows.append(
                f"{card} & {estimator.replace('_', ' ')} & "
                f"{_int(point)} & {region} & {arm['status']} & "
                f"{_sec(cost['C_time_epsilon_us'])} & "
                rf"\texttt{{{spread['cheapest_mapping']}}} & "
                f"{_num(spread['spread'])} \\\\")
    inadmissible = sorted({
        k
        for estimator in ("single_assignment", "pooled")
        for arm in beh2["k_star"]["superconducting-like"][estimator][
            "by_arm"].values()
        for k in arm["inadmissible_block_sizes"]
    })
    rows.append(r"\colrule")
    rows.append(
        rf"\multicolumn{{8}}{{p{{0.96\textwidth}}}}{{Rungs the "
        rf"superconducting-like card refuses "
        rf"to price: $k={{{', '.join(str(k) for k in inadmissible)}}}$. "
        rf"Accuracy target {_num(record['accuracy_target_millihartree'], 1)} mHa; "
        rf"H$_4$ is not priced at all, its own subspace bias being "
        rf"{_num(record['systems']['h4']['max_exact_subspace_bias_millihartree'], 3)} mHa.}} \\")
    _write("cost.tex", rows)


def qr3_table() -> None:
    """Mapping spread against instance spread, on every tier that measured it.

    The three readouts aggregate differently and the column says how: a ratio
    column would invite reading them as one quantity, which is exactly the
    mistake the record's own claim boundaries forbid.
    """
    mapping = json.loads(MAPPING_AXIS.read_text(encoding="utf-8"))
    r3c = json.loads(R3C.read_text(encoding="utf-8"))
    r3d = json.loads(R3D.read_text(encoding="utf-8"))

    structural = mapping["qr3"]["structural"]
    rows = []
    for metric, label in (
            ("qwc_settings_matched_greedy", "QWC settings, matched greedy"),
            ("mean_word_weight", "Mean Pauli word weight"),
            ("word_universe", "Word universe"),
    ):
        block = structural.get(metric)
        if block is None:
            continue
        rows.append(
            f"{label} & structural & widest mapping vs.\\ narrowest instance & "
            f"{_num(block['max_mapping_spread_factor'])} & "
            f"{_num(block['min_instance_spread_factor'])} & "
            f"{block['verdict'].replace('_', ' ')} \\\\")
    rows.append(r"\colrule")
    comparison = r3c["qr3_second_instance"]["comparison"]
    widest = comparison["widest_mapping_spread"]
    narrowest = comparison["narrowest_instance_spread"]
    rows.append(
        r"Accuracy-matched $C_{\mathrm{time}}(\epsilon)$ & exact & "
        "point estimate on the supporting cells & "
        f"{_num(widest['point'])} & {_num(narrowest['point'])} & "
        f"{comparison['verdict'].replace('_', ' ')} \\\\")
    rows.append(
        r"Accuracy-matched $C_{\mathrm{time}}(\epsilon)$ & exact & "
        "conservative support of the same cells & "
        f"{_num(widest['minimum_possible'])} & "
        f"{_num(narrowest['maximum_possible'])} & "
        f"{comparison['verdict'].replace('_', ' ')} \\\\")
    readout = r3d["readout"]
    rows.append(
        r"R3d midpoint refinement & exact & "
        "support after added endpoints & "
        f"{_num(readout['mapping_support_minimum'])} & "
        f"{_num(readout['instance_support_maximum'])} & "
        f"{readout['classification'].replace('_', ' ')} \\\\")
    _write("qr3.tex", rows)


def r4a_table() -> None:
    record = json.loads(R4A.read_text(encoding="utf-8"))
    ceiling = record["gates"]["word_universe_ceiling"]
    admissible_bias = record["gates"]["admissible_bias_millihartree"]
    rows = []
    for rung in record["contextual_rungs"]:
        arms = {arm["name"]: arm for arm in rung["arms"]}
        cs_qse, cs_acase = arms["cs_qse"], arms["cs_acase"]
        admitted = sum(1 for arm in rung["arms"] if arm["admissible"])
        rows.append(
            f"{_int(rung['fixed_qubits'])} & "
            f"{_int(cs_qse['active_qubits'])} & "
            f"{_num(cs_qse['bias_millihartree'], 2)} & "
            f"{_int(cs_qse['word_universe'])} & "
            f"{_num(cs_acase['bias_millihartree'], 2)} & "
            f"{_int(cs_acase['word_universe'])} & "
            f"{_pct(cs_qse['hamiltonian_removed_hs_fraction'])} & "
            f"{_int(admitted)} \\\\")
    rows.append(r"\colrule")
    unrestricted = {arm["name"]: arm for arm in record["contextual_rungs"][0]["arms"]}
    rows.append(
        rf"\multicolumn{{8}}{{p{{0.96\textwidth}}}}{{Unrestricted arms, "
        rf"identical at every rung: "
        rf"\texttt{{full\_qse}} bias {_num(unrestricted['full_qse']['bias_millihartree'], 4)} mHa, "
        rf"word universe {_int(unrestricted['full_qse']['word_universe'])} "
        rf"(ceiling {_int(ceiling)}); "
        rf"\texttt{{acase}} bias {_num(unrestricted['acase']['bias_millihartree'], 4)} mHa, "
        rf"word universe {_int(unrestricted['acase']['word_universe'])}. "
        rf"Admissible bias {_num(admissible_bias, 3)} mHa.}} \\")
    _write("r4a.tex", rows)


def ledger_table() -> None:
    status = json.loads(PHASE_STATUS.read_text(encoding="utf-8"))
    rows = []
    for program in status["adjunct_programs"]:
        outcome = program.get("outcome", "---").replace("_", " ")
        rows.append(
            rf"\texttt{{{program['id'].replace('_', '-')}}} & "
            f"{program['title']} & "
            f"{program['status'].replace('_', ' ')} & {outcome} \\\\")
    rows.append(r"\colrule")
    summary = status["summary"]
    open_or_partial = [
        phase for phase in status["numbered_phases"]
        if phase["status"] != "complete"
    ]
    rows.append(
        rf"\multicolumn{{4}}{{p{{0.96\textwidth}}}}{{Numbered phases: "
        rf"{_int(summary['complete_phase_count'])} of "
        rf"{_int(summary['numbered_phase_count'])} complete "
        rf"({_pct(summary['strict_complete_fraction'])} strict, "
        rf"{_pct(summary['progress_weighted_fraction'])} progress-weighted); "
        rf"{_int(len(open_or_partial))} open, partial, or proposed.}} \\")
    _write("ledger.tex", rows)


# ---------------------------------------------------------------------------
# Inline numbers
# ---------------------------------------------------------------------------
def numbers_macros(census: dict) -> None:
    """Every number the prose states, as a macro bound to its record.

    Sec. VIII claims that no number in this manuscript is typed by hand.  That
    claim is only checkable if the prose quantities are generated too, so each
    one is defined here and ``check_manuscript.py`` rejects every body numeral
    outside the short allowlist of conceptual notation.
    """
    source = census["source"]
    gates = census["gates"]
    records = census["records"]
    hier_h4 = json.loads(HIER_H4.read_text(encoding="utf-8"))
    hier_beh2 = json.loads(HIER_BEH2.read_text(encoding="utf-8"))
    p14b = json.loads(PHASE14B.read_text(encoding="utf-8"))
    cost = json.loads(PROTOCOL_COST.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING_AXIS.read_text(encoding="utf-8"))
    r3c = json.loads(R3C.read_text(encoding="utf-8"))
    r3d = json.loads(R3D.read_text(encoding="utf-8"))
    r4a = json.loads(R4A.read_text(encoding="utf-8"))
    g1 = json.loads(G1.read_text(encoding="utf-8"))
    status = json.loads(PHASE_STATUS.read_text(encoding="utf-8"))

    h4 = cost["systems"]["h4"]
    beh2 = cost["systems"]["beh2"]
    sc_full = p14b["protocols"]["fully_commuting"]["device_costs"][
        "superconducting-like"]["ledger"]
    r4a_arms = {arm["name"]: arm for arm in r4a["contextual_rungs"][0]["arms"]}
    jw_ion = beh2["k_star"]["ion-like"]["single_assignment"]["by_arm"]["jw"]
    jw_logical = beh2["k_star"]["logical-alltoall"]["single_assignment"][
        "by_arm"]["jw"]
    structural = mapping["qr3"]["structural"]["qwc_settings_matched_greedy"]

    macros: dict[str, str] = {
        # Sec. II -- the package as measured
        "cqcModules": _int(source["package_modules"]),
        "cqcLines": _int(source["package_lines"]),
        "cqcTestModules": _int(source["test_modules"]),
        "cqcTestLines": _int(source["test_lines"]),
        "cqcSubspaceLines": _int(source["layers"]["subspace"]["lines"]),
        "cqcKernelLines": _int(source["layers"]["kernel"]["lines"]),
        "cqcMeasurementLines": _int(source["layers"]["measurement"]["lines"]),
        "cqcBridgeModules": _int(source["layers"]["bridges"]["modules"]),
        # Secs. III and IV -- the evidence contract as measured
        "cqcProducers": _int(gates["producers"]),
        "cqcGates": _int(gates["gates"]),
        "cqcSameStemPairs": _int(gates["same_stem_pairs"]),
        "cqcProducersWithoutGate": _int(gates["producers_without_gate"]),
        "cqcGatesInWorkflow": _int(gates["named_in_workflow"]),
        "cqcGatesAutomatic": _int(gates["run_on_every_pull_request"]),
        "cqcGatesDispatch": _int(gates["manual_dispatch_only"]),
        "cqcGatesAbsent": _int(len(gates["absent_from_workflow"])),
        "cqcJsonRecords": _int(records["json_records"]),
        "cqcSeriesFiles": _int(records["series_files"]),
        "cqcRecordsLabelled": _int(records["forms"]["top_level_tier"]),
        "cqcRecordsRoleWithNested": _int(
            records["forms"]["role_with_nested_tier"]),
        "cqcRecordsRoleOnly": _int(records["forms"]["role_only"]),
        "cqcRecordsUnlabelled": _int(records["forms"]["none"]),
        "cqcRecordsPerQuantity": _int(records["forms"]["per_quantity_mapping"]),
        "cqcRecordsNested": _int(records["forms"]["nested_in_subobject"]),
        "cqcDistinctTiers": _int(records["distinct_top_level_tiers"]),
        # Sec. V A -- the block-commuting hierarchy
        "cqcHFourSettingsQWC": _int(hier_h4["rows"][0]["settings"]),
        "cqcHFourSettingsFull": _int(hier_h4["rows"][-1]["settings"]),
        "cqcHFourPrepReduction": _pct(
            hier_h4["qwc_to_full"]["preparation_reduction_fraction"]),
        "cqcHFourCXRatio": _num(
            hier_h4["qwc_to_full"]["max_cx_to_preparation_cost_ratio"]),
        "cqcHFourCXAvoided": _num(
            hier_h4["qwc_to_full"]["logical_cx_per_preparation_avoided"]),
        "cqcBeHSettingsQWC": _int(hier_beh2["rows"][0]["settings"]),
        "cqcBeHSettingsFull": _int(hier_beh2["rows"][-1]["settings"]),
        "cqcBeHPrepReduction": _pct(
            hier_beh2["qwc_to_full"]["preparation_reduction_fraction"]),
        "cqcBeHCXRatio": _num(
            hier_beh2["qwc_to_full"]["max_cx_to_preparation_cost_ratio"]),
        "cqcBeHCXRatioWorst": _num(hier_beh2["rows"][1][
            "cx_to_preparation_cost_ratio_vs_qwc"]),
        # Sec. V B -- QWC against fully commuting
        "cqcShotRatio": _num(
            p14b["decision"]["fully_commuting_to_qwc_certified_shot_ratio"], 5),
        "cqcShotRatioInverse": _int(
            1.0 / p14b["decision"]["fully_commuting_to_qwc_certified_shot_ratio"]),
        "cqcQWCCertifiedShots": _pow2(p14b["protocols"]["qwc"][
            "certification"]["certified_total_physical_shots"]),
        "cqcFullCertifiedShots": _pow2(p14b["protocols"]["fully_commuting"][
            "certification"]["certified_total_physical_shots"]),
        "cqcIonRuntimeRatio": _num(p14b["decision"]["card_specific"][
            "ion-like"]["fully_commuting_to_qwc_runtime_ratio"], 3),
        "cqcLogicalRuntimeRatio": _num(p14b["decision"]["card_specific"][
            "logical-alltoall"]["fully_commuting_to_qwc_runtime_ratio"], 3),
        "cqcSCInadmissibleSettings": _int(sc_full["inadmissible_settings"]),
        "cqcSCSettings": _int(sc_full["settings"]),
        "cqcSCMeanFidelity": _num(sc_full["fidelity"]["mean"], 3),
        "cqcSCFidelityFloor": _num(sc_full["fidelity"]["floor"], 2),
        "cqcBankWords": _int(p14b["system"]["measured_word_universe"]),
        "cqcBankBasis": _int(p14b["system"]["basis_size"]),
        "cqcRitzSupport": _int(p14b["system"]["ritz_functional_measured_support"]),
        # Sec. V C -- accuracy-matched cost
        "cqcTargetMilliHartree": _num(cost["accuracy_target_millihartree"], 1),
        "cqcHFourBias": _num(h4["max_exact_subspace_bias_millihartree"], 3),
        "cqcBeHBias": _num(beh2["max_exact_subspace_bias_millihartree"], 4),
        "cqcCrossChecked": _int(beh2["r1_cross_check"]["compared_crossings"]),
        "cqcCrossAgreeing": _int(beh2["r1_cross_check"]["agreeing_crossings"]),
        "cqcRThreeCCells": _int(r3c["pricing"]["evaluated_cells"]),
        "cqcRThreeCFinite": _int(r3c["pricing"]["cells_with_a_finite_interval"]),
        "cqcRThreeCCensored": _int(r3c["pricing"]["right_censored_cells"]),
        "cqcRThreeCSolves": _int(r3c["pricing"]["confirmatory_solves_attempted"]),
        "cqcRThreeCFailures": _int(r3c["pricing"]["confirmatory_solve_failures"]),
        "cqcRThreeCSCPriced": _int(
            r3c["pricing"]["priced_cells_by_card"]["superconducting-like"]),
        "cqcKStarIon": _int(jw_ion["k_star_point"]),
        "cqcKStarLogical": _int(jw_logical["k_star_point"]),
        "cqcKStarIonRegion": ", ".join(
            str(k) for k in sorted(set(jw_ion["k_star_region"]))),
        "cqcMappingSpreadKOne": _num(
            beh2["k_star"]["ion-like"]["mapping_cost_spread"][
                "single_assignment"]["all_arms"]["by_block_size"]["1"]["spread"]),
        # Sec. VI B -- the mapping-versus-instance question
        "cqcStructuralMapping": _num(structural["max_mapping_spread_factor"]),
        "cqcStructuralInstance": _num(structural["min_instance_spread_factor"]),
        "cqcStructuralMargin": _num(structural["margin_factor"], 3),
        "cqcRThreeDMappingSupport": _num(
            r3d["readout"]["mapping_support_minimum"], 2),
        "cqcRThreeDInstanceSupport": _num(
            r3d["readout"]["instance_support_maximum"], 2),
        "cqcRThreeDCells": _int(r3d["refinement_run"]["new_endpoint_cell_count"]),
        # Secs. VI C and VI D -- the screens that stopped
        "cqcRFourARungs": _int(len(r4a["contextual_rungs"])),
        "cqcRFourAPassing": _int(len(r4a["structural_gate"]["passing_contextual_rungs"])),
        "cqcRFourARemovedFraction": _pct(
            r4a_arms["cs_qse"]["hamiltonian_removed_hs_fraction"]),
        "cqcRFourACSBias": _num(r4a_arms["cs_qse"]["bias_millihartree"], 2),
        "cqcRFourACSACaseBias": _num(
            r4a_arms["cs_acase"]["bias_millihartree"], 2),
        "cqcRFourAACaseBias": _num(r4a_arms["acase"]["bias_millihartree"], 4),
        "cqcRFourAACaseWords": _int(r4a_arms["acase"]["word_universe"]),
        "cqcRFourAAdmissibleBias": _num(
            r4a["gates"]["admissible_bias_millihartree"], 3),
        "cqcRFourACeiling": _int(r4a["gates"]["word_universe_ceiling"]),
        "cqcRFourAFullWords": _int(r4a_arms["full_qse"]["word_universe"]),
        "cqcGOneMajorana": _int(
            g1["gate_outcomes"]["e_marginal_by_pool"]["majorana_monomials"][0]),
        "cqcGOneExcitation": _int(
            g1["gate_outcomes"]["e_marginal_by_pool"]["determinant_excitations"][0]),
        # Sec. VI E -- the ledger
        "cqcNumberedPhases": _int(status["summary"]["numbered_phase_count"]),
        "cqcCompletePhases": _int(status["summary"]["complete_phase_count"]),
        "cqcStrictFraction": _pct(status["summary"]["strict_complete_fraction"]),
        "cqcWeightedFraction": _pct(
            status["summary"]["progress_weighted_fraction"]),
        "cqcAdjunctPrograms": _int(len(status["adjunct_programs"])),
        "cqcNegativePrograms": _int(sum(
            1 for program in status["adjunct_programs"]
            if program["status"] in ("closed_negative", "retired", "not_authorized"))),
    }
    rows = [rf"\newcommand{{\{name}}}{{{value}}}"
            for name, value in sorted(macros.items())]
    _write("numbers.tex", rows)


def main(tables: Path | None = None, data: Path | None = None) -> Path:
    """Generate every fragment, optionally into a scratch tree.

    ``check_manuscript.py`` passes a temporary directory and compares what
    comes out against what is committed.  Without that, a hand-edited value in
    a fragment passes every check: the provenance headers digest the generator
    and the source records, neither of which a hand edit touches.
    """
    global TABLES, DATA, CENSUS
    original = TABLES, DATA, CENSUS
    target_tables = tables if tables is not None else TABLES
    target_data = data if data is not None else DATA
    try:
        TABLES = target_tables
        DATA = target_data
        CENSUS = target_data / original[2].name
        census = write_census()
        layers_table(census)
        evidence_coverage_table(census)
        gate_classes_table(census)
        hierarchy_table()
        phase14b_table()
        cost_table()
        qr3_table()
        r4a_table()
        ledger_table()
        numbers_macros(census)
        return target_tables
    finally:
        TABLES, DATA, CENSUS = original


if __name__ == "__main__":
    print(main())
