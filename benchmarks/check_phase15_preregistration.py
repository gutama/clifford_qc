"""Gate the result-free Phase 15 preregistration: the H² support/cost preflight.

The declaration asks one question per frozen bank. Over the bank's retained
block, would the second-moment rows ``K_ij = A_i† H² A_j`` keep the measured
word universe within a decade of the ``(S, H)`` universe it already measures?
And would they keep the bank's resident coefficients within the largest count
the package has demonstrably held? This gate checks that the declaration can
answer that question as frozen, and that it was frozen before any answer
existed:

* completeness, and the absence of any result-shaped value;
* the SHA-256 of every input the run reads: the FCIDUMPs and their provenance,
  the mapping-axis config that fixes each basis, and the Phase 2M-A ledger
  that supplies the baseline and the storage anchor. The multivector product
  and the Pauli kernel, which define what a row's support *is*, stay bound
  until a record exists; after that the record's provenance names the code.
* an exhaustive status ladder, and a total verdict rule;
* the population: exactly the five fixed-label banks the Phase 2M-A ledger
  priced, at the labels the mapping-axis config freezes;
* the clauses against each other, the Phase 16B v3 lesson. The storage anchor
  must be the largest committed coefficient count and must clear every
  baseline, and the strict pruning threshold must sit between the multivector
  product's own and every Hamiltonian coefficient;
* a recomputation, from the committed inputs alone, of every number in
  ``measured_before_freezing``: each ``(S, H)`` bank's size, word universe and
  coefficient count, which must also match the ledger. It then checks the
  premise of the symmetry ceiling. The Hamiltonian, every generator and the
  whole ``(S, H)`` universe must commute with both spin-block parities, so no
  second-moment word can leave a sector of ``4^n / 4`` words. From that
  ceiling it derives which banks can bind which clause, and so which verdicts
  are reachable at all;
* commit order, from git history, once a record exists: the config's last
  change must strictly precede the record;
* the revision log.

It forms no ``H²``, builds no second-moment row, counts no second-moment word,
and computes no status or verdict.

    python benchmarks/check_phase15_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

CONFIG = HERE / "configs" / "phase15_h2_preflight.json"
RECORD = HERE / "reference_results" / "phase15_h2_preflight.json"
LEDGER = HERE / "reference_results" / "bank_storage_ledger.json"
MAPPING = HERE / "configs" / "mapping_axis.json"
SCHEMA = "clifford_qc.phase15_h2_preflight_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "question_id", "purpose", "contains_results",
    "claim_boundary", "question", "row_definition", "banks", "baseline",
    "clauses", "pruning", "symmetry_ceiling", "decision_rule", "consequences",
    "measured_before_freezing", "decisive_banks", "reachable_verdicts",
    "deterministic_checks", "evidence", "excluded_from_cost",
    "exploratory_disclosure", "prespecified_followup", "implementation_lineage",
    "record_requirements", "revisions",
)
BOUND_INPUTS = (
    "benchmarks/configs/mapping_axis.json",
    "benchmarks/reference_results/bank_storage_ledger.json",
)
BOUND_IMPLEMENTATIONS = ("clifford_qc/multivector.py", "clifford_qc/pauli_kernel.py")
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "status_observed",
    "bank_status", "bank_statuses", "omega", "omega_observed", "h2_terms",
    "k_word_universe", "combined_word_universe", "k_coefficient_occurrences",
    "total_coefficient_occurrences", "conclusion", "elapsed_seconds",
})
BANK_STATUSES = ("ELIGIBLE", "WORD_PROHIBITIVE", "STORAGE_PROHIBITIVE",
                 "BOTH_PROHIBITIVE", "CONVENTION_SENSITIVE")
VERDICTS = ("FULL", "RESTRICTED", "NONE")
UNTENSED = ("has been", "was run", "were run", "no count has been")

_INT_FIELDS = ("n_qubits", "basis_size", "block_pairs", "hamiltonian_terms",
               "max_generator_terms", "sh_word_universe", "sh_coefficient_occurrences",
               "sector_word_ceiling", "operator_sector_violations",
               "sh_sector_violations")
_BOOL_FIELDS = ("word_clause_can_bind", "storage_clause_can_bind")
_FLOAT_FIELDS = ("min_abs_hamiltonian_coefficient",)
MEASURED_FIELDS = _INT_FIELDS + _BOOL_FIELDS + _FLOAT_FIELDS


# ------------------------------------------------------------------ loading

def _walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def load_config(path: Path = CONFIG) -> dict:
    """Read the declaration, refusing one that carries a result."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for where, key, value in _walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            raise ValueError(f"config carries a result field {where}.{key}")
    if config.get("contains_results") is not False:
        raise ValueError("contains_results must be literally false")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


# ------------------------------------------------------------------ the rule

def clause_passes(config: dict, *, sh_words: int, combined_words: int,
                  total_coefficients: int) -> tuple[bool, bool]:
    """``(word, storage)`` for one bank at one pruning threshold."""
    clauses = config["clauses"]
    word = combined_words <= int(clauses["word"]["max_ratio"]) * sh_words
    storage = total_coefficients <= int(clauses["storage"]["max_coefficient_occurrences"])
    return word, storage


def status_of(declared: tuple[bool, bool], strict: tuple[bool, bool]) -> str:
    """The frozen ladder, as a function of both clauses at both thresholds.

    Strict pruning only removes terms, so a clause that passes at the declared
    threshold passes at the strict one. The one disagreement possible is a
    clause that fails only because of terms at or below the strict threshold,
    and a status that rests on those is not read as eligibility.
    """
    if declared != strict:
        return "CONVENTION_SENSITIVE"
    word, storage = declared
    if word and storage:
        return "ELIGIBLE"
    if storage:
        return "WORD_PROHIBITIVE"
    if word:
        return "STORAGE_PROHIBITIVE"
    return "BOTH_PROHIBITIVE"


def verdict_of(statuses) -> str:
    statuses = tuple(statuses)
    if "INVALID" in statuses:
        return "INVALID"
    eligible = sum(value == "ELIGIBLE" for value in statuses)
    if eligible == len(statuses):
        return "FULL"
    return "NONE" if eligible == 0 else "RESTRICTED"


def ladder_problems(config: dict) -> list[str]:
    problems = []
    rule = config["decision_rule"]
    order = [row["status"] for row in rule["bank_status_order"]]
    if order != list(BANK_STATUSES):
        problems.append(f"status ladder order moved: {order}")
    pairs = list(itertools.product((False, True), repeat=2))
    reached = {status_of(d, s) for d in pairs for s in pairs
               if all(sd <= ss for sd, ss in zip(d, s))}
    if reached != set(BANK_STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(BANK_STATUSES) - reached}")
    count = len(config["banks"]["systems"])
    for tuple_ in itertools.product(BANK_STATUSES, repeat=count):
        verdict = verdict_of(tuple_)
        if verdict == "FULL" and any(v != "ELIGIBLE" for v in tuple_):
            problems.append(f"{tuple_} gives FULL with a bank that is not eligible")
            break
    if verdict_of(("ELIGIBLE",) * count) != "FULL":
        problems.append("all-ELIGIBLE must give FULL")
    if verdict_of(("WORD_PROHIBITIVE",) * count) != "NONE":
        problems.append("no eligible bank must give NONE")
    if rule.get("combination_rule") != (
            "every bank ELIGIBLE gives FULL; no bank ELIGIBLE gives NONE; "
            "anything else is RESTRICTED; any INVALID bank gives INVALID"):
        problems.append("combination_rule text moved from the rule this gate enforces")
    if set(config["consequences"]) != set(VERDICTS) | {"INVALID"}:
        problems.append("every verdict, INVALID included, must declare its consequence")
    return problems


def _ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def ledger_baseline() -> dict[str, dict]:
    """The Phase 2M-A fixed-label banks, keyed by system."""
    return {row["system"]: row for row in _ledger()["measured_banks"]
            if row.get("kind") == "fixed_label_bank"}


def population_problems(config: dict) -> list[str]:
    problems = []
    banks = config["banks"]
    systems = list(banks["systems"])
    if len(set(systems)) != len(systems):
        problems.append("a bank is declared twice")
    ledger = ledger_baseline()
    if sorted(systems) != sorted(ledger):
        problems.append(f"the banks must be exactly the ledger's fixed-label banks "
                        f"{sorted(ledger)}, not {sorted(systems)}")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    keys = [spec["key"] for spec in mapping["systems"]]
    missing = sorted(set(systems) - set(keys))
    if missing:
        problems.append(f"banks absent from the mapping-axis config: {missing}")
    if banks.get("source_config") != "benchmarks/configs/mapping_axis.json":
        problems.append("the bases must come from benchmarks/configs/mapping_axis.json")
    if config["baseline"].get("record") != "benchmarks/reference_results/bank_storage_ledger.json":
        problems.append("the baseline must be the committed Phase 2M-A ledger")
    if set(config["measured_before_freezing"]) != set(systems):
        problems.append("measured_before_freezing must cover exactly the declared banks")
    return problems


def anchor_problems(config: dict) -> list[str]:
    """The storage anchor is the largest coefficient count a committed run held."""
    problems = []
    storage = config["clauses"]["storage"]
    anchor = storage["max_coefficient_occurrences"]
    committed = _ledger()["committed_records"]
    by_record = {row["record"]: row for row in committed}
    source = storage.get("anchor_record")
    row = by_record.get(source)
    if row is None:
        return [f"anchor record {source!r} is not in the ledger's committed records"]
    if row["coefficient_occurrences"] != anchor:
        problems.append(f"anchor {anchor} is not {source}'s committed "
                        f"{row['coefficient_occurrences']}")
    largest = max(r["coefficient_occurrences"] for r in committed)
    if anchor != largest:
        problems.append(f"the anchor must be the largest committed count, {largest}")
    if row["coefficient_occurrences"] * 24 != row["packed_operator_bytes"]:
        problems.append("the anchor row breaks cached_operator_bytes == 24 * T_coeff")
    if not row.get("peak_rss_bytes"):
        problems.append("the anchor row must carry the peak RSS it completed at")
    return problems


def consistency_problems(config: dict) -> list[str]:
    """The declaration's clauses against each other (the Phase 16B v3 lesson)."""
    from clifford_qc.multivector import TOL

    problems = []
    clauses = config["clauses"]
    ratio = clauses["word"]["max_ratio"]
    if not isinstance(ratio, int) or ratio <= 1:
        problems.append("the word clause's ratio must be an integer above one: the "
                        "combined universe contains the (S, H) one, so a ratio of "
                        "one or less fails before K is formed")
    anchor = clauses["storage"]["max_coefficient_occurrences"]
    if not isinstance(anchor, int):
        problems.append("the storage anchor must be an integer coefficient count")
    pruning = config["pruning"]
    declared, strict = float(pruning["declared_tolerance"]), float(pruning["strict_tolerance"])
    if declared != TOL:
        problems.append(f"the declared tolerance must be the multivector product's own "
                        f"{TOL}, not {declared}")
    if not declared < strict:
        problems.append("the strict tolerance must lie above the declared one")
    for name, measured in config["measured_before_freezing"].items():
        if isinstance(anchor, int) and measured["sh_coefficient_occurrences"] > anchor:
            problems.append(f"{name}: the (S, H) baseline alone exceeds the storage anchor")
        if not strict < measured["min_abs_hamiltonian_coefficient"]:
            problems.append(f"{name}: the strict tolerance would prune a Hamiltonian word")
        if measured["sh_word_universe"] > measured["sector_word_ceiling"]:
            problems.append(f"{name}: the (S, H) universe exceeds its sector ceiling")
    decisive = sorted(name for name, measured in config["measured_before_freezing"].items()
                      if measured["word_clause_can_bind"]
                      or measured["storage_clause_can_bind"])
    if sorted(config["decisive_banks"]) != decisive:
        problems.append(f"decisive_banks must be the banks a clause can bind: {decisive}")
    if not decisive:
        problems.append("no bank can bind either clause, so the preflight decides nothing")
    reachable = reachable_verdicts(config["measured_before_freezing"])
    if sorted(config["reachable_verdicts"]) != sorted(reachable):
        problems.append(f"reachable_verdicts must be {sorted(reachable)}")
    return problems


def reachable_verdicts(measured: dict) -> set[str]:
    """Verdicts some outcome can produce, given which banks can bind a clause."""
    options = []
    for row in measured.values():
        can_fail = row["word_clause_can_bind"] or row["storage_clause_can_bind"]
        options.append((True, False) if can_fail else (True,))
    reached = set()
    for eligible in itertools.product(*options):
        count = sum(eligible)
        reached.add("FULL" if count == len(eligible) else
                    "NONE" if count == 0 else "RESTRICTED")
    return reached


def claim_problems(config: dict) -> list[str]:
    problems = []
    boundary = config["claim_boundary"]
    for phrase in UNTENSED:
        if phrase in boundary:
            problems.append(f"claim boundary is tensed ({phrase!r}); a record that "
                            "inherits it would contradict its own counts")
    record = config["record_requirements"]
    if record.get("must_carry", {}).get("quantum_advantage_claim") is not False:
        problems.append("records must carry quantum_advantage_claim: false")
    if config["prespecified_followup"].get("permitted") != "none":
        problems.append("this declaration permits no follow-up; say so literally")
    if config["evidence"].get("label") != "structural":
        problems.append("the preflight's evidence label must be structural")
    return problems


def lineage_problems(config: dict, *, record_exists: bool | None = None) -> list[str]:
    """Inputs stay bound; the implementation is bound until a record names it."""
    problems = []
    record_exists = RECORD.exists() if record_exists is None else record_exists
    lineage = config["implementation_lineage"]
    for group in ("inputs", "implementation"):
        if group == "implementation" and record_exists:
            continue
        for row in lineage[group]:
            path = ROOT / row["path"]
            if not path.is_file():
                problems.append(f"lineage file missing: {row['path']}")
            elif _sha256(path) != row["sha256"]:
                problems.append(f"declared sha256 drifted for {row['path']}")
    inputs = {row["path"] for row in lineage["inputs"]}
    implementation = {row["path"] for row in lineage["implementation"]}
    for path in BOUND_INPUTS:
        if path not in inputs:
            problems.append(f"input {path} must be bound in implementation_lineage")
    for path in BOUND_IMPLEMENTATIONS:
        if path not in implementation:
            problems.append(f"implementation {path} must be bound in implementation_lineage")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    by_key = {spec["key"]: spec for spec in mapping["systems"]}
    for name in config["banks"]["systems"]:
        spec = by_key.get(name, {})
        for key in ("source", "provenance"):
            if key in spec and spec[key] not in inputs:
                problems.append(f"{name}: {key} is not bound in implementation_lineage")
    return problems


def revision_problems(config: dict) -> list[str]:
    """Revisions are numbered from zero, and none after the first saw a count."""
    problems = []
    revisions = config["revisions"]
    if not revisions or [row.get("revision") for row in revisions] != list(
            range(len(revisions))):
        return ["revisions must be numbered 0, 1, 2, ... in order"]
    for row in revisions[1:]:
        stated = str(row.get("second_moment_counts_at_revision", ""))
        if not stated.startswith("none"):
            problems.append(f"revision {row['revision']} does not state that no "
                            "second-moment count existed when it was made")
        if not row.get("changes"):
            problems.append(f"revision {row['revision']} lists no changes")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    for check in (ladder_problems, population_problems, anchor_problems,
                  consistency_problems, claim_problems, lineage_problems,
                  revision_problems):
        problems.extend(check(config))
    return problems


# ------------------------------------------------------------------ structural

def spin_block_masks(n: int) -> tuple[int, int]:
    """Low bits of the two-bit lanes on even (up) and odd (down) qubits.

    Every declared bank uses the interleaved spin convention, orbital ``p``
    spin ``s`` on qubit ``2p + s``, so the up block is the even qubits.
    """
    even = sum(1 << (2 * j) for j in range(0, n, 2))
    odd = sum(1 << (2 * j) for j in range(1, n, 2))
    return even, odd


def in_spin_parity_sector(n: int, code: int) -> bool:
    """True when a word commutes with both spin-block parities.

    ``Z`` over a block anticommutes with ``X`` and ``Y`` on that block, so a
    word commutes with it exactly when it carries an even number of ``X``/``Y``
    letters there. An operator conserving ``N_up`` and ``N_down`` has only
    such words, and so does every product of such operators.
    """
    even, odd = spin_block_masks(n)
    low = code & (even | odd)
    high = (code >> 1) & (even | odd)
    xy = low ^ high  # a lane is X or Y exactly when its two bits differ
    return (xy & even).bit_count() % 2 == 0 and (xy & odd).bit_count() % 2 == 0


def sector_ceiling(n: int) -> int:
    """Words commuting with both spin-block parities: ``4^n / 4`` for even ``n``."""
    if n < 2 or n % 2:
        raise ValueError("the interleaved spin sector needs an even qubit count")
    return 4 ** n // 4


def structural_quantities(config: dict, name: str) -> dict:
    """Every frozen number of one bank, recomputed from committed inputs."""
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace.elements import MatrixElementBank

    try:
        from benchmarks.run_mapping_axis import _build_model, _selected_generators
        from benchmarks.run_mapping_axis import load_config as load_mapping
    except ImportError:  # pragma: no cover - direct script execution
        from run_mapping_axis import _build_model, _selected_generators
        from run_mapping_axis import load_config as load_mapping

    spec = next(s for s in load_mapping()["systems"] if s["key"] == name)
    model, _ = _build_model(spec)
    if model.metadata.get("spin_convention") != "interleaved":
        raise ValueError(f"{name}: the sector ceiling assumes the interleaved convention")
    _, selected = _selected_generators(model, spec)
    rho = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho, model.hamiltonian, selected)
    bank.matrices()
    resources = bank.resources()
    n = model.n
    hamiltonian = bank.hamiltonian
    operator_words = list(hamiltonian.terms)
    for generator in selected:
        operator_words.extend(generator.mv.terms)
    universe = bank.word_set()
    ceiling = sector_ceiling(n)
    block_pairs = len(selected) * (len(selected) + 1) // 2
    sh_words = int(resources["word_universe"])
    sh_coefficients = int(resources["coefficient_occurrences"])
    clauses = config["clauses"]
    return {
        "n_qubits": n,
        "basis_size": len(selected),
        "block_pairs": block_pairs,
        "hamiltonian_terms": hamiltonian.nnz(),
        "max_generator_terms": max(g.mv.nnz() for g in selected),
        "sh_word_universe": sh_words,
        "sh_coefficient_occurrences": sh_coefficients,
        "sector_word_ceiling": ceiling,
        "operator_sector_violations": sum(
            not in_spin_parity_sector(n, code) for code in operator_words),
        "sh_sector_violations": sum(not in_spin_parity_sector(n, code) for code in universe),
        "word_clause_can_bind": ceiling > int(clauses["word"]["max_ratio"]) * sh_words,
        "storage_clause_can_bind": (
            sh_coefficients + block_pairs * ceiling
            > int(clauses["storage"]["max_coefficient_occurrences"])),
        "min_abs_hamiltonian_coefficient": min(abs(c) for c in hamiltonian.terms.values()),
    }


def structural_problems(config: dict, notes: list[str] | None = None,
                        computed: dict | None = None) -> list[str]:
    """Recompute and compare every frozen number, and the ceiling's premises."""
    notes = [] if notes is None else notes
    problems = []
    ledger = ledger_baseline()
    for name in config["banks"]["systems"]:
        declared = config["measured_before_freezing"][name]
        got = structural_quantities(config, name)
        if computed is not None:
            computed[name] = got
        for key in _INT_FIELDS + _BOOL_FIELDS:
            if declared.get(key) != got[key]:
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        for key in _FLOAT_FIELDS:
            if not _close(declared.get(key, float("nan")), got[key]):
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        row = ledger[name]
        for key, field in (("sh_word_universe", "word_universe"),
                           ("sh_coefficient_occurrences", "coefficient_occurrences"),
                           ("block_pairs", "complete_block_pairs"),
                           ("basis_size", "basis_size"), ("n_qubits", "n_qubits")):
            if got[key] != row[field]:
                problems.append(f"{name}: {key} {got[key]} is not the ledger's {row[field]}")
        if got["operator_sector_violations"] or got["sh_sector_violations"]:
            problems.append(f"{name}: a word leaves the spin-parity sector, so the "
                            "sector ceiling does not bound this bank")
        binds = [clause for clause in ("word", "storage")
                 if got[f"{clause}_clause_can_bind"]]
        notes.append(f"  {name}: n={got['n_qubits']} M={got['basis_size']} "
                     f"W_SH={got['sh_word_universe']} T_SH={got['sh_coefficient_occurrences']} "
                     f"ceiling={got['sector_word_ceiling']}; can bind: "
                     f"{', '.join(binds) if binds else 'neither clause'}")
    return problems


# ------------------------------------------------------------------ order

def _first_commit(path: Path):
    try:
        out = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", str(path)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commits = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return commits[-1] if commits else None


def _last_commit(path: Path):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip() if out.returncode == 0 else ""
    return commit or None


def _shallow_boundary() -> set[str]:
    """Commits whose parents a shallow clone cut off; git cannot date a path there."""
    try:
        out = subprocess.run(["git", "rev-parse", "--git-path", "shallow"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    shallow = ROOT / out.stdout.strip() if out.returncode == 0 else None
    if shallow is None or not shallow.is_file():
        return set()
    return {line.strip() for line in shallow.read_text().splitlines() if line.strip()}


def commit_order_problems(notes: list[str]) -> list[str]:
    """The config's last change must strictly precede the record's first commit."""
    if not RECORD.exists():
        notes.append("  commit order: no record yet, nothing to order against")
        return []
    config_commit, record_commit = _last_commit(CONFIG), _first_commit(RECORD)
    if config_commit is None or record_commit is None:
        notes.append("  commit order: SKIP (not committed yet, or no git history)")
        return []
    if {config_commit, record_commit} & _shallow_boundary():
        notes.append("  commit order: SKIP (the clone is too shallow to reach "
                     "the commits that order them)")
        return []
    if config_commit == record_commit:
        return ["the config's last change and the record share a commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config_commit, record_commit],
        cwd=ROOT, capture_output=True, timeout=30)
    if ancestor.returncode != 0:
        return [f"config commit {config_commit[:12]} is not an ancestor of record "
                f"commit {record_commit[:12]}"]
    notes.append(f"  commit order: config's last change {config_commit[:12]} precedes "
                 f"record {record_commit[:12]}")
    return []


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    notes: list[str] = []
    problems = static_problems(config)
    if not problems:
        problems.extend(structural_problems(config, notes))
        problems.extend(commit_order_problems(notes))
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print("OK Phase 15 preflight preregistration is complete, count-free, jointly "
          "satisfiable, and bound to its inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
