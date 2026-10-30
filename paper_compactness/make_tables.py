"""Generate every number the compactness paper typesets.

The paper reads six committed records under ``benchmarks/reference_results/``
and one derived diagnostic, ``data/start_weights.json``:

* ``packet_budget_convergence.json`` -- the declared column-budget comparison
  that is the paper's subject.  Its trajectories are decoded from the
  record's content-addressed payloads and nothing is recomputed: columns,
  Hamiltonian actions and determinant support are read prefix by prefix
  exactly as the campaign recorded them, and the declared readouts are applied
  with the campaign summarizer's own functions.
* ``packet_work_reuse.json``, ``packet_work_large.json``,
  ``sparse_packet_work.json``, ``exact_projection_cache.json`` and
  ``frame_coordinate_projection.json`` -- the separately declared campaigns
  that reduced the packet arm's cost.  Only the quantities the ledger quotes
  are extracted, with the medians their own summarizers use.

Where the prose states a pattern rather than a number -- which input reads
mixed, that Davidson stopped on an excited root, that a cache left the
trajectories bitwise identical -- the generator asserts it against the record
and refuses to emit fragments that would contradict the text.

Run ``python paper_compactness/make_tables.py`` after any change to a record.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TABLES = HERE / "tables"
DATA = HERE / "data"

RECORDS = ROOT / "benchmarks" / "reference_results"
BUDGET = RECORDS / "packet_budget_convergence.json"
REUSE = RECORDS / "packet_work_reuse.json"
LARGE = RECORDS / "packet_work_large.json"
SPARSE = RECORDS / "sparse_packet_work.json"
CACHE = RECORDS / "exact_projection_cache.json"
COORD = RECORDS / "frame_coordinate_projection.json"
SPARSE_DECLARATION = ROOT / "benchmarks" / "SPARSE_PACKET_WORK_DECLARATION.md"
START = DATA / "start_weights.json"

TABLE_SOURCES: dict[str, tuple[Path, ...]] = {
    "inputs.tex": (BUDGET, START),
    "thresholds.tex": (BUDGET,),
    "support.tex": (BUDGET,),
    "leaders.tex": (BUDGET,),
    "readouts.tex": (BUDGET,),
    "ledger.tex": (LARGE, SPARSE, CACHE, COORD),
    "numbers.tex": (BUDGET, START, REUSE, LARGE, SPARSE, SPARSE_DECLARATION, CACHE, COORD),
}

ARMS = ("packets_exhaustive", "packets_early", "selected_ci", "davidson")
# The leader comparison is between the three growth rules; the early policy is
# a screening variant of packets and is reported beside them, not against them.
RULES = ("packets_exhaustive", "selected_ci", "davidson")
ARM_LABELS = {
    "packets_exhaustive": "Packets",
    "packets_early": "Packets, early",
    "selected_ci": "Selected CI",
    "davidson": "Davidson",
}
ARM_SHORT = {"packets_exhaustive": "P", "selected_ci": "S", "davidson": "D"}

# Typeset names.  An input the record introduces without a name here stops the
# generator, so a new system is named deliberately rather than as a raw key.
INPUT_LABELS = {
    "hubbard_2x4_u4": r"$2{\times}4$ Hubbard",
    "hubbard_1x8_u4": r"$1{\times}8$ Hubbard chain",
    "beh2_stretched_full": r"BeH$_2$, stretched",
    "hubbard_2x3_u8": r"$2{\times}3$ Hubbard",
    "h2o_cas8e6o": r"H$_2$O CAS(8e,6o)",
}
INPUT_MACROS = {
    "hubbard_2x4_u4": "Primary",
    "hubbard_1x8_u4": "Chain",
    "beh2_stretched_full": "Beryllium",
    "hubbard_2x3_u8": "Ladder",
    "h2o_cas8e6o": "Water",
}
ROLE_LABELS = {
    "primary": "primary",
    "held_out_lattice": "held-out lattice",
    "held_out_molecule": "held-out molecule",
    "complete_span_lattice_control": "complete-span control",
    "development_complete_span_control": "development control",
}
UNIT_LABELS = {"t": "$t$", "Ha": "Ha"}
DECISIVE = ("hubbard_2x4_u4", "hubbard_1x8_u4", "beh2_stretched_full")


def git_blob_sha(path: Path) -> str:
    """Git blob identity, so a regenerated source forces a regenerated table."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _write(name: str, rows: list[str]) -> None:
    """Write one fragment with the blob identities of its sources and generator."""
    TABLES.mkdir(parents=True, exist_ok=True)
    header = [f"% source-git-blob-sha: {git_blob_sha(path)}" for path in TABLE_SOURCES[name]]
    header.append(f"% generator-git-blob-sha: {git_blob_sha(Path(__file__).resolve())}")
    (TABLES / name).write_text("\n".join(header + rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Formatting.  Every value is emitted through \ensuremath so one macro works in
# a table cell, in running prose and inside displayed mathematics.
# ---------------------------------------------------------------------------
def _math(body: str) -> str:
    return rf"\ensuremath{{{body}}}"


def _int(value: float | int) -> str:
    return _math(f"{int(round(value)):,}".replace(",", r"{,}"))


def _num(value: float, digits: int = 2) -> str:
    return _math(f"{value:.{digits}f}")


def _sig(value: float, figures: int = 3) -> str:
    """Fixed significant figures without switching to exponent form."""
    if value == 0:
        return _math("0")
    decimals = max(0, figures - 1 - math.floor(math.log10(abs(value))))
    return _math(f"{value:,.{decimals}f}".replace(",", r"{,}"))


def _sci(value: float, figures: int = 2) -> str:
    mantissa, exponent = f"{value:.{figures - 1}e}".split("e")
    if float(mantissa) == 1.0:  # a declared tolerance reads as a power of ten
        return _math(rf"10^{{{int(exponent)}}}")
    return _math(rf"{mantissa}\times10^{{{int(exponent)}}}")


SMALL_COUNTS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _count(value: int) -> str:
    """Design counts below eleven read as words in running prose."""
    return SMALL_COUNTS[value] if 0 <= value < len(SMALL_COUNTS) else _int(value)


def _floor(value: float, decimals: int = 1) -> str:
    """Round down, for 'at least' statements: 5.46 reads as 5.4, never 5.5."""
    scale = 10 ** decimals
    return _math(f"{math.floor(value * scale) / scale:.{decimals}f}")


def _pct(fraction: float, digits: int = 1) -> str:
    return _math(rf"{100.0 * fraction:.{digits}f}\%")


def _threshold(value: float) -> str:
    return _math(f"{value:g}")


def _list(items: list[str]) -> str:
    """'a, b and c', with the separators outside math so they typeset as text."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _error(value: float, floor: float) -> str:
    """Final errors at or below the declared complete-span tolerance are rounding."""
    if abs(value) <= floor:
        return _math(rf"<10^{{{round(math.log10(floor))}}}")
    return _sig(value)


def _input_label(name: str) -> str:
    try:
        return INPUT_LABELS[name]
    except KeyError:
        raise SystemExit(f"no typeset name for input {name!r} in make_tables.INPUT_LABELS") from None


def _require(condition: bool, message: str) -> None:
    """The prose states this pattern; refuse to generate if the record no longer shows it."""
    if not condition:
        raise SystemExit(f"record contradicts the manuscript: {message}")


# ---------------------------------------------------------------------------
# The budget-convergence comparison
# ---------------------------------------------------------------------------
class Budget:
    """Decoded trajectories of the declared comparison, one per input and arm."""

    def __init__(self) -> None:
        from benchmarks.metadata_io import read_json
        from benchmarks.run_packet_budget_convergence import decode_science

        self.record = read_json(BUDGET)
        self.config = self.record["config"]
        self.decisions = self.config["decisions"]
        self.thresholds = list(self.config["thresholds"])
        self.inputs = {case["name"]: case for case in self.config["inputs"]}
        for name in self.inputs:
            _input_label(name)
        self.sciences = {key: decode_science(payload)
                         for key, payload in self.record["scientific_payloads"].items()}
        self.workers: dict[tuple[str, str], list[dict]] = {}
        for worker in self.record["workers"]:
            _require(worker["status"] == "ok", f"cell {worker['input']}/{worker['arm']} did not complete")
            self.workers.setdefault((worker["input"], worker["arm"]), []).append(worker)
        for (name, arm), workers in self.workers.items():
            _require(len(workers) == self.inputs[name]["repeats"], f"{name}/{arm} lost a repeat")
            _require(len({w["scientific_sha256"] for w in workers}) == 1,
                     f"{name}/{arm} repeats differ scientifically")
        for name in self.inputs:
            for arm in ARMS:
                science = self.science(name, arm)
                _require(len(science["support"]) == len(science["trajectory"]),
                         f"{name}/{arm} support is not recorded at every prefix")

    def science(self, name: str, arm: str) -> dict:
        return self.sciences[self.workers[name, arm][0]["scientific_sha256"]]

    def reference(self, name: str) -> float:
        return self.record["references"][name]["singlet_reference"]

    def overlap(self, name: str, arm: str) -> float:
        return self.record["final_ground_state_overlaps"][self.workers[name, arm][0]["scientific_sha256"]]

    def hit(self, name: str, arm: str, threshold: float) -> int | None:
        from benchmarks.summarize_packet_budget_convergence import first_hit
        return first_hit(self.science(name, arm), self.reference(name), threshold)

    def at(self, name: str, arm: str, threshold: float) -> tuple[int, int, int] | None:
        """(columns, actions, support) at the first prefix within the threshold."""
        index = self.hit(name, arm, threshold)
        if index is None:
            return None
        science = self.science(name, arm)
        row = science["trajectory"][index]
        return row["dimension"], row["h_action_calls"], science["support"][index][0]

    def final(self, name: str, arm: str) -> dict:
        science = self.science(name, arm)
        row = science["trajectory"][-1]
        return {"columns": row["dimension"], "actions": row["h_action_calls"],
                "support": science["support"][-1][0], "error": row["energy"] - self.reference(name),
                "residual": row["residual_norm"], "stop": science["stop_reason"]}

    def support_at_columns(self, name: str, arm: str, columns: int) -> int:
        from benchmarks.summarize_packet_budget_convergence import support_at
        return support_at(self.science(name, arm), columns)

    def frontiers(self, name: str) -> list[int]:
        budget = self.inputs[name]["maximum_columns"]
        return [k for k in self.config["frontiers"] if k < budget] + [budget]

    def trivial(self, name: str, threshold: float) -> bool:
        """The shared start span already meets this threshold, for every arm."""
        return all(self.hit(name, arm, threshold) == 0 for arm in ARMS)

    def compactness(self, name: str) -> str:
        from benchmarks.summarize_packet_budget_convergence import compactness
        packets, sci = self.science(name, "packets_exhaustive"), self.science(name, "selected_ci")
        limited = [k for k in self.frontiers(name) if k >= self.decisions["frontier_minimum_columns"]]
        args = (self.reference(name), self.thresholds, limited, self.decisions["column_compactness_ratio"])
        p_more, s_more = compactness(packets, sci, *args), compactness(sci, packets, *args)
        return "mixed" if p_more == s_more else "packets" if p_more else "selected_ci"

    def action_ratios(self, name: str) -> list[float]:
        """Packets' actions over selected CI's at every threshold both reach."""
        ratios = []
        for threshold in self.thresholds:
            p, s = self.at(name, "packets_exhaustive", threshold), self.at(name, "selected_ci", threshold)
            if p and s:
                ratios.append(p[1] / s[1])
        return ratios

    def early_within_band(self, name: str) -> tuple[bool, list[tuple[int, float]]]:
        """Readout 3: columns within the ratio and frontier errors within the band."""
        from benchmarks.summarize_packet_budget_convergence import frontier_error, hit_columns
        packets, early = self.science(name, "packets_exhaustive"), self.science(name, "packets_early")
        reference = self.reference(name)
        slow = [t for t in self.thresholds if math.isfinite(hit_columns(packets, reference, t))
                and hit_columns(early, reference, t) > self.decisions["early_column_ratio"] * hit_columns(packets, reference, t)]
        gaps = [(k, frontier_error(early, reference, k) - frontier_error(packets, reference, k),
                 frontier_error(packets, reference, k)) for k in self.frontiers(name)]
        outside = [(k, gap) for k, gap, base in gaps
                   if abs(gap) > self.decisions["early_relative_frontier_error"] * abs(base)]
        return (not slow and not outside), outside

    def davidson_fewer(self, name: str) -> bool:
        """Readout 4: Davidson reaches every threshold in fewer columns than both restricted arms."""
        from benchmarks.summarize_packet_budget_convergence import hit_columns
        sciences = {arm: self.science(name, arm) for arm in RULES}
        reference = self.reference(name)
        return all(hit_columns(sciences["davidson"], reference, t)
                   < min(hit_columns(sciences["packets_exhaustive"], reference, t),
                         hit_columns(sciences["selected_ci"], reference, t))
                   for t in self.thresholds)


@functools.lru_cache(maxsize=None)
def budget() -> Budget:
    return Budget()


@functools.lru_cache(maxsize=None)
def start_weights() -> dict[str, float]:
    payload = json.loads(START.read_text(encoding="utf-8"))
    if payload.get("record_git_blob_sha") != git_blob_sha(BUDGET):
        raise SystemExit("data/start_weights.json was computed from a different budget record "
                         "(run paper_compactness/make_start_weights.py)")
    weights = {name: entry["ground_space_weight"] for name, entry in payload["weights"].items()}
    if set(weights) != set(budget().inputs):
        raise SystemExit("data/start_weights.json does not cover the budget record's inputs")
    return weights


def _weight(value: float) -> str:
    """A weight below the frame's numerical floor is rounding, not a measured overlap."""
    floor = budget().config["support_tolerance"]
    if value < floor:
        return _math(rf"<10^{{{round(math.log10(floor))}}}")
    return _sci(value) if value < 1e-2 else _num(value)


# ---------------------------------------------------------------------------
# Tables of the comparison
# ---------------------------------------------------------------------------
def inputs_table() -> None:
    b = budget()
    rows = []
    for name, case in b.inputs.items():
        groups = b.workers[name, "packets_exhaustive"][0]["cache"]["group_block_builds"]
        rows.append(" & ".join([
            _input_label(name), ROLE_LABELS[case["role"]], _int(case["sector_dimension"]),
            _int(case["singlet_dimension"]), _int(groups), _int(case["maximum_columns"]),
            UNIT_LABELS[case["energy_unit"]], _weight(start_weights()[name])]) + r" \\")
    _write("inputs.tex", rows)


def thresholds_table() -> None:
    b = budget()
    floor = b.decisions["complete_span_tolerance"]
    rows = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\le{t:g}$}}" for t in b.thresholds)
            + r" & \\",
            "Arm & " + " & ".join(r"$K$ & $W$" for _ in b.thresholds) + r" & Final error \\",
            r"\hline"]
    for index, (name, case) in enumerate(b.inputs.items()):
        if index:
            rows.append(r"\hline")
        rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{_input_label(name)}}}, "
                    rf"{ROLE_LABELS[case['role']]}; budget {_int(case['maximum_columns'])}, "
                    rf"energies in {UNIT_LABELS[case['energy_unit']]}}} \\")
        for arm in ARMS:
            cells = []
            for threshold in b.thresholds:
                point = b.at(name, arm, threshold)
                cells += [_int(point[0]), _int(point[1])] if point else ["---", "---"]
            rows.append(" & ".join([ARM_LABELS[arm], *cells, _error(b.final(name, arm)["error"], floor)]) + r" \\")
    _write("thresholds.tex", rows)


def support_table() -> None:
    b = budget()
    rows = ["Arm & " + " & ".join(rf"$\le{t:g}$" for t in b.thresholds) + r" & Matched $K$ \\", r"\hline"]
    for index, name in enumerate(DECISIVE):
        matched = b.final(name, "davidson")["columns"]
        if index:
            rows.append(r"\hline")
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{_input_label(name)}}}; "
                    rf"matched at $K={_int(matched)}$}} \\")
        for arm in ARMS:
            cells = []
            for threshold in b.thresholds:
                point = b.at(name, arm, threshold)
                cells.append(_int(point[2]) if point else "---")
            rows.append(" & ".join([ARM_LABELS[arm], *cells, _int(b.support_at_columns(name, arm, matched))]) + r" \\")
    _write("support.tex", rows)


def _leaders(name: str, threshold: float) -> dict[str, list[str]] | None:
    """Arms with the smallest columns, actions and support among those reaching the threshold."""
    b = budget()
    reached = {arm: b.at(name, arm, threshold) for arm in RULES}
    reached = {arm: point for arm, point in reached.items() if point is not None}
    if not reached or b.trivial(name, threshold):
        return None
    leaders = {}
    for axis, label in enumerate(("columns", "actions", "support")):
        best = min(point[axis] for point in reached.values())
        leaders[label] = [arm for arm in RULES if arm in reached and reached[arm][axis] == best]
    leaders["reached"] = list(reached)
    return leaders


def leaders_table() -> None:
    b = budget()
    rows = [r"Input & $\le$ & Columns & Actions & Support & $K_P/K_S$ & $W_P/W_S$ & $S_P/S_S$ \\", r"\hline"]
    for name in DECISIVE:
        for threshold in b.thresholds:
            leaders = _leaders(name, threshold)
            if leaders is None:
                continue
            cells = []
            for axis, label in enumerate(("columns", "actions", "support")):
                cells.append(", ".join(f"{ARM_SHORT[arm]} ({_int(b.at(name, arm, threshold)[axis])})"
                                       for arm in leaders[label]))
            p, s = b.at(name, "packets_exhaustive", threshold), b.at(name, "selected_ci", threshold)
            ratios = [_num(p[axis] / s[axis]) if p and s else "---" for axis in range(3)]
            rows.append(" & ".join([_input_label(name), _threshold(threshold), *cells, *ratios]) + r" \\")
    _write("leaders.tex", rows)


def _readout_cells(name: str) -> list[str]:
    b = budget()
    verdict = b.compactness(name)
    columns = {"mixed": "mixed", "packets": "packets", "selected_ci": "selected CI"}[verdict]
    if verdict == "mixed":
        work = "---"
    else:
        ratios = b.action_ratios(name)
        flagged = [r for r in ratios if r >= b.decisions["work_ratio"]]
        work = (rf"packets $\ge${_floor(min(ratios))}$\times$ actions" if flagged and len(flagged) == len(ratios)
                else "no penalty" if not flagged else "partial penalty")
    within, outside = b.early_within_band(name)
    if within:
        early = "within band"
    else:
        lower = all(gap < 0 for _, gap in outside)
        early = "outside band, lower error" if lower else "outside band"
    final = b.final(name, "davidson")
    if final["error"] > b.decisions["davidson_uninformative_if_converged_above_reference_by"]:
        davidson = "excited root"
    elif b.davidson_fewer(name):
        davidson = "fewer at every threshold"
    else:
        davidson = "ties at the start span"
    return [columns, work, early, davidson]


def readouts_table() -> None:
    b = budget()
    rows = []
    for name in b.inputs:
        rows.append(" & ".join([_input_label(name), *_readout_cells(name)]) + r" \\")
    _write("readouts.tex", rows)


# ---------------------------------------------------------------------------
# The engineering ledger
# ---------------------------------------------------------------------------
MIB = float(1 << 20)


@functools.lru_cache(maxsize=None)
def ledger() -> dict:
    """Quantities the ledger quotes, each a median over a campaign's repetitions."""
    from benchmarks.metadata_io import read_json
    from benchmarks.run_packet_work_large import unpack_observation
    from benchmarks.run_sparse_packet_work import unpack
    from benchmarks.run_frame_coordinate_projection import decode_science as decode_coordinates

    out: dict = {}
    median = statistics.median

    # Fixed-frame action reuse and shortlisting on H2O, first prefix within 1.6 mHa.
    reuse = read_json(REUSE)

    def first(worker):
        return next((r for r in worker["result"]["trajectory"] if r["target_reached"]), None)

    for arm in ("original", "reuse", "shortlist", "selected_ci"):
        hits = [first(w) for w in reuse["workers"] if w["arm"] == arm and w["status"] == "ok"]
        _require(hits and all(h is not None for h in hits), f"H2O {arm} did not reach its target")
        out[f"reuse_{arm}"] = {k: median(h[k] for h in hits) for k in ("dimension", "h_action_calls", "candidates_scored")}
    out["reuse_target"] = reuse["config"]["accuracy_target_hartree"]

    # The same changes on the 4,900-determinant lattice at a fixed column budget.
    large = unpack_observation(read_json(LARGE))
    out["large_budget"] = large["config"]["maximum_columns"]
    for arm in ("original", "reuse", "shortlist", "selected_ci"):
        group = [w for w in large["workers"] if w["arm"] == arm and w["status"] == "ok"]
        _require(len(group) == large["config"]["repeats"], f"larger-sector {arm} lost a repeat")
        out[f"large_{arm}"] = {
            "actions": median(w["result"]["trajectory"][-1]["h_action_calls"] for w in group),
            "scores": median(w["result"]["trajectory"][-1]["candidates_scored"] for w in group),
            "error": median(w["result"]["trajectory"][-1]["energy_error"] for w in group),
            "seconds": median(w["result"]["setup"]["wall_seconds"] + w["result"]["solver_wall_seconds"] for w in group),
            "rss": median(w["result"]["final_peak_rss_bytes"] for w in group) / MIB,
            "selections": [[r["selection"] for r in w["result"]["trajectory"]] for w in group],
        }
    _require(out["large_original"]["selections"] == out["large_reuse"]["selections"],
             "action reuse changed the larger-sector selections")
    _require(out["large_shortlist"]["error"] > out["large_original"]["error"],
             "shortlisting no longer changes the larger-sector trajectory")

    # Sparse setup and early marking before admissibility.
    sparse = unpack(read_json(SPARSE))
    for name in ("hubbard_2x4_u4", "hubbard_2x3_u8"):
        for arm in ("dense", "sparse", "early"):
            ok = [w for w in sparse["workers"] if w["input"] == name and w["arm"] == arm and w["status"] == "ok"]
            _require(bool(ok), f"sparse campaign has no {name}/{arm} observation")
            audits = [d for d in ok[0]["result"]["decisions"] if d["audit"]]
            out[f"sparse_{name}_{arm}"] = {
                "seconds": median(w["result"]["setup"]["wall_seconds"] + w["result"]["solver_wall_seconds"] for w in ok),
                "rss": median(w["result"]["final_peak_rss_bytes"] for w in ok) / MIB,
                "error": median(w["result"]["trajectory"][-1]["error"] for w in ok),
                "audits": len(audits), "missed": sum(d["audit_missed_winner"] for d in audits),
            }

    # The sparse campaign stated its speed clause in prose, not in its config.
    match = re.search(r">=(\d+)% lower\s+median setup-plus-solver time", SPARSE_DECLARATION.read_text(encoding="utf-8"))
    _require(match is not None, "the sparse declaration no longer states its speed clause")
    out["sparse_declared_reduction"] = int(match.group(1)) / 100

    # Exact projection caches: bitwise-identical payloads, then the joint speed/memory clause.
    cache = read_json(CACHE)
    promotion = cache["config"]["promotion"]
    out["promotion"] = promotion
    primary = [w for w in cache["workers"] if w["input"] == "hubbard_2x4_u4" and w["policy"] == "exhaustive"]
    for mode in ("none", "basis", "frame", "combined"):
        group = [w for w in primary if w["cache_mode"] == mode]
        out[f"cache_{mode}"] = {"seconds": median(w["setup_seconds"] + w["solver_seconds"] for w in group),
                                "rss": median(w["worker_peak_rss_bytes"] for w in group) / MIB}
    for case in cache["config"]["inputs"]:
        for comparison in cache["config"]["comparisons"]:
            group = [w for w in cache["workers"] if w["input"] == case["name"] and w["policy"] == comparison["policy"]]
            _require(len({w["scientific_sha256"] for w in group}) == 1,
                     f"caching changed the {case['name']}/{comparison['policy']} payload")
    out["cache_payloads"] = len(cache["scientific_payloads"])
    out["cache_cells"] = len(cache["workers"])

    # Frame-coordinate projection: arithmetic changes, so equality is checked decision by decision.
    coord = read_json(COORD)
    out["coord_promotion"] = coord["config"]["promotion"]
    sciences = {k: decode_coordinates(v) for k, v in coord["scientific_payloads"].items()}
    for policy in ("exhaustive", "early"):
        group = [w for w in coord["workers"] if w["input"] == "hubbard_2x4_u4" and w["policy"] == policy]
        for mode in ("combined", "coordinates"):
            cells = [w for w in group if w["cache_mode"] == mode]
            out[f"coord_{policy}_{mode}"] = {
                "seconds": median(w["setup_seconds"] + w["solver_seconds"] for w in cells),
                "rss": median(w["worker_peak_rss_bytes"] for w in cells) / MIB}
        swaps, winners_equal = set(), True
        for repeat in sorted({w["repeat"] for w in group}):
            pair = {w["cache_mode"]: sciences[w["scientific_sha256"]] for w in group if w["repeat"] == repeat}
            for a, b in zip(pair["coordinates"]["decisions"], pair["combined"]["decisions"]):
                winners_equal &= a["selected"] == b["selected"]
                if a["shortlist"] != b["shortlist"]:
                    _require(set(a["shortlist"]) == set(b["shortlist"]), "coordinate shortlist membership differs")
                    swaps.add(a["step"])
        out[f"coord_{policy}_swaps"] = sorted(swaps)
        out[f"coord_{policy}_winners_equal"] = winners_equal
    _require(not out["coord_exhaustive_swaps"] and out["coord_early_swaps"],
             "the ordered-shortlist failure is no longer confined to the early policy")
    _require(out["coord_early_winners_equal"] and out["coord_exhaustive_winners_equal"],
             "frame coordinates changed a selected candidate")
    out["coord_cells"] = len(coord["workers"])
    return out


def _arrow(before: float, after: float, fmt) -> str:
    return rf"{fmt(before)}$\to${fmt(after)}"


def ledger_table() -> None:
    led = ledger()
    one = functools.partial(_sig, figures=3)
    rss = functools.partial(_num, digits=0)
    rows = [
        " & ".join([
            "Frame action reuse", "identical selections",
            _arrow(led["large_original"]["actions"], led["large_reuse"]["actions"], _int) + " actions",
            _arrow(led["large_original"]["seconds"], led["large_reuse"]["seconds"], one),
            _arrow(led["large_original"]["rss"], led["large_reuse"]["rss"], rss),
            "kept experimental"]) + r" \\",
        " & ".join([
            "Residual shortlist", "heuristic; trajectory changes",
            _arrow(led["large_original"]["scores"], led["large_shortlist"]["scores"], _int) + " scores",
            _arrow(led["large_original"]["seconds"], led["large_shortlist"]["seconds"], one),
            _arrow(led["large_original"]["rss"], led["large_shortlist"]["rss"], rss),
            "kept experimental"]) + r" \\",
        " & ".join([
            "Sparse setup", "identical up to phase", "---",
            _arrow(led["sparse_hubbard_2x4_u4_dense"]["seconds"], led["sparse_hubbard_2x4_u4_sparse"]["seconds"], one),
            _arrow(led["sparse_hubbard_2x4_u4_dense"]["rss"], led["sparse_hubbard_2x4_u4_sparse"]["rss"], rss)
            + r"$^{a}$",
            "speed clause fails"]) + r" \\",
        " & ".join([
            "Projection caches", "bitwise identical", "---",
            _arrow(led["cache_none"]["seconds"], led["cache_combined"]["seconds"], one),
            _arrow(led["cache_none"]["rss"], led["cache_combined"]["rss"], rss),
            "memory clause fails"]) + r" \\",
        " & ".join([
            "Frame coordinates", "shortlist order differs", "---",
            _arrow(led["coord_exhaustive_combined"]["seconds"], led["coord_exhaustive_coordinates"]["seconds"], one),
            _arrow(led["coord_exhaustive_combined"]["rss"], led["coord_exhaustive_coordinates"]["rss"], rss),
            "equivalence fails"]) + r" \\",
    ]
    _write("ledger.tex", rows)


# ---------------------------------------------------------------------------
# Every quantity quoted in the prose
# ---------------------------------------------------------------------------
def _assert_patterns() -> None:
    """Patterns the prose states in words; each must still hold in the record."""
    b = budget()
    d = b.decisions
    _require(b.compactness("hubbard_2x4_u4") == "mixed", "the primary no longer reads mixed")
    _require(b.compactness("hubbard_1x8_u4") == "packets", "the chain no longer favours packets")
    _require(b.compactness("beh2_stretched_full") == "mixed", "BeH2 no longer reads mixed")
    _require(any(b.compactness(name) != "packets" for name in DECISIVE), "the decision is no longer negative")
    for name, case in b.inputs.items():
        if "control" in case["role"]:
            for arm in ARMS:
                final = b.final(name, arm)
                _require(abs(final["error"]) <= b.decisions["complete_span_tolerance"],
                         f"{name}/{arm} misses the complete-span reference")
    primary = b.final("hubbard_2x4_u4", "davidson")
    _require(primary["stop"] == "no_admissible_direction", "the primary Davidson stop changed")
    _require(b.overlap("hubbard_2x4_u4", "davidson") < 1e-6, "the primary Davidson state now overlaps the ground space")
    _require(b.davidson_fewer("hubbard_1x8_u4"), "Davidson no longer leads the chain at every threshold")
    for name in DECISIVE:
        for threshold in b.thresholds:
            leaders = _leaders(name, threshold)
            if leaders is None or "davidson" not in leaders["reached"] or len(leaders["reached"]) < 2:
                continue
            _require("davidson" in leaders["columns"],
                     f"Davidson does not lead columns on {name} at {threshold}")
            _require("davidson" in leaders["actions"], f"Davidson does not lead actions on {name} at {threshold}")
            _require("davidson" not in leaders["support"], f"Davidson leads support on {name} at {threshold}")
            _require(not set(leaders["columns"]) & set(leaders["actions"]) & set(leaders["support"]),
                     f"one rule leads on every yardstick on {name} at {threshold}")
    # Selected CI needs fewer actions than packets wherever both reach a threshold,
    # except one early BeH2 threshold that packets reach before selected CI's
    # up-front action table pays for itself.  The prose names that exception.
    exceptions = []
    for name in DECISIVE:
        for threshold in b.thresholds:
            p, s = b.at(name, "packets_exhaustive", threshold), b.at(name, "selected_ci", threshold)
            if p and s and not b.trivial(name, threshold) and p[1] < s[1]:
                exceptions.append((name, threshold))
    _require(exceptions == [("beh2_stretched_full", b.thresholds[1])],
             f"the thresholds where packets need fewer actions changed: {exceptions}")
    # Packets never need more columns than selected CI at a shared threshold,
    # and clear the declared margin only on the chain.
    for name in DECISIVE:
        for threshold in b.thresholds:
            p, s = b.at(name, "packets_exhaustive", threshold), b.at(name, "selected_ci", threshold)
            if p and s and not b.trivial(name, threshold):
                _require(p[0] <= s[0], f"selected CI needs fewer columns on {name} at {threshold}")
                _require((p[0] <= d["column_compactness_ratio"] * s[0]) == (name == "hubbard_1x8_u4"),
                         f"the declared column margin pattern changed on {name} at {threshold}")
    # Selected CI's action count saturates at the singlet frame: it pays for the whole table.
    for name in DECISIVE:
        _require(b.final(name, "selected_ci")["actions"] == b.inputs[name]["singlet_dimension"],
                 f"selected CI's actions on {name} differ from the frame dimension")
    weights = start_weights()
    _require(weights["hubbard_1x8_u4"] < weights["hubbard_2x4_u4"],
             "the chain's start weight is no longer below the primary's")


def numbers_macros() -> None:
    b = budget()
    led = ledger()
    weights = start_weights()
    _assert_patterns()
    d = b.decisions
    macros: dict[str, str] = {}

    # Design.
    macros["pkCells"] = _int(len(b.record["workers"]))
    macros["pkPayloads"] = _int(len(b.sciences))
    macros["pkInputCount"] = _count(len(b.inputs))
    macros["pkThresholdList"] = _list([_threshold(t) for t in b.thresholds])
    for key, value in zip(("One", "Tenth"), b.thresholds):
        macros[f"pkThreshold{key}"] = _threshold(value)
    macros["pkThresholdTight"] = _threshold(b.thresholds[-1])
    repeats = sorted({case["repeats"] for case in b.inputs.values()})
    macros["pkMinRepeats"] = _count(repeats[0])
    macros["pkMaxRepeats"] = _count(repeats[-1])
    macros["pkTightTarget"] = _math(f"{1000 * b.thresholds[-1]:g}") + r"\,mHa"
    macros["pkCompactRatio"] = _threshold(d["column_compactness_ratio"])
    macros["pkWorkRatio"] = _threshold(d["work_ratio"])
    macros["pkEarlyColumnRatio"] = _threshold(d["early_column_ratio"])
    macros["pkEarlyFrontierBand"] = _pct(d["early_relative_frontier_error"], 0)
    macros["pkFrontierMinimum"] = _int(d["frontier_minimum_columns"])
    macros["pkFrontierList"] = _list([_int(k) for k in b.config["frontiers"]])
    macros["pkCompleteSpanTolerance"] = _sci(d["complete_span_tolerance"], 1)
    macros["pkResidualTolerance"] = _sci(b.config["residual_tolerance"], 1)
    macros["pkSupportTolerance"] = _sci(b.config["support_tolerance"], 1)
    macros["pkTieTolerance"] = _sci(b.config["score_tie_tolerance"], 1)
    macros["pkShift"] = _threshold(b.config["preconditioner_shift"])
    macros["pkShortlistFraction"] = _pct(b.config["shortlist_fraction"], 0)
    macros["pkAuditInterval"] = _count(b.config["audit_interval"])
    macros["pkWorkerTimeout"] = _int(b.config["worker_timeout_seconds"])
    from benchmarks.check_packet_budget_convergence import FULL_REPLAY_DIMENSION
    macros["pkFullReplayDimension"] = _int(FULL_REPLAY_DIMENSION)
    starts = {len(b.science(name, arm)["trajectory"][0]["selection"]["initial_csf_indices"])
              for name in b.inputs for arm in ARMS}
    _require(len(starts) == 1, "the arms no longer share one start-span size")
    macros["pkStartColumns"] = _count(starts.pop())

    # Inputs.
    macros["pkPrimaryDeterminants"] = _int(b.inputs["hubbard_2x4_u4"]["sector_dimension"])
    for name, key in (("hubbard_2x4_u4", "Primary"), ("hubbard_2x3_u8", "Ladder")):
        model = b.inputs[name]["model"]
        macros[f"pk{key}Interaction"] = _threshold(model["U"] / model["t"])
    _require(b.inputs["hubbard_1x8_u4"]["model"]["U"] == b.inputs["hubbard_2x4_u4"]["model"]["U"]
             and b.inputs["hubbard_1x8_u4"]["sector_dimension"] == b.inputs["hubbard_2x4_u4"]["sector_dimension"],
             "the chain no longer shares the primary's parameters and sector")
    macros["pkPrimaryName"] = r"$2{\times}4$ Hubbard lattice"
    macros["pkChainName"] = r"$1{\times}8$ Hubbard chain"
    macros["pkLadderName"] = r"$2{\times}3$ Hubbard lattice"
    macros["pkWaterName"] = r"H$_2$O CAS(8e,6o)"
    macros["pkStartSpace"] = r"CAS(2e,2o)"
    macros["pkPrimarySpinBlock"] = _int(b.workers["hubbard_2x4_u4", "packets_exhaustive"][0]["setup"]["largest_spin_block"])

    # Columns.
    one, tenth, hundredth, tight = b.thresholds
    p1, s1, d1 = (b.at("hubbard_2x4_u4", arm, one) for arm in RULES)
    macros["pkPrimaryPacketsColumnsOne"] = _int(p1[0])
    macros["pkPrimarySciColumnsOne"] = _int(s1[0])
    macros["pkPrimaryColumnRatioOne"] = _num(p1[0] / s1[0])
    _require(p1[0] / s1[0] > d["column_compactness_ratio"], "the primary column ratio now meets the rule")
    pt = b.at("hubbard_2x4_u4", "packets_exhaustive", tenth)
    _require(pt is not None and b.at("hubbard_2x4_u4", "selected_ci", tenth) is None,
             "the primary 0.1 threshold is no longer packets-only")
    macros["pkPrimaryPacketsColumnsTenth"] = _int(pt[0])
    macros["pkPrimaryPacketsFinalError"] = _sig(b.final("hubbard_2x4_u4", "packets_exhaustive")["error"])
    macros["pkPrimarySciFinalError"] = _sig(b.final("hubbard_2x4_u4", "selected_ci")["error"])
    chain_p = [b.at("hubbard_1x8_u4", "packets_exhaustive", t) for t in (one, tenth, hundredth)]
    chain_s = [b.at("hubbard_1x8_u4", "selected_ci", t) for t in (one, tenth, hundredth)]
    _require(all(chain_p) and chain_s[2] is None and chain_s[0] and chain_s[1], "the chain threshold pattern changed")
    macros["pkChainPacketsColumns"] = _list([_int(p[0]) for p in chain_p])
    macros["pkChainSciColumns"] = _list([_int(s[0]) for s in chain_s[:2]])
    macros["pkChainPacketThresholds"] = _list([_threshold(t) for t in (one, tenth, hundredth)])
    macros["pkChainPacketsFinalError"] = _sig(b.final("hubbard_1x8_u4", "packets_exhaustive")["error"])
    macros["pkChainEarlyFinalError"] = _sig(b.final("hubbard_1x8_u4", "packets_early")["error"])
    bp, bs, bd = (b.at("beh2_stretched_full", arm, tight) for arm in RULES)
    macros["pkBerylliumPacketsColumnsTight"] = _int(bp[0])
    macros["pkBerylliumSciColumnsTight"] = _int(bs[0])
    macros["pkBerylliumColumnRatioTight"] = _num(bp[0] / bs[0])
    _require(bp[0] / bs[0] > d["column_compactness_ratio"], "the BeH2 column ratio now meets the rule")
    macros["pkBerylliumDavidsonColumnsTight"] = _int(bd[0])
    bp_tenth, bs_tenth = (b.at("beh2_stretched_full", arm, tenth) for arm in ("packets_exhaustive", "selected_ci"))
    macros["pkBerylliumPacketsActionsTenth"] = _int(bp_tenth[1])
    macros["pkBerylliumSciActionsTenth"] = _int(bs_tenth[1])
    macros["pkBerylliumColumnsTenth"] = _int(bp_tenth[0])
    _require(bp[2] == bs[2], "packets and selected CI no longer meet on one BeH2 support")
    macros["pkBerylliumPacketsSupportTight"] = _int(bp[2])
    macros["pkBerylliumDavidsonSupportTight"] = _int(bd[2])
    dc = b.at("hubbard_1x8_u4", "davidson", tight)
    macros["pkChainDavidsonColumnsTight"] = _int(dc[0])
    macros["pkChainDavidsonSupportTight"] = _int(dc[2])
    macros["pkLadderDavidsonColumnsTight"] = _int(b.at("hubbard_2x3_u8", "davidson", tight)[0])
    macros["pkLadderPacketsColumnsTight"] = _int(b.at("hubbard_2x3_u8", "packets_exhaustive", tight)[0])
    macros["pkLadderSciColumnsTight"] = _int(b.at("hubbard_2x3_u8", "selected_ci", tight)[0])

    # Actions.
    ratios = b.action_ratios("hubbard_1x8_u4")
    macros["pkChainMinActionRatio"] = _floor(min(ratios))
    pf, sf, ef = (b.final("hubbard_2x4_u4", arm) for arm in ("packets_exhaustive", "selected_ci", "packets_early"))
    _require(pf["columns"] == ef["columns"], "early and exhaustive primary budgets differ")
    macros["pkPrimaryPacketsActions"] = _int(pf["actions"])
    macros["pkPrimarySciActions"] = _int(sf["actions"])
    macros["pkPrimaryEarlyActions"] = _int(ef["actions"])
    macros["pkPrimaryActionRatio"] = _int(pf["actions"] / sf["actions"])
    macros["pkPrimaryEarlyActionFactor"] = _num(pf["actions"] / ef["actions"], 1)

    # Support.
    macros["pkPrimaryPacketsSupportOne"] = _int(p1[2])
    macros["pkPrimarySciSupportOne"] = _int(s1[2])
    macros["pkPrimaryDavidsonColumnsOne"] = _int(d1[0])
    macros["pkPrimaryDavidsonSupportOne"] = _int(d1[2])
    cp1, cs1 = (b.at("hubbard_1x8_u4", arm, one) for arm in ("packets_exhaustive", "selected_ci"))
    cp2, cs2 = (b.at("hubbard_1x8_u4", arm, tenth) for arm in ("packets_exhaustive", "selected_ci"))
    _require(cp1[2] < cs1[2] and cs2[2] < cp2[2], "the chain support lead no longer changes hands")
    macros["pkChainPacketsSupportOne"] = _int(cp1[2])
    macros["pkChainSciSupportOne"] = _int(cs1[2])
    macros["pkChainPacketsSupportTenth"] = _int(cp2[2])
    macros["pkChainSciSupportTenth"] = _int(cs2[2])
    matched = b.final("hubbard_2x4_u4", "davidson")["columns"]
    macros["pkPrimaryMatchedColumns"] = _int(matched)
    macros["pkPrimaryPacketsSupportMatched"] = _int(b.support_at_columns("hubbard_2x4_u4", "packets_exhaustive", matched))
    macros["pkPrimarySciSupportMatched"] = _int(b.support_at_columns("hubbard_2x4_u4", "selected_ci", matched))

    # The Davidson reference on the primary.
    final = b.final("hubbard_2x4_u4", "davidson")
    macros["pkPrimaryDavidsonStopColumns"] = _int(final["columns"])
    macros["pkPrimaryDavidsonResidual"] = _sci(final["residual"])
    macros["pkPrimaryDavidsonError"] = _sig(final["error"], 4)
    macros["pkPrimaryDavidsonOverlap"] = _sci(b.overlap("hubbard_2x4_u4", "davidson"))
    macros["pkPrimaryStartWeight"] = _weight(weights["hubbard_2x4_u4"])
    macros["pkChainStartWeight"] = _weight(weights["hubbard_1x8_u4"])
    macros["pkPrimaryGap"] = _sig(b.record["references"]["hubbard_2x4_u4"]["singlet_gap_to_next"])

    # The earlier campaigns that motivated the comparison.
    macros["pkWaterPacketColumns"] = _int(led["reuse_original"]["dimension"])
    macros["pkWaterSciColumns"] = _int(led["reuse_selected_ci"]["dimension"])
    macros["pkWaterPacketActions"] = _int(led["reuse_original"]["h_action_calls"])
    macros["pkWaterSciActions"] = _int(led["reuse_selected_ci"]["h_action_calls"])
    macros["pkWaterReuseActions"] = _int(led["reuse_reuse"]["h_action_calls"])
    macros["pkWaterShortlistScores"] = _int(led["reuse_shortlist"]["candidates_scored"])
    macros["pkWaterPacketScores"] = _int(led["reuse_original"]["candidates_scored"])
    macros["pkLargeBudget"] = _int(led["large_budget"])
    macros["pkLargePacketsError"] = _num(led["large_original"]["error"])
    macros["pkLargeSciError"] = _num(led["large_selected_ci"]["error"])

    # The ledger in prose.
    macros["pkReuseActionsBefore"] = _int(led["large_original"]["actions"])
    macros["pkReuseActionsAfter"] = _int(led["large_reuse"]["actions"])
    macros["pkReuseSecondsBefore"] = _sig(led["large_original"]["seconds"])
    macros["pkReuseSecondsAfter"] = _sig(led["large_reuse"]["seconds"])
    macros["pkReuseRssBefore"] = _num(led["large_original"]["rss"], 0)
    macros["pkReuseRssAfter"] = _num(led["large_reuse"]["rss"], 0)
    macros["pkLargeSciSeconds"] = _sig(led["large_selected_ci"]["seconds"])
    macros["pkShortlistErrorChange"] = _pct(led["large_shortlist"]["error"] / led["large_original"]["error"] - 1, 2)
    dense, sparse_, early = (led[f"sparse_hubbard_2x4_u4_{a}"] for a in ("dense", "sparse", "early"))
    macros["pkSparseRssBefore"] = _num(dense["rss"], 0)
    macros["pkSparseRssAfter"] = _num(sparse_["rss"], 0)
    sparse_reduction = 1 - sparse_["seconds"] / dense["seconds"]
    _require(sparse_reduction < led["sparse_declared_reduction"], "sparse setup now meets its speed clause")
    macros["pkSparseReduction"] = _pct(sparse_reduction, 1)
    macros["pkSparseDeclaredReduction"] = _pct(led["sparse_declared_reduction"], 0)
    ladder_sparse, ladder_early = led["sparse_hubbard_2x3_u8_sparse"], led["sparse_hubbard_2x3_u8_early"]
    macros["pkLadderEarlyAudits"] = _count(ladder_early["audits"])
    macros["pkLadderEarlyMissed"] = _count(ladder_early["missed"])
    macros["pkLadderEarlyErrorChange"] = _pct(ladder_early["error"] / ladder_sparse["error"] - 1, 2)
    _require(ladder_early["missed"] > 0, "the held-out lattice audit no longer misses a winner")
    reduction = 1 - led["cache_combined"]["seconds"] / led["cache_none"]["seconds"]
    growth = led["cache_combined"]["rss"] / led["cache_none"]["rss"] - 1
    _require(reduction >= led["promotion"]["primary_minimum_setup_plus_solver_reduction"]
             and growth > 0, "the projection-cache clause no longer fails on memory alone")
    macros["pkCacheReduction"] = _pct(reduction, 2)
    macros["pkCacheRssGrowth"] = _pct(growth, 2)
    macros["pkCacheMinimumReduction"] = _pct(led["promotion"]["primary_minimum_setup_plus_solver_reduction"], 0)
    macros["pkCacheCells"] = _int(led["cache_cells"])
    macros["pkCachePayloads"] = _count(led["cache_payloads"])
    exh = 1 - led["coord_exhaustive_coordinates"]["seconds"] / led["coord_exhaustive_combined"]["seconds"]
    ear = 1 - led["coord_early_coordinates"]["seconds"] / led["coord_early_combined"]["seconds"]
    macros["pkCoordinateExhaustiveReduction"] = _pct(exh, 1)
    macros["pkCoordinateEarlyReduction"] = _pct(ear, 1)
    macros["pkCoordinateSwapSteps"] = _list([_int(step) for step in led["coord_early_swaps"]])

    rows = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    _write("numbers.tex", rows)


def main(tables: Path | None = None) -> Path:
    """Generate every fragment, optionally into a scratch tree for the gate to compare."""
    global TABLES
    original = TABLES
    TABLES = tables if tables is not None else TABLES
    try:
        inputs_table()
        thresholds_table()
        support_table()
        leaders_table()
        readouts_table()
        ledger_table()
        numbers_macros()
        return TABLES
    finally:
        TABLES = original


if __name__ == "__main__":
    print(main())
