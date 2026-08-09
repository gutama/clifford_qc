"""Generate all numerical table fragments for the standalone DA-CASE paper."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
LADDER = ROOT / "benchmarks" / "reference_results" / "acase_ladder_summary.csv"
H4 = ROOT / "benchmarks" / "reference_results" / "fcidump_h4.json"
WARM = ROOT / "benchmarks" / "reference_results" / "warm_start_h4.json"
KRYLOV_WIDTH = ROOT / "benchmarks" / "reference_results" / "krylov_width.json"
MATCHED = ROOT / "benchmarks" / "reference_results" / "matched_h4.json"
PHASE12_PRIMARY = ROOT / "benchmarks" / "results" / "phase12_paper_b_five_system.json"
CLIFFORD_HIERARCHY = tuple(
    ROOT / "benchmarks" / "reference_results" / f"clifford_hierarchy_{system}.json"
    for system in ("h4", "beh2"))
DIMER = ROOT / "examples" / "data" / "wannier_hubbard_dimer.json"
RESPONSE = HERE / "data" / "response_bootstrap.json"
RESPONSE_ILL = HERE / "data" / "response_bootstrap_illconditioned.json"


def _git_blob_sha(path: Path) -> str:
    """Git blob identity for byte-exact generated-artifact bindings."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _write(name: str, rows: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text("\n".join(rows) + "\n")


def _sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10 ** exponent
    if exponent == 0:
        return rf"${mantissa:.{digits}f}$"
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def _kappa(value: float) -> str:
    if abs(value) < 10.0:
        return f"{value:.3g}"
    return _sci(value, 1)


def dimer_table() -> None:
    model = json.loads(DIMER.read_text())
    t = abs(float(model["one_body"][0][1]))
    u = float(model["onsite_u"][0])
    root = math.sqrt(u * u + 16.0 * t * t)
    energy = 0.5 * (u - root)
    double = 0.25 * (1.0 - u / root)
    spin = -0.75 * (1.0 - 2.0 * double)
    gap = -energy
    weight = 1.0 - 2.0 * double
    chi = 2.0 * weight / gap
    rows = [
        rf"Ground energy $E_0$ & {energy:.9f} eV & "
        r"$(U-\sqrt{U^2+16t^2})/2$ \\",
        rf"Double occupancy per site $d$ & {double:.9f} & "
        r"$(\partial E_0/\partial U)/2$ \\",
        rf"$\langle\mathbf S_0\!\cdot\!\mathbf S_1\rangle$ & "
        rf"{spin:.9f} & $-\frac34(1-2d)$ \\",
        r"$\langle S^2\rangle$ & $<10^{-12}$ & singlet invariant \\",
        rf"Staggered-spin gap $\omega$ & {gap:.9f} eV & $-E_0$ \\",
        rf"Staggered-spin weight $w$ & {weight:.9f} & $1-2d$ \\",
        rf"Static susceptibility $\chi(0)$ & {chi:.9f} eV$^{{-1}}$ & "
        r"$2w/\omega$ \\",
    ]
    _write("dimer_results.tex", rows)


def h4_table() -> None:
    record = json.loads(H4.read_text())
    ref = record["independent_reference"]["fci_energy"]
    mapped = record["mapped_oracle"]
    adaptive = record["adaptive_acase"]
    sd = record["complete_singles_doubles"]
    rows = [
        rf"External determinant FCI & {ref:.15f} & --- & --- \\",
        rf"Mapped sector oracle & {mapped['ground_energy']:.15f} & "
        rf"{_sci(mapped['external_fci_error'])} Ha & 36 \\",
        rf"DA-CASE & {adaptive['ground_energy']:.15f} & "
        rf"{adaptive['error_millihartree']:.6f} mHa & "
        rf"{adaptive['basis_size']} \\",
        rf"Complete singles/doubles & {sd['ground_energy']:.15f} & "
        rf"{sd['error_millihartree']:.6f} mHa & {sd['basis_size']} \\",
    ]
    _write("h4_results.tex", rows)


def warm_start_table() -> None:
    """Hybrid accuracy plus the exact-simulation resource ledger."""
    record = json.loads(WARM.read_text())
    cold = record["cold"]
    rows = [
        rf"Hartree--Fock determinant & 0 & 0 & 0 & --- & "
        rf"{cold['basis_size']} & {cold['error_millihartree']:.3f} & "
        rf"{_kappa(cold['condition_number'])} & {cold['word_universe']} \\",
    ]
    for row in record["warm"]:
        rows.append(
            rf"ADAPT-VQE state, $k={row['adapt_operators']}$ & "
            rf"{row['adapt_gradient_evaluations']} & "
            rf"{row['adapt_optimizer_evaluations']} & "
            rf"{row['adapt_state_preparation_operators']} & "
            rf"{row['adapt_error_millihartree']:.3f} & {row['basis_size']} & "
            rf"{row['error_millihartree']:.3f} & "
            rf"{_kappa(row['condition_number'])} & {row['word_universe']} \\")
    _write("warm_start_results.tex", rows)


def _krylov_widths() -> dict[str, int]:
    if not KRYLOV_WIDTH.exists():
        return {}
    record = json.loads(KRYLOV_WIDTH.read_text())
    widths = {}
    for row in record["rows"]:
        if not row.get("word_universe_certificate_passed", False):
            raise ValueError(f"uncertified Krylov width for {row['system']}")
        widths[row["system"]] = row["word_universe"]
    return widths


def _ladder_rows() -> list[dict[str, str]]:
    with LADDER.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ladder_table() -> None:
    rows = _ladder_rows()
    by_key = {(r["system"], r["method"]): r for r in rows}
    specs = [
        ("H$_4$, 0.9 \\AA", "h4_chain(r=0.9)", "acase_exact", "krylov", "Ha"),
        ("H$_4$, 1.8 \\AA", "h4_chain(r=1.8)", "acase_exact", "krylov", "Ha"),
        ("H$_2$O (4e,4o), $2R_e$", "h2o_4e4o(scale=2.0)",
         "acase_exact", "krylov", "Ha"),
        ("H$_2$O (8e,6o)", "h2o_8e6o(scale=1.0)",
         "acase_exact", "generator_coordinate", "Ha"),
        ("Hubbard $2\\times2$", "hubbard(2x2,t=1.0,U=4.0,obc)",
         "acase_level4", "krylov", "$t$"),
        ("Hubbard $2\\times3$", "hubbard(2x3,t=1.0,U=4.0,obc)",
         "acase_level4", "generator_coordinate", "$t$"),
        ("Kitaev $2\\times2$", "kitaev(2x2,K=1.0,1.0,1.0,obc)",
         "acase_exact", "krylov", "$K$"),
    ]
    out = []
    labels = {
        "krylov": "Krylov",
        "generator_coordinate": "gen.-coord.",
    }
    krylov_width = _krylov_widths()
    for display, system, acase_method, comparator_method, unit in specs:
        a = by_key[(system, acase_method)]
        b = by_key[(system, comparator_method)]
        if comparator_method == "krylov":
            width = krylov_width.get(system)
            comparator_w = "n/a" if width is None else str(width)
        else:
            comparator_w = b["word_universe"] or "n/a"
        out.append(
            f"{display} & {a['basis_size']} & "
            f"{_sci(abs(float(a['error'])))} {unit} & "
            f"{_kappa(float(a['condition_number']))} & {a['word_universe']} & "
            f"{labels[comparator_method]} & {b['basis_size']} & "
            f"{_sci(abs(float(b['error'])))} {unit} & "
            f"{_kappa(float(b['condition_number']))} & {comparator_w} \\\\")
    _write("ladder_results.tex", out)


def phase12_primary_table() -> None:
    """Matched-budget Phase 12 rows from the committed five-system record."""
    record = json.loads(PHASE12_PRIMARY.read_text())
    by_key = {row["system_key"]: row for row in record["systems"]}
    methods = (
        "budget_selected_ci",
        "matched_selected_ci",
        "acase",
        "qsci_haar_dressed_acase",
        "fixed_krylov",
    )
    labels = (
        ("Hubbard $2\\times2$", "hubbard_2x2"),
        ("Hubbard $2\\times3$", "hubbard_2x3"),
        ("H$_4$, 0.9 \\AA", "h4_equilibrium"),
        ("H$_4$, 1.8 \\AA", "h4_stretched"),
        ("H$_4$ FCIDUMP", "fcidump_h4_equilibrium"),
    )

    def fmt(value: float) -> str:
        """Four significant figures, with small values in scientific form."""
        if value == 0.0:
            return "$0$"
        exponent = int(math.floor(math.log10(abs(value))))
        if abs(value) >= 0.1:
            decimals = max(0, 3 - exponent)
            return "$" + f"{value:.{decimals}f}" + "$"
        return _sci(value, 3)

    out = [
        f"% source-git-blob-sha: {_git_blob_sha(PHASE12_PRIMARY)}",
        f"% generator-git-blob-sha: {_git_blob_sha(Path(__file__).resolve())}",
    ]
    for display, key in labels:
        arms = {row["method"]: row for row in by_key[key]["arms"]}
        missing = [method for method in methods if method not in arms]
        if missing:
            raise ValueError(f"Phase 12 {key} is missing arms: {missing}")
        if any(arms[method]["M"] != 7 for method in methods):
            raise ValueError(f"Phase 12 {key} does not satisfy the M=7 table contract")
        out.append(
            display + " & "
            + " & ".join(fmt(abs(float(arms[method]["absolute_error"])))
                         for method in methods)
            + r" \\"
        )
    _write("phase12_primary.tex", out)


def response_table() -> None:
    record = json.loads(RESPONSE.read_text())
    exact = record["exact"]
    measured = record["measured"]
    exact_line = exact["lines"][0]
    line = measured["lines"][0]
    chi = measured["susceptibility_per_ev"]
    out = [
        rf"Gap $\omega$ (eV) & {exact_line['gap_ev']:.9f} & "
        rf"{line['gap_ev']['estimate']:.9f} & "
        rf"[{line['gap_ev']['lower']:.9f}, {line['gap_ev']['upper']:.9f}] \\",
        rf"Weight $w$ & {exact_line['weight']:.9f} & "
        rf"{line['weight']['estimate']:.9f} & "
        rf"[{line['weight']['lower']:.9f}, {line['weight']['upper']:.9f}] \\",
        rf"$\chi(0)$ (eV$^{{-1}}$) & "
        rf"{exact['susceptibility_per_ev']:.9f} & {chi['estimate']:.9f} & "
        rf"[{chi['lower']:.9f}, {chi['upper']:.9f}] \\",
    ]
    _write("response_results.tex", out)


def matched_table() -> None:
    """Every arm on one H4 contract with words and physical QWC settings.

    ``state_evaluation_contexts`` is deliberately not called a preparation
    count: every shot of every physical measurement setting requires a fresh
    preparation. ``selection_qwc_group_evaluations`` sums settings across
    changing-state selection rounds; for fixed-reference DA-CASE it is the one
    cached union. ``---`` marks a column an arm does not have. ADAPT-GCIM's
    final object is instead counted as Hamiltonian/overlap transition pairs.
    """
    record = json.loads(MATCHED.read_text())
    # The matched table is where the width comparison is actually made, so the
    # Krylov arm's width is carried over from the record that can compute it
    # rather than left blank; the tracked route cannot reach it here.
    krylov = _krylov_widths().get("h4_chain(r=0.9)")
    rows = []
    for row in record["rows"]:
        size = row["basis_size"]
        basis = "---" if size is None else str(size)
        kappa = ("---" if row["condition_number"] is None
                 else _kappa(row["condition_number"]))
        selection = row["selection_evaluations"]
        selection_cell = "---" if not selection else f"{selection:,}"
        words = row["selection_words"]
        selection_groups = row.get("selection_qwc_group_evaluations")
        words_cell = ("---" if not words else
                      f"{words:,}/{selection_groups:,}")
        final = row["final_words"]
        if final is None and row["arm"] == "Krylov" and krylov is not None:
            final_cell = rf"{krylov:,}/---\footnotemark[1]"
        else:
            final_groups = row.get("final_qwc_groups")
            final_cell = ("n/t" if final is None else
                          f"{final:,}/{final_groups:,}")
        h_pairs = row.get("hamiltonian_matrix_pairs")
        s_pairs = row.get("overlap_offdiagonal_pairs")
        pair_cell = ("---" if h_pairs is None
                     else f"{h_pairs:,}/{s_pairs:,}")
        # Three decimals renders the Krylov arm's 1.05e-5 mHa as a flat zero,
        # which reads as exactness rather than as a small number.
        error = row["error_millihartree"]
        error_cell = (f"{error:.3f}" if abs(error) >= 5e-4
                      else _sci(error, 2).replace("$", "$"))
        display_arm = row["arm"].replace("A-CASE", "DA-CASE")
        rows.append(
            rf"{display_arm} & {basis} & {error_cell} & "
            rf"{kappa} & {row['state_evaluation_contexts']:,} & "
            rf"{row['ansatz_rotors']} & {selection_cell} & {words_cell} & "
            rf"{final_cell} & {pair_cell} \\")
    _write("matched_results.tex", rows)


def clifford_hierarchy_table() -> None:
    """Both eight-qubit hierarchy records, one block per system."""
    out: list[str] = []
    for path in CLIFFORD_HIERARCHY:
        record = json.loads(path.read_text())
        if out:
            out.append(r"\colrule")
        # A leading system column rather than a spanning header row: a
        # \multicolumn may not be the first token of an \input fragment inside
        # a TeX alignment.
        system = (rf"{record['label']} ($M={record['basis_size']}$, "
                  rf"$W={record['word_universe']:,}$)")
        for row in record["rows"]:
            block = row["block_size"]
            label = "$1$ (QWC)" if block == 1 else (
                f"${block}$ (full)" if block == record["n_qubits"]
                else f"${block}$")
            ratio = row["cx_to_preparation_cost_ratio_vs_qwc"]
            out.append(
                f"{system} & {label} & {row['settings']} & "
                f"{row['word_samples_per_preparation']:.2f} & "
                f"{row['logical_cx_per_sweep']} & "
                f"{row['mean_logical_cx_depth']:.2f} & "
                f"{row['max_logical_cx_depth']} & "
                f"{row['state_preparations_at_uniform_shots'] / 1e6:.3f} & "
                + ("---" if ratio is None else f"{ratio:.3f}")
                + r" \\")
            system = ""
    _write("clifford_hierarchy.tex", out)


def conditioning_table() -> None:
    rows = []
    for path in (RESPONSE, RESPONSE_ILL):
        record = json.loads(path.read_text())
        basis, boot = record["basis"], record["bootstrap"]
        chi = record["measured"]["susceptibility_per_ev"]
        exact_chi = record["exact"]["susceptibility_per_ev"]
        failures = boot["failures"]
        width = chi["upper"] - chi["lower"]
        rows.append(
            rf"{basis['family']} & {_kappa(basis['condition_number'])} & "
            rf"{boot['replicates_succeeded']}/{boot['replicates_requested']} & "
            rf"{failures['rank']} & {failures['root_collision']} & "
            rf"{failures['solver']} & {_sci(width)} & "
            rf"{'yes' if chi['lower'] <= exact_chi <= chi['upper'] else 'no'} \\")
    _write("conditioning_results.tex", rows)


def main() -> None:
    dimer_table()
    h4_table()
    warm_start_table()
    matched_table()
    clifford_hierarchy_table()
    ladder_table()
    phase12_primary_table()
    response_table()
    conditioning_table()
    print(TABLES)


if __name__ == "__main__":
    main()
