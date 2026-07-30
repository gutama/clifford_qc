"""Generate all numerical table fragments for the standalone A-CASE paper."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
LADDER = ROOT / "benchmarks" / "reference_results" / "acase_ladder_summary.csv"
H4 = ROOT / "benchmarks" / "reference_results" / "fcidump_h4.json"
DIMER = ROOT / "examples" / "data" / "wannier_hubbard_dimer.json"
RESPONSE = HERE / "data" / "response_bootstrap.json"


def _write(name: str, rows: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text("\n".join(rows) + "\n")


def _sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10 ** exponent
    if exponent == 0:
        return rf"${mantissa:.{digits}f}$"
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def _kappa(value: float) -> str:
    """Overlap condition numbers, in the same notation as the error column.

    ``f"{v:.2g}"`` renders 6.6e+10 as text in a physics table; the errors beside
    it are typeset. Well-conditioned arms report exactly 1, which should stay a
    bare 1 rather than becoming $1.0\\times10^{0}$.
    """
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
        rf"Adaptive A-CASE & {adaptive['ground_energy']:.15f} & "
        rf"{adaptive['error_millihartree']:.6f} mHa & "
        rf"{adaptive['basis_size']} \\",
        rf"Complete singles/doubles & {sd['ground_energy']:.15f} & "
        rf"{sd['error_millihartree']:.6f} mHa & {sd['basis_size']} \\",
    ]
    _write("h4_results.tex", rows)


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
    for display, system, acase_method, comparator_method, unit in specs:
        a = by_key[(system, acase_method)]
        b = by_key[(system, comparator_method)]
        out.append(
            f"{display} & {a['basis_size']} & "
            f"{_sci(abs(float(a['error'])))} {unit} & "
            f"{_kappa(float(a['condition_number']))} & {a['word_universe']} & "
            f"{labels[comparator_method]} & {b['basis_size']} & "
            f"{_sci(abs(float(b['error'])))} {unit} & "
            f"{_kappa(float(b['condition_number']))} \\\\")
    _write("ladder_results.tex", out)


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


def main() -> None:
    dimer_table()
    h4_table()
    ladder_table()
    response_table()
    print(TABLES)


if __name__ == "__main__":
    main()

