"""Generate the numeric body of every manuscript table from committed records.

Companion to ``make_figures.py``. Reads only ``benchmarks/reference_results/``
and writes one LaTeX fragment per table into ``paper/tables/``; the manuscript
``\\input``s those fragments, so no table value is transcribed by hand and a
stale record cannot survive a regeneration.

Each fragment contains only the rows of the tabular body (no preamble, no
header, no ``\\end{tabular}``), so the column specification, header row, and
caption stay in the manuscript where they belong.

    python paper/make_tables.py
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "reference_results"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def load(name):
    path = DATA / name
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def med(xs):
    return st.median(sorted(xs)) if xs else float("nan")


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 5.2e-05 -> $5.2\\times10^{-5}$."""
    if x is None:
        return "---"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def thousands(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def write(name, rows):
    (OUT / f"{name}.tex").write_text("\n".join(rows) + "\n")
    print(f"wrote tables/{name}.tex ({len(rows)} lines)")


# -- Clopper-Pearson ---------------------------------------------------------

def cp_upper(k, n, conf=0.95):
    """One-sided upper confidence limit on a binomial rate."""
    if n == 0:
        return None
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    try:
        from scipy.stats import beta
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


# -- Table: strict-selection calibration -------------------------------------

def tab_calibration():
    rows = [r for r in load("calibration.jsonl") if r.get("scope", "pooled") == "pooled"]
    label = {"normal": "normal", "eb": "emp.-Bernstein"}
    out, seen = [], set()
    for bound in ("normal", "eb"):
        block = sorted((r for r in rows if r["bound"] == bound), key=lambda r: r["delta"])
        if out:
            out.append(r"\colrule")
        for r in block:
            n = r["runs"]
            n_res = round(r["resolved_rate"] * n)
            wrong = round(r["wrong_selection_rate"] * n)
            head = label[bound] if bound not in seen else ""
            seen.add(bound)
            cond = cp_upper(wrong, n_res)
            out.append(
                f"{head} & {r['delta']:.2f} & {r['resolved_rate']:.4f} & "
                f"{thousands(n_res)} & {wrong} & {sci(cp_upper(wrong, n))} & "
                f"{'---' if cond is None else (sci(cond, 2) if cond < 0.01 else f'${cond:.3f}$')} & "
                f"{r['coverage']:.5f} & {thousands(r['median_shots'])} & "
                f"{int(r['median_circuits'])}\\\\")
    write("calibration", out)


# -- Tables: certified eps-best trajectories ---------------------------------

CERT_LABEL = {"tfim_n4_h1": r"TFIM ($h{=}1$)", "random_ising_n4": "random-field Ising",
              "h2_sto3g": r"H$_2$ (STO-3G)", "lih_2e2o": r"LiH ($2e,2o$)"}
CERT_SHORT = {"tfim_n4_h1": r"TFIM ($h{=}1$)", "random_ising_n4": r"rand.\ Ising",
              "h2_sto3g": r"H$_2$", "lih_2e2o": "LiH"}


def tab_certified():
    rows = load("certified_trajectories.jsonl")
    body, scale = [], []
    for r in rows:
        appended = [s for s in r["trajectory"] if s["label"]]
        argmax = sum(1 for s in appended if s.get("exact_rank") == 1)
        body.append(
            f"{CERT_LABEL[r['system']]:20s} & {r['n']} & {r['pool_size']} & "
            f"{r['operators']}\\,({r['certified_steps']}) & {argmax} & "
            f"{sci(r['final_rel_error'])} & {sci(r['total_shots'])}\\\\")
        if scale:
            scale.append(r"\colrule")
        for i, s in enumerate(appended):
            head = CERT_SHORT[r["system"]] if i == 0 else ""
            gmax, gsel = s["grad_max"], s["grad_selected"]
            eps_over = s.get("eps_over_gmax")
            short = s.get("rel_shortfall")
            scale.append(
                f"{head} & {s['step']} & ${gmax:.4g}$ & ${gsel:.4g}$ & "
                f"${eps_over:.3g}$ & "
                f"{'$0$' if not short else f'${short:.3f}$'} & "
                f"{s.get('exact_rank')}\\\\")
    write("certified_trajectories", body)
    write("eps_scale", scale)


# -- Table: shot-ceiling sweep -----------------------------------------------

STRAT_LABEL = {"exact_tie": "exact tie", "small_gap": "small gap",
               "clear_gap": "clear gap", "tight": "tight", "narrow": "narrow",
               "moderate": "moderate", "wide": "wide"}


def tab_ceiling():
    rows = load("ceiling_sweep.jsonl")
    if rows is None:
        return
    out = []
    for rule in ("exact_best", "eps_best"):
        for stratum in ("exact_tie", "small_gap", "clear_gap"):
            block = sorted((r for r in rows if r["rule"] == rule
                            and r["stratum"] == stratum),
                           key=lambda r: r["ceiling_factor"])
            if not block:
                continue
            if out:
                out.append(r"\colrule")
            for i, r in enumerate(block):
                head = (rule.replace("_", "-") + ", " + STRAT_LABEL[stratum]) if i == 0 else ""
                shots = r["median_shots_to_resolution"]
                eta = r["median_eta_required"]
                radius = r["median_terminal_radius"]
                out.append(
                    f"{head} & ${r['ceiling_factor']}\\times$ & "
                    f"{r['resolved_rate']:.3f} & {r['wrong_rate_unconditional']:.4f} & "
                    f"{'---' if shots is None else sci(shots)} & "
                    f"{'---' if radius is None else f'${radius:.4f}$'} & "
                    f"{'---' if eta is None else f'${eta:.3f}$'}\\\\")
    write("ceiling_sweep", out)


# -- Table: gap-stratified hard instances ------------------------------------

ARM_LABEL = {"fixed_shot": "fixed-shot", "fallback": "fallback",
             "strict_exact_eb": "strict exact-best", "strict_eps_eb": r"strict $\varepsilon$-best"}


def tab_hard():
    rows = load("hard_instances.jsonl")
    if rows is None:
        return
    sel = [r for r in rows if r["part"] == "selection"]
    out = []
    for stratum in ("tight", "narrow", "moderate", "wide"):
        block = [r for r in sel if r["stratum"] == stratum]
        if not block:
            continue
        if out:
            out.append(r"\colrule")
        order = {a: i for i, a in enumerate(ARM_LABEL)}
        for i, r in enumerate(sorted(block, key=lambda r: order[r["arm"]])):
            head = STRAT_LABEL[stratum] if i == 0 else ""
            w = r["wrong_rate_given_committed"]
            v = r["eps_violation_rate_given_committed"]
            rr = r["mean_rel_regret"]
            out.append(
                f"{head} & {ARM_LABEL[r['arm']]} & {r['commit_rate']:.3f} & "
                f"{'---' if w is None else f'{w:.3f}'} & "
                f"{'---' if v is None else f'{v:.3f}'} & "
                f"{'---' if rr is None else (f'{rr:.4f}' if rr else '$0$')} & "
                f"{thousands(r['median_shots'])} & {int(r['median_circuits'])}\\\\")
    write("hard_instances", out)

    traj = [r for r in rows if r["part"] == "trajectory"]
    if traj:
        tlab = {"fixed_shot_4k": "fixed-shot (4k/word)",
                "fixed_shot_64k": "fixed-shot (64k/word)",
                "fallback_doubling": "fallback (doubling)",
                "strict_eps_eb": r"strict $\varepsilon$-best"}
        write("hard_trajectories", [
            f"{tlab.get(r['arm'], r['arm'])} & {sci(r['rel_err_median'])} & "
            f"{sci(r['rel_err_p90'])} & {thousands(r['shots_median'])} & "
            f"{int(r['circuits_median'])} & {int(r['operators_median'])} & "
            f"{r['abstentions_total']}\\\\" for r in traj])


# -- Table: baseline ladder --------------------------------------------------

BASE_ARMS = [("exact", "exact (ceiling)"), ("random", "random (floor)"),
             ("fixed_shot", "fixed-shot"), ("shared_only", "shared-only"),
             ("shared_grouped", "shared-grouped"), ("variance_reuse", "variance-reuse"),
             ("strict", "strict"), ("fallback", "fallback")]
FAM_BASE = {"tfim_crit": r"TFIM $h{=}1$", "random_ising": r"rand.\ Ising"}


def tab_baselines():
    rows = load("baselines.jsonl")
    by = {(r["family"], r["arm"]): r for r in rows}
    out = []
    for fi, fam in enumerate(("tfim_crit", "random_ising")):
        if fi:
            out.append(r"\colrule")
        for i, (arm, lab) in enumerate(BASE_ARMS):
            r = by.get((fam, arm))
            if r is None:
                continue
            head = FAM_BASE[fam] if i == 0 else ""
            err = ("abstains" if r["abstentions_total"] == r["runs"]
                   else sci(r["rel_err_median"]))
            out.append(f"{head} & {lab} & {err} & {thousands(r['shots_median'])} & "
                       f"{int(r['circuits_median'])} & {r['abstentions_total']}\\\\")
    write("baselines", out)
    sizes = rows[0]
    write("plan_sizes", [
        f"naive word-measurements & {sizes['naive_words']}\\\\",
        f"shared unique words & {sizes['shared_words']}\\\\",
        f"QWC groups & {sizes['qwc_groups']}\\\\",
        f"sharing factor & ${sizes['shared_reuse']:.1f}\\times$\\\\",
        f"grouping factor & ${sizes['naive_words'] / sizes['qwc_groups'] / sizes['shared_reuse']:.1f}\\times$\\\\",
        f"combined factor & ${sizes['grouped_reuse']:.1f}\\times$\\\\",
    ])


# -- Table: headline n=4 spin trajectories -----------------------------------

FAM_LABEL = {
    "tfim(n=4,J=1.0,h=0.5,obc)": r"TFIM $h{=}0.5$",
    "tfim(n=4,J=1.0,h=1.0,obc)": r"TFIM $h{=}1.0$",
    "tfim(n=4,J=1.0,h=1.5,obc)": r"TFIM $h{=}1.5$",
    "tfim(n=4,J=1.0,h=1.0,pbc)": r"TFIM $h{=}1.0$ (PBC)",
    "random_ising(n=4,obc)": r"rand.\ Ising",
}
METHODS = [("random", "random"), ("doubling", "doubling (ungrouped)"),
           ("doubling_grouped", "doubling (grouped)"),
           ("variance_grouped", "variance (grouped)")]


def _fam(name):
    return re.sub(r"seed=\d+,?", "", name).replace(",)", ")")


def ambiguity_rate(rows):
    ambiguous = sum(
        row["status_counts"].get("budget_exhausted_ambiguous", 0)
        for row in rows
    )
    steps = sum(row["selection_steps"] for row in rows)
    return ambiguous / steps if steps else 0.0


def tab_headline():
    rows = load("spin_headline_n4.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(_fam(r["model"]), r["method"])].append(r)

    out, group = [], []
    for fi, fam in enumerate(FAM_LABEL):
        if fi:
            out.append(r"\colrule")
        for i, (m, lab) in enumerate(METHODS):
            rs = g[(fam, m)]
            if not rs:
                continue
            head = FAM_LABEL[fam] if i == 0 else ""
            a = "---" if m == "random" else f"{ambiguity_rate(rs):.2f}"
            out.append(f"{head} & {lab} & {sci(med([r['relative_error'] for r in rs]))} & "
                       f"{thousands(med([r['total_shots'] for r in rs]))} & "
                       f"{int(med([r['total_circuits'] for r in rs]))} & {a}\\\\")
        ung, grp = g[(fam, "doubling")], g[(fam, "doubling_grouped")]
        if ung and grp:
            cu = med([r["total_circuits"] for r in ung])
            cg = med([r["total_circuits"] for r in grp])
            su = med([r["total_shots"] for r in ung])
            sg = med([r["total_shots"] for r in grp])
            group.append(f"{FAM_LABEL[fam]} & {int(cu)} & {int(cg)} & "
                         f"${cu / cg:.1f}\\times$ & {thousands(su)} & "
                         f"{thousands(sg)} & ${su / sg:.1f}\\times$\\\\")
    write("headline", out)
    write("grouping", group)


# -- Table: exploratory n=6 spin trajectories -------------------------------

FAM_N6_LABEL = {
    "tfim(n=6,J=1.0,h=1.0,obc)": r"TFIM $h{=}1.0$",
    "random_ising(n=6,obc)": r"rand.\ Ising",
    "xxz(n=6,J=1.0,delta=1.0,obc)": r"XXZ $\Delta{=}1$",
}
N6_METHODS = [
    ("exact", "exact"),
    ("random", "random"),
    ("doubling_grouped", "doubling (grouped)"),
    ("variance_grouped", "variance (grouped)"),
]


def tab_spin_n6():
    rows = load("spin_n6.jsonl")
    if rows is None:
        return
    g = defaultdict(list)
    for r in rows:
        g[(_fam(r["model"]), r["method"])].append(r)

    out = []
    for fi, fam in enumerate(FAM_N6_LABEL):
        if fi:
            out.append(r"\colrule")
        for i, (method, label) in enumerate(N6_METHODS):
            rs = g[(fam, method)]
            if not rs:
                continue
            head = FAM_N6_LABEL[fam] if i == 0 else ""
            near_values = [
                r["near_optimal_rate"] for r in rs
                if r["near_optimal_rate"] is not None
            ]
            near = f"{st.mean(near_values):.2f}" if near_values else "---"
            ambiguity = (
                f"{ambiguity_rate(rs):.2f}"
                if method in ("doubling_grouped", "variance_grouped")
                else "---"
            )
            out.append(
                f"{head} & {label} & "
                f"{sci(med([r['relative_error'] for r in rs]))} & "
                f"{thousands(med([r['total_shots'] for r in rs]))} & "
                f"{thousands(med([r['total_circuits'] for r in rs]))} & "
                f"{near} & {ambiguity}\\\\"
            )
    write("spin_n6", out)


# -- Tables: chemistry -------------------------------------------------------

MOL_LABEL = {"h2": r"H$_2$", "lih": "LiH", "beh2": r"BeH$_2$", "h4_chain": r"H$_4$"}
CHEM_ARMS = [("exact", "exact"), ("fast", "FAST"),
             ("confidence", r"conf.-guided"), ("random", "random")]


def tab_chemistry():
    rows = load("chemistry.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(r["model"].split("(")[0], r["arm"])].append(r)
    out = []
    for mi, mol in enumerate(("h2", "lih", "beh2", "h4_chain")):
        if mi:
            out.append(r"\colrule")
        first = True
        for arm, lab in CHEM_ARMS:
            rs = g[(mol, arm)]
            if not rs:
                continue
            head = MOL_LABEL[mol] if first else ""
            first = False
            errs = [r["final_error_mha"] for r in rs]
            m = med(errs)
            if m < 1e-6:
                err = r"$\approx0$"
            elif len(set(round(e, 6) for e in errs)) > 1:
                err = f"${m:.2f}$ [${min(errs):.2f}$--${max(errs):.2f}$]"
            else:
                err = f"${m:.4g}$" if m >= 1 else f"${m:.3g}$"
            ops = [r["ops_to_accuracy"] for r in rs]
            if any(o is None for o in ops):
                opstr, shotstr = "---", "---"
            else:
                opstr = (f"{min(ops)}--{max(ops)}" if min(ops) != max(ops)
                         else str(int(ops[0])))
                sh = med([r["shots_to_accuracy"] for r in rs])
                shotstr = "0" if sh == 0 else sci(sh)
            out.append(f"{head} & {lab} & {err} & {opstr} & {shotstr}\\\\")
    write("chemistry", out)


def main():
    tab_calibration()
    tab_certified()
    tab_ceiling()
    tab_hard()
    tab_baselines()
    tab_headline()
    tab_spin_n6()
    tab_chemistry()
    print("wrote table fragments to", OUT)


if __name__ == "__main__":
    main()
