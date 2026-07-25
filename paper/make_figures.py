"""Generate Paper A figures from the committed benchmark JSONL.

Reads only ``benchmarks/reference_results/*.jsonl`` and writes PDF assets to
``paper/paper_assets/``. Every plotted value is a median (or rate) over the
committed seeds; nothing is recomputed by re-running experiments.

    python paper/make_figures.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "reference_results"
OUT = Path(__file__).resolve().parent / "paper_assets"
OUT.mkdir(exist_ok=True)

# colorblind-safe, print-legible
C = {"random": "#999999", "doubling": "#4477AA", "doubling_grouped": "#66CCEE",
     "variance_grouped": "#EE6677", "exact": "#228833", "fast": "#CCBB44",
     "confidence": "#4477AA"}
# Sized for a two-column PRA layout: at ~3.4 in per column the previous 7.5 pt
# legends and 7 pt annotations rendered below the journal's 6 pt floor after
# scaling. Base font 11 pt with 9.5 pt legends keeps every label legible.
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.axisbelow": True,
})
ANNOT = 9.0      # in-axes annotation size


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def fam(name):
    return re.sub(r"seed=\d+,?", "", name).replace(",)", ")")


def med(xs):
    return st.median(sorted(xs)) if xs else float("nan")


FAM_LABEL = {
    "tfim(n=4,J=1.0,h=0.5,obc)": "TFIM h=0.5",
    "tfim(n=4,J=1.0,h=1.0,obc)": "TFIM h=1.0",
    "tfim(n=4,J=1.0,h=1.5,obc)": "TFIM h=1.5",
    "tfim(n=4,J=1.0,h=1.0,pbc)": "TFIM h=1.0 (PBC)",
    "random_ising(n=4,obc)": "rand. Ising",
}
ORDER = list(FAM_LABEL)


def group_headline():
    rows = load("spin_headline_n4.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(fam(r["model"]), r["method"])].append(r)
    return g


def fig_selection_quality():
    """Median relative energy error: random vs confidence-guided variants."""
    g = group_headline()
    methods = ["random", "doubling_grouped", "variance_grouped"]
    # These arms accept the empirical leader when the budget is spent, so they
    # are *confidence-guided*, never strict-certified. The legend must not say
    # "certified": that label is reserved for strict empirical-Bernstein
    # decisions (Table II, Sec. V B).
    labels = {"random": "random", "doubling_grouped": "confidence-guided (doubling)",
              "variance_grouped": "confidence-guided (variance)"}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = range(len(ORDER))
    w = 0.26
    for i, m in enumerate(methods):
        vals = [max(med([r["relative_error"] for r in g[(f, m)]]), 1e-16) for f in ORDER]
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=labels[m], color=C[m])
    ax.set_yscale("log")
    ax.set_ylabel(r"median relative energy error $\epsilon_E$")
    ax.set_xticks(list(x))
    ax.set_xticklabels([FAM_LABEL[f] for f in ORDER], rotation=25, ha="right")
    ax.axhline(1e-3, ls="--", lw=0.8, color="k", alpha=0.6)
    ax.text(len(ORDER) - 0.5, 1.3e-3, r"$10^{-3}$", fontsize=ANNOT, ha="right", va="bottom")
    # above the axes: the log range spans 14 decades, so any in-axes corner
    # collides with a bar on some family
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.4)
    fig.savefig(OUT / "selection_quality.pdf")
    plt.close(fig)


def fig_grouping():
    """Distinct measurement circuits: ungrouped vs QWC-grouped."""
    g = group_headline()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = range(len(ORDER))
    w = 0.38
    ung = [med([r["total_circuits"] for r in g[(f, "doubling")]]) for f in ORDER]
    grp = [med([r["total_circuits"] for r in g[(f, "doubling_grouped")]]) for f in ORDER]
    ax.bar([xi - w / 2 for xi in x], ung, w, label="ungrouped", color=C["doubling"])
    ax.bar([xi + w / 2 for xi in x], grp, w, label="QWC-grouped", color=C["doubling_grouped"])
    for xi, u, gg in zip(x, ung, grp):
        ax.text(xi + w / 2, gg, f"{u/gg:.1f}x", fontsize=ANNOT, ha="center", va="bottom")
    ax.set_ylabel("median distinct measurement circuits")
    ax.set_xticks(list(x))
    ax.set_xticklabels([FAM_LABEL[f] for f in ORDER], rotation=25, ha="right")
    ax.legend(frameon=False)
    fig.savefig(OUT / "grouping.pdf")
    plt.close(fig)


def fig_allocation():
    """Ambiguous-selection rate and shot cost: doubling vs variance-proportional."""
    g = group_headline()

    def ambig_rate(rs):
        amb = sum(r["status_counts"].get("budget_exhausted_ambiguous", 0) for r in rs)
        steps = sum(r["selection_steps"] for r in rs)
        return amb / steps if steps else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = range(len(ORDER))
    w = 0.38
    for ax, key, ylab, title in [
        (axes[0], "ambig", "ambiguous-selection rate", "Ambiguity"),
        (axes[1], "shots", "median total shots", "Measurement cost")]:
        d = [g[(f, "doubling_grouped")] for f in ORDER]
        v = [g[(f, "variance_grouped")] for f in ORDER]
        if key == "ambig":
            dv = [ambig_rate(rs) for rs in d]
            vv = [ambig_rate(rs) for rs in v]
        else:
            dv = [med([r["total_shots"] for r in rs]) for rs in d]
            vv = [med([r["total_shots"] for r in rs]) for rs in v]
        ax.bar([xi - w / 2 for xi in x], dv, w, label="doubling", color=C["doubling_grouped"])
        ax.bar([xi + w / 2 for xi in x], vv, w, label="variance-prop.", color=C["variance_grouped"])
        ax.set_ylabel(ylab)
        ax.set_xticks(list(x))
        ax.set_xticklabels([FAM_LABEL[f] for f in ORDER], rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False)
    fig.savefig(OUT / "allocation.pdf")
    plt.close(fig)


def fig_chemistry():
    """Final energy error per molecule per selection arm."""
    rows = load("chemistry.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(r["model"].split("(")[0], r["arm"])].append(r)
    mols = ["h2", "lih", "beh2", "h4_chain"]
    mol_lab = {"h2": "H$_2$", "lih": "LiH", "beh2": "BeH$_2$", "h4_chain": "H$_4$"}
    arms = ["exact", "confidence", "fast", "random"]
    arm_lab = {"exact": "exact gradient", "confidence": "confidence-guided",
               "fast": "FAST-inspired proxy", "random": "random"}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = range(len(mols))
    w = 0.2
    off = (len(arms) - 1) / 2.0  # center the group of bars on each tick
    for i, a in enumerate(arms):
        vals = [max(med([r["final_error_mha"] for r in g[(m, a)]]), 1e-4)
                if g[(m, a)] else 0.0 for m in mols]
        ax.bar([xi + (i - off) * w for xi in x], vals, w, label=arm_lab[a], color=C[a])
        # an arm that was not run leaves a gap; say so rather than let the gap
        # read as a zero
        for xi, m, v in zip(x, mols, vals):
            if v == 0.0:
                ax.text(xi + (i - off) * w, 1.3e-4, "not run", fontsize=7.5,
                        rotation=90, ha="center", va="bottom", color="0.35")
    ax.set_yscale("log")
    ax.set_ylabel("median final error (mHa)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([mol_lab[m] for m in mols])
    ax.axhline(1.6, ls="--", lw=0.8, color="k", alpha=0.6)
    ax.text(len(mols) - 0.5, 1.9, "chemical accuracy", fontsize=ANNOT, ha="right", va="bottom")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "chemistry.pdf")
    plt.close(fig)




def fig_calibration():
    """Headline: empirical wrong-selection vs delta, plus abstention/cost."""
    # calibration.jsonl carries per-instance rows (scope="instance") alongside
    # the pooled summary (scope="pooled"); the figure uses the pooled rows only.
    rows = [r for r in load("calibration.jsonl")
            if r.get("scope", "pooled") == "pooled"]
    by = {}
    for r in rows:
        by.setdefault(r["bound"], []).append(r)
    for b in by:
        by[b].sort(key=lambda r: r["delta"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    bcol = {"normal": C["doubling"], "eb": C["variance_grouped"]}
    blab = {"normal": "normal (Gaussian)", "eb": "empirical-Bernstein"}

    ax = axes[0]
    dd = [r["delta"] for r in by["normal"]]
    ax.plot([0, max(dd)], [0, max(dd)], ls="--", lw=0.8, color="k", label=r"$y=\delta$")
    for b in ("normal", "eb"):
        ax.plot([r["delta"] for r in by[b]], [r["wrong_selection_rate"] for r in by[b]],
                "o-", color=bcol[b], label=blab[b], ms=4)
    ax.set_xlabel(r"nominal error budget $\delta$")
    ax.set_ylabel("empirical wrong-selection rate")
    ax.set_title("Observed ranking errors", fontsize=11)
    ax.set_xticks(dd)
    ax.set_ylim(-0.005, max(dd) + 0.02)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for b in ("normal", "eb"):
        ax.plot([r["delta"] for r in by[b]], [r["abstention_rate"] for r in by[b]],
                "o-", color=bcol[b], label=f"{blab[b]}", ms=4)
    ax.set_xlabel(r"nominal error budget $\delta$")
    ax.set_ylabel("abstention rate")
    ax.set_title("Strict selection: abstention", fontsize=11)
    ax.set_xticks(dd)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(OUT / "calibration.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_selection_quality()
    fig_grouping()
    fig_allocation()
    fig_chemistry()
    fig_calibration()
    print("wrote figures to", OUT)
