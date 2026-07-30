"""Generate vector figures for the standalone A-CASE paper."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/clifford-qc-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "paper_assets"
LADDER = ROOT / "benchmarks" / "reference_results" / "acase_ladder_summary.csv"
RESPONSE = HERE / "data" / "response_bootstrap.json"


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.25, 3.05, 1.55, 1.1, "Effective model", "FCIDUMP or\nversioned JSON"),
        (2.15, 3.05, 1.55, 1.1, "Qubit map", "fermions $\\to$\nPauli words"),
        (4.05, 3.05, 1.55, 1.1, "Single reference", "$\\rho$ and\ngenerator pool"),
        (5.95, 3.05, 1.55, 1.1, "Element bank", "$S$, $H$, $Q$\nshared words"),
        (7.85, 3.05, 1.8, 1.1, "A-CASE solve", "adaptive GEP\n+ conditioning"),
    ]
    colors = ["#d9edf7", "#d9edf7", "#e8f5e9", "#fff3cd", "#f3e5f5"]
    for (x, y, w, h, title, subtitle), color in zip(boxes, colors):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#263238", linewidth=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.70, title, ha="center", va="center",
                fontweight="bold")
        ax.text(x + w / 2, y + 0.31, subtitle, ha="center", va="center")
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch(
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
            arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
            color="#455a64"))

    lower = [
        (2.0, 0.65, 2.0, 1.0, "Shared QWC cache",
         "one joint histogram\nper commuting group"),
        (4.65, 0.65, 2.0, 1.0, "Whole-pipeline bootstrap",
         "threshold $\\to$ GEP $\\to$\nroots $\\to$ weights"),
        (7.3, 0.65, 2.1, 1.0, "Reported outputs",
         "energy, observables,\nresponse + evidence label"),
    ]
    for x, y, w, h, title, subtitle in lower:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#fce4ec", edgecolor="#263238", linewidth=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.64, title, ha="center", va="center",
                fontweight="bold")
        ax.text(x + w / 2, y + 0.27, subtitle, ha="center", va="center")
    for left, right in zip(lower[:-1], lower[1:]):
        ax.add_patch(FancyArrowPatch(
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
            arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
            color="#455a64"))
    ax.add_patch(FancyArrowPatch(
        (6.72, 3.05), (3.0, 1.65),
        connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
        mutation_scale=9, linewidth=0.9, linestyle="--", color="#ad1457"))
    ax.add_patch(FancyArrowPatch(
        (8.75, 3.05), (8.35, 1.65),
        connectionstyle="arc3,rad=-0.08", arrowstyle="-|>",
        mutation_scale=9, linewidth=0.9, color="#455a64"))
    ax.text(5.0, 2.15, "finite-shot route", color="#ad1457",
            ha="center", fontstyle="italic")
    ax.text(5.0, 4.65,
            "DFT / Wannier / embedding is upstream; A-CASE begins at the "
            "effective many-body Hamiltonian",
            ha="center", va="center", fontweight="bold")
    fig.tight_layout(pad=0.2)
    fig.savefig(ASSETS / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_heatmap() -> None:
    with LADDER.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(r["system"], r["method"]): r for r in rows}
    systems = [
        # Matplotlib is not LaTeX: outside mathtext a "\AA" control sequence is
        # drawn literally, backslash and all. The unicode glyph is in the
        # default DejaVu font, so it renders in both text and PDF output.
        ("H$_4$ 0.9 Å", "h4_chain(r=0.9)"),
        ("H$_4$ 1.8 Å", "h4_chain(r=1.8)"),
        ("H$_2$O (4e,4o), $2R_e$", "h2o_4e4o(scale=2.0)"),
        ("H$_2$O (8e,6o)", "h2o_8e6o(scale=1.0)"),
        ("Hubbard $2\\times2$", "hubbard(2x2,t=1.0,U=4.0,obc)"),
        ("Hubbard $2\\times3$", "hubbard(2x3,t=1.0,U=4.0,obc)"),
        ("Kitaev $2\\times2$", "kitaev(2x2,K=1.0,1.0,1.0,obc)"),
    ]
    methods = [
        ("QSE", "qse"),
        ("Krylov", "krylov"),
        ("Gen.-coord.", "generator_coordinate"),
        ("ADAPT", "adapt_exact"),
        ("A-CASE", "acase_exact"),
    ]
    values = np.full((len(systems), len(methods)), np.nan)
    annotations: list[list[str]] = [["" for _ in methods] for _ in systems]
    for i, (_, system) in enumerate(systems):
        exact = abs(float(by_key[(system, "exact")]["energy"]))
        for j, (_, method) in enumerate(methods):
            row = by_key.get((system, method))
            if row is None or not row["error"]:
                continue
            relative = max(abs(float(row["error"])) / exact, 1e-15)
            values[i, j] = relative
            annotations[i][j] = f"{relative:.1e}"

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(
        masked, cmap="viridis_r", norm=LogNorm(vmin=1e-13, vmax=5e-1),
        aspect="auto")
    ax.set_xticks(range(len(methods)), [x[0] for x in methods])
    ax.set_yticks(range(len(systems)), [x[0] for x in systems])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] < 2e-4 else "black"
                ax.text(j, i, annotations[i][j], ha="center", va="center",
                        fontsize=6.4, color=color)
            else:
                ax.text(j, i, "n/t", ha="center", va="center",
                        fontsize=6.4, color="#777777")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"absolute relative energy error")
    ax.set_title("Matched-budget validation ladder (default A-CASE uses $M=9$)")
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(ASSETS / "validation_ladder.pdf", bbox_inches="tight")
    plt.close(fig)


def response_plot() -> None:
    record = json.loads(RESPONSE.read_text())
    spectrum = record["spectrum"]
    x = np.asarray(spectrum["frequency_ev"])
    exact = np.asarray(spectrum["exact"])
    estimate = np.asarray(spectrum["estimate"])
    lower = np.asarray(spectrum["lower"])
    upper = np.asarray(spectrum["upper"])

    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.fill_between(x, lower, upper, color="#6a51a3", alpha=0.24,
                    linewidth=0, label="95% pointwise percentile band")
    ax.plot(x, estimate, color="#54278f", linewidth=1.4,
            label="finite-shot estimate")
    ax.plot(x, exact, color="#111111", linewidth=1.0, linestyle="--",
            label="exact projected response")
    ax.set_xlabel(r"frequency $\omega$ (eV)")
    ax.set_ylabel(r"$S(\omega)$ (eV$^{-1}$)")
    ax.set_xlim(0.35, 1.30)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Grouped-bootstrap staggered-spin response")
    ax.text(
        0.02, 0.04,
        "heuristic; conditional on surviving replicas",
        transform=ax.transAxes, fontsize=6.8, color="#54278f")
    fig.tight_layout(pad=0.4)
    fig.savefig(ASSETS / "response_bootstrap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    _style()
    pipeline()
    benchmark_heatmap()
    response_plot()
    print(ASSETS)


if __name__ == "__main__":
    main()
