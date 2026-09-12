"""Generate the vector figures for the architecture paper.

Three of the five figures plot a committed record or the source census; the
other two are schematics and plot nothing.  ``manifest.json`` binds each figure to
its generator and its exact inputs, because PDF streams differ across
Matplotlib and font builds even when they draw the same paths, so a byte
comparison is the wrong drift gate and a digest of the inputs is the right one.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/clifford-qc-mpl")

import numpy as np

try:  # package import in tests/tools versus direct script execution
    from .make_tables import LAYERS
except ImportError:  # pragma: no cover - exercised by the documented CLI
    from make_tables import LAYERS

# Matplotlib is a figure-time dependency, not an import-time one.  The package
# keeps optional dependencies out of every module initializer for the same
# reason, and check_manuscript.py imports this module for FIGURE_SOURCES alone:
# the manuscript gate must run wherever the test suite runs, which is an
# environment with numpy and no plotting stack.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / "clifford_qc"
ASSETS = HERE / "paper_assets"
CENSUS = HERE / "data" / "source_census.json"
TABLE_GENERATOR = HERE / "make_tables.py"
RECORDS = ROOT / "benchmarks" / "reference_results"
HIER_H4 = RECORDS / "clifford_hierarchy_h4_v2.json"
HIER_BEH2 = RECORDS / "clifford_hierarchy_beh2_v2.json"
PROTOCOL_COST = RECORDS / "protocol_cost.json"
PDF_METADATA = {"CreationDate": None, "ModDate": None}

CARDS = ("ion-like", "logical-alltoall", "superconducting-like")


def _pyplot():
    """Return the configured pyplot module, importing it on first use."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style() -> None:
    plt = _pyplot()
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


def _save(fig, name: str) -> None:
    fig.savefig(ASSETS / name, bbox_inches="tight", metadata=PDF_METADATA)
    _pyplot().close(fig)


def _source_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def architecture_sources_digest() -> str:
    """Bind the layer diagram to every package source that can change an edge."""
    entries = [
        f"{path.relative_to(ROOT).as_posix()}:{_git_blob_sha(path)}"
        for path in sorted(PACKAGE.rglob("*.py"))
    ]
    payload = "\n".join(entries) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FIGURE_SOURCES: dict[str, tuple[Path, ...]] = {
    "architecture_flow.pdf": (),
    "layer_stack.pdf": (CENSUS, TABLE_GENERATOR),
    "evidence_path.pdf": (),
    "hierarchy_tradeoff.pdf": (HIER_H4, HIER_BEH2),
    "cost_bracket.pdf": (PROTOCOL_COST,),
}


def _write_manifest() -> None:
    figures = {
        name: {
            "sources": {
                str(path.resolve().relative_to(ROOT)): _source_digest(path)
                for path in paths
            },
        }
        for name, paths in FIGURE_SOURCES.items()
    }
    figures["layer_stack.pdf"][
        "architecture_sources_sha256"] = architecture_sources_digest()
    manifest = {
        "schema": "clifford_qc.architecture_figure_manifest.v1",
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _source_digest(Path(__file__).resolve()),
        },
        "figures": figures,
    }
    (ASSETS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _box(ax, x, y, w, h, title, subtitle, color, title_size=7.4,
         subtitle_size=6.1):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=0.7, edgecolor="#37474f", facecolor=color))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color="#37474f")


def _arrow(ax, start, end, *, dashed=False, color="#263238"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.7,
        color=color, linestyle="--" if dashed else "-",
        shrinkA=1.0, shrinkB=1.0))


def architecture_flow() -> None:
    """The numerical path and the claim path, joined by typed contracts."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.05)
    ax.axis("off")

    ax.text(0.05, 4.72, "scientific path", fontsize=7.1, fontweight="bold",
            color="#263238", va="center")
    scientific = [
        (0.05, 3.35, 1.55, 1.0, "Model",
         "$H$, observables,\nreference state", "#e8f5e9"),
        (1.88, 3.35, 2.15, 1.0, "Reduction policy",
         "pass through\nsector storage / tapering: exact\ncontextual: approximate",
         "#d9edf7"),
        (4.31, 3.35, 2.15, 1.0, "Solver policy",
         "A-CASE | QSCI / selected CI\nADAPT-VQE | hybrid", "#f3e5f5"),
        (6.74, 3.35, 3.16, 1.0, "Estimator and cost",
         "exact oracle or compiled measurement\nplan + backend + cumulative cache",
         "#fff3cd"),
    ]
    for x, y, w, h, title, subtitle, color in scientific:
        _box(ax, x, y, w, h, title, subtitle, color,
             title_size=7.1, subtitle_size=5.8)
    for left, right in zip(scientific, scientific[1:]):
        _arrow(ax, (left[0] + left[2], left[1] + left[3] / 2),
               (right[0], right[1] + right[3] / 2))

    ax.text(0.05, 2.86, "claim path", fontsize=7.1, fontweight="bold",
            color="#263238", va="center")
    claim = [
        (0.05, 1.55, 1.55, 0.92, "Config / plan",
         "acceptance rule\npreregistered when declared",
         "#eceff1"),
        (1.88, 1.55, 2.15, 0.92, "Labelled result",
         "value + uncertainty + evidence", "#ffe0b2"),
        (4.31, 1.55, 2.15, 0.92, "Versioned record",
         "inputs + provenance + outcome", "#eceff1"),
        (6.74, 1.55, 1.35, 0.92, "Gate",
         "rebuild | audit", "#ffebee"),
        (8.37, 1.55, 1.53, 0.92, "Disposition",
         "admit | refuse |\nindeterminate", "#ffebee"),
    ]
    for x, y, w, h, title, subtitle, color in claim:
        _box(ax, x, y, w, h, title, subtitle, color,
             title_size=6.4, subtitle_size=5.4)
    for left, right in zip(claim, claim[1:]):
        _arrow(ax, (left[0] + left[2], left[1] + left[3] / 2),
               (right[0], right[1] + right[3] / 2))
    ax.plot([8.32, 8.32, 3.0], [3.35, 2.78, 2.78],
            linewidth=0.7, color="#b71c1c")
    _arrow(ax, (3.0, 2.78), (3.0, 2.47), color="#b71c1c")
    ax.text(5.65, 2.94, "measurement decides the evidence label",
            fontsize=5.8, color="#b71c1c", ha="center", va="center")

    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.25), 9.85, 0.62,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=0.7, edgecolor="#37474f", facecolor="#f5f7f8"))
    ax.text(4.975, 0.56,
            "shared contracts: MV | Program / PauliSum | Backend | Restriction | "
            "CompiledMeasurementPlan | EvidenceLevel",
            ha="center", va="center", fontsize=6.1, color="#263238")
    ax.text(2.96, 3.17,
            "generator filtering changes the candidate pool, not Hilbert-space width",
            ha="center", va="top", fontsize=5.5, color="#455a64")

    fig.tight_layout(pad=0.25)
    _save(fig, "architecture_flow.pdf")


def layer_stack() -> None:
    """Measured size of every layer, beside the edges that cross them."""
    plt = _pyplot()
    census = json.loads(CENSUS.read_text(encoding="utf-8"))["source"]
    labels = sorted(
        LAYERS, key=lambda item: census["layers"][item[1]]["lines"],
        reverse=True)
    lines = [census["layers"][key]["lines"] for _, key in labels]
    modules = [census["layers"][key]["modules"] for _, key in labels]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.1, 3.15),
        gridspec_kw={"width_ratios": [0.92, 1.18]})

    position = np.arange(len(labels))[::-1]
    left.barh(position, lines, height=0.66, color="#90a4ae",
              edgecolor="#37474f", linewidth=0.5)
    for y, value, count in zip(position, lines, modules):
        left.text(value + max(lines) * 0.015, y, f"{value:,} / {count}",
                  va="center", fontsize=6.2, color="#263238")
    left.set_yticks(position)
    left.set_yticklabels([label for label, _ in labels], fontsize=6.6)
    left.set_xlabel("physical lines (annotated: lines / modules)")
    left.set_xlim(0, max(lines) * 1.28)
    left.spines[["top", "right"]].set_visible(False)
    left.set_title("(a) Declared-layer size", loc="left")

    right.set_xlim(0, 10)
    right.set_ylim(-0.75, 7.6)
    right.axis("off")
    right.set_title("(b) Selected dependency edges", loc="left")
    # Boxes occupy the middle; the two gutters carry the deferred upward edges,
    # so no arrow crosses a box it does not touch.
    tiers = [
        (1.25, 6.05, 7.3, 0.82, "Evidence and reproduction",
         "benchmarks/, verify.py, reproducibility.py", "#eceff1"),
        (1.25, 4.85, 3.5, 0.82, "Solvers",
         "subspace/, algorithms/", "#f3e5f5"),
        (5.05, 4.85, 3.5, 0.82, "Problem definition", "models/", "#e8f5e9"),
        (1.25, 3.65, 3.5, 0.82, "Measurement", "measurement/", "#fff3cd"),
        (5.05, 3.65, 3.5, 0.82, "Execution", "backends/", "#fff3cd"),
        (1.25, 2.45, 7.3, 0.82, "Program IR",
         "ir.py, qasm3.py, fermion_mapping.py", "#d9edf7"),
        (1.25, 1.25, 7.3, 0.82, "Algebra kernel",
         "multivector.py, pauli_kernel.py, ...", "#d9edf7"),
        (1.25, 0.15, 7.3, 0.68, "Optional bridges",
         "stim, OpenFermion, pytket, PennyLane, PyZX", "#ffffff"),
    ]
    for x, y, w, h, title, subtitle, color in tiers:
        _box(right, x, y, w, h, title, subtitle, color,
             title_size=6.4, subtitle_size=5.2)
    down = [
        ((3.0, 6.05), (3.0, 5.67)),
        ((6.8, 6.05), (6.8, 5.67)),
        ((2.6, 4.85), (2.6, 4.47)),
        ((2.6, 3.65), (2.6, 3.27)),
        ((6.8, 3.65), (6.8, 3.27)),
        ((4.9, 2.45), (4.9, 2.07)),
    ]
    for start, end in down:
        _arrow(right, start, end)
    # The upward edges are real; only their timing is constrained.  They are
    # unlabelled in the gutter and named once below, which is legible at column
    # width where three rotated captions are not.
    # measurement -> subspace, routed through the left gutter
    _arrow(right, (1.05, 3.95), (1.05, 5.15), dashed=True, color="#b71c1c")
    # kernel -> execution, routed through the right gutter into that box
    right.plot([8.55, 8.95, 8.95], [1.66, 1.66, 4.06],
               linestyle="--", linewidth=0.8, color="#b71c1c")
    _arrow(right, (8.95, 4.06), (8.55, 4.06),
           dashed=True, color="#b71c1c")
    # Program IR -> optional bridges, a separate downward deferred call
    right.plot([8.55, 9.5, 9.5], [2.86, 2.86, 0.49],
               linestyle="--", linewidth=0.8, color="#1565c0")
    _arrow(right, (9.5, 0.49), (8.55, 0.49),
           dashed=True, color="#1565c0")
    right.text(
        1.25, -0.32,
        "red upward: measurement $\\to$ subspace; kernel $\\to$ backends",
        fontsize=5.6, color="#b71c1c", va="top")
    right.text(
        1.25, -0.62,
        "blue downward: Program IR $\\to$ optional bridges (deferred)",
        fontsize=5.6, color="#1565c0", va="top")
    right.text(1.25, 7.25,
               "solid: import-time    red dashed: upward runtime    "
               "blue dashed: optional bridge",
               fontsize=6.0, color="#37474f")
    fig.tight_layout()
    _save(fig, "layer_stack.pdf")


def evidence_path() -> None:
    """One pass from a word universe to a labelled interval."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    row = [
        (0.1, 2.25, 1.55, 0.95, "Word universe", "PauliSum,\nbank rows", "#d9edf7"),
        (1.85, 2.25, 1.7, 0.95, "Grouping", "QWC $k=1$ to\nfully commuting $k=n$", "#d9edf7"),
        (3.75, 2.25, 1.75, 0.95, "Compiled plan", "frozen groups, Clifford\nsettings, ledger", "#e8f5e9"),
        (5.7, 2.25, 1.6, 0.95, "Allocation", "uniform, variance,\ngroup-optimal", "#e8f5e9"),
        (7.5, 2.25, 2.4, 0.95, "Sampling backend", "GroupSample per setting", "#fff3cd"),
    ]
    lower = [
        (7.5, 0.5, 2.4, 0.95, "Cumulative cache",
         "a shared word is\npaid for once", "#fff3cd"),
        (5.5, 0.5, 1.75, 0.95, "Word functional",
         "linear read of\ncached words", "#f3e5f5"),
        (3.4, 0.5, 1.85, 0.95, "Confidence bound",
         "empirical Bernstein,\nSidak, Jeffreys", "#f3e5f5"),
        (0.1, 0.5, 3.05, 0.95, "Labelled interval",
         "exact | asymptotic | finite sample | heuristic", "#ffe0b2"),
    ]
    for x, y, w, h, title, subtitle, color in row + lower:
        _box(ax, x, y, w, h, title, subtitle, color)
    for index in range(len(row) - 1):
        x, y, w, h = row[index][0], row[index][1], row[index][2], row[index][3]
        _arrow(ax, (x + w, y + h / 2), (row[index + 1][0], y + h / 2))
    _arrow(ax, (8.7, 2.25), (8.7, 1.45))
    for index in range(len(lower) - 1):
        x, y, w, h = lower[index][0], lower[index][1], lower[index][2], lower[index][3]
        next_x = lower[index + 1][0] + lower[index + 1][2]
        _arrow(ax, (x, y + h / 2), (next_x, y + h / 2))
    ax.plot([8.7, 6.45], [1.78, 1.78], linestyle="--", linewidth=0.7,
            color="#1565c0")
    ax.plot([8.7, 8.7], [1.45, 1.78], linestyle="--", linewidth=0.7,
            color="#1565c0")
    _arrow(ax, (6.45, 1.78), (6.5, 2.25), dashed=True, color="#1565c0")
    ax.text(3.75, 1.87, "covariance-aware variance feeds the allocator",
            fontsize=6.0, color="#1565c0", ha="center")
    ax.text(0.1, 0.03,
            "Every box is a seam: the label a number carries is decided here, "
            "not at the solver.", fontsize=6.2, color="#37474f")
    fig.tight_layout()
    _save(fig, "evidence_path.pdf")


def hierarchy_tradeoff() -> None:
    """Settings fall and entangling depth rises along the same axis."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharex=True)
    for ax, (label, path) in zip(axes, (("H$_4$", HIER_H4), ("BeH$_2$", HIER_BEH2))):
        record = json.loads(path.read_text(encoding="utf-8"))
        blocks = [row["block_size"] for row in record["rows"]]
        settings = [row["settings"] for row in record["rows"]]
        depth = [row["max_logical_cx_depth"] for row in record["rows"]]
        ratio = [row["cx_to_preparation_cost_ratio_vs_qwc"] for row in record["rows"]]

        ax.plot(blocks, settings, marker="o", markersize=3.4, linewidth=1.0,
                color="#1565c0", label="compiled settings")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(blocks)
        ax.set_xticklabels([str(block) for block in blocks])
        ax.set_xlabel("block size $k$")
        ax.set_ylabel("compiled settings", color="#1565c0")
        ax.tick_params(axis="y", labelcolor="#1565c0")
        ax.spines[["top"]].set_visible(False)

        twin = ax.twinx()
        twin.plot(blocks, depth, marker="s", markersize=3.4, linewidth=1.0,
                  color="#b71c1c", label="max CX depth per setting")
        twin.set_ylabel("max logical CX depth", color="#b71c1c")
        twin.tick_params(axis="y", labelcolor="#b71c1c")
        twin.spines[["top"]].set_visible(False)

        for block, value in zip(blocks, ratio):
            if value is None:
                continue
            twin.annotate(f"{value:.2f}", (block, depth[blocks.index(block)]),
                          textcoords="offset points", xytext=(0, 7),
                          fontsize=5.8, color="#4e342e", ha="center")
        twin.set_ylim(top=max(depth) * 1.25)
        ax.set_xlim(0.8, 10.5)
        worst = max(value for value in ratio if value is not None)
        verdict = "trade pays" if worst < 1.0 else "trade does not pay"
        ax.set_title(f"{label}: {verdict}", loc="left")
    fig.tight_layout()
    _save(fig, "hierarchy_tradeoff.pdf")


def cost_bracket() -> None:
    """Accuracy-matched cost brackets, and what the cards do to them."""
    plt = _pyplot()
    record = json.loads(PROTOCOL_COST.read_text(encoding="utf-8"))
    beh2 = record["systems"]["beh2"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=False)
    for ax, card in zip(axes, CARDS):
        arm = beh2["k_star"][card]["single_assignment"]["by_arm"]["jw"]
        blocks = [cost["block_size"] for cost in arm["costs"]]
        point = np.array([cost["C_time_epsilon_us"] for cost in arm["costs"]]) / 1e6
        lower = np.array([cost["C_time_lower_us"] for cost in arm["costs"]]) / 1e6
        upper = np.array([cost["C_time_upper_us"] for cost in arm["costs"]]) / 1e6
        ax.vlines(blocks, lower, upper, color="#90a4ae", linewidth=3.2)
        ax.plot(blocks, point, marker="o", markersize=3.6, linewidth=0.0,
                color="#1565c0", label="point estimate")
        region = set(arm["k_star_region"])
        ax.plot([block for block in blocks if block in region],
                [value for block, value in zip(blocks, point) if block in region],
                marker="o", markersize=6.5, markerfacecolor="none",
                linewidth=0.0, color="#1b5e20", label="$k^*$ region")
        # A rung the card refuses has no priced cost at all, so it cannot be
        # drawn at a y value: it is marked at the top of the panel instead.
        refused = [block for block in arm["inadmissible_block_sizes"]
                   if block not in blocks]
        ceiling = float(upper.max())
        for block in refused:
            ax.plot([block], [ceiling * 1.8], marker="x", markersize=5.0,
                    color="#b71c1c", linewidth=0.0)
            ax.annotate("refused", (block, ceiling * 1.8),
                        textcoords="offset points", xytext=(0, -11),
                        fontsize=5.8, color="#b71c1c", ha="center")
        for block in arm["inadmissible_block_sizes"]:
            if block in blocks:
                ax.annotate("refused", (block, point[blocks.index(block)]),
                            textcoords="offset points", xytext=(-9, 6),
                            fontsize=5.8, color="#b71c1c")
        ticks = sorted(set(blocks) | set(refused))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(block) for block in ticks])
        ax.set_xlabel("block size $k$")
        ax.set_title(card, loc="left", fontsize=7.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$C_{\mathrm{time}}(\epsilon)$  (s)")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "cost_bracket.pdf")


def main() -> None:
    _style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    architecture_flow()
    layer_stack()
    evidence_path()
    hierarchy_tradeoff()
    cost_bracket()
    _write_manifest()
    print(ASSETS)


if __name__ == "__main__":
    main()
