"""Generate the vector figures for the compactness paper.

``yardsticks.pdf`` plots every prefix of the declared budget comparison against
three axes; ``support_spectrum.pdf`` is a schematic and plots nothing.
``manifest.json`` binds each figure to this generator, to the digests of its
inputs and to a checksum of the committed asset.  The manuscript gate checks
the manifest rather than re-rendering, because PDF streams differ across
Matplotlib and font builds even when they draw the same paths.

Matplotlib is imported when a figure is drawn, not when this module is
imported: ``check_manuscript.py`` imports ``FIGURE_SOURCES`` from here and must
run where no plotting stack is installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/clifford-qc-mpl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "paper_assets"
RECORDS = ROOT / "benchmarks" / "reference_results"
BUDGET = RECORDS / "packet_budget_convergence.json"
TABLE_GENERATOR = HERE / "make_tables.py"
PDF_METADATA = {"CreationDate": None, "ModDate": None}

FIGURE_SOURCES: dict[str, tuple[Path, ...]] = {
    "support_spectrum.pdf": (),
    # The plot decodes trajectories through the table generator's loader, so a
    # change to that loader is a change to this figure's input.
    "yardsticks.pdf": (BUDGET, TABLE_GENERATOR),
}

# Categorical slots one to three of the validated reference palette, in fixed
# order: packets, selected CI, Davidson.  The early policy is a variant of
# packets and keeps their hue, distinguished by a dashed line.  Slot three sits
# below 3:1 contrast on white, so every panel row is also direct-labelled and
# each arm has its own line style.
COLORS = {
    "packets_exhaustive": "#2a78d6",
    "packets_early": "#2a78d6",
    "selected_ci": "#eb6834",
    "davidson": "#1baf7a",
}
STYLES = {
    "packets_exhaustive": "-",
    "packets_early": (0, (3, 1.6)),
    "selected_ci": "-",
    "davidson": "-",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
NEUTRAL_FILL = "#d9d8d3"


def _source_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    return plt


def _save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, metadata=PDF_METADATA, bbox_inches="tight", pad_inches=0.02)


def _write_manifest() -> None:
    figures = {
        name: {
            "asset_git_blob_sha": _git_blob_sha(ASSETS / name),
            "sources": {str(path.resolve().relative_to(ROOT)): _source_digest(path) for path in paths},
        }
        for name, paths in FIGURE_SOURCES.items()
    }
    manifest = {
        "schema": "clifford_qc.compactness_figure_manifest.v1",
        "generator": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                      "sha256": _source_digest(Path(__file__).resolve())},
        "figures": figures,
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fig. 1: what one new column may contain under each growth rule
# ---------------------------------------------------------------------------
# Spin blocks of an illustrative frame, as (determinant rows, singlet columns),
# grouped by occupation outside the active space.  The frame is block diagonal
# once determinants are ordered by spatial pattern: each CSF lives on one block.
SCHEMATIC_GROUPS = (
    ((1, 1), (2, 1)),
    ((3, 2), (2, 1)),
    ((2, 1), (3, 2)),
    ((1, 1), (2, 1), (2, 1)),
)


def support_spectrum() -> None:
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    blocks, group_of = [], []
    row = column = 0
    for g, group in enumerate(SCHEMATIC_GROUPS):
        for rows, columns in group:
            blocks.append((row, rows, column, columns))
            group_of.append(g)
            row += rows
            column += columns
    n_rows, n_columns = row, column

    panels = (
        ("selected_ci", "Selected CI", "one CSF", lambda b, c: (b, c) == (3, blocks[3][2])),
        ("packets_exhaustive", "Packet", "one group", lambda b, c: group_of[b] == 1),
        ("davidson", "Davidson", "whole frame", lambda b, c: True),
    )
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.95), sharey=True)
    for ax, (arm, title, caption, chosen) in zip(axes, panels):
        color = COLORS[arm]
        support = set()
        for b, (r0, rows, c0, columns) in enumerate(blocks):
            for c in range(c0, c0 + columns):
                picked = chosen(b, c)
                for r in range(r0, r0 + rows):
                    ax.add_patch(Rectangle((c + 0.06, n_rows - r - 1 + 0.06), 0.88, 0.88, linewidth=0,
                                           facecolor=color if picked else NEUTRAL_FILL))
                if picked:
                    support.update(range(r0, r0 + rows))
        # The new column's support: every determinant it may occupy.
        for r in range(n_rows):
            ax.add_patch(Rectangle((n_columns + 0.6, n_rows - r - 1 + 0.06), 0.88, 0.88, linewidth=0,
                                   facecolor=color if r in support else "white",
                                   edgecolor=GRID if r not in support else color))
        # Brackets for the occupation groups outside the active space.
        start = 0
        for group in SCHEMATIC_GROUPS:
            width = sum(columns for _, columns in group)
            ax.plot([start + 0.15, start + width - 0.15], [n_rows + 0.45] * 2, color=INK_MUTED, linewidth=0.6)
            start += width
        ax.set_xlim(-0.3, n_columns + 1.8)
        ax.set_ylim(-0.3, n_rows + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=INK, pad=2)
        ax.text(n_columns / 2, -1.0, caption, ha="center", va="top", fontsize=6.5, color=INK_SECONDARY)
        ax.text(n_columns + 1.05, n_rows + 0.45, "S", ha="center", va="center", fontsize=6.5,
                color=INK_SECONDARY, style="italic")
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.88, bottom=0.12)
    _save(fig, "support_spectrum.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2: one set of trajectories, three yardsticks
# ---------------------------------------------------------------------------
ROWS = (
    ("hubbard_2x4_u4", r"$2{\times}4$ Hubbard (primary)"),
    ("hubbard_1x8_u4", r"$1{\times}8$ chain (held out)"),
    ("beh2_stretched_full", r"BeH$_2$, stretched (held out)"),
)
AXES = (
    ("columns", "retained columns $K$"),
    ("actions", "Hamiltonian actions $W$"),
    ("support", "determinant support $S$"),
)
ARM_LABELS = {
    "packets_exhaustive": "packets",
    "packets_early": "packets, early",
    "selected_ci": "selected CI",
    "davidson": "Davidson",
}
ERROR_FLOOR = 1e-5


def yardsticks() -> None:
    plt = _pyplot()
    import sys
    sys.path.insert(0, str(HERE))
    from make_tables import ARMS, budget

    b = budget()
    fig, grid = plt.subplots(len(ROWS), len(AXES), figsize=(7.0, 5.6), sharey="row")
    for i, (name, row_title) in enumerate(ROWS):
        unit = b.inputs[name]["energy_unit"]
        reference = b.reference(name)
        for j, (axis, label) in enumerate(AXES):
            ax = grid[i][j]
            for threshold in b.thresholds:
                ax.axhline(threshold, color=GRID, linewidth=0.8, zorder=0)
            for arm in ARMS:
                science = b.science(name, arm)
                trajectory = science["trajectory"]
                x = {"columns": [r["dimension"] for r in trajectory],
                     "actions": [r["h_action_calls"] for r in trajectory],
                     "support": [s[0] for s in science["support"]]}[axis]
                y = [max(r["energy"] - reference, ERROR_FLOOR) for r in trajectory]
                ax.plot(x, y, color=COLORS[arm], linestyle=STYLES[arm], linewidth=1.3 if arm != "packets_early" else 1.0,
                        solid_capstyle="round", zorder=3 if arm == "davidson" else 2)
                ax.plot([x[-1]], [y[-1]], marker="o", markersize=3.2, color=COLORS[arm],
                        markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ERROR_FLOOR * 0.6, 20)
            ax.grid(False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == len(ROWS) - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"error above reference ({'Ha' if unit == 'Ha' else '$t$'})")
                ax.text(0.02, 1.02, row_title, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7.5, color=INK, fontweight="bold")
    # Direct labels where the four curves separate best: the chain's action panel.
    # Each label sits beside the arm's last prefix, except Davidson's, which
    # labels its descent rather than its endpoint on the floor.
    ax = grid[1][1]
    name = ROWS[1][0]
    reference = b.reference(name)
    placements = {"packets_exhaustive": ((0, -8), "center"), "packets_early": ((-5, -2), "right"),
                  "selected_ci": ((5, 0), "left"), "davidson": ((6, 0), "left")}
    for arm in ARMS:
        trajectory = b.science(name, arm)["trajectory"]
        point = trajectory[-1]
        if arm == "davidson":
            point = next(r for r in trajectory if r["energy"] - reference <= 1e-2)
        x, y = point["h_action_calls"], max(point["energy"] - reference, ERROR_FLOOR)
        offset, align = placements[arm]
        ax.annotate(ARM_LABELS[arm], (x, y), xytext=offset, textcoords="offset points",
                    ha=align, va="center", fontsize=6.5, color=INK_SECONDARY)
    handles = [plt.Line2D([], [], color=COLORS[arm], linestyle=STYLES[arm], linewidth=1.3, label=ARM_LABELS[arm])
               for arm in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(ARMS), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.08, wspace=0.08, hspace=0.32)
    _save(fig, "yardsticks.pdf")
    plt.close(fig)


def main() -> None:
    support_spectrum()
    yardsticks()
    _write_manifest()


if __name__ == "__main__":
    main()
