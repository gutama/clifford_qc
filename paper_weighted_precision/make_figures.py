"""Generate deterministic vector figures for the weighted precision manuscript.

All four figures are plotted directly from the frozen Commit 1 declaration and
Commit 2 result record (``docs/restart/weighted_precision_results.json.gz``)
and bound via SHA-256 digests in ``paper_assets/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/clifford-qc-mpl")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_weighted_precision.make_tables import (  # noqa: E402
    DECLARATION_PATH,
    RESULT_PATH,
    load_verified_payload,
)

ASSETS = HERE / "paper_assets"
MANIFEST_PATH = ASSETS / "manifest.json"
PDF_METADATA = {"CreationDate": None, "ModDate": None}

# Colorblind-safe Okabe-Ito palette
C_LDF = "#D55E00"       # vermilion
C_SI = "#0072B2"        # blue
C_RITZ = "#009E73"      # bluish green
C_WARN = "#E69F00"      # orange
C_QWC = "#6A5ACD"       # slate purple
C_GRAY = "#555555"

FIGURE_NAMES = (
    "fig1_shot_savings_and_grid.pdf",
    "fig2_r_emp_vs_r_hat.pdf",
    "fig3_stretched_rank_and_width.pdf",
    "fig4_noisy_bias_variance.pdf",
)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style() -> None:
    plt = _plt()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: Any, name: str, preview_dir: Path | None = None) -> None:
    plt = _plt()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, bbox_inches="tight", metadata=PDF_METADATA)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest() -> dict[str, Any]:
    sources = (DECLARATION_PATH, RESULT_PATH)
    return {
        "schema": "clifford_qc.weighted_precision_figure_manifest.v1",
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256_text(Path(__file__).resolve()),
        },
        "figures": {
            name: {
                "sha256": _sha256_file(ASSETS / name) if (ASSETS / name).is_file() else None,
                "sources": {
                    str(path.relative_to(ROOT)): _sha256_file(path) for path in sources
                }
            }
            for name in FIGURE_NAMES
        },
    }


def fig1_shot_savings_and_grid(
    decl: dict[str, Any],
    record: dict[str, Any],
    summary: dict[str, Any],
    preview_dir: Path | None = None,
) -> None:
    """Fig 1: (a) Matched-k adaptive shot savings + setting counts; (b) LiH look-by-look SE."""
    from matplotlib.ticker import NullFormatter

    plt = _plt()
    part_a_map = {(r["case"], r["arm"]): r for r in summary["part_a"]}
    sys_by_name = {s["name"]: s for s in record["systems"]}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    groups = [
        ("beh2_8q", 1, "ldf_qwc", "si_qwc", None, "BeH$_2$\n$k=1$"),
        ("beh2_8q", 2, "ldf_k2", "si_k2", None, "BeH$_2$\n$k=2$"),
        ("beh2_8q", 8, "ldf_fc", "si_fc", "ritz_si_fc", "BeH$_2$\n$k=8$"),
        ("lih_8q", 1, "ldf_qwc", "si_qwc", None, "LiH\n$k=1$"),
        ("lih_8q", 2, "ldf_k2", "si_k2", None, "LiH\n$k=2$"),
        ("lih_8q", 8, "ldf_fc", "si_fc", "ritz_si_fc", "LiH\n$k=8$"),
        ("h4_8q", 1, "ldf_qwc", "si_qwc", None, "H$_4$\n$k=1$"),
        ("h4_8q", 2, "ldf_k2", "si_k2", None, "H$_4$\n$k=2$"),
        ("h4_8q", 8, "ldf_fc", "si_fc", "ritz_si_fc", "H$_4$\n$k=8$"),
    ]
    x = np.arange(len(groups), dtype=float)
    w = 0.27

    ldf_vals = [part_a_map[(c, ldf)]["mean_total_shots"] / 1e6 for c, _, ldf, _, _, _ in groups]
    si_vals = [part_a_map[(c, si)]["mean_total_shots"] / 1e6 for c, _, _, si, _, _ in groups]

    ax_a.bar(x - w, ldf_vals, width=w, color=C_LDF, label="Unweighted LDF", edgecolor="white", linewidth=0.4)
    ax_a.bar(x, si_vals, width=w, color=C_SI, label="Bank-weighted SI", edgecolor="white", linewidth=0.4)

    ritz_x = [i + w for i, (_, _, _, _, rz, _) in enumerate(groups) if rz is not None]
    ritz_vals = [
        part_a_map[(c, rz)]["mean_total_shots"] / 1e6
        for c, _, _, _, rz, _ in groups
        if rz is not None
    ]
    ax_a.bar(
        ritz_x,
        ritz_vals,
        width=w,
        color=C_RITZ,
        label="Ritz-SI FC (paid pilot)",
        edgecolor="white",
        linewidth=0.4,
    )

    for i, (c, k, ldf, si, _rz, _) in enumerate(groups):
        ratio = si_vals[i] / ldf_vals[i]
        top_y = max(ldf_vals[i], si_vals[i])
        if abs(ratio - 1.0) < 1e-6:
            txt = "1.0×\n(tie)"
            col = C_GRAY
        else:
            txt = f"{1.0 / ratio:.1f}×"
            col = C_SI
        ax_a.text(i - 0.12, top_y * 1.15, txt, ha="center", va="bottom", fontsize=6.3, color=col, fontweight="bold")
        if k == 1:
            n_ldf = sys_by_name[c]["part_a_plans"][ldf]["settings"]
            n_si = sys_by_name[c]["part_a_plans"][si]["settings"]
            ax_a.text(
                i,
                0.105,
                f"{n_si}>{n_ldf} set.",
                ha="center",
                va="bottom",
                fontsize=5.3,
                color="#333333",
                bbox={"boxstyle": "round,pad=0.14", "fc": "#f5f5f5", "ec": "#dddddd", "lw": 0.4, "alpha": 0.94},
            )
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.095, 28.0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([g[5] for g in groups], fontsize=6.4)
    ax_a.set_ylabel("Mean physical shots ($\\times 10^6$, log)")
    ax_a.set_title("(a) Matched-$k$ shot cost (P1a & P1b)")
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")
    for sep in (2.5, 5.5):
        ax_a.axvline(sep, color="#bbbbbb", linestyle=":", linewidth=0.8)

    # Panel (b): Look-by-look SE trajectory on LiH (8q)
    lih_rows = [r for r in record["rows"] if r["part"] == "A" and r["case_name"] == "lih_8q"]
    look_specs = [
        ("ldf_fc", "LDF FC ($k=8$)", C_LDF, "s", "--"),
        ("si_fc", "Weighted SI FC ($k=8$)", C_SI, "o", "-"),
        ("si_qwc", "Weighted SI QWC ($k=1$)", C_QWC, "^", "-."),
    ]
    for arm_key, lbl, col, mk, ls in look_specs:
        arows = [r for r in lih_rows if r["arm"] == arm_key]
        xs, meds, los, his = [], [], [], []
        for lk in range(4):
            pts = [
                (r["looks"][lk]["shots"] / 1e6, r["looks"][lk]["std_error"] * 1000.0)
                for r in arows
                if len(r["looks"]) > lk
            ]
            if not pts:
                continue
            xs.append(float(np.median([p[0] for p in pts])))
            ses = [p[1] for p in pts]
            meds.append(float(np.median(ses)))
            los.append(float(min(ses)))
            his.append(float(max(ses)))
        ax_b.fill_between(xs, los, his, color=col, alpha=0.14)
        ax_b.plot(xs, meds, color=col, marker=mk, linestyle=ls, linewidth=1.4, markersize=4.2, label=lbl)

    ax_b.axhline(1.6, color="#222222", linestyle=":", linewidth=1.1, label="Target SE ($1.6$ mHa)")
    ax_b.annotate(
        "SI QWC stops\nat 0.25M (1.54 mHa)",
        xy=(0.25, 1.539),
        xytext=(0.125, 0.72),
        fontsize=6.1,
        color=C_QWC,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": C_QWC, "lw": 0.9},
    )
    ax_b.annotate(
        "Both FC arms\n>1.6 mHa at 0.25M;\nboth step to 0.625M",
        xy=(0.25, 2.03),
        xytext=(0.30, 6.2),
        fontsize=6.0,
        color="#222222",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9},
    )
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.032, 0.85)
    ax_b.set_ylim(0.45, 130.0)
    ax_b.set_xticks([0.04, 0.10, 0.25, 0.625])
    ax_b.set_xticklabels(["0.04", "0.10", "0.25", "0.625"])
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xlabel("Cumulative physical shots ($\\times 10^6$, log)")
    ax_b.set_ylabel("Estimated Ritz SE (mHa, median & range)")
    ax_b.set_title("(b) LiH (8q) look-grid tie & QWC stop")
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout(w_pad=1.6)
    _save(fig, "fig1_shot_savings_and_grid.pdf", preview_dir)


def fig2_r_emp_vs_r_hat(
    decl: dict[str, Any],
    record: dict[str, Any],
    summary: dict[str, Any],
    preview_dir: Path | None = None,
) -> None:
    """Fig 2: (a) Empirical covariance-aware R vs state-independent R_hat; (b) P1b shot ratios."""
    from matplotlib.ticker import NullFormatter

    plt = _plt()
    part_a = summary["part_a"]
    p1b = summary["criteria"]["P1b_ritz_replan_vs_bank_fc"]["cells"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.12, 1.0]}
    )

    case_style = {
        "beh2_8q": ("BeH$_2$ (8q)", C_SI, "o"),
        "lih_8q": ("LiH (8q)", C_LDF, "s"),
        "h4_8q": ("H$_4$ (8q)", C_RITZ, "^"),
        "lih_10q_stretched": ("LiH (10q, str.)", C_WARN, "D"),
    }

    grid = np.geomspace(1.3, 36.0, 100)
    ax_a.plot(grid, grid, color=C_GRAY, linestyle="--", linewidth=1.0, label="$R_{\\mathrm{emp}} = \\hat{R}$")
    ax_a.fill_between(grid, grid, 36.0, color="#eaf7f0", alpha=0.35)
    ax_a.fill_between(grid, 0.9, grid, color="#fdf0ec", alpha=0.30)

    seen_cases: set[str] = set()
    for r in part_a:
        if r["arm"] == "si_fc_width":
            continue
        cname = r["case"]
        lbl, col, mk = case_style[cname]
        r_hat = r["median_r_hat_ritz"]
        r_emp = r["median_r_emp_ritz"]
        ax_a.scatter(
            r_hat,
            r_emp,
            color=col,
            marker=mk,
            s=36,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=lbl if cname not in seen_cases else None,
        )
        seen_cases.add(cname)

    # Clean non-overlapping callouts
    lih_qwc = next(r for r in part_a if r["case"] == "lih_8q" and r["arm"] == "si_qwc")
    ax_a.annotate(
        f"LiH SI QWC\n($R/\\hat{{R}}={lih_qwc['median_r_emp_over_r_hat_ritz']:.2f}$)",
        xy=(lih_qwc["median_r_hat_ritz"], lih_qwc["median_r_emp_ritz"]),
        xytext=(2.2, 28.0),
        fontsize=6.3,
        color=C_LDF,
        fontweight="bold",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": C_LDF, "lw": 0.9},
    )
    ax_a.text(1.5, 13.0, "Empirical > proxy", fontsize=6.0, color="#1d6f42", style="italic")
    ax_a.text(25.0, 2.2, "Empirical < proxy", fontsize=6.0, color="#a03c1e", style="italic", ha="center")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(1.3, 36.0)
    ax_a.set_ylim(0.9, 36.0)
    ax_a.set_xticks([2, 5, 10, 20])
    ax_a.set_xticklabels(["2", "5", "10", "20"])
    ax_a.xaxis.set_minor_formatter(NullFormatter())
    ax_a.set_xlabel("State-independent shot ratio $\\hat{R}$ (Ritz functional)")
    ax_a.set_ylabel("Empirical covariance-aware $R_{\\mathrm{emp}}$")
    ax_a.set_title("(a) Terminal Ritz-shot ratio calibration")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=6.0)

    # Panel (b): P1b mean shot-cost ratios
    pair_order = [
        ("beh2_8q", "ritz_si_fc", "si_fc", "BeH$_2$: Ritz-FC / SI-FC", C_RITZ),
        ("beh2_8q", "ritz_si_fc", "si_qwc", "BeH$_2$: Ritz-FC / SI-QWC", C_RITZ),
        ("beh2_8q", "si_fc", "si_qwc", "BeH$_2$: SI-FC / SI-QWC", C_SI),
        ("lih_8q", "ritz_si_fc", "si_fc", "LiH: Ritz-FC / SI-FC", C_RITZ),
        ("lih_8q", "ritz_si_fc", "si_qwc", "LiH: Ritz-FC / SI-QWC", C_LDF),
        ("lih_8q", "si_fc", "si_qwc", "LiH: SI-FC / SI-QWC", C_LDF),
        ("h4_8q", "ritz_si_fc", "si_fc", "H$_4$: Ritz-FC / SI-FC", C_RITZ),
        ("h4_8q", "ritz_si_fc", "si_qwc", "H$_4$: Ritz-FC / SI-QWC", C_RITZ),
        ("h4_8q", "si_fc", "si_qwc", "H$_4$: SI-FC / SI-QWC", C_SI),
    ]
    p1b_map = {(c["case"], c["treatment"], c["comparator"]): c for c in p1b}
    y_pos = np.arange(len(pair_order))[::-1]
    ax_b.axvline(1.0, color="#222222", linestyle="--", linewidth=1.0)

    for yi, (c, tr, cp, label, col) in zip(y_pos, pair_order, strict=True):
        sr = p1b_map[(c, tr, cp)]["shot_ratio"]
        m = sr["mean_ratio"]
        lo, hi = sr["mean_ci"]
        ax_b.errorbar(
            m,
            yi,
            xerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="o",
            color=col,
            markersize=5.2,
            capsize=2.5,
            linewidth=1.3,
        )
        if m < 1.8:
            ax_b.text(m * 1.11, yi + 0.12, f"{m:.2f}×", fontsize=6.4, color=col, fontweight="bold", ha="left")
        else:
            ax_b.text(m * 0.85, yi + 0.12, f"{m:.2f}×", fontsize=6.4, color=col, fontweight="bold", ha="right")

    ax_b.set_xscale("log")
    ax_b.set_xlim(0.28, 3.6)
    ax_b.set_xticks([0.4, 0.7, 1.0, 1.5, 2.5])
    ax_b.set_xticklabels(["0.4×", "0.7×", "1.0×", "1.5×", "2.5×"])
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([p[3] for p in pair_order], fontsize=6.6)
    ax_b.set_xlabel("Mean physical shot ratio (95% bootstrap CI, log)")
    ax_b.set_title("(b) P1b: FC vs. QWC & paid-pilot replanning")
    ax_b.text(0.32, -0.78, "← Cheaper", fontsize=6.3, color=C_RITZ, ha="left")
    ax_b.text(3.4, -0.78, "Costlier →", fontsize=6.3, color=C_LDF, ha="right")
    ax_b.set_ylim(-1.0, len(pair_order) - 0.35)

    fig.tight_layout(w_pad=1.6, rect=(0, 0.075, 1, 1))
    _save(fig, "fig2_r_emp_vs_r_hat.pdf", preview_dir)


def fig3_stretched_rank_and_width(
    decl: dict[str, Any],
    record: dict[str, Any],
    summary: dict[str, Any],
    preview_dir: Path | None = None,
) -> None:
    """Fig 3: (a) P2 stretched 10q rank transfer; (b) Opt-in bootstrap width-guard tradeoff."""
    from matplotlib.ticker import NullFormatter

    plt = _plt()
    part_a_map = {(r["case"], r["arm"]): r for r in summary["part_a"]}
    width_rows = {w["case"]: w for w in summary["criteria"]["width_guard_overhead"]}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    arms = [
        ("ldf_qwc", "LDF QWC", C_LDF),
        ("si_qwc", "SI QWC", C_QWC),
        ("si_k2", "SI $k=2$\n(P2 PASS)", C_SI),
        ("ritz_si_k2", "Ritz-SI $k=2$\n(P2 PASS)", C_RITZ),
    ]
    x = np.arange(len(arms), dtype=float)
    means = [part_a_map[("lih_10q_stretched", a)]["mean_total_shots"] / 1e6 for a, _, _ in arms]
    meds = [part_a_map[("lih_10q_stretched", a)]["median_total_shots"] / 1e6 for a, _, _ in arms]
    rmses = [part_a_map[("lih_10q_stretched", a)]["rmse_mHa_all_nonfailed"] for a, _, _ in arms]
    trues = [part_a_map[("lih_10q_stretched", a)]["true_precision_stops"] for a, _, _ in arms]

    ax_a.bar(
        x - 0.16,
        means,
        width=0.32,
        color=[c for _, _, c in arms],
        alpha=0.88,
        edgecolor="white",
        label="Mean shots ($\\times 10^6$)",
    )
    ax_a.scatter(
        x - 0.16,
        meds,
        color="#111111",
        marker="_",
        s=110,
        linewidth=1.8,
        zorder=5,
        label="Median shots ($3.125$M)",
    )

    ax_a2 = ax_a.twinx()
    ax_a2.spines["top"].set_visible(False)
    ax_a2.grid(False)
    ax_a2.bar(
        x + 0.18,
        rmses,
        width=0.28,
        color=[c for _, _, c in arms],
        alpha=0.35,
        edgecolor=[c for _, _, c in arms],
        hatch="///",
        linewidth=0.9,
        label="Terminal RMSE (mHa)",
    )
    for i, (tr, m_val, rm) in enumerate(zip(trues, means, rmses, strict=True)):
        ax_a.text(
            i - 0.16,
            m_val + 0.14,
            f"{m_val:.2f}M\n({tr}/16)",
            ha="center",
            va="bottom",
            fontsize=6.1,
            fontweight="bold",
            color="#222222",
        )
        ax_a2.text(
            i + 0.18,
            rm + 0.03,
            f"{rm:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="#444444",
        )

    ax_a.set_ylim(0, 6.75)
    ax_a2.set_ylim(0, 1.45)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([lbl for _, lbl, _ in arms], fontsize=6.6)
    ax_a.set_ylabel("Physical shots ($\\times 10^6$)")
    ax_a2.set_ylabel("Terminal RMSE (mHa)", color="#444444")
    ax_a.set_title("(a) P2: Stretched 10q LiH (0 reduced-rank stops)")

    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, framealpha=0.94, edgecolor="#cccccc", fontsize=6.0)

    # Panel (b): Opt-in width guard Pareto shift (si_fc -> si_fc_width)
    mols = [
        ("beh2_8q", "BeH$_2$ (8q)", C_SI, "o", (1.02, 1.44)),
        ("lih_8q", "LiH (8q)", C_LDF, "s", (1.18, 0.56)),
        ("h4_8q", "H$_4$ (8q)", C_RITZ, "^", (5.8, 0.87)),
    ]
    for cname, lbl, col, mk, label_xy in mols:
        base = part_a_map[(cname, "si_fc")]
        wide = part_a_map[(cname, "si_fc_width")]
        bx, by = base["mean_total_shots"] / 1e6, base["rmse_mHa_all_nonfailed"]
        wx, wy = wide["mean_total_shots"] / 1e6, wide["rmse_mHa_all_nonfailed"]
        ax_b.scatter(bx, by, color=col, marker=mk, s=46, facecolors="white", linewidth=1.5, zorder=5)
        ax_b.scatter(wx, wy, color=col, marker=mk, s=50, edgecolor="white", linewidth=0.5, zorder=5, label=lbl)
        ax_b.annotate(
            "",
            xy=(wx, wy),
            xytext=(bx, by),
            arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3, "shrinkA": 4, "shrinkB": 5},
        )
        wratio = width_rows[cname]["shot_ratio"]["mean_ratio"]
        bratio = width_rows[cname]["bootstrap_solve_ratio"]
        exh_note = f"\n({wide['exhausted']}/{wide['attempts']} exh.)" if wide["exhausted"] else ""
        ax_b.text(
            label_xy[0],
            label_xy[1],
            f"{wratio:.1f}× shots, {bratio:.1f}× boot{exh_note}",
            fontsize=5.9,
            color=col,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax_b.scatter([], [], color=C_GRAY, marker="o", s=36, facecolors="white", linewidth=1.3, label="Rank-only (si_fc)")
    ax_b.scatter([], [], color=C_GRAY, marker="o", s=36, label="+ Width guard (si_fc_width)")
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.45, 14.5)
    ax_b.set_ylim(0.28, 1.82)
    ax_b.set_xticks([0.5, 1.0, 2.0, 4.0, 10.0])
    ax_b.set_xticklabels(["0.5", "1", "2", "4", "10"])
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xlabel("Mean physical shots ($\\times 10^6$, log)")
    ax_b.set_ylabel("Terminal RMSE (mHa)")
    ax_b.set_title("(b) Opt-in bootstrap half-width guard cost")
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.99, 0.87), frameon=True, framealpha=0.94, edgecolor="#cccccc", fontsize=5.8)

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig3_stretched_rank_and_width.pdf", preview_dir)


def fig4_noisy_bias_variance(
    decl: dict[str, Any],
    record: dict[str, Any],
    summary: dict[str, Any],
    preview_dir: Path | None = None,
) -> None:
    """Fig 4: Part B bias-variance decomposition across pure-2Q noise and sc_card_full."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = _plt()
    part_b_map = {(r["case"], r["regime"], r["arm"]): r for r in summary["part_b"]}

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), sharey="row")
    mols = [("beh2_8q", "BeH$_2$ (8q)"), ("lih_8q", "LiH (8q)"), ("h4_8q", "H$_4$ (8q)")]
    arms = [
        ("si_qwc", "QWC", C_QWC),
        ("si_fc_galic_tight", "Tight", C_RITZ),
        ("si_fc_galic_moderate", "Moderate", C_WARN),
        ("si_fc_unconstrained", "FC", C_LDF),
    ]
    x = np.arange(len(arms), dtype=float)
    w = 0.36

    for col_i, (cname, mtitle) in enumerate(mols):
        ax = axes[0, col_i]
        for reg_i, (reg_key, offset, alpha_b, hatch_v) in enumerate([
            ("depol2q_p002_r3", -w / 2, 0.92, None),
            ("depol2q_p010_r3", +w / 2, 0.55, "//"),
        ]):
            rmses = [part_b_map[(cname, reg_key, a)]["rmse_mHa"] for a, _, _ in arms]
            inf_b = [abs(part_b_map[(cname, reg_key, a)]["exact_inf_bias_mHa"]) for a, _, _ in arms]
            false_s = [part_b_map[(cname, reg_key, a)]["false_precision_stops"] for a, _, _ in arms]
            ax.bar(
                x + offset,
                rmses,
                width=w * 0.92,
                color=[c for _, _, c in arms],
                alpha=alpha_b,
                edgecolor="#222222" if reg_i == 1 else "white",
                linewidth=0.5,
                hatch=hatch_v,
            )
            ax.scatter(
                x + offset,
                inf_b,
                marker="D",
                s=14,
                color="#111111",
                facecolors="white",
                linewidth=0.9,
                zorder=5,
            )
            for xi, rm, ib, fs in zip(x + offset, rmses, inf_b, false_s, strict=True):
                ax.text(
                    xi,
                    max(rm, ib) + 0.72,
                    f"{fs}",
                    ha="center",
                    va="bottom",
                    fontsize=5.6,
                    color="#222222",
                    fontweight="bold" if fs >= 12 else "normal",
                )

        ax.axhline(1.6, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_title(f"{mtitle}: Pure 2Q noise ($r=3$)", fontsize=8.0)
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{label}\n({part_b_map[(cname, reg_key, key)]['cz_max']} CZ)"
            for key, label, _ in arms
        ], fontsize=6.0)
        ax.set_ylim(-0.5, 20.5)
        if col_i == 0:
            ax.set_ylabel("Finite-shot RMSE & $|$bias$|$ (mHa)\n[numbers = false stops / 16]")
            handles = [
                Patch(facecolor="#666666", alpha=0.92, label="$p_{2q}=0.002$ (solid)"),
                Patch(facecolor="#666666", alpha=0.55, edgecolor="#222222", hatch="//", label="$p_{2q}=0.010$ (hatched)"),
                Line2D([0], [0], marker="D", color="none", markeredgecolor="#111111", markerfacecolor="white", markersize=4.5, label="Exact $|E_\\infty^{\\mathrm{noisy}} - E|$"),
            ]
            ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.94, edgecolor="#cccccc", fontsize=5.8)

    for col_i, (cname, mtitle) in enumerate(mols):
        ax = axes[1, col_i]
        reg_key = "sc_card_full"
        rmses = [part_b_map[(cname, reg_key, a)]["rmse_mHa"] for a, _, _ in arms]
        inf_b = [abs(part_b_map[(cname, reg_key, a)]["exact_inf_bias_mHa"]) for a, _, _ in arms]
        ax.bar(
            x,
            rmses,
            width=0.58,
            color=[c for _, _, c in arms],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.scatter(
            x,
            inf_b,
            marker="D",
            s=18,
            color="#111111",
            facecolors="white",
            linewidth=0.9,
            zorder=5,
        )
        for xi, rm, ib in zip(x, rmses, inf_b, strict=True):
            ax.text(xi, max(rm, ib) + 4.5, f"{rm:.1f}", ha="center", va="bottom", fontsize=6.0, color="#222222")
        ax.axhline(1.6, color="#222222", linestyle=":", linewidth=0.9)
        ax.set_title(f"{mtitle}: Full card ($p_{{\\mathrm{{ro}}}}=0.02$)", fontsize=8.0)
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{label}\n({part_b_map[(cname, reg_key, key)]['cz_max']} CZ)"
            for key, label, _ in arms
        ], fontsize=6.0)
        ax.set_ylim(0, 180.0)
        if col_i == 0:
            ax.set_ylabel("RMSE & $|$bias$|$ (mHa)\n[16/16 false stops in all arms]")
        ax.text(
            1.5,
            172.0,
            "Empirical bias² ≥ 99.9% of MSE",
            ha="center",
            va="top",
            fontsize=5.8,
            color="#8b1e1e",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fff3f3", "ec": "#e0b0b0", "lw": 0.6},
        )

    fig.tight_layout(h_pad=1.25, w_pad=0.9)
    _save(fig, "fig4_noisy_bias_variance.pdf", preview_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    _style()
    decl, record, summary = load_verified_payload()
    fig1_shot_savings_and_grid(decl, record, summary, preview_dir=args.preview_dir)
    fig2_r_emp_vs_r_hat(decl, record, summary, preview_dir=args.preview_dir)
    fig3_stretched_rank_and_width(decl, record, summary, preview_dir=args.preview_dir)
    fig4_noisy_bias_variance(decl, record, summary, preview_dir=args.preview_dir)

    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(FIGURE_NAMES)} vector figures and {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
