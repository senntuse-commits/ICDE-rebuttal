#!/usr/bin/env python3
"""Generate the figures for the additional experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPERIMENTS = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "input"
OUT = EXPERIMENTS / "figures"

SERIF = "Times New Roman"
SANS = "Arial"
COLORS = {"JFYan": "#dcecf5", "ParYan": "#fff0df", "ObliYan": "#f4dfe1"}
LINE_COLORS = {"JFYan": "#5aa6c8", "ParYan": "#e9a65f", "ObliYan": "#d98c95"}
HATCHES = {"JFYan": "//", "ParYan": "\\\\", "ObliYan": ".."}
MARKERS = {"JFYan": "o", "ParYan": "s", "ObliYan": "^"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [SERIF],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def finish_axes(ax, grid=True):
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    ax.tick_params(axis="both", which="major", width=0.55, length=2.4, labelsize=5.7, pad=1.5)
    if grid:
        ax.grid(True, which="major", color="#888888", linestyle="--", linewidth=0.38)
    ax.set_axisbelow(True)


def draw_runtime():
    snap = rows(DATA / "snap_runtime.csv")
    job = rows(DATA / "job13d_runtime.csv")
    panels = [
        ("depth", "depth", [2, 3, 4], "Join-Tree Depth", "(a)"),
        ("branching", "branching", [2, 3, 4], "Branching Factor", "(b)"),
        ("relations", "relations", [5, 7, 9], "Number of Relations", "(c)"),
    ]

    modes = ("JFYan", "ParYan", "ObliYan")
    three_group_width = 0.13
    three_group_offset = 0.14
    fig, axes = plt.subplots(1, 4, figsize=(6.55, 1.55), sharey=True)

    def grouped(
        ax,
        xlabels,
        values_by_mode,
        xlabel,
        panel_label,
        bar_width=three_group_width,
        bar_offset=three_group_offset,
    ):
        x = np.arange(len(xlabels), dtype=float)
        for mode_offset, mode in zip((-bar_offset, 0.0, bar_offset), modes):
            ax.bar(
                x + mode_offset,
                values_by_mode[mode],
                width=bar_width,
                color=COLORS[mode],
                edgecolor="black",
                linewidth=0.48,
                hatch=HATCHES[mode],
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_ylim(8e2, 1e7)
        ax.set_yticks([1e3, 1e4, 1e5, 1e6, 1e7])
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in xlabels], fontfamily=SERIF, fontsize=5.8)
        if len(xlabels) == 3:
            ax.set_xlim(-0.55, len(xlabels) - 0.45)
        ax.set_xlabel(xlabel, fontfamily=SANS, fontsize=6.4, labelpad=1.5)
        ax.text(0.5, -0.23, panel_label, transform=ax.transAxes, ha="center", va="top",
                fontfamily=SERIF, fontsize=6.5)
        finish_axes(ax)

    for ax, (series, field, xvals, xlabel, panel_label) in zip(axes[:3], panels):
        selected = {int(r[field]): r for r in snap if r["series"] == series}
        values = {mode: [float(selected[x][f"{mode}_ms"]) for x in xvals] for mode in modes}
        grouped(ax, xvals, values, xlabel, panel_label)

    job = sorted(job, key=lambda r: int(r["sample_percent"]))
    job_labels = [f"{r['sample_percent']}%" for r in job]
    job_values = {mode: [float(r[f"{mode}_ms"]) for r in job] for mode in modes}
    grouped(
        axes[3],
        job_labels,
        job_values,
        "IMDb Data Scale",
        "(d)",
        bar_width=0.19,
        bar_offset=0.21,
    )

    axes[0].set_ylabel("Running Time [ms]", fontfamily=SANS, fontsize=6.5, labelpad=1.5)
    handles = [
        Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.45,
              hatch=HATCHES[m], label=m)
        for m in modes
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               frameon=True, handlelength=1.35, columnspacing=0.75,
               prop={"family": SANS, "size": 5.8})
    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.23, wspace=0.18)
    fig.savefig(OUT / "real_data_runtime.pdf")
    plt.close(fig)


def line_panel(ax, x, series, y_limits, y_ticks, xlabel="Scale Factor", ylabel=False):
    for label, values in series:
        ax.plot(
            x,
            values,
            color=LINE_COLORS.get(label, "#555555"),
            marker=MARKERS.get(label, "o"),
            markersize=2.25,
            linewidth=0.72,
            markerfacecolor=LINE_COLORS.get(label, "#555555"),
            markeredgecolor="black",
            markeredgewidth=0.38,
        )
    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], fontfamily=SERIF, fontsize=5.7)
    ax.set_xlabel(xlabel, fontfamily=SANS, fontsize=6.4, labelpad=1.3)
    if ylabel:
        ax.set_ylabel("Running Time [ms]", fontfamily=SANS, fontsize=6.4, labelpad=1.2)
    finish_axes(ax)


def draw_nonoblivious():
    data = rows(DATA / "tpcds_nonoblivious.csv")
    x = [int(r["scale_factor"]) for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.50))
    for ax, query, ymax, ticks in (
        (axes[0], "Q18", 1e6, [1e2, 1e3, 1e4, 1e5, 1e6]),
        (axes[1], "Q85", 1e5, [1e2, 1e3, 1e4, 1e5]),
    ):
        line_panel(
            ax,
            x,
            [
                ("JFYan", [float(r[f"{query}_JFYan_ms"]) for r in data]),
                ("NonObliJFYan", [float(r[f"{query}_NonObliJFYan_ms"]) for r in data]),
            ],
            (1e2, ymax),
            ticks,
            ylabel=ax is axes[0],
        )
        ax.text(0.5, -0.29, f"({'a' if query == 'Q18' else 'b'}) TPC-DS Query {query[1:]}",
                transform=ax.transAxes, ha="center", va="top", fontfamily=SERIF, fontsize=6.4)
    legend = [
        Line2D([0], [0], color=LINE_COLORS["JFYan"], marker="o", linewidth=0.72,
               markersize=2.4, markeredgecolor="black", markeredgewidth=0.35, label="JFYan"),
        Line2D([0], [0], color="#777777", marker="s", linewidth=0.72,
               markersize=2.4, markeredgecolor="black", markeredgewidth=0.35, label="NonObliJFYan"),
    ]
    # Use one compact legend for the two panels.
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
               handlelength=1.45, columnspacing=0.8, frameon=True,
               prop={"family": SANS, "size": 5.7})
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.24, wspace=0.31)
    fig.savefig(OUT / "tpcds_nonoblivious_runtime.pdf")
    plt.close(fig)


def draw_memory():
    tpc = rows(DATA / "tpcds_memory.csv")
    tpch = sorted(rows(DATA / "tpch9_memory.csv"), key=lambda r: int(r["target_rows"]))
    metrics = (
        ("peak_mb", "Peak memory", "#5aa6c8", "o"),
        ("copy_tables_mb", "Copy tables", "#e9a65f", "s"),
        ("final_output_mb", "Final output", "#d98c95", "^"),
    )

    def linear(ax, x, data_rows, xlabel, panel, ymax, yticks, ylabel=False):
        for field, label, color, marker in metrics:
            ax.plot(x, [float(r[field]) for r in data_rows], color=color, marker=marker,
                    markersize=2.15, linewidth=0.7, markerfacecolor=color,
                    markeredgecolor="black", markeredgewidth=0.35)
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontfamily=SERIF, fontsize=5.5)
        ax.set_xlabel(xlabel, fontfamily=SANS, fontsize=6.1, labelpad=1.2)
        if ylabel:
            ax.set_ylabel("Memory [MiB]", fontfamily=SANS, fontsize=6.2, labelpad=1.2)
        ax.text(0.5, -0.29, panel, transform=ax.transAxes, ha="center", va="top",
                fontfamily=SERIF, fontsize=6.3)
        finish_axes(ax)

    # Two compact panels occupy one IEEE column.
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.94))
    for idx, (query, ymax, yticks) in enumerate(
        (("Q18", 25000, [0, 5000, 10000, 15000, 20000, 25000]),
         ("Q85", 4000, [0, 1000, 2000, 3000, 4000]))
    ):
        selected = sorted((r for r in tpc if r["query"] == query), key=lambda r: int(r["sf"]))
        x = [int(r["sf"]) for r in selected]
        linear(axes[idx], x, selected, "Scale Factor", f"({chr(97 + idx)}) TPC-DS Query {query[1:]}",
               ymax, yticks, ylabel=True)

    handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=0.7, markersize=2.3,
               markeredgecolor="black", markeredgewidth=0.35, label=label)
        for _field, label, color, marker in metrics
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               handlelength=1.15, columnspacing=0.52, frameon=True,
               prop={"family": SANS, "size": 5.35})
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.28, wspace=0.31)
    fig.savefig(OUT / "tpcds_memory.pdf")
    plt.close(fig)

    # TPC-H Q9 is deliberately one panel with the same physical panel size as
    # either TPC-DS panel above; it will sit beside the EPC table in LaTeX.
    fig, ax = plt.subplots(1, 1, figsize=(1.78, 1.94))
    targets = [int(r["target_rows"]) / 1_000_000 for r in tpch]
    target_positions = list(range(len(targets)))
    linear(ax, target_positions, tpch, r"Public Target $\Lambda$ [M rows]", "TPC-H Query 9",
           12000, [0, 3000, 6000, 9000, 12000], ylabel=True)
    ax.set_xticklabels([f"{v:.2f}" for v in targets], fontfamily=SERIF, fontsize=4.9)
    short_handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=0.7, markersize=2.2,
               markeredgecolor="black", markeredgewidth=0.35, label=short)
        for (_field, _label, color, marker), short in zip(metrics, ("Peak", "Copy", "Output"))
    ]
    fig.legend(handles=short_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               handlelength=0.9, columnspacing=0.3, frameon=True,
               prop={"family": SANS, "size": 4.9})
    fig.subplots_adjust(left=0.23, right=0.985, top=0.84, bottom=0.28)
    fig.savefig(OUT / "tpch9_memory.pdf")
    plt.close(fig)

    # Full-width paper figure: three equally sized, slightly wider panels.
    # Use a compact three-panel layout.
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.98))
    for idx, (query, ymax, yticks) in enumerate(
        (("Q18", 25000, [0, 5000, 10000, 15000, 20000, 25000]),
         ("Q85", 4000, [0, 1000, 2000, 3000, 4000]))
    ):
        selected = sorted((r for r in tpc if r["query"] == query), key=lambda r: int(r["sf"]))
        x = [int(r["sf"]) for r in selected]
        linear(axes[idx], x, selected, "Scale Factor",
               f"({chr(97 + idx)}) TPC-DS Query {query[1:]}", ymax, yticks, ylabel=True)

    targets = [int(r["target_rows"]) / 1_000_000 for r in tpch]
    target_positions = list(range(len(targets)))
    linear(axes[2], target_positions, tpch, r"Public Target $\Lambda$ [M rows]",
           "(c) TPC-H Query 9", 12000, [0, 3000, 6000, 9000, 12000], ylabel=True)
    axes[2].set_xticklabels([f"{v:.2f}" for v in targets], fontfamily=SERIF, fontsize=5.5)

    handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=0.7, markersize=2.3,
               markeredgecolor="black", markeredgewidth=0.35, label=label)
        for _field, label, color, marker in metrics
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3,
               handlelength=1.15, columnspacing=0.65, frameon=True,
               prop={"family": SANS, "size": 5.6})
    fig.subplots_adjust(left=0.072, right=0.995, top=0.84, bottom=0.29, wspace=0.30)
    fig.savefig(OUT / "memory_combined.pdf")
    fig.savefig(OUT / "memory_combined.png", dpi=300)
    plt.close(fig)


TOPOLOGIES = {
    "depth_d2_m9_medium": [-1, 0, 0, 0, 1, 1, 2, 2, 3],
    "depth_d3_m9_medium": [-1, 0, 0, 0, 1, 1, 2, 2, 4],
    "depth_d4_m9_medium": [-1, 0, 0, 0, 1, 1, 4, 4, 6],
    "branch_b2_m9_medium": [-1, 0, 0, 1, 1, 2, 2, 3, 3],
    "branch_b3_m9_medium": [-1, 0, 0, 0, 1, 1, 1, 4, 2],
    "branch_b4_m9_medium": [-1, 0, 0, 0, 0, 1, 1, 1, 5],
    "relations_m5_d2_b3": [-1, 0, 0, 0, 1],
    "relations_m7_d2_b3": [-1, 0, 0, 0, 1, 2, 3],
    "relations_m9_d2_b3": [-1, 0, 0, 0, 1, 1, 2, 2, 3],
}


def positions(parents):
    children = {i: [] for i in range(len(parents))}
    root = 0
    for child, parent in enumerate(parents):
        if parent < 0:
            root = child
        else:
            children[parent].append(child)
    pos, leaf = {}, 0

    def walk(node, depth):
        nonlocal leaf
        if not children[node]:
            x = leaf
            leaf += 1
        else:
            xs = [walk(child, depth + 1) for child in children[node]]
            x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    walk(root, 0)
    scale = max(1, leaf - 1)
    return root, children, {n: (0.08 + 0.84 * x / scale, y) for n, (x, y) in pos.items()}


def tree_axis(ax, parents, title, sizes, labels=None):
    root, children, pos = positions(parents)
    max_depth = max(-y for _, y in pos.values())
    for parent, kids in children.items():
        for child in kids:
            x1, y1 = pos[parent]
            x2, y2 = pos[child]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.48, zorder=1)
    for node, (x, y) in pos.items():
        text = labels[node] if labels else rf"$R_{{{node}}}$"
        ax.text(x, y, text, ha="center", va="center", fontsize=4.8,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#dddddd",
                          edgecolor="black", linewidth=0.42), zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-max_depth - 0.55, 0.43)
    ax.axis("off")
    ax.text(0.5, -0.04, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.4, fontfamily=SERIF)
    if sizes:
        ax.text(0.5, -0.17, sizes, transform=ax.transAxes, ha="center", va="top",
                fontsize=4.4, linespacing=0.95, fontfamily=SERIF)


def draw_trees():
    snap = {r["case"]: r for r in rows(DATA / "snap_runtime.csv")}
    series = (
        (
            "snap_depth_join_trees.pdf",
            (("depth_d2_m9_medium", "(a) Depth = 2"),
             ("depth_d3_m9_medium", "(b) Depth = 3"),
             ("depth_d4_m9_medium", "(c) Depth = 4")),
        ),
        (
            "snap_branching_join_trees.pdf",
            (("branch_b2_m9_medium", "(a) Branching factor = 2"),
             ("branch_b3_m9_medium", "(b) Branching factor = 3"),
             ("branch_b4_m9_medium", "(c) Branching factor = 4")),
        ),
        (
            "snap_relations_join_trees.pdf",
            (("relations_m5_d2_b3", "(a) 5 relations"),
             ("relations_m7_d2_b3", "(b) 7 relations"),
             ("relations_m9_d2_b3", "(c) 9 relations")),
        ),
    )
    for filename, cases in series:
        fig, axes = plt.subplots(1, 3, figsize=(6.55, 1.30))
        for ax, (case, title) in zip(axes, cases):
            meta = snap[case]
            size = f"Input={int(meta['input_rows']):,}\nOutput={int(meta['output_rows']):,}"
            tree_axis(ax, TOPOLOGIES[case], title, size)
        fig.subplots_adjust(left=0.015, right=0.99, top=0.98, bottom=0.27, wspace=0.40)
        fig.savefig(OUT / filename)
        plt.close(fig)

    job_labels = {
        0: "title", 1: "kind_type", 2: "movie_companies", 3: "company_name",
        4: "company_type", 5: "movie_info", 6: "info_type\n(release dates)",
        7: "movie_info_idx", 8: "info_type\n(rating)",
    }
    job_parents = [-1, 0, 0, 2, 2, 0, 5, 0, 7]
    fig, ax = plt.subplots(1, 1, figsize=(4.65, 1.16))
    tree_axis(ax, job_parents, "JOB 13d acyclic join tree", "", job_labels)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.98, bottom=0.22)
    fig.savefig(OUT / "job13d_join_tree.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_trees()
    draw_runtime()
    draw_nonoblivious()
    draw_memory()


if __name__ == "__main__":
    main()
