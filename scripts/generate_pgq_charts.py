#!/usr/bin/env python3
"""Generate Oracle 26ai SQL/PGQ topology diagrams and benchmark comparison charts (LIN-273).

Usage:
    # 1. Generate the hindsight_memory_graph architecture & query-path diagram only:
    python3 scripts/generate_pgq_charts.py --topology-only --output-dir docs/oracle/charts

    # 2. Generate both topology diagram and benchmark comparison charts from JSON:
    python3 scripts/generate_pgq_charts.py --input pgq_benchmark_results.json --output-dir docs/oracle/charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def render_topology_diagram(output_path: Path) -> Path:
    """Render a clean architectural diagram of hindsight_memory_graph (Oracle 26ai SQL/PGQ)."""
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        0.5,
        8.05,
        "Oracle AI Database 26ai — hindsight_memory_graph (SQL/PGQ Zero-Copy Topology)",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        0.5,
        7.68,
        "OPTIONS (TRUSTED MODE) over relational tables • Isolated in engine/memories/ora/ • Automatic relational fallback • Zero PG SQL changes",
        fontsize=9.8,
        color="#475569",
    )

    def draw_box(x, y, w, h, title, body, fc, ec, title_color="#0F172A"):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.14,rounding_size=0.18",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.18, y + h - 0.32, title, fontsize=10.5, fontweight="bold", color=title_color)
        ax.text(x + 0.18, y + h - 0.62, body, fontsize=8.6, color="#334155", va="top", linespacing=1.35)

    draw_box(
        0.6,
        5.0,
        5.1,
        2.15,
        "VERTEX: MEMORY_UNITS  (label: memory_unit)",
        "KEY (id)\n"
        "Props: id, bank_id, fact_type, event_date,\n"
        "  occurred_start, occurred_end, mentioned_at,\n"
        "  updated_at, proof_count",
        fc="#EFF6FF",
        ec="#2563EB",
        title_color="#1D4ED8",
    )
    draw_box(
        8.3,
        5.0,
        5.1,
        2.15,
        "VERTEX: ENTITIES  (label: entity)",
        "KEY (id)\n"
        "Props: id, bank_id, canonical_name,\n"
        "  mention_count\n"
        "Used for multi-hop entity neighborhood & recall",
        fc="#ECFDF5",
        ec="#059669",
        title_color="#047857",
    )

    draw_box(
        0.6,
        2.65,
        2.95,
        1.75,
        "EDGE: memory_link",
        "MEMORY_LINKS\nmemory_unit → memory_unit\nsemantic | causes | caused_by\nenables | prevents + weight",
        fc="#FFF7ED",
        ec="#EA580C",
        title_color="#C2410C",
    )
    draw_box(
        3.9,
        2.65,
        2.95,
        1.75,
        "EDGE: derived_from",
        "OBSERVATION_SOURCES\nobservation → source_unit\nZero-copy lineage for\nexpand_observations_pgq",
        fc="#FAF5FF",
        ec="#9333EA",
        title_color="#7E22CE",
    )
    draw_box(
        7.2,
        2.65,
        2.95,
        1.75,
        "EDGE: mentions",
        "UNIT_ENTITIES\nmemory_unit → entity\nDrives entity_expansion\nwith ROW_NUMBER cap",
        fc="#F0FDF4",
        ec="#16A34A",
        title_color="#15803D",
    )
    draw_box(
        10.5,
        2.65,
        2.9,
        1.75,
        "EDGE: cooccurs_with",
        "ENTITY_COOCCURRENCES\nentity ↔ entity (1–2 hops)\nExplicit 1-hop / 2-hop arms\n(avoids ORA-40998)",
        fc="#FEF2F2",
        ec="#DC2626",
        title_color="#B91C1C",
    )

    arrow_kw = dict(arrowstyle="-|>", color="#64748B", lw=1.4, mutation_scale=12)
    ax.annotate("", xy=(2.0, 4.45), xytext=(2.4, 4.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.35, 4.45), xytext=(4.2, 4.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.15, 4.45), xytext=(5.5, 4.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.0, 4.45), xytext=(9.6, 4.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.9, 4.45), xytext=(11.4, 4.98), arrowprops=arrow_kw)

    draw_box(
        0.6,
        0.4,
        12.8,
        1.72,
        "LIN-273 SQL/PGQ Query Paths (OracleLinkExpansionRetriever + engine/memories/ora/pgq.py)",
        "1. build_pgq_entity_expansion_cte      →  GRAPH_TABLE(mentions) + ROW_NUMBER per_entity_limit + window filter\n"
        "2. build_pgq_semantic_causal_cte       →  GRAPH_TABLE(memory_link) 1-hop semantic + 1–2 hop causal decay (0.7×), seed exclusion\n"
        "3. expand_observations_pgq             →  GRAPH_TABLE(derived_from + mentions + memory_link) fused single-fetch with fallback\n"
        "4. query_entity_neighborhood_pgq       →  Explicit 0/1/2 hop_distance + bi-temporal occurred/mentioned filter | Indexes: IDX_ML_FROM/TO_TYPE_WEIGHT",
        fc="#FFFFFF",
        ec="#334155",
        title_color="#0F172A",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return output_path


def render_benchmark_charts(data: dict[str, Any], output_path: Path) -> Path:
    """Render side-by-side latency (p50/p95) and buffer_gets comparison charts from benchmark JSON."""
    workloads = data.get("workloads", [])
    if not workloads:
        raise ValueError("Benchmark JSON contains no workloads.")

    labels = [w["name"].replace(" (", "\n(") for w in workloads]
    p50 = [float(w["p50_ms"]) for w in workloads]
    p95 = [float(w["p95_ms"]) for w in workloads]
    modes = [w.get("mode", "relational") for w in workloads]
    gets = [w.get("buffer_gets_per_exec") for w in workloads]
    has_gets = any(g is not None for g in gets)

    ncols = 2 if has_gets else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 5.4), dpi=180)
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F8FAFC")

    x = list(range(len(labels)))
    width = 0.36
    colors_p50 = ["#2563EB" if m == "pgq" else "#64748B" for m in modes]
    colors_p95 = ["#60A5FA" if m == "pgq" else "#94A3B8" for m in modes]

    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    bars1 = ax0.bar([i - width / 2 for i in x], p50, width, label="p50 (ms)", color=colors_p50)
    bars2 = ax0.bar([i + width / 2 for i in x], p95, width, label="p95 (ms)", color=colors_p95)
    ax0.set_title(
        f"Latency: Relational CTE vs Oracle 26ai SQL/PGQ\n(bank={data.get('bank_id', 'n/a')}, seeds={data.get('seed_count', 'n/a')})",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax0.set_ylabel("Milliseconds (lower is better)", color="#334155")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    ax0.legend(frameon=False)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax0.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1E293B",
        )

    if has_gets:
        ax1 = axes[1]
        ax1.set_facecolor("#FFFFFF")
        vals = [float(g) if g is not None else 0.0 for g in gets]
        bar_colors = ["#059669" if m == "pgq" else "#94A3B8" for m in modes]
        bars_g = ax1.bar(x, vals, width=0.52, color=bar_colors)
        ax1.set_title("Logical Reads (V$SQL buffer_gets / exec)", fontsize=11.5, fontweight="bold", color="#0F172A")
        ax1.set_ylabel("Buffer Gets (lower is better)", color="#334155")
        ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=12, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, raw in zip(bars_g, gets):
            if raw is not None:
                h = bar.get_height()
                ax1.annotate(
                    f"{h:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1E293B",
                )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SQL/PGQ topology and benchmark charts.")
    parser.add_argument("--input", default=None, help="Path to benchmark JSON from benchmark_oracle_pgq.py")
    parser.add_argument("--output-dir", default="docs/oracle/charts", help="Directory for generated PNGs")
    parser.add_argument("--topology-only", action="store_true", help="Only generate the graph topology diagram")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    topo_path = render_topology_diagram(out_dir / "hindsight_memory_graph_topology.png")
    print(f"Generated topology diagram: {topo_path}")

    if not args.topology_only:
        if not args.input:
            raise SystemExit("Pass --input <benchmark.json> or use --topology-only.")
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        bench_path = render_benchmark_charts(data, out_dir / "oracle_pgq_benchmark_comparison.png")
        print(f"Generated benchmark comparison chart: {bench_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())