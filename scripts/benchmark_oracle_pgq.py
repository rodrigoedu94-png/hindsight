#!/usr/bin/env python3
"""Reproducible Oracle AI Database 26ai SQL/PGQ vs Relational CTE benchmark (LIN-272 / LIN-273).

Reads HINDSIGHT_API_DATABASE_URL from the environment, measures p50/p95 latency
and optional V$SQL buffer_gets across Relational CTE vs SQL/PGQ workloads, and
writes structured JSON for scripts/generate_pgq_charts.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIM_DIR = REPO_ROOT / "hindsight-api-slim"
if str(SLIM_DIR) not in sys.path:
    sys.path.insert(0, str(SLIM_DIR))


@dataclass
class WorkloadMetrics:
    name: str
    mode: str
    iterations: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    row_count: int
    buffer_gets_per_exec: float | None = None
    # Server-side time per execution and the plan it ran, from the same V$SQL cursors.
    # Pass a plan_hash_value's sql_id to DBMS_XPLAN.DISPLAY_CURSOR for the full plan.
    elapsed_ms_per_exec: float | None = None
    plan_hash_values: list[int] | None = None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize_timings(
    name: str,
    mode: str,
    latencies_ms: list[float],
    row_count: int,
    *,
    buffer_gets_per_exec: float | None = None,
    elapsed_ms_per_exec: float | None = None,
    plan_hash_values: list[int] | None = None,
) -> WorkloadMetrics:
    return WorkloadMetrics(
        name=name,
        mode=mode,
        iterations=len(latencies_ms),
        p50_ms=round(_percentile(latencies_ms, 0.50), 3),
        p95_ms=round(_percentile(latencies_ms, 0.95), 3),
        mean_ms=round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
        min_ms=round(min(latencies_ms), 3) if latencies_ms else 0.0,
        max_ms=round(max(latencies_ms), 3) if latencies_ms else 0.0,
        row_count=row_count,
        buffer_gets_per_exec=round(buffer_gets_per_exec, 2) if buffer_gets_per_exec is not None else None,
        elapsed_ms_per_exec=round(elapsed_ms_per_exec, 3) if elapsed_ms_per_exec is not None else None,
        plan_hash_values=plan_hash_values,
    )


class _MarkedConnection:
    """Prefix a marker comment to the SQL a builder sends, so V$SQL can find its cursor."""

    def __init__(self, conn: Any, marker: str) -> None:
        self._conn = conn
        self._marker = marker

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(f"/* {self._marker} */ {query}", *args)


async def _try_fetch_sql_stats(conn: Any, marker: str) -> dict[str, Any]:
    """buffer_gets and elapsed_time per execution, and the plan hash values, for one marker.

    Needs SELECT on V$SQL; without it every field stays None and only client latency is kept.
    """
    try:
        rows = await conn.fetch(
            """
            SELECT plan_hash_value, SUM(buffer_gets) AS gets, SUM(elapsed_time) AS elapsed_us,
                   SUM(executions) AS execs
            FROM v$sql
            WHERE sql_text LIKE $1
              AND sql_text NOT LIKE '%v$sql%'
            GROUP BY plan_hash_value
            """,
            f"%{marker}%",
        )
    except Exception:
        return {}
    execs = sum(int(r["execs"] or 0) for r in rows)
    if not execs:
        return {}
    return {
        "buffer_gets_per_exec": sum(float(r["gets"] or 0) for r in rows) / execs,
        "elapsed_ms_per_exec": sum(float(r["elapsed_us"] or 0) for r in rows) / execs / 1000.0,
        "plan_hash_values": sorted(int(r["plan_hash_value"]) for r in rows),
    }


async def _benchmark_coro(
    label: str,
    mode: str,
    fn,
    *,
    warmup: int,
    iterations: int,
    conn: Any = None,
    sql_marker: str | None = None,
) -> WorkloadMetrics:
    for _ in range(warmup):
        await fn()

    timings_ms: list[float] = []
    last_count = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = await fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        if hasattr(result, "entity"):
            last_count = len(result.entity) + len(result.semantic) + len(result.causal)
        else:
            last_count = len(result)

    stats = await _try_fetch_sql_stats(conn, sql_marker) if (conn and sql_marker) else {}
    return summarize_timings(label, mode, timings_ms, last_count, **stats)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    db_url = args.database_url or os.getenv("HINDSIGHT_API_DATABASE_URL", "")
    if not db_url:
        raise SystemExit("Missing database URL. Set HINDSIGHT_API_DATABASE_URL or pass --database-url.")
    # fq_store_table and the SQL/PGQ capability check read the backend from the config.
    os.environ["HINDSIGHT_API_DATABASE_BACKEND"] = "oracle"

    from hindsight_api.engine.db.ops import UpdatedWindow
    from hindsight_api.engine.db.ops_oracle import OracleOps
    from hindsight_api.engine.db.oracle import OracleBackend, OracleLiteralStr
    from hindsight_api.engine.memories.ora import pgq as ora_pgq
    from hindsight_api.engine.schema import fq_store_table as fq_table

    backend = OracleBackend()
    await backend.initialize(db_url, min_size=1, max_size=2)
    ops = OracleOps()
    # V$SQL keeps cursors across runs: a per-run tag keeps one run's reads out of the next.
    run = uuid.uuid4().hex[:8]
    window = UpdatedWindow(after=None, before=None, first_param_index=4)
    obs_window = UpdatedWindow(after=None, before=None, first_param_index=3)

    try:
        async with backend.acquire() as conn:
            has_pgq = await ops.has_property_graph(conn)
            mu = fq_table("memory_units")
            ue = fq_table("unit_entities")
            ml = fq_table("memory_links")
            graph_name = fq_table("hindsight_memory_graph")

            seed_rows = await conn.fetch(
                f"SELECT id FROM {mu} WHERE bank_id = $1 AND fact_type = $2 ORDER BY updated_at DESC FETCH FIRST $3 ROWS ONLY",
                OracleLiteralStr(args.bank_id),  # VARCHAR2: a UUID-shaped bank id must not bind as RAW(16)
                args.fact_type,
                args.seed_count,
            )
            seed_ids = [r["id"] for r in seed_rows]
            if not seed_ids:
                raise SystemExit(
                    f"No seed memory_units found for bank_id={args.bank_id!r}, fact_type={args.fact_type!r}."
                )

            obs_seed_rows = await conn.fetch(
                f"SELECT id FROM {mu} WHERE bank_id = $1 AND fact_type = 'observation' ORDER BY updated_at DESC FETCH FIRST $2 ROWS ONLY",
                OracleLiteralStr(args.bank_id),
                args.seed_count,
            )
            obs_seed_ids = [r["id"] for r in obs_seed_rows]

            rel_sql = f"""
                /* BENCH_LIN273_COMBINED_REL {run} */
                WITH {ops.build_entity_expansion_cte(mu, ue, args.per_entity_limit, window)},
                {ops.build_semantic_causal_cte(ml, mu, window)}
                SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded
            """
            # What v1.3.0 runs today when the graph exists: relational entity arm, SQL/PGQ
            # semantic/causal arms (OracleOps.build_pgq_semantic_causal_cte).
            v130_sql = f"""
                /* BENCH_LIN273_COMBINED_V130 {run} */
                WITH {ops.build_entity_expansion_cte(mu, ue, args.per_entity_limit, window)},
                {ops.build_pgq_semantic_causal_cte(graph_name, mu, window, max_causal_hops=2)}
                SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded
            """
            pgq_1_sql = f"""
                /* BENCH_LIN273_COMBINED_PGQ_1HOP {run} */
                WITH {ora_pgq.build_pgq_entity_expansion_cte(graph_name, mu, args.per_entity_limit, window)},
                {ora_pgq.build_pgq_semantic_causal_cte(graph_name, mu, window, max_causal_hops=1)}
                SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded
            """
            pgq_2_sql = f"""
                /* BENCH_LIN273_COMBINED_PGQ_2HOP {run} */
                WITH {ora_pgq.build_pgq_entity_expansion_cte(graph_name, mu, args.per_entity_limit, window)},
                {ora_pgq.build_pgq_semantic_causal_cte(graph_name, mu, window, max_causal_hops=2)}
                SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded
            """

            workloads: list[WorkloadMetrics] = [
                await _benchmark_coro(
                    "Combined LinkExpansion (Relational CTE)",
                    "relational",
                    lambda: conn.fetch(rel_sql, seed_ids, args.fact_type, args.budget),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    conn=conn,
                    sql_marker=f"BENCH_LIN273_COMBINED_REL {run}",
                )
            ]
            if has_pgq:
                workloads.append(
                    await _benchmark_coro(
                        "Combined LinkExpansion (v1.3.0 hybrid)",
                        "hybrid",
                        lambda: conn.fetch(v130_sql, seed_ids, args.fact_type, args.budget),
                        warmup=args.warmup,
                        iterations=args.iterations,
                        conn=conn,
                        sql_marker=f"BENCH_LIN273_COMBINED_V130 {run}",
                    )
                )
                workloads.append(
                    await _benchmark_coro(
                        "Combined LinkExpansion (SQL/PGQ 1-hop)",
                        "pgq",
                        lambda: conn.fetch(pgq_1_sql, seed_ids, args.fact_type, args.budget),
                        warmup=args.warmup,
                        iterations=args.iterations,
                        conn=conn,
                        sql_marker=f"BENCH_LIN273_COMBINED_PGQ_1HOP {run}",
                    )
                )
                workloads.append(
                    await _benchmark_coro(
                        "Combined LinkExpansion (SQL/PGQ 2-hop)",
                        "pgq",
                        lambda: conn.fetch(pgq_2_sql, seed_ids, args.fact_type, args.budget),
                        warmup=args.warmup,
                        iterations=args.iterations,
                        conn=conn,
                        sql_marker=f"BENCH_LIN273_COMBINED_PGQ_2HOP {run}",
                    )
                )
            if obs_seed_ids:
                workloads.append(
                    await _benchmark_coro(
                        "Observation Expansion (Relational CTE)",
                        "relational",
                        lambda: ops.expand_observations(
                            _MarkedConnection(conn, f"BENCH_LIN273_OBS_REL {run}"),
                            mu,
                            ue,
                            ml,
                            obs_seed_ids,
                            args.budget,
                            args.per_entity_limit,
                            obs_window,
                        ),
                        warmup=args.warmup,
                        iterations=args.iterations,
                        conn=conn,
                        sql_marker=f"BENCH_LIN273_OBS_REL {run}",
                    )
                )
                if has_pgq:
                    workloads.append(
                        await _benchmark_coro(
                            "Observation Expansion (SQL/PGQ derived_from)",
                            "pgq",
                            lambda: ora_pgq.expand_observations_pgq(
                                _MarkedConnection(conn, f"BENCH_LIN273_OBS_PGQ {run}"),
                                graph_name,
                                mu,
                                obs_seed_ids,
                                args.budget,
                                args.per_entity_limit,
                                obs_window,
                                max_causal_hops=2,
                            ),
                            warmup=args.warmup,
                            iterations=args.iterations,
                            conn=conn,
                            sql_marker=f"BENCH_LIN273_OBS_PGQ {run}",
                        )
                    )

            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bank_id": args.bank_id,
                "fact_type": args.fact_type,
                "seed_count": len(seed_ids),
                "observation_seed_count": len(obs_seed_ids),
                "budget": args.budget,
                "per_entity_limit": args.per_entity_limit,
                "has_property_graph": has_pgq,
                "workloads": [asdict(w) for w in workloads],
            }
    finally:
        await backend.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Oracle 26ai SQL/PGQ vs Relational CTEs.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--fact-type", default="world")
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--per-entity-limit", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", default="pgq_benchmark_results.json")
    args = parser.parse_args(argv)
    for flag, val in (
        ("--seed-count", args.seed_count),
        ("--budget", args.budget),
        ("--per-entity-limit", args.per_entity_limit),
        ("--iterations", args.iterations),
    ):
        if val <= 0:
            parser.error(f"{flag} must be >= 1 (got {val})")
    if args.warmup < 0:
        parser.error(f"--warmup must be >= 0 (got {args.warmup})")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run_benchmark(args))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {out_path} ({len(report['workloads'])} workloads).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
