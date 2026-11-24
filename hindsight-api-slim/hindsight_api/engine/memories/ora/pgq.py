"""Oracle AI Database 26ai SQL/PGQ (GRAPH_TABLE) query builders (LIN-273).

Isolated under ``engine/memories/ora/`` so Oracle SQL/PGQ logic stays decoupled
from PostgreSQL SQL and avoids bloating ``ops_oracle.py``. All queries use
explicit 1-hop / 2-hop patterns (never referencing properties on quantified
variable-length edges) to avoid ``ORA-40998``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from ...db.ops import LinkExpansionRows, UpdatedWindow, memory_unit_columns


def build_pgq_entity_expansion_cte(
    graph_name: str,
    mu_table: str,
    per_entity_limit: int,
    window: UpdatedWindow,
) -> str:
    """Build entity expansion CTE using Oracle 26ai SQL/PGQ GRAPH_TABLE over mentions edges."""
    safe_limit = max(1, int(per_entity_limit))
    return f"""
            entity_units_raw AS (
                SELECT DISTINCT entity_id, unit_id
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (src IS memory_unit) -[m1 IS mentions]-> (e IS entity)
                          <-[m2 IS mentions]- (dst IS memory_unit)
                    WHERE src.id = ANY($1::uuid[])
                      AND dst.id != ALL($1::uuid[])
                      AND dst.fact_type = $2
                      {window.clause("dst")}
                    COLUMNS (
                        e.id AS entity_id,
                        dst.id AS unit_id
                    )
                )
            ),
            entity_units_ranked AS (
                SELECT entity_id,
                       unit_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY entity_id
                           ORDER BY unit_id DESC
                       ) AS rn
                FROM entity_units_raw
            ),
            entity_scores AS (
                SELECT eur.unit_id, COUNT(DISTINCT eur.entity_id) AS score
                FROM entity_units_ranked eur
                WHERE eur.rn <= {safe_limit}
                GROUP BY eur.unit_id
            ),
            entity_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       es.score, 'entity' AS source
                FROM entity_scores es
                JOIN {mu_table} mu ON mu.id = es.unit_id
                ORDER BY es.score DESC
                FETCH FIRST $3 ROWS ONLY
            )"""


def build_pgq_semantic_causal_cte(
    graph_name: str,
    mu_table: str,
    window: UpdatedWindow,
    *,
    max_causal_hops: int = 2,
) -> str:
    """Build semantic and multi-hop causal CTEs using Oracle 26ai SQL/PGQ GRAPH_TABLE."""
    causal_two_hop = ""
    if max_causal_hops >= 2:
        causal_two_hop = f"""
                    UNION ALL
                    SELECT id, weight
                    FROM GRAPH_TABLE (
                        {graph_name}
                        MATCH (src IS memory_unit) -[l1 IS memory_link]-> (mid IS memory_unit)
                              -[l2 IS memory_link]-> (dst IS memory_unit)
                        WHERE src.id = ANY($1::uuid[])
                          AND l1.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND l2.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND dst.fact_type = $2
                          AND dst.id != ALL($1::uuid[])
                          {window.clause("dst")}
                        COLUMNS (
                            dst.id AS id,
                            (l1.weight * l2.weight * 0.7) AS weight
                        )
                    )"""

    return f"""
            sem_scores AS (
                SELECT id, MAX(weight) AS score
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (src IS memory_unit) -[l IS memory_link]- (dst IS memory_unit)
                    WHERE src.id = ANY($1::uuid[])
                      AND l.link_type = 'semantic'
                      AND dst.fact_type = $2
                      AND dst.id != ALL($1::uuid[])
                      {window.clause("dst")}
                    COLUMNS (
                        dst.id AS id,
                        l.weight AS weight
                    )
                )
                GROUP BY id
            ),
            semantic_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       ss.score, 'semantic' AS source
                FROM sem_scores ss
                JOIN {mu_table} mu ON mu.id = ss.id
                ORDER BY ss.score DESC
                FETCH FIRST $3 ROWS ONLY
            ),
            causal_scores AS (
                SELECT id, MAX(weight) AS score
                FROM (
                    SELECT id, weight
                    FROM GRAPH_TABLE (
                        {graph_name}
                        MATCH (src IS memory_unit) -[l1 IS memory_link]-> (dst IS memory_unit)
                        WHERE src.id = ANY($1::uuid[])
                          AND l1.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND dst.fact_type = $2
                          AND dst.id != ALL($1::uuid[])
                          {window.clause("dst")}
                        COLUMNS (
                            dst.id AS id,
                            l1.weight AS weight
                        )
                    ){causal_two_hop}
                ) causal_raw
                GROUP BY id
            ),
            causal_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       cs.score, 'causal' AS source
                FROM causal_scores cs
                JOIN {mu_table} mu ON mu.id = cs.id
                ORDER BY cs.score DESC
                FETCH FIRST $3 ROWS ONLY
            )"""


async def expand_observations_pgq(
    conn: Any,
    graph_name: str,
    mu_table: str,
    seed_ids: list,
    budget: int,
    per_entity_limit: int,
    window: UpdatedWindow,
    *,
    max_causal_hops: int = 2,
) -> LinkExpansionRows:
    """Expand observation memories via Oracle 26ai SQL/PGQ (derived_from, mentions, memory_link)."""
    if not seed_ids or budget <= 0:
        return LinkExpansionRows(entity=[], semantic=[], causal=[])
    safe_limit = max(1, int(per_entity_limit))
    causal_two_hop = ""
    if max_causal_hops >= 2:
        causal_two_hop = f"""
                    UNION ALL
                    SELECT id, weight
                    FROM GRAPH_TABLE (
                        {graph_name}
                        MATCH (src IS memory_unit) -[l1 IS memory_link]-> (mid IS memory_unit)
                              -[l2 IS memory_link]-> (dst IS memory_unit)
                        WHERE src.id = ANY($1::uuid[])
                          AND l1.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND l2.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND dst.fact_type = 'observation'
                          AND dst.id != ALL($1::uuid[])
                          {window.clause("dst")}
                        COLUMNS (
                            dst.id AS id,
                            (l1.weight * l2.weight * 0.7) AS weight
                        )
                    )"""

    all_rows = await conn.fetch(
        f"""
            WITH seed_sources AS (
                SELECT DISTINCT source_id
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (obs IS memory_unit) -[d IS derived_from]-> (src IS memory_unit)
                    WHERE obs.id = ANY($1::uuid[])
                    COLUMNS (
                        src.id AS source_id
                    )
                )
            ),
            source_entity_units_raw AS (
                SELECT DISTINCT entity_id, unit_id
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (obs IS memory_unit) -[d IS derived_from]-> (src IS memory_unit)
                          -[m1 IS mentions]-> (e IS entity)
                          <-[m2 IS mentions]- (tgt IS memory_unit)
                    WHERE obs.id = ANY($1::uuid[])
                      AND tgt.id != src.id
                    COLUMNS (
                        e.id AS entity_id,
                        tgt.id AS unit_id
                    )
                )
            ),
            entity_units_ranked AS (
                SELECT unit_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY entity_id
                           ORDER BY unit_id DESC
                       ) AS rn
                FROM source_entity_units_raw
            ),
            connected_sources AS (
                SELECT DISTINCT eur.unit_id AS source_id
                FROM entity_units_ranked eur
                WHERE eur.rn <= {safe_limit}
                  AND NOT EXISTS (
                      SELECT 1 FROM seed_sources ss WHERE ss.source_id = eur.unit_id
                  )
            ),
            obs_derived_edges AS (
                SELECT observation_id, source_id
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (obs IS memory_unit) -[d IS derived_from]-> (src IS memory_unit)
                    WHERE obs.fact_type = 'observation'
                      AND obs.id != ALL($1::uuid[])
                      {window.clause("obs")}
                    COLUMNS (
                        obs.id AS observation_id,
                        src.id AS source_id
                    )
                )
            ),
            obs_scores AS (
                SELECT ode.observation_id, COUNT(DISTINCT cs.source_id) AS score
                FROM connected_sources cs
                JOIN obs_derived_edges ode ON ode.source_id = cs.source_id
                GROUP BY ode.observation_id
            ),
            observation_entity_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       s.score,
                       'entity' AS source
                FROM obs_scores s
                JOIN {mu_table} mu ON mu.id = s.observation_id
                WHERE mu.fact_type = 'observation'
                  {window.clause("mu")}
                ORDER BY s.score DESC
                FETCH FIRST $2 ROWS ONLY
            ),
            sem_scores AS (
                SELECT id, MAX(weight) AS score
                FROM GRAPH_TABLE (
                    {graph_name}
                    MATCH (src IS memory_unit) -[l IS memory_link]- (dst IS memory_unit)
                    WHERE src.id = ANY($1::uuid[])
                      AND l.link_type = 'semantic'
                      AND dst.fact_type = 'observation'
                      AND dst.id != ALL($1::uuid[])
                      {window.clause("dst")}
                    COLUMNS (
                        dst.id AS id,
                        l.weight AS weight
                    )
                )
                GROUP BY id
            ),
            semantic_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       ss.score, 'semantic' AS source
                FROM sem_scores ss
                JOIN {mu_table} mu ON mu.id = ss.id
                ORDER BY ss.score DESC
                FETCH FIRST $2 ROWS ONLY
            ),
            causal_scores AS (
                SELECT id, MAX(weight) AS score
                FROM (
                    SELECT id, weight
                    FROM GRAPH_TABLE (
                        {graph_name}
                        MATCH (src IS memory_unit) -[l1 IS memory_link]-> (dst IS memory_unit)
                        WHERE src.id = ANY($1::uuid[])
                          AND l1.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                          AND dst.fact_type = 'observation'
                          AND dst.id != ALL($1::uuid[])
                          {window.clause("dst")}
                        COLUMNS (
                            dst.id AS id,
                            l1.weight AS weight
                        )
                    ){causal_two_hop}
                ) causal_raw
                GROUP BY id
            ),
            causal_expanded AS (
                SELECT {memory_unit_columns("mu", indent=23)},
                       cs.score, 'causal' AS source
                FROM causal_scores cs
                JOIN {mu_table} mu ON mu.id = cs.id
                ORDER BY cs.score DESC
                FETCH FIRST $2 ROWS ONLY
            )
            SELECT * FROM observation_entity_expanded
            UNION ALL
            SELECT * FROM semantic_expanded
            UNION ALL
            SELECT * FROM causal_expanded
            """,
        seed_ids,
        budget,
        *window.params,
    )
    return LinkExpansionRows(
        entity=[r for r in all_rows if r["source"] == "entity"],
        semantic=[r for r in all_rows if r["source"] == "semantic"],
        causal=[r for r in all_rows if r["source"] == "causal"],
    )


async def query_entity_neighborhood_pgq(
    conn: Any,
    *,
    mu_table: str,
    graph_name: str,
    bank_id: str,
    entity_ids: list,
    max_hops: int = 2,
    occurred_after: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int = 50,
) -> list[Any]:
    """Traverse entity co-occurrences and linked bi-temporal facts with explicit 0/1/2 hop distances."""
    if not entity_ids or limit <= 0:
        return []
    hops = max(1, min(int(max_hops), 2))
    params: list[object] = [entity_ids, bank_id]
    temporal_clauses: list[str] = []
    if occurred_after is not None:
        params.append(occurred_after)
        temporal_clauses.append(
            f"AND COALESCE(mu.occurred_end, mu.occurred_start, mu.mentioned_at, mu.event_date) >= ${len(params)}"
        )
    if occurred_before is not None:
        params.append(occurred_before)
        temporal_clauses.append(
            f"AND COALESCE(mu.occurred_start, mu.mentioned_at, mu.event_date) <= ${len(params)}"
        )
    params.append(limit)
    limit_param = f"${len(params)}"
    temporal_sql = " ".join(temporal_clauses)

    two_hop_arm = ""
    if hops >= 2:
        two_hop_arm = f"""
                        UNION ALL
                        SELECT unit_id, neighbor_entity_id, neighbor_entity_name, 2 AS hop_distance
                        FROM GRAPH_TABLE (
                            {graph_name}
                            MATCH (e1 IS entity) -[co1 IS cooccurs_with]- (e_mid IS entity)
                                  -[co2 IS cooccurs_with]- (e2 IS entity)
                                  <-[m IS mentions]- (u IS memory_unit)
                            WHERE e1.id = ANY($1::uuid[])
                              AND e_mid.id != e1.id
                              AND e2.id != e1.id
                              AND e2.id != e_mid.id
                              AND u.bank_id = $2
                            COLUMNS (
                                u.id AS unit_id,
                                e2.id AS neighbor_entity_id,
                                e2.canonical_name AS neighbor_entity_name
                            )
                        )"""

    return cast(
        "list[Any]",
        await conn.fetch(
            f"""
                SELECT {memory_unit_columns("mu", indent=23)},
                       n.neighbor_entity_id,
                       n.neighbor_entity_name,
                       n.hop_distance
                FROM (
                    SELECT unit_id,
                           neighbor_entity_id,
                           MAX(neighbor_entity_name) AS neighbor_entity_name,
                           MIN(hop_distance) AS hop_distance
                    FROM (
                        SELECT unit_id, neighbor_entity_id, neighbor_entity_name, 0 AS hop_distance
                        FROM GRAPH_TABLE (
                            {graph_name}
                            MATCH (e1 IS entity) <-[m IS mentions]- (u IS memory_unit)
                            WHERE e1.id = ANY($1::uuid[])
                              AND u.bank_id = $2
                            COLUMNS (
                                u.id AS unit_id,
                                e1.id AS neighbor_entity_id,
                                e1.canonical_name AS neighbor_entity_name
                            )
                        )
                        UNION ALL
                        SELECT unit_id, neighbor_entity_id, neighbor_entity_name, 1 AS hop_distance
                        FROM GRAPH_TABLE (
                            {graph_name}
                            MATCH (e1 IS entity) -[co1 IS cooccurs_with]- (e2 IS entity)
                                  <-[m IS mentions]- (u IS memory_unit)
                            WHERE e1.id = ANY($1::uuid[])
                              AND e2.id != e1.id
                              AND u.bank_id = $2
                            COLUMNS (
                                u.id AS unit_id,
                                e2.id AS neighbor_entity_id,
                                e2.canonical_name AS neighbor_entity_name
                            )
                        ){two_hop_arm}
                    )
                    GROUP BY unit_id, neighbor_entity_id
                ) n
                JOIN {mu_table} mu ON mu.id = n.unit_id
                WHERE mu.bank_id = $2
                  {temporal_sql}
                ORDER BY n.hop_distance ASC, mu.event_date DESC
                LIMIT {limit_param}
                """,
            *params,
        ),
    )