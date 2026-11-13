from __future__ import annotations

import json

import pytest

from hindsight_api.engine.memories.ora.counts import _tags_list
from hindsight_api.engine.memories.postgres import PostgresMemories


class _Connection:
    backend_type = "oracle"

    def __init__(self, tags: list[list[str] | None]):
        self.tags = tags
        self.queries = []
        self.rewritten_queries = []
        self.args = ()

    async def fetch(self, query, *args):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        rewritten = _rewrite_pg_to_oracle(query).query
        self.queries.append(query)
        self.rewritten_queries.append(rewritten)
        self.args = args
        tags_to_return = self.tags
        if "tags &&" in query:
            requested = set(args[1])
            tags_to_return = [tags for tags in self.tags if tags and requested.intersection(tags)]
        elif "<@" in query and "@>" in query:
            requested = set(args[1])
            tags_to_return = [tags for tags in self.tags if set(tags or []) == requested]
        if "UNION ALL" in query:
            counts = {}
            for tags in tags_to_return:
                encoded = json.dumps(tags or [])
                if len(encoded) <= 900:
                    counts[encoded] = counts.get(encoded, 0) + 1
            rows = [{"tags": encoded, "cnt": count} for encoded, count in counts.items()]
            rows.extend(
                {"tags": json.dumps(tags), "cnt": 1}
                for tags in tags_to_return
                if len(json.dumps(tags or [])) > 900
            )
            return rows
        return []


async def _counts(tags, *, limit=100, offset=0, tag_groups=None):
    conn = _Connection(tags)
    result = await PostgresMemories({}).observation_scope_counts(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank",
        limit=limit,
        offset=offset,
        tag_groups=tag_groups,
    )
    return result, conn


@pytest.mark.asyncio
async def test_oracle_query_avoids_postgres_only_scope_sql():
    _, conn = await _counts([["a"]])
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    for sql in conn.queries:
        query = _rewrite_pg_to_oracle(sql).query.upper()
        for forbidden in ("::TEXT[]", "UNNEST(", "AS MATERIALIZED", "ON TRUE"):
            assert forbidden not in query


@pytest.mark.asyncio
async def test_oracle_counts_merge_tag_orders_and_global_scopes():
    result, _ = await _counts([["b", "a"], ["a", "b"], None, []])
    assert result["scopes"] == [
        {"tags": [], "count": 2},
        {"tags": ["a", "b"], "count": 2},
    ]
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_oracle_counts_sort_by_count_then_tags():
    result, _ = await _counts([["z"], ["b"], ["a"], ["a"]])
    assert result["scopes"] == [
        {"tags": ["a"], "count": 2},
        {"tags": ["b"], "count": 1},
        {"tags": ["z"], "count": 1},
    ]


@pytest.mark.asyncio
async def test_oracle_counts_paginates_and_preserves_total_for_empty_page():
    result, _ = await _counts([["a"], ["b"], ["c"]], limit=1, offset=1)
    assert result == {"scopes": [{"tags": ["b"], "count": 1}], "total": 3, "limit": 1, "offset": 1}

    result, _ = await _counts([["a"], ["b"], ["c"]], limit=1, offset=9)
    assert result == {"scopes": [], "total": 3, "limit": 1, "offset": 9}


@pytest.mark.asyncio
async def test_oracle_counts_applies_tag_groups_predicate():
    from hindsight_api.engine.search.tags import TagGroupLeaf

    result, conn = await _counts(
        [["wanted"], ["other"]],
        tag_groups=[TagGroupLeaf(tags=["wanted"], match="any")],
    )
    assert "TAGS && $2" in " ".join(conn.queries).upper()
    assert result["scopes"] == [{"tags": ["wanted"], "count": 1}]
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_oracle_counts_rewrites_exact_tag_group_operators():
    from hindsight_api.engine.search.tags import TagGroupLeaf

    result, conn = await _counts(
        [["wanted"], ["wanted", "extra"], ["other"]],
        tag_groups=[TagGroupLeaf(tags=["wanted"], match="exact")],
    )
    assert result["scopes"] == [{"tags": ["wanted"], "count": 1}]
    assert result["total"] == 1
    rewritten = conn.rewritten_queries[0]
    for forbidden in ("$1", "$2", "::varchar[]", "::text[]", "@>", "<@", "&&"):
        assert forbidden not in rewritten.lower()
    assert "JSON_TABLE" in rewritten.upper()


def test_tags_list_preserves_duplicate_tags_when_sorting():
    assert _tags_list(["b", "a", "a"]) == ["a", "a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"limit": -1}, {"offset": -1}])
async def test_negative_pagination_is_rejected_before_database_query(kwargs):
    conn = _Connection([["a"]])
    with pytest.raises(ValueError, match="limit and offset must be non-negative"):
        await PostgresMemories({}).observation_scope_counts(
            conn=conn,
            fq_table=lambda name: name,
            bank_id="bank",
            **kwargs,
        )
    assert conn.queries == []


@pytest.mark.asyncio
async def test_zero_limit_returns_empty_page_and_exact_total():
    result, conn = await _counts([["a"], ["b"]], limit=0)
    assert result == {"scopes": [], "total": 2, "limit": 0, "offset": 0}
    assert len(conn.queries) == 1


@pytest.mark.asyncio
async def test_short_and_long_tag_queries_share_one_union_statement():
    _, conn = await _counts([["short"], ["x" * 1000]])
    assert len(conn.queries) == 1
    query = conn.queries[0].upper()
    assert "WITH FILTERED AS" in query
    assert "COMPACT_GROUPS AS" in query
    assert "UNION ALL" in query
    assert "SELECT TO_CLOB(SHORT_TAGS) AS TAGS, CNT" in query
    assert "TAGS AS TAGS, 1 AS CNT" in query
