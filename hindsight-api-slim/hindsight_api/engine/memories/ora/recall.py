"""Oracle-specific semantic + BM25 recall arm."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ....config import get_config
from ...schema import fq_store_table as fq_table
from ...search.bm25_term_selection import build_bm25_query_text
from ...search.retrieval import tokenize_query
from ...search.tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause_simple
from ...search.types import RetrievalResult
from ...sql import create_sql_dialect
from ..base import GRAPH_SEED_LIMIT, SemanticBm25Result

logger = logging.getLogger(__name__)


async def retrieve_semantic_bm25_combined_sql(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_semantic: float | None = None,
    min_keyword: float | None = None,
    graph_seed_min_similarity: float | None = None,
    enable_text_search: bool = True,
) -> dict[str, SemanticBm25Result]:
    """Oracle dense + Oracle Text BM25 retrieval with single-pass exact scan and DRG fallback."""
    result_dict = {ft: SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None) for ft in fact_types}

    config = get_config()
    tokens = tokenize_query(query_text) if enable_text_search else []

    sem_min = min_semantic if min_semantic is not None else config.semantic_min_similarity
    bm25_min = min_keyword if min_keyword is not None else config.bm25_min_score

    graph_seed_threshold = (
        graph_seed_min_similarity
        if graph_seed_min_similarity is not None and sem_min <= graph_seed_min_similarity
        else None
    )
    semantic_fetch = max(limit, GRAPH_SEED_LIMIT if graph_seed_threshold is not None else 0)

    cols = (
        "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
        "fact_type, document_id, chunk_id, tags, metadata, proof_count"
    )
    table = fq_table("memory_units")
    dialect = create_sql_dialect(getattr(conn, "backend_type", "oracle"))

    _include_bm25 = bool(tokens)
    tags_param_idx = 5 if _include_bm25 else 3
    tags_clause = build_tags_where_clause_simple(tags, tags_param_idx, match=tags_match)

    tag_groups_param_start = tags_param_idx + (1 if tags else 0)
    built = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)
    groups_clause = built.sql
    groups_params = built.params

    _next_idx = tag_groups_param_start + len(groups_params)
    updated_range_clause = ""
    updated_range_params: list[Any] = []
    if created_after is not None:
        updated_range_params.append(created_after)
        updated_range_clause += f" AND updated_at > ${_next_idx}"
        _next_idx += 1
    if created_before is not None:
        updated_range_params.append(created_before)
        updated_range_clause += f" AND updated_at < ${_next_idx}"
        _next_idx += 1

    build_single_pass = getattr(dialect, "build_semantic_single_pass_arm", None)
    use_oracle_single_pass = (
        len(fact_types) > 1
        and callable(build_single_pass)
        and getattr(config, "oracle_vector_search", "exact") != "approx"
    )
    if use_oracle_single_pass and callable(build_single_pass):
        arms = [
            build_single_pass(
                table=table,
                cols=cols,
                fact_types=fact_types,
                embedding_param="$1",
                bank_id_param="$2",
                fetch_limit=semantic_fetch,
                min_similarity=sem_min,
                tags_clause=tags_clause,
                groups_clause=groups_clause,
                extra_where=updated_range_clause,
            )
        ]
    else:
        arms = [
            dialect.build_semantic_arm(
                table=table,
                cols=cols,
                fact_type=ft,
                embedding_param="$1",
                bank_id_param="$2",
                fetch_limit=semantic_fetch,
                min_similarity=sem_min,
                tags_clause=tags_clause,
                groups_clause=groups_clause,
                extra_where=updated_range_clause,
            )
            for ft in fact_types
        ]

    if _include_bm25:
        text_ext = config.text_search_extension
        bm25_text_param: str = await build_bm25_query_text(
            conn,
            dialect,
            tokens=tokens,
            query_text=query_text,
            table="memory_units",
            language=config.text_search_extension_native_language,
            config=config,
        )
        for i, ft in enumerate(fact_types):
            arms.append(
                dialect.build_bm25_arm(
                    table=table,
                    cols=cols,
                    fact_type=ft,
                    bank_id_param="$2",
                    limit_param="$3",
                    text_param="$4",
                    tags_clause=tags_clause,
                    groups_clause=groups_clause,
                    arm_index=i,
                    text_search_extension=text_ext,
                    bm25_language=config.text_search_extension_native_language,
                    bm25_min_score=bm25_min,
                    pg_search_function_schema=config.text_search_extension_pg_search_function_schema,
                    pg_search_tokenizer=config.text_search_extension_pg_search_tokenizer,
                    max_query_terms=config.bm25_max_query_terms,
                    extra_where=updated_range_clause,
                )
            )

    query = "\nUNION ALL\n".join(arms)

    params: list[Any] = [query_emb_str, bank_id]
    if _include_bm25:
        params.append(limit)
        params.append(bm25_text_param)
    if tags:
        params.append(tags)
    params.extend(groups_params)
    params.extend(updated_range_params)

    try:
        rows = await conn.fetch(query, *params)
    except Exception as e:
        err_str = str(e)
        if _include_bm25 and ("DRG-10599" in err_str or "ORA-30600" in err_str or "ORA-29902" in err_str):
            logger.warning("Oracle Text CONTAINS failed (%s), falling back to semantic-only search", err_str[:120])
            fb_tags_idx = 3
            fb_tags_clause = build_tags_where_clause_simple(tags, fb_tags_idx, match=tags_match)
            fb_groups_start = fb_tags_idx + (1 if tags else 0)
            built = build_tag_groups_where_clause(tag_groups, fb_groups_start)
            fb_groups_clause = built.sql
            fb_next_idx = fb_groups_start + len(groups_params)
            fb_updated_clause = ""
            if created_after is not None:
                fb_updated_clause += f" AND updated_at > ${fb_next_idx}"
                fb_next_idx += 1
            if created_before is not None:
                fb_updated_clause += f" AND updated_at < ${fb_next_idx}"
                fb_next_idx += 1
            if use_oracle_single_pass and callable(build_single_pass):
                fb_arms = [
                    build_single_pass(
                        table=table,
                        cols=cols,
                        fact_types=fact_types,
                        embedding_param="$1",
                        bank_id_param="$2",
                        fetch_limit=semantic_fetch,
                        min_similarity=sem_min,
                        tags_clause=fb_tags_clause,
                        groups_clause=fb_groups_clause,
                        extra_where=fb_updated_clause,
                    )
                ]
            else:
                fb_arms = [
                    dialect.build_semantic_arm(
                        table=table,
                        cols=cols,
                        fact_type=ft,
                        embedding_param="$1",
                        bank_id_param="$2",
                        fetch_limit=semantic_fetch,
                        min_similarity=sem_min,
                        tags_clause=fb_tags_clause,
                        groups_clause=fb_groups_clause,
                        extra_where=fb_updated_clause,
                    )
                    for ft in fact_types
                ]
            fb_query = "\nUNION ALL\n".join(fb_arms)
            fb_params: list[Any] = [query_emb_str, bank_id]
            if tags:
                fb_params.append(tags)
            fb_params.extend(groups_params)
            fb_params.extend(updated_range_params)
            rows = await conn.fetch(fb_query, *fb_params)
        else:
            raise

    semantic_candidates: dict[str, list[RetrievalResult]] = {ft: [] for ft in fact_types}
    for r in rows:
        row = dict(r)
        source = row.pop("source")
        ft = row.get("fact_type")
        if ft not in result_dict:
            continue
        if source == "semantic":
            if len(semantic_candidates[ft]) < semantic_fetch:
                semantic_candidates[ft].append(RetrievalResult.from_db_row(row))
        else:
            result_dict[ft].bm25.append(RetrievalResult.from_db_row(row))

    for ft, candidates in semantic_candidates.items():
        result_dict[ft].semantic.extend(candidates[:limit])
        if graph_seed_threshold is not None:
            result_dict[ft].graph_seeds = [
                candidate
                for candidate in candidates
                if candidate.similarity is not None and candidate.similarity >= graph_seed_threshold
            ][:GRAPH_SEED_LIMIT]

    return result_dict
