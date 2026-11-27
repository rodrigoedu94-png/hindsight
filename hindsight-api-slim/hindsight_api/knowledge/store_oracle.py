"""Oracle AI Database 26ai implementation of :class:`KnowledgeStore`."""

from __future__ import annotations

import array
import json
import logging
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, cast

from ..engine.db.oracle import OracleLiteralStr
from ..engine.db_utils import acquire_with_retry
from ..engine.schema import fq_table
from ..engine.search.tags import TagGroup, TagGroupAnd, TagGroupLeaf, TagGroupNot, TagGroupOr, TagsMatch
from ..schema_context import get_current_schema
from .store import (
    _DOC_COLUMNS,
    _MOVE_WINDOW,
    _PASSAGE_COLUMNS,
    CANDIDATE_POOL_CAP,
    CANDIDATE_ROLES_SQL,
    KEYWORD_IDF_SHARE,
    RESCORE_ONLY_ROLES,
    RESCORE_POOL,
    ArmHit,
    ArmName,
    BankStats,
    DocumentPage,
    DocumentStatus,
    KnowledgeBankNotFound,
    KnowledgeDocument,
    KnowledgePassage,
    KnowledgeStore,
    MediaType,
    PassageReplacement,
    PassagesReplaced,
    PassageStage,
    ScoreExplain,
    SearchFilters,
    SearchHit,
    StatusUpdate,
    StoreSearchResult,
    StoreSearchTrace,
    TagCount,
    TagPage,
    TimeBucket,
    _chunks,
    _merged_ranges,
    _passage,
    fuse_keyword,
    page_score,
    with_document_prior,
)

logger = logging.getLogger(__name__)

_ORACLE_IN_CHUNK = 900
_MAX_KEYWORD_TERMS = 16
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _vec(values: Sequence[float]) -> array.array:
    """Bind a float sequence natively as Oracle ``DB_TYPE_VECTOR`` (FLOAT32)."""
    return array.array("f", values)


def _in_chunks(values: Sequence[Any], chunk_size: int = _ORACLE_IN_CHUNK) -> list[list[Any]]:
    return [list(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]


def _lit(value: str) -> OracleLiteralStr:
    """Wrap a VARCHAR2 string so ``OracleConnection`` never converts UUID-shaped IDs to RAW(16)."""
    return OracleLiteralStr(value)


def _lit_opt(value: str | None) -> OracleLiteralStr | None:
    return OracleLiteralStr(value) if value is not None else None


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except ValueError:
            logger.warning("Malformed JSON in kb_documents.tags; returning empty tags list")
            return []
    return list(raw)


def _oracle_document(row: Any) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=row["document_id"],
        filename=row["filename"],
        content_type=row["content_type"],
        media_type=cast(MediaType, row["media_type"]),
        size_bytes=int(row["size_bytes"] or 0),
        storage_key=row["storage_key"],
        content_hash=row["content_hash"],
        status=cast(DocumentStatus, row["status"]),
        error=row["error"],
        passage_count=int(row["passage_count"] or 0),
        tags=_parse_tags(row["tags"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_document_id=row["source_document_id"],
        content_tokens=int(row["content_tokens"] or 0),
    )


def _escape_oracle_text_term(term: str) -> str:
    """Escape a plain token for Oracle Text ``CONTAINS``."""
    return "{" + term.replace("}", "}}") + "}"


def _oracle_tags_predicate(
    tags: Sequence[str],
    params: list[Any],
    alias: str,
    match: TagsMatch,
) -> str:
    """Build an Oracle JSON predicate over ``<alias>tags`` matching PostgreSQL semantics."""
    col = f"{alias}tags"
    if match == "exact":
        if not tags:
            return f"AND ({col} IS NULL OR NOT JSON_EXISTS({col}, '$[*]'))"
        unique_tags = list(dict.fromkeys(tags))
        and_clauses = []
        for tag in unique_tags:
            params.append(_lit(tag))
            and_clauses.append(f"JSON_EXISTS({col}, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
        params.append(len(unique_tags))
        count_expr = (
            f"(SELECT COUNT(DISTINCT jt.tag) FROM JSON_TABLE({col}, '$[*]' COLUMNS (tag VARCHAR2(4000) PATH '$')) jt)"
        )
        return "AND (" + " AND ".join(and_clauses) + f" AND {count_expr} = ${len(params)})"
    if not tags:
        return ""
    if match == "any":
        or_clauses = []
        for tag in tags:
            params.append(_lit(tag))
            or_clauses.append(f"JSON_EXISTS({col}, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
        return f"AND (({col} IS NULL OR NOT JSON_EXISTS({col}, '$[*]')) OR (" + " OR ".join(or_clauses) + "))"
    if match == "any_strict":
        or_clauses = []
        for tag in tags:
            params.append(_lit(tag))
            or_clauses.append(f"JSON_EXISTS({col}, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
        return "AND (" + " OR ".join(or_clauses) + ")"
    if match == "all":
        and_clauses = []
        for tag in tags:
            params.append(_lit(tag))
            and_clauses.append(f"JSON_EXISTS({col}, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
        return f"AND (({col} IS NULL OR NOT JSON_EXISTS({col}, '$[*]')) OR (" + " AND ".join(and_clauses) + "))"
    # all_strict
    and_clauses = []
    for tag in tags:
        params.append(_lit(tag))
        and_clauses.append(f"JSON_EXISTS({col}, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
    return "AND (" + " AND ".join(and_clauses) + ")"


def _oracle_tag_group_expr(group: TagGroup, params: list[Any], alias: str) -> str:
    """Compile a recursive ``TagGroup`` filter tree into Oracle SQL."""
    if isinstance(group, TagGroupLeaf):
        pred = _oracle_tags_predicate(group.tags, params, alias, group.match)
        return pred.removeprefix("AND ") if pred else "1=1"
    if isinstance(group, TagGroupAnd):
        parts = [_oracle_tag_group_expr(child, params, alias) for child in group.filters]
        return "(" + " AND ".join(parts) + ")" if parts else "1=1"
    if isinstance(group, TagGroupOr):
        parts = [_oracle_tag_group_expr(child, params, alias) for child in group.filters]
        return "(" + " OR ".join(parts) + ")" if parts else "1=0"
    if isinstance(group, TagGroupNot):
        return f"NOT ({_oracle_tag_group_expr(group.filter, params, alias)})"
    return "1=1"


def _oracle_in_clause(col: str, values: Sequence[str], params: list[Any], *, literal_str: bool = False) -> str:
    """Build a chunked ``col IN (...)`` clause safe for >1000 elements."""
    groups: list[str] = []
    for chunk in _in_chunks(values):
        placeholders = []
        for val in chunk:
            params.append(_lit(val) if literal_str else val)
            placeholders.append(f"${len(params)}")
        groups.append(f"{col} IN ({', '.join(placeholders)})")
    return "(" + " OR ".join(groups) + ")" if groups else "1=0"


class OracleKnowledgeStore(KnowledgeStore):
    """Oracle AI Database 26ai implementation of :class:`KnowledgeStore`."""

    def __init__(self, backend_provider: Any) -> None:
        self._backend_provider = backend_provider
        self._ready: dict[str, int] = {}

    async def _pool(self) -> Any:
        return await self._backend_provider()

    async def ensure_ready(self, dimension: int) -> None:
        schema = get_current_schema() or "default"
        if self._ready.get(schema) == dimension:
            return
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT dim FROM ("
                f"SELECT VECTOR_DIMENSION_COUNT(embedding) AS dim FROM {fq_table('kb_passages')} "
                f"UNION SELECT VECTOR_DIMENSION_COUNT(embedding) AS dim FROM {fq_table('kb_passage_vectors')}"
                ") dims WHERE dim IS NOT NULL FETCH FIRST 2 ROWS ONLY"
            )
            dims = {int(r["dim"]) for r in rows if r["dim"] is not None}
            if dims and dims != {dimension}:
                raise RuntimeError(
                    f"Knowledge passages already store vectors of dimension {sorted(dims)}, "
                    f"which does not match configured dimension {dimension}."
                )
        self._ready[schema] = dimension

    async def get_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, KnowledgeDocument]:
        if not document_ids:
            return {}
        out: dict[str, KnowledgeDocument] = {}
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            for chunk in _in_chunks(document_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT {_DOC_COLUMNS} FROM {table} WHERE bank_id = $1 AND {in_sql}",
                    *params,
                )
                for r in rows:
                    doc = _oracle_document(r)
                    out[doc.document_id] = doc
        return out

    async def put_documents(self, bank_id: str, documents: list[KnowledgeDocument]) -> dict[str, str | None]:
        if not documents:
            return {}
        table = fq_table("kb_documents")
        ordered = sorted(documents, key=lambda d: d.document_id)
        doc_ids = [d.document_id for d in ordered]
        merge_sql = (
            f"MERGE INTO {table} target "
            "USING (SELECT $1 AS bank_id, $2 AS document_id, $3 AS filename, $4 AS content_type, "
            "$5 AS media_type, $6 AS size_bytes, $7 AS storage_key, $8 AS content_hash, "
            "$9 AS status, $10 AS tags, $11 AS source_document_id, $12 AS content_tokens FROM DUAL) src "
            "ON (target.bank_id = src.bank_id AND target.document_id = src.document_id) "
            "WHEN MATCHED THEN UPDATE SET "
            "target.filename = src.filename, target.content_type = src.content_type, "
            "target.media_type = src.media_type, target.size_bytes = src.size_bytes, "
            "target.storage_key = src.storage_key, target.content_hash = src.content_hash, "
            "target.status = src.status, target.error = NULL, target.tags = src.tags, "
            "target.source_document_id = src.source_document_id, target.content_tokens = src.content_tokens, "
            "target.updated_at = SYSTIMESTAMP "
            "WHEN NOT MATCHED THEN INSERT "
            "(bank_id, document_id, filename, content_type, media_type, size_bytes, storage_key, "
            "content_hash, status, error, tags, source_document_id, content_tokens) "
            "VALUES (src.bank_id, src.document_id, src.filename, src.content_type, src.media_type, "
            "src.size_bytes, src.storage_key, src.content_hash, src.status, NULL, src.tags, "
            "src.source_document_id, src.content_tokens)"
        )
        batch = [
            (
                _lit(bank_id),
                _lit(d.document_id),
                d.filename,
                d.content_type,
                d.media_type,
                d.size_bytes,
                d.storage_key,
                d.content_hash,
                d.status,
                json.dumps(d.tags),
                _lit_opt(d.source_document_id),
                d.content_tokens,
            )
            for d in ordered
        ]
        for attempt in range(2):
            found: dict[str, str | None] = {}
            try:
                async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
                    for chunk in _in_chunks(doc_ids):
                        params: list[Any] = [_lit(bank_id)]
                        in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                        prev_rows = await conn.fetch(
                            f"SELECT document_id, storage_key FROM {table} "
                            f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id FOR UPDATE",
                            *params,
                        )
                        for r in prev_rows:
                            found[r["document_id"]] = r["storage_key"]

                    for chunk in _chunks(batch):
                        await conn.executemany(merge_sql, chunk)
                return {d.document_id: found.get(d.document_id) for d in documents}
            except Exception as exc:
                if attempt == 0 and "ORA-00001" in str(exc):
                    continue
                raise
        return {}

    async def list_documents(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        status: str | None = None,
        media_type: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentPage:
        params: list[Any] = [_lit(bank_id)]
        where = "bank_id = $1"
        if query:
            params.append(f"%{query}%")
            where += (
                f" AND (UPPER(document_id) LIKE UPPER(${len(params)}) OR UPPER(filename) LIKE UPPER(${len(params)}))"
            )
        if status:
            params.append(status)
            where += f" AND status = ${len(params)}"
        if media_type:
            params.append(media_type)
            where += f" AND media_type = ${len(params)}"
        if tags:
            or_parts = []
            for tag in tags:
                params.append(_lit(tag))
                or_parts.append(f"JSON_EXISTS(tags, '$[*]?(@ == $t)' PASSING ${len(params)} AS \"t\")")
            where += " AND (" + " OR ".join(or_parts) + ")"
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE {where}", *params)
            rows = await conn.fetch(
                f"SELECT {_DOC_COLUMNS} FROM {table} WHERE {where} "
                f"ORDER BY updated_at DESC, document_id OFFSET ${len(params) + 1} ROWS FETCH NEXT ${len(params) + 2} ROWS ONLY",
                *params,
                offset,
                limit,
            )
        return DocumentPage(documents=[_oracle_document(r) for r in rows], total=int(total or 0))

    async def set_statuses(self, bank_id: str, updates: list[StatusUpdate]) -> None:
        if not updates:
            return
        table = fq_table("kb_documents")
        staged = fq_table("kb_staged_passages")
        ordered = sorted(updates, key=lambda u: u.document_id)
        doc_ids = [u.document_id for u in ordered]
        async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
            held: dict[str, str | None] = {}
            for chunk in _in_chunks(doc_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                locked_rows = await conn.fetch(
                    f"SELECT document_id, storage_key FROM {table} "
                    f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id FOR UPDATE",
                    *params,
                )
                for r in locked_rows:
                    held[r["document_id"]] = r["storage_key"]

            to_update: list[StatusUpdate] = []
            for u in ordered:
                if u.document_id not in held:
                    continue
                if u.storage_key is not None and held[u.document_id] != u.storage_key:
                    continue
                to_update.append(u)

            if to_update:
                batch = [(u.status, u.error, _lit(bank_id), _lit(u.document_id)) for u in to_update]
                for chunk in _chunks(batch):
                    await conn.executemany(
                        f"UPDATE {table} SET status = $1, error = $2, updated_at = SYSTIMESTAMP "
                        "WHERE bank_id = $3 AND document_id = $4",
                        chunk,
                    )
                failed = [u.document_id for u in to_update if u.status == "failed"]
                for chunk in _in_chunks(failed):
                    params = [_lit(bank_id)]
                    in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                    await conn.execute(f"DELETE FROM {staged} WHERE bank_id = $1 AND {in_sql}", *params)

    async def _held(self, conn: Any, bank_id: str, document_ids: list[str]) -> dict[str, str | None]:
        if not document_ids:
            return {}
        table = fq_table("kb_documents")
        out: dict[str, str | None] = {}
        for chunk in _in_chunks(sorted(set(document_ids))):
            params: list[Any] = [_lit(bank_id)]
            in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
            rows = await conn.fetch(
                f"SELECT document_id, storage_key FROM {table} "
                f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id FOR UPDATE",
                *params,
            )
            for r in rows:
                out[r["document_id"]] = r["storage_key"]
        return out

    @staticmethod
    async def _insert_passages(
        conn: Any,
        bank_id: str,
        items: Sequence[PassageStage | PassageReplacement],
        table: str,
        vectors_table: str,
    ) -> None:
        rows = [
            (
                _lit(bank_id),
                _lit(i.document_id),
                p.index,
                p.modality,
                p.text,
                p.page,
                p.start_ms,
                p.end_ms,
                p.media_key,
                _vec(p.embedding or []),
            )
            for i in items
            for p in i.passages
        ]
        if rows:
            columns = "bank_id, document_id, idx, modality, text, page, start_ms, end_ms, media_key, embedding"
            values = "$1, $2, $3, $4, $5, $6, $7, $8, $9, $10"
            for chunk in _chunks(rows):
                await conn.executemany(f"INSERT INTO {table} ({columns}) VALUES ({values})", chunk)
        extras = [
            (_lit(bank_id), _lit(i.document_id), p.index, v.role, v.n, _vec(v.embedding))
            for i in items
            for p in i.passages
            for v in p.extra_vectors
        ]
        for chunk in _chunks(extras):
            await conn.executemany(
                f"INSERT INTO {vectors_table} (bank_id, document_id, idx, role, n, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                chunk,
            )

    async def stage_passages(self, bank_id: str, items: list[PassageStage]) -> list[str]:
        if not items:
            return []
        staged = fq_table("kb_staged_passages")
        async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
            held = await self._held(conn, bank_id, [i.document_id for i in items])

            def holds(i: PassageStage) -> bool:
                return i.document_id in held and (i.storage_key is None or held[i.document_id] == i.storage_key)

            skipped = [i.document_id for i in items if not holds(i)]
            kept = [i for i in items if holds(i)]
            restart = [i.document_id for i in kept if i.staged == 0]
            for chunk in _in_chunks(restart):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                await conn.execute(f"DELETE FROM {staged} WHERE bank_id = $1 AND {in_sql}", *params)
            await self._insert_passages(conn, bank_id, kept, staged, fq_table("kb_staged_passage_vectors"))
        return skipped

    async def replace_passages(self, bank_id: str, items: list[PassageReplacement]) -> PassagesReplaced:
        if not items:
            return PassagesReplaced()
        table = fq_table("kb_passages")
        staged_table = fq_table("kb_staged_passages")
        vectors_table = fq_table("kb_passage_vectors")
        staged_vectors_table = fq_table("kb_staged_passage_vectors")
        async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
            held = await self._held(conn, bank_id, [i.document_id for i in items])

            def holds(i: PassageReplacement) -> bool:
                return i.document_id in held and (i.storage_key is None or held[i.document_id] == i.storage_key)

            skipped = [i.document_id for i in items if not holds(i)]
            kept = [i for i in items if holds(i)]
            if not kept:
                return PassagesReplaced(skipped_document_ids=skipped)

            document_ids = [i.document_id for i in kept]
            old_keys: list[str] = []
            for chunk in _in_chunks(document_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                old_rows = await conn.fetch(
                    f"SELECT media_key FROM {table} WHERE bank_id = $1 AND {in_sql} AND media_key IS NOT NULL",
                    *params,
                )
                old_keys.extend(r["media_key"] for r in old_rows)
                await conn.execute(f"DELETE FROM {table} WHERE bank_id = $1 AND {in_sql}", *params)

            with_staged = [i for i in kept if i.staged]
            moved_count: dict[str, int] = {}
            moved_keys: set[str] = set()
            lit_bank_id = _lit(bank_id)
            for item in with_staged:
                lit_doc_id = _lit(item.document_id)
                for lo in range(0, item.staged, _MOVE_WINDOW):
                    hi = min(item.staged, lo + _MOVE_WINDOW)
                    staged_rows = await conn.fetch(
                        f"SELECT media_key FROM {staged_table} "
                        "WHERE bank_id = $1 AND document_id = $2 AND idx >= $3 AND idx < $4",
                        lit_bank_id,
                        lit_doc_id,
                        lo,
                        hi,
                    )
                    moved_count[item.document_id] = moved_count.get(item.document_id, 0) + len(staged_rows)
                    moved_keys.update(r["media_key"] for r in staged_rows if r["media_key"])
                    await conn.execute(
                        f"INSERT INTO {table} (bank_id, document_id, idx, modality, text, page, start_ms, end_ms, "
                        "media_key, embedding) "
                        "SELECT bank_id, document_id, idx, modality, text, page, start_ms, end_ms, media_key, embedding "
                        f"FROM {staged_table} WHERE bank_id = $1 AND document_id = $2 AND idx >= $3 AND idx < $4",
                        lit_bank_id,
                        lit_doc_id,
                        lo,
                        hi,
                    )
                    await conn.execute(
                        f"INSERT INTO {vectors_table} (bank_id, document_id, idx, role, n, embedding) "
                        "SELECT bank_id, document_id, idx, role, n, embedding "
                        f"FROM {staged_vectors_table} "
                        "WHERE bank_id = $1 AND document_id = $2 AND idx >= $3 AND idx < $4",
                        lit_bank_id,
                        lit_doc_id,
                        lo,
                        hi,
                    )

            for chunk in _in_chunks(document_ids):
                params = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                await conn.execute(f"DELETE FROM {staged_table} WHERE bank_id = $1 AND {in_sql}", *params)

            await self._insert_passages(conn, bank_id, kept, table, vectors_table)

            doc_updates = [
                (moved_count.get(i.document_id, 0) + len(i.passages), _lit(bank_id), _lit(i.document_id)) for i in kept
            ]
            for chunk in _chunks(doc_updates):
                await conn.executemany(
                    f"UPDATE {fq_table('kb_documents')} "
                    "SET status = 'ready', error = NULL, passage_count = $1, updated_at = SYSTIMESTAMP "
                    "WHERE bank_id = $2 AND document_id = $3",
                    chunk,
                )

        new_keys = {p.media_key for i in kept for p in i.passages if p.media_key} | moved_keys
        return PassagesReplaced(
            stale_media_keys=[k for k in old_keys if k not in new_keys],
            skipped_document_ids=skipped,
        )

    async def list_passages(self, bank_id: str, document_ids: list[str]) -> dict[str, list[KnowledgePassage]]:
        out: dict[str, list[KnowledgePassage]] = {d: [] for d in document_ids}
        if not document_ids:
            return out
        table = fq_table("kb_passages")
        async with acquire_with_retry(await self._pool()) as conn:
            for chunk in _in_chunks(document_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT document_id, {_PASSAGE_COLUMNS} FROM {table} "
                    f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id, idx",
                    *params,
                )
                for r in rows:
                    out[r["document_id"]].append(_passage(r))
        return out

    async def _fetch_ranges(self, conn: Any, bank_id: str, ranges: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
        if not ranges:
            return []
        table = fq_table("kb_passages")
        out: list[dict[str, Any]] = []
        # Group by bounded OR clauses to stay well below Oracle's bind limit.
        for batch in _in_chunks(ranges, 200):
            params: list[Any] = [_lit(bank_id)]
            conds: list[str] = []
            for doc_id, first, last in batch:
                params.extend([_lit(doc_id), first, last])
                b = len(params)
                conds.append(f"(document_id = ${b - 2} AND idx BETWEEN ${b - 1} AND ${b})")
            rows = await conn.fetch(
                f"SELECT document_id, {_PASSAGE_COLUMNS} FROM {table} "
                f"WHERE bank_id = $1 AND (" + " OR ".join(conds) + ") ORDER BY document_id, idx",
                *params,
            )
            by_doc: dict[str, list[Any]] = {}
            for r in rows:
                by_doc.setdefault(r["document_id"], []).append(r)
            for doc_id, first, last in batch:
                for r in by_doc.get(doc_id, []):
                    if first <= r["idx"] <= last:
                        item = dict(r)
                        item["range_document_id"] = doc_id
                        item["first"] = first
                        item["last"] = last
                        out.append(item)
        return out

    async def passage_ranges(
        self, bank_id: str, ranges: list[tuple[str, int, int]]
    ) -> dict[tuple[str, int, int], list[KnowledgePassage]]:
        ranges = list(dict.fromkeys(ranges))
        out: dict[tuple[str, int, int], list[KnowledgePassage]] = {r: [] for r in ranges}
        if not ranges:
            return out
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await self._fetch_ranges(conn, bank_id, ranges)
        for r in rows:
            out[(r["range_document_id"], r["first"], r["last"])].append(_passage(r))
        return out

    async def media_keys(self, bank_id: str, document_ids: list[str]) -> list[str]:
        if not document_ids:
            return []
        table = fq_table("kb_passages")
        out: list[str] = []
        async with acquire_with_retry(await self._pool()) as conn:
            for chunk in _in_chunks(document_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT media_key FROM {table} WHERE bank_id = $1 AND {in_sql} AND media_key IS NOT NULL",
                    *params,
                )
                out.extend(r["media_key"] for r in rows)
        return out

    async def derived_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {d: [] for d in document_ids}
        if not document_ids:
            return out
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            for chunk in _in_chunks(document_ids):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("source_document_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT source_document_id, document_id FROM {table} "
                    f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id",
                    *params,
                )
                for r in rows:
                    out[r["source_document_id"]].append(r["document_id"])
        return out

    async def delete_documents(self, bank_id: str, document_ids: list[str]) -> list[KnowledgeDocument]:
        if not document_ids:
            return []
        table = fq_table("kb_documents")
        deleted: list[KnowledgeDocument] = []
        async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
            for chunk in _in_chunks(sorted(set(document_ids))):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT {_DOC_COLUMNS} FROM {table} "
                    f"WHERE bank_id = $1 AND {in_sql} ORDER BY document_id FOR UPDATE",
                    *params,
                )
                if rows:
                    deleted.extend(_oracle_document(r) for r in rows)
                    await conn.execute(f"DELETE FROM {table} WHERE bank_id = $1 AND {in_sql}", *params)
        return deleted

    async def delete_some_documents(self, bank_id: str, limit: int) -> int:
        if limit <= 0:
            return 0
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn, conn.transaction():
            rows = await conn.fetch(
                f"SELECT document_id FROM {table} WHERE bank_id = $1 AND ROWNUM <= $2 FOR UPDATE SKIP LOCKED",
                _lit(bank_id),
                limit,
            )
            picked = [r["document_id"] for r in rows]
            if not picked:
                return 0
            total_deleted = 0
            for chunk in _in_chunks(picked):
                params: list[Any] = [_lit(bank_id)]
                in_sql = _oracle_in_clause("document_id", chunk, params, literal_str=True)
                await conn.execute(f"DELETE FROM {table} WHERE bank_id = $1 AND {in_sql}", *params)
                total_deleted += len(chunk)
            return total_deleted

    async def search(
        self,
        bank_id: str,
        vector: list[float],
        *,
        top_k: int,
        filters: SearchFilters,
        text_weight: float = 0.6,
        query_text: str | None = None,
        keyword_weight: float = 0.2,
        around: int = 0,
        trace: StoreSearchTrace | None = None,
    ) -> StoreSearchResult:
        docs_table = fq_table("kb_documents")
        passages_table = fq_table("kb_passages")
        vectors_table = fq_table("kb_passage_vectors")
        banks_table = fq_table("banks")
        qvec = _vec(vector)
        pool_size = max(100, min(top_k * 10, CANDIDATE_POOL_CAP), top_k)

        async with acquire_with_retry(await self._pool()) as conn:
            found_bank = await conn.fetchval(
                f"SELECT 1 FROM {banks_table} WHERE bank_id = $1 AND kind = 'knowledge'",
                _lit(bank_id),
            )
            if not found_bank:
                raise KnowledgeBankNotFound(bank_id)

            filter_params: list[Any] = [_lit(bank_id)]
            where = ""
            if filters.document_ids:
                where += " AND " + _oracle_in_clause(
                    "x.document_id", filters.document_ids, filter_params, literal_str=True
                )
            if filters.media_types:
                where += " AND " + _oracle_in_clause("d.media_type", filters.media_types, filter_params)
            if filters.tags or filters.tags_match == "exact":
                where += " " + _oracle_tags_predicate(filters.tags or [], filter_params, "d.", filters.tags_match)
            if filters.tag_groups:
                for grp in filters.tag_groups:
                    where += " AND " + _oracle_tag_group_expr(grp, filter_params, "d.")

            # Join kb_documents only when a document-level filter is active (#6747).
            needs_doc_join = bool(
                filters.media_types or filters.tags or filters.tags_match == "exact" or filters.tag_groups
            )
            join_docs = (
                f"JOIN {docs_table} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id "
                if needs_doc_join
                else ""
            )

            # Arm 1: primary passage embeddings (EXACT cosine distance).
            vec_params = [qvec, *filter_params]
            p_where = re.sub(r"\$(\d+)", lambda m: f"${int(m.group(1)) + 1}", where)
            primary_rows = await conn.fetch(
                f"SELECT x.document_id, x.idx, 1 - VECTOR_DISTANCE(x.embedding, $1, COSINE) AS arm_score "
                f"FROM {passages_table} x "
                f"{join_docs}"
                f"WHERE x.bank_id = $2{p_where} "
                f"ORDER BY VECTOR_DISTANCE(x.embedding, $1, COSINE), x.document_id ASC, x.idx ASC "
                f"FETCH EXACT FIRST {pool_size} ROWS ONLY",
                *vec_params,
            )

            # Arm 2: extra role vectors (bands, page_text, units, windows).
            extra_roles_sql = CANDIDATE_ROLES_SQL.replace("role", "x.role")
            extra_rows = await conn.fetch(
                f"SELECT x.document_id, x.idx, MAX(1 - VECTOR_DISTANCE(x.embedding, $1, COSINE)) AS arm_score "
                f"FROM {vectors_table} x "
                f"{join_docs}"
                f"WHERE x.bank_id = $2{p_where} AND {extra_roles_sql} "
                f"GROUP BY x.document_id, x.idx "
                f"ORDER BY MAX(1 - VECTOR_DISTANCE(x.embedding, $1, COSINE)) DESC, x.document_id ASC, x.idx ASC "
                f"FETCH FIRST {pool_size} ROWS ONLY",
                *vec_params,
            )

            # Arm 3: Oracle Text keyword search with BM25 IDF filtering.
            terms_idf: dict[str, float] = {}
            keyword_rows: list[Any] = []
            keywords_enabled = bool(query_text) and keyword_weight > 0
            if keywords_enabled and query_text:
                raw_terms = list(dict.fromkeys(w.lower() for w in _WORD_RE.findall(query_text)))[:_MAX_KEYWORD_TERMS]
                if raw_terms:
                    total_n = float(
                        await conn.fetchval(
                            f"SELECT COALESCE(SUM(passage_count), 0) FROM {docs_table} WHERE bank_id = $1",
                            _lit(bank_id),
                        )
                        or 0.0
                    )
                    df_map: dict[str, float] = {}
                    batched_df_ok = False
                    if len(raw_terms) > 1:
                        escaped_terms = [_escape_oracle_text_term(t) for t in raw_terms]
                        union_sql = " UNION ALL ".join(
                            f"SELECT {i} AS term_idx, COUNT(*) AS df FROM {passages_table} "
                            f"WHERE bank_id = $1 AND CONTAINS(text, ${i + 2}, {i + 1}) > 0"
                            for i in range(len(escaped_terms))
                        )
                        try:
                            df_rows = await conn.fetch(union_sql, _lit(bank_id), *escaped_terms)
                            for row in df_rows:
                                idx_val = int(row["term_idx"])
                                df_val = row["df"]
                                if 0 <= idx_val < len(raw_terms) and df_val and float(df_val) > 0:
                                    df_map[raw_terms[idx_val]] = float(df_val)
                            batched_df_ok = True
                        except Exception:
                            logger.debug(
                                "Batched Oracle Text df probe failed; falling back to per-term probes", exc_info=True
                            )
                    if not batched_df_ok:
                        for term in raw_terms:
                            escaped = _escape_oracle_text_term(term)
                            try:
                                df = await conn.fetchval(
                                    f"SELECT COUNT(*) FROM {passages_table} WHERE bank_id = $1 AND CONTAINS(text, $2) > 0",
                                    _lit(bank_id),
                                    escaped,
                                )
                                if df and float(df) > 0:
                                    df_map[term] = float(df)
                            except Exception:
                                logger.debug("Oracle Text term df probe skipped for %r", term, exc_info=True)
                    if df_map:
                        raw_idf = {
                            t: math.log(1.0 + max(total_n - df, 0.0) / (df + 0.5) + 0.5 / (df + 0.5))
                            for t, df in df_map.items()
                        }
                        max_idf = max(raw_idf.values())
                        terms_idf = {t: val for t, val in raw_idf.items() if val >= KEYWORD_IDF_SHARE * max_idf}
                    if terms_idf:
                        # ACCUM, not OR: OR scores a passage by its best single term, so one that matches
                        # every query term ties one that matches only the strongest. ACCUM adds the terms'
                        # scores, so the candidate pool (ORDER BY SCORE) favors passages matching more terms.
                        or_expr = " ACCUM ".join(_escape_oracle_text_term(t) for t in terms_idf)
                        kw_params = [*filter_params, or_expr]
                        kw_pos = len(kw_params)
                        try:
                            keyword_rows = await conn.fetch(
                                f"SELECT x.document_id, x.idx, SCORE(1) AS arm_score "
                                f"FROM {passages_table} x "
                                f"{join_docs}"
                                f"WHERE x.bank_id = $1{where} AND CONTAINS(x.text, ${kw_pos}, 1) > 0 "
                                f"ORDER BY SCORE(1) DESC, x.document_id ASC, x.idx ASC "
                                f"FETCH FIRST {pool_size} ROWS ONLY",
                                *kw_params,
                            )
                        except Exception:
                            logger.debug("Oracle Text keyword arm failed for query %r", or_expr, exc_info=True)

            found_by: dict[tuple[str, int], list[tuple[str, float]]] = {}
            for arm_name, arm_rows in (
                ("vector", primary_rows),
                ("extra_vectors", extra_rows),
                ("keyword", keyword_rows),
            ):
                for r in arm_rows:
                    key = (r["document_id"], int(r["idx"]))
                    found_by.setdefault(key, []).append((arm_name, float(r["arm_score"] or 0.0)))

            if trace is not None:
                trace.keyword_terms = terms_idf
                trace.candidates = len(found_by)
                trace.keyword_weight = keyword_weight
                for (doc_id, idx), arm_list in found_by.items():
                    for arm_name, score in arm_list:
                        trace.arms.setdefault(cast(ArmName, arm_name), []).append(ArmHit(doc_id, idx, score))
                for arm_hits in trace.arms.values():
                    arm_hits.sort(key=lambda a: a.score, reverse=True)

            if not found_by:
                return StoreSearchResult([])

            candidate_keys = list(found_by.keys())
            doc_cols = ", ".join(f"d.{col.strip()}" for col in _DOC_COLUMNS.split(",") if col.strip() != "document_id")
            passage_cols = ", ".join(f"p.{col.strip()}" for col in _PASSAGE_COLUMNS.split(","))
            rows: list[Any] = []
            for key_batch in _in_chunks(candidate_keys, 200):
                params = [qvec, _lit(bank_id)]
                conds = []
                for doc_id, idx in key_batch:
                    params.extend([_lit(doc_id), idx])
                    b = len(params)
                    conds.append(f"(p.document_id = ${b - 1} AND p.idx = ${b})")
                fetched = await conn.fetch(
                    f"SELECT {passage_cols}, {doc_cols}, p.document_id, "
                    f"1 - VECTOR_DISTANCE(p.embedding, $1, COSINE) AS score "
                    f"FROM {passages_table} p "
                    f"JOIN {docs_table} d ON d.bank_id = p.bank_id AND d.document_id = p.document_id "
                    f"WHERE p.bank_id = $2 AND (" + " OR ".join(conds) + ")",
                    *params,
                )
                rows.extend(fetched)

            if not rows:
                return StoreSearchResult([])
            # Break equal-score ties deterministically by (document_id, idx) (#6697).
            rows.sort(key=lambda r: (r["document_id"], int(r["idx"])))

            # Fetch extra non-sentence vector role scores for candidates.
            role_scores: dict[tuple[str, int], dict[str, list[float]]] = {}
            extra_filter = CANDIDATE_ROLES_SQL.replace("role", "v.role")
            for key_batch in _in_chunks(candidate_keys, 200):
                params = [qvec, _lit(bank_id)]
                conds = []
                for doc_id, idx in key_batch:
                    params.extend([_lit(doc_id), idx])
                    b = len(params)
                    conds.append(f"(v.document_id = ${b - 1} AND v.idx = ${b})")
                vrows = await conn.fetch(
                    f"SELECT v.document_id, v.idx, v.role, 1 - VECTOR_DISTANCE(v.embedding, $1, COSINE) AS sim "
                    f"FROM {vectors_table} v "
                    f"WHERE v.bank_id = $2 AND {extra_filter} AND (" + " OR ".join(conds) + ")",
                    *params,
                )
                for vr in vrows:
                    key = (vr["document_id"], int(vr["idx"]))
                    role_scores.setdefault(key, {}).setdefault(vr["role"], []).append(float(vr["sim"] or 0.0))

            roles = [role_scores.get((r["document_id"], int(r["idx"])), {}) for r in rows]
            first = [page_score(float(r["score"] or 0.0), ro, text_weight) for r, ro in zip(rows, roles, strict=True)]

            # Stage 3: rescore top pages with sentence vectors.
            top_indices = sorted(range(len(rows)), key=lambda i: first[i], reverse=True)[: max(top_k, RESCORE_POOL)]
            top_indices = [i for i in top_indices if rows[i]["page"] is not None]
            if top_indices:
                rescore_roles = ", ".join(f"'{r}'" for r in RESCORE_ONLY_ROLES)
                best: dict[tuple[str, int], float] = {}
                for idx_batch in _in_chunks(top_indices, 200):
                    s_params: list[Any] = [qvec, _lit(bank_id)]
                    conds = []
                    for i in idx_batch:
                        s_params.extend([_lit(rows[i]["document_id"]), int(rows[i]["idx"])])
                        b = len(s_params)
                        conds.append(f"(v.document_id = ${b - 1} AND v.idx = ${b})")
                    srows = await conn.fetch(
                        f"SELECT v.document_id, v.idx, MAX(1 - VECTOR_DISTANCE(v.embedding, $1, COSINE)) AS best "
                        f"FROM {vectors_table} v "
                        f"WHERE v.bank_id = $2 AND v.role IN ({rescore_roles}) AND (" + " OR ".join(conds) + ") "
                        "GROUP BY v.document_id, v.idx",
                        *s_params,
                    )
                    for sr in srows:
                        best[(sr["document_id"], int(sr["idx"]))] = float(sr["best"] or 0.0)
                for i in top_indices:
                    sentence = best.get((rows[i]["document_id"], int(rows[i]["idx"])))
                    if sentence is not None:
                        roles[i]["sentence"] = [sentence]
                        first[i] = page_score(float(rows[i]["score"] or 0.0), roles[i], text_weight)

            dense = with_document_prior([r["document_id"] for r in rows], first)
            keyword_scores: list[float] = []
            for r in rows:
                if not terms_idf or not r["text"]:
                    keyword_scores.append(0.0)
                else:
                    doc_words = {w.lower() for w in _WORD_RE.findall(r["text"])}
                    keyword_scores.append(sum(idf for term, idf in terms_idf.items() if term in doc_words))

            has_text = [bool((r["text"] or "").strip()) for r in rows]
            scores = fuse_keyword(dense, keyword_scores, keyword_weight, has_text)
            hits = [
                SearchHit(
                    passage=_passage(r),
                    document=_oracle_document(r),
                    score=scores[i],
                    explain=ScoreExplain(
                        found_by=sorted({arm for arm, _ in found_by[(r["document_id"], int(r["idx"]))]}),
                        similarity=float(r["score"] or 0.0),
                        roles={role: max(values) for role, values in roles[i].items() if values},
                        passage_score=first[i],
                        with_document=dense[i],
                        keyword=keyword_scores[i],
                    ),
                )
                for i, r in enumerate(rows)
            ]
            hits.sort(key=lambda h: h.score, reverse=True)
            if trace is not None:
                trace.scored = hits
            hits = hits[:top_k]
            spans = [
                (h.document.document_id, max(0, h.passage.index - around), h.passage.index + around)
                for h in hits
                if h.passage.modality == "text"
            ]
            nearby: dict[tuple[str, int], KnowledgePassage] = {}
            if around > 0 and spans:
                for r in await self._fetch_ranges(conn, bank_id, _merged_ranges(spans)):
                    nearby[(r["range_document_id"], r["idx"])] = _passage(r)
        return StoreSearchResult(hits, nearby)

    async def list_tags(self, bank_id: str, *, pattern: str | None, limit: int, offset: int) -> TagPage:
        like = pattern.replace("*", "%") if pattern else None
        table = fq_table("kb_documents")
        base_sql = (
            f"SELECT jt.tag AS tag FROM {table} d, "
            "JSON_TABLE(d.tags, '$[*]' COLUMNS (tag VARCHAR2(4000) PATH '$')) jt "
            "WHERE d.bank_id = $1"
        )
        params: list[Any] = [_lit(bank_id)]
        match = ""
        if like is not None:
            params.append(like)
            match = f" WHERE UPPER(t.tag) LIKE UPPER(${len(params)})"
        async with acquire_with_retry(await self._pool()) as conn:
            total = await conn.fetchval(f"SELECT COUNT(DISTINCT t.tag) FROM ({base_sql}) t{match}", *params)
            rows = await conn.fetch(
                f"SELECT t.tag, COUNT(*) AS documents FROM ({base_sql}) t{match} "
                f"GROUP BY t.tag ORDER BY documents DESC, t.tag ASC "
                f"OFFSET ${len(params) + 1} ROWS FETCH NEXT ${len(params) + 2} ROWS ONLY",
                *params,
                offset,
                limit,
            )
        return TagPage(
            tags=[TagCount(tag=r["tag"], documents=int(r["documents"])) for r in rows], total=int(total or 0)
        )

    async def bank_stats(self, bank_ids: list[str]) -> dict[str, BankStats]:
        if not bank_ids:
            return {}
        stats = {b: BankStats() for b in bank_ids}
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            for chunk in _in_chunks(bank_ids):
                params: list[Any] = []
                in_sql = _oracle_in_clause("bank_id", chunk, params, literal_str=True)
                rows = await conn.fetch(
                    f"SELECT bank_id, media_type, status, COUNT(*) AS documents, "
                    "COALESCE(SUM(passage_count), 0) AS passages, "
                    "COALESCE(SUM(CASE WHEN passage_count > 0 THEN content_tokens ELSE 0 END), 0) AS content_tokens, "
                    "MAX(updated_at) AS last_at "
                    f"FROM {table} WHERE {in_sql} "
                    "GROUP BY bank_id, media_type, status",
                    *params,
                )
                for r in rows:
                    s = stats[r["bank_id"]]
                    n = int(r["documents"])
                    s.document_count += n
                    s.passage_count += int(r["passages"])
                    s.content_tokens += int(r["content_tokens"])
                    s.by_media_type[r["media_type"]] = s.by_media_type.get(r["media_type"], 0) + n
                    if r["last_at"] is not None and (s.last_document_at is None or r["last_at"] > s.last_document_at):
                        s.last_document_at = r["last_at"]
                    if r["status"] == "ready":
                        s.ready_count += n
                    elif r["status"] == "failed":
                        s.failed_count += n
                    else:
                        s.pending_count += n
        return stats

    async def timeseries(
        self, bank_id: str, *, since: datetime, trunc: str, media_type: str | None = None
    ) -> list[TimeBucket]:
        if trunc not in ("hour", "day"):
            raise ValueError("trunc must be 'hour' or 'day'")
        ora_fmt = "HH24" if trunc == "hour" else "DD"
        params: list[Any] = [_lit(bank_id), since]
        media_sql = ""
        if media_type is not None:
            params.append(media_type)
            media_sql = f" AND media_type = ${len(params)}"
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT TRUNC(SYS_EXTRACT_UTC(created_at), '{ora_fmt}') AS t, COUNT(*) AS documents, "
                f"COALESCE(SUM(passage_count), 0) AS passages "
                f"FROM {fq_table('kb_documents')} WHERE bank_id = $1 AND created_at >= $2{media_sql} "
                f"GROUP BY TRUNC(SYS_EXTRACT_UTC(created_at), '{ora_fmt}') ORDER BY t",
                *params,
            )
        buckets: list[TimeBucket] = []
        for r in rows:
            ts = r["t"]
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            buckets.append(TimeBucket(time=ts, documents=int(r["documents"]), passages=int(r["passages"])))
        return buckets
