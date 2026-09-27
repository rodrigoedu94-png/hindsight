"""Embedding-dimension management on the Oracle backend.

The Oracle baseline declares ``VECTOR(384, FLOAT32)`` and, unlike PostgreSQL, nothing used to
reconcile it with the configured embeddings model: the startup path never ran the check and
``run-db-migration --embedding-dimension`` reached PostgreSQL-only SQL. These unit tests drive
``_ensure_oracle_table_embedding_dimension`` with a scripted cursor; the ``oracle``-marked tests
at the bottom exercise the same code against a live database.
"""

import array
import re
import uuid

import pytest

from hindsight_api.migrations import _ensure_oracle_table_embedding_dimension


class _ScriptedCursor:
    """Answers the dictionary/data queries the dimension check issues; records every statement."""

    def __init__(
        self,
        *,
        vector_info: str | None,
        stored_dimensions: list[int] | None = None,
        vector_indexes: list[tuple[str, str, str]] | None = None,  # (name, INDEX_SUBTYPE, PARTITIONED)
        has_legacy: bool = False,
    ) -> None:
        self.vector_info = vector_info
        self.stored_dimensions = stored_dimensions or []
        self.vector_indexes = vector_indexes or []
        self.has_legacy = has_legacy
        self.statements: list[str] = []
        self._result: list[tuple] = []

    def execute(self, sql: str, binds: dict | None = None) -> None:
        self.statements.append(sql)
        # Column DDL changes what the dictionary reports next, like the real catalog.
        if "RENAME COLUMN embedding TO embedding_legacy" in sql:
            self.vector_info, self.has_legacy = None, True
        elif added := re.search(r"ADD \(embedding VECTOR\((\d+), FLOAT32\)\)", sql):
            self.vector_info = f"VECTOR({added.group(1)},FLOAT32,DENSE)"
        elif "DROP COLUMN embedding_legacy" in sql:
            self.has_legacy = False
        if "vector_info" in sql.lower():
            self._result = ([("EMBEDDING", self.vector_info)] if self.vector_info is not None else []) + (
                [("EMBEDDING_LEGACY", None)] if self.has_legacy else []
            )
        elif "VECTOR_DIMENSION_COUNT" in sql:
            # With :dim bound the query looks for a row of any OTHER dimension; without it, any row.
            wanted = (binds or {}).get("dim")
            matches = [d for d in self.stored_dimensions if wanted is None or d != wanted]
            self._result = [(d,) for d in matches[:1]]
        elif "index_type = 'VECTOR'" in sql:
            self._result = list(self.vector_indexes)
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


def _ddl(cursor: _ScriptedCursor) -> list[str]:
    return [s for s in cursor.statements if re.match(r"\s*(ALTER TABLE|DROP INDEX|CREATE)", s)]


def test_matching_fixed_dimension_changes_nothing():
    cursor = _ScriptedCursor(vector_info="VECTOR(1536,FLOAT32,DENSE)", stored_dimensions=[1536])
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


@pytest.mark.parametrize("required", [384, 768, 1536, 3072])
def test_empty_table_is_resized_and_its_vector_index_rebuilt(required):
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)" if required != 384 else "VECTOR(1024,FLOAT32,DENSE)",
        vector_indexes=[("IDX_MU_EMBEDDING_HNSW", "NEIGHBOR_PARTITIONS_IVF", "NO")],
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", required)
    # MODIFY is not an option: Oracle rejects any VECTOR dimension change with ORA-51859.
    assert _ddl(cursor) == [
        'DROP INDEX "IDX_MU_EMBEDDING_HNSW"',
        "ALTER TABLE MEMORY_UNITS RENAME COLUMN embedding TO embedding_legacy",
        f"ALTER TABLE MEMORY_UNITS ADD (embedding VECTOR({required}, FLOAT32))",
        "ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy",
        'CREATE VECTOR INDEX "IDX_MU_EMBEDDING_HNSW" ON MEMORY_UNITS (embedding) '
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95",
    ]


def test_resize_keeps_a_local_hnsw_index_local():
    cursor = _ScriptedCursor(
        vector_info="VECTOR(384,FLOAT32,DENSE)",
        vector_indexes=[("MU_HNSW", "INMEMORY_NEIGHBOR_GRAPH_HNSW", "YES")],
    )
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor)[-1] == (
        'CREATE VECTOR INDEX "MU_HNSW" ON MEMORY_UNITS (embedding) '
        "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95 LOCAL"
    )


def test_resize_refuses_an_index_it_cannot_rebuild():
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)", vector_indexes=[("X", "SOMETHING_NEW", "NO")])
    with pytest.raises(RuntimeError, match="unknown subtype 'SOMETHING_NEW'"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_resize_interrupted_after_the_rename_is_finished():
    cursor = _ScriptedCursor(vector_info=None, has_legacy=True)
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == [
        "ALTER TABLE MEMORY_UNITS ADD (embedding VECTOR(1536, FLOAT32))",
        "ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy",
    ]


def test_resize_interrupted_after_the_new_column_is_finished():
    cursor = _ScriptedCursor(vector_info="VECTOR(1536,FLOAT32,DENSE)", has_legacy=True)
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == ["ALTER TABLE MEMORY_UNITS DROP COLUMN embedding_legacy"]


def test_table_with_embeddings_of_another_dimension_fails_explicitly():
    cursor = _ScriptedCursor(vector_info="VECTOR(384,FLOAT32,DENSE)", stored_dimensions=[384])
    with pytest.raises(RuntimeError, match=r"from 384 to 1536.*MEMORY_UNITS"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_flexible_column_holding_only_the_model_dimension_is_accepted():
    """A VECTOR(*, *) column (as found on a live Autonomous deployment) is left as is."""
    cursor = _ScriptedCursor(vector_info="VECTOR(*,*,DENSE)", stored_dimensions=[1536])
    _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_flexible_column_holding_another_dimension_fails_explicitly():
    cursor = _ScriptedCursor(vector_info="VECTOR(*,*,DENSE)", stored_dimensions=[384])
    with pytest.raises(RuntimeError, match=r"MEMORY_UNITS.*384.*1536"):
        _ensure_oracle_table_embedding_dimension(cursor, "MEMORY_UNITS", 1536)
    assert _ddl(cursor) == []


def test_missing_table_is_skipped():
    cursor = _ScriptedCursor(vector_info=None)
    _ensure_oracle_table_embedding_dimension(cursor, "MENTAL_MODELS", 1536)
    assert _ddl(cursor) == []


@pytest.fixture()
def oracle_cursor(_oracle_admin_dsn):
    """A live cursor on the test schema plus a scratch table name, dropped afterwards."""
    oracledb = pytest.importorskip("oracledb")
    conn = oracledb.connect(**_oracle_admin_dsn)
    conn.autocommit = True
    cursor = conn.cursor()
    table = f"HS_DIM_{uuid.uuid4().hex[:8].upper()}"
    try:
        yield cursor, table
    finally:
        try:
            cursor.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        conn.close()


def _vector_info(cursor, table: str) -> str:
    cursor.execute(
        "SELECT vector_info FROM user_tab_columns WHERE table_name = :t AND column_name = 'EMBEDDING'", {"t": table}
    )
    return cursor.fetchone()[0]


def _vector_indexes(cursor, table: str) -> list[str]:
    cursor.execute(
        "SELECT index_name FROM user_indexes WHERE table_name = :t AND index_type = 'VECTOR' ORDER BY 1", {"t": table}
    )
    return [r[0] for r in cursor.fetchall()]


@pytest.mark.oracle
def test_live_resize_replaces_the_column_and_keeps_the_vector_index(oracle_cursor):
    cursor, table = oracle_cursor
    # Same shape as the baseline: VECTOR(384, FLOAT32) with an IVF (neighbor partitions) index.
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, text CLOB, embedding VECTOR(384, FLOAT32))")
    cursor.execute(
        f"CREATE VECTOR INDEX {table}_IVF ON {table} (embedding) ORGANIZATION NEIGHBOR PARTITIONS "
        "DISTANCE COSINE WITH TARGET ACCURACY 95"
    )

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)

    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"
    assert _vector_indexes(cursor, table) == [f"{table}_IVF"]

    # Idempotent: a second run with the same model changes nothing.
    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)
    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"

    # Once embeddings are stored, a model with another dimension is refused, not mixed in.
    cursor.execute(f"INSERT INTO {table} VALUES (1, 'x', :v)", {"v": array.array("f", [0.1] * 1536)})
    with pytest.raises(RuntimeError, match="from 1536 to 768"):
        _ensure_oracle_table_embedding_dimension(cursor, table, 768)
    assert _vector_info(cursor, table) == "VECTOR(1536,FLOAT32,DENSE)"


@pytest.mark.oracle
def test_live_flexible_column_is_validated_not_altered(oracle_cursor):
    cursor, table = oracle_cursor
    cursor.execute(f"CREATE TABLE {table} (id NUMBER PRIMARY KEY, embedding VECTOR)")
    cursor.execute(f"INSERT INTO {table} VALUES (1, :v)", {"v": array.array("f", [0.1] * 1536)})

    _ensure_oracle_table_embedding_dimension(cursor, table, 1536)
    assert _vector_info(cursor, table) == "VECTOR(*,*,DENSE)"

    with pytest.raises(RuntimeError, match="flexible VECTOR column holding 1536"):
        _ensure_oracle_table_embedding_dimension(cursor, table, 384)


def test_admin_migration_of_an_oracle_url_skips_the_postgresql_steps(monkeypatch):
    """`hindsight-admin run-db-migration --embedding-dimension N` on Oracle.

    It used to go through the PostgreSQL per-schema unit (libpq URL, information_schema,
    pgvector extension checks) and fail; it now runs Alembic and the Oracle dimension step,
    as the migration user, and maps PG's "public" default to the connecting user's schema.
    """
    from hindsight_api import migrations

    calls: list[str] = []
    monkeypatch.setattr(migrations, "_should_isolate_migrations", lambda: False)
    monkeypatch.setattr(
        migrations,
        "run_migrations",
        lambda url, *, schema=None, migration_database_url=None, **_: calls.append(
            f"migrate {url} schema={schema} as={migration_database_url}"
        ),
    )
    monkeypatch.setattr(
        migrations,
        "ensure_embedding_dimension",
        lambda url, dim, *, schema=None, store_owned_memories=False, **_: calls.append(
            f"dimension {url} {dim} schema={schema}"
        ),
    )
    monkeypatch.setattr(migrations, "_migrate_one_schema_pg", lambda *a, **k: calls.append("postgresql path"))

    migrations.run_migrations_for_schemas(
        "oracle+oracledb://app@/?dsn=x",
        ["public", "TENANT_B"],
        migration_database_url="oracle+oracledb://owner@/?dsn=x",
        embedding_dimension=1536,
    )

    assert calls == [
        "migrate oracle+oracledb://app@/?dsn=x schema=None as=oracle+oracledb://owner@/?dsn=x",
        "dimension oracle+oracledb://owner@/?dsn=x 1536 schema=None",
        "migrate oracle+oracledb://app@/?dsn=x schema=TENANT_B as=oracle+oracledb://owner@/?dsn=x",
        "dimension oracle+oracledb://owner@/?dsn=x 1536 schema=TENANT_B",
    ]
