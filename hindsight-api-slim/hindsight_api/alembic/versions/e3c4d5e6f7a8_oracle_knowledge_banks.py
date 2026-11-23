"""Add Oracle Knowledge Banks tables, indexes, Oracle Text index, and runtime grants.

Revision ID: e3c4d5e6f7a8
Revises: o2b3c4d5e6f7
Create Date: 2026-11-22

Creates the five ``kb_*`` tables on Oracle AI Database 26ai so Knowledge Banks
run natively alongside PostgreSQL:

- ``kb_documents`` (FK -> ``banks(bank_id)`` ON DELETE CASCADE)
- ``kb_passages`` (FK -> ``kb_documents(bank_id, document_id)`` ON DELETE CASCADE)
- ``kb_passage_vectors`` (FK -> ``kb_passages(bank_id, document_id, idx)`` ON DELETE CASCADE)
- ``kb_staged_passages`` (FK -> ``kb_documents(bank_id, document_id)`` ON DELETE CASCADE)
- ``kb_staged_passage_vectors`` (FK -> ``kb_staged_passages(bank_id, document_id, idx)`` ON DELETE CASCADE)

All foreign keys are leading prefixes of their composite primary keys, so every
foreign key is indexed (zero unindexed FKs). Also creates ``idx_kb_docs_source``,
``idx_kb_docs_updated``, ``idx_kb_passages_bank``, the Oracle Text index
``idx_kb_passages_text`` (``CTXSYS.CONTEXT``, ``SYNC (ON COMMIT)``), and grants
``SELECT, INSERT, UPDATE, DELETE`` to every grantee holding ``INSERT`` on
``MEMORY_UNITS``.

Oracle-only: PostgreSQL skips both upgrade and downgrade cleanly.
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e3c4d5e6f7a8"
down_revision: str | Sequence[str] | None = "o2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE kb_documents (
        bank_id VARCHAR2(256) NOT NULL,
        document_id VARCHAR2(512) NOT NULL,
        filename VARCHAR2(1024),
        content_type VARCHAR2(256) NOT NULL,
        media_type VARCHAR2(64) NOT NULL,
        language VARCHAR2(32),
        size_bytes NUMBER(19) DEFAULT 0 NOT NULL,
        storage_key VARCHAR2(2048),
        content_hash VARCHAR2(128) NOT NULL,
        status VARCHAR2(32) DEFAULT 'pending' NOT NULL,
        error CLOB,
        passage_count NUMBER(10) DEFAULT 0 NOT NULL,
        tags CLOB DEFAULT '[]' NOT NULL,
        source_document_id VARCHAR2(512),
        content_tokens NUMBER(19) DEFAULT 0 NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        PRIMARY KEY (bank_id, document_id),
        CONSTRAINT fk_kb_docs_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE TABLE kb_passages (
        bank_id VARCHAR2(256) NOT NULL,
        document_id VARCHAR2(512) NOT NULL,
        idx NUMBER(10) NOT NULL,
        modality VARCHAR2(64) NOT NULL,
        text CLOB,
        page NUMBER(10),
        start_ms NUMBER(10),
        end_ms NUMBER(10),
        media_key VARCHAR2(2048),
        embedding VECTOR NOT NULL,
        PRIMARY KEY (bank_id, document_id, idx),
        CONSTRAINT fk_kb_passages_doc FOREIGN KEY (bank_id, document_id)
            REFERENCES kb_documents(bank_id, document_id) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE TABLE kb_passage_vectors (
        bank_id VARCHAR2(256) NOT NULL,
        document_id VARCHAR2(512) NOT NULL,
        idx NUMBER(10) NOT NULL,
        role VARCHAR2(64) NOT NULL,
        n NUMBER(10) NOT NULL,
        embedding VECTOR NOT NULL,
        PRIMARY KEY (bank_id, document_id, idx, role, n),
        CONSTRAINT fk_kb_pvec_passage FOREIGN KEY (bank_id, document_id, idx)
            REFERENCES kb_passages(bank_id, document_id, idx) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE TABLE kb_staged_passages (
        bank_id VARCHAR2(256) NOT NULL,
        document_id VARCHAR2(512) NOT NULL,
        idx NUMBER(10) NOT NULL,
        modality VARCHAR2(64) NOT NULL,
        text CLOB,
        page NUMBER(10),
        start_ms NUMBER(10),
        end_ms NUMBER(10),
        media_key VARCHAR2(2048),
        embedding VECTOR NOT NULL,
        PRIMARY KEY (bank_id, document_id, idx),
        CONSTRAINT fk_kb_staged_doc FOREIGN KEY (bank_id, document_id)
            REFERENCES kb_documents(bank_id, document_id) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE TABLE kb_staged_passage_vectors (
        bank_id VARCHAR2(256) NOT NULL,
        document_id VARCHAR2(512) NOT NULL,
        idx NUMBER(10) NOT NULL,
        role VARCHAR2(64) NOT NULL,
        n NUMBER(10) NOT NULL,
        embedding VECTOR NOT NULL,
        PRIMARY KEY (bank_id, document_id, idx, role, n),
        CONSTRAINT fk_kb_spvec_staged FOREIGN KEY (bank_id, document_id, idx)
            REFERENCES kb_staged_passages(bank_id, document_id, idx) ON DELETE CASCADE
    )
    """.strip(),
)

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX idx_kb_docs_source ON kb_documents (bank_id, source_document_id)",
    "CREATE INDEX idx_kb_docs_updated ON kb_documents (bank_id, updated_at DESC)",
    "CREATE INDEX idx_kb_passages_bank ON kb_passages (bank_id)",
)

_TEXT_INDEX = "CREATE INDEX idx_kb_passages_text ON kb_passages (text) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC (ON COMMIT)')"

_GRANT_KB_TABLES_PLSQL = """
BEGIN
  FOR r IN (
    SELECT DISTINCT grantee
    FROM user_tab_privs_made
    WHERE table_name = 'MEMORY_UNITS'
      AND privilege = 'INSERT'
      AND grantee NOT IN ('PUBLIC')
  ) LOOP
    FOR t IN (
      SELECT column_value AS tbl
      FROM TABLE(SYS.ODCIVARCHAR2LIST(
        'KB_DOCUMENTS',
        'KB_PASSAGES',
        'KB_PASSAGE_VECTORS',
        'KB_STAGED_PASSAGES',
        'KB_STAGED_PASSAGE_VECTORS'
      ))
    ) LOOP
      BEGIN
        EXECUTE IMMEDIATE
          'GRANT SELECT, INSERT, UPDATE, DELETE ON ' || t.tbl || ' TO "' || REPLACE(r.grantee, '"', '""') || '"';
      EXCEPTION
        WHEN OTHERS THEN
          IF SQLCODE IN (-1031, -1917, -942, -905, -922) THEN
            NULL;
          ELSE
            RAISE;
          END IF;
      END;
    END LOOP;
  END LOOP;
END;
"""


def _oracle_exec_ignoring(sql: str, *codes: int) -> None:
    """Run one DDL statement, swallowing the given ORA- codes so a rerun is a no-op."""
    tolerated = " OR ".join(f"SQLCODE = {code}" for code in codes)
    block = (
        f"BEGIN EXECUTE IMMEDIATE :stmt; EXCEPTION WHEN OTHERS THEN IF {tolerated} THEN NULL; ELSE RAISE; END IF; END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql})


def _oracle_upgrade() -> None:
    # -1430: column already exists.
    _oracle_exec_ignoring("ALTER TABLE banks ADD (kind VARCHAR2(32) DEFAULT 'memory' NOT NULL)", -1430)

    # -955: name already used by an existing object.
    for ddl in _TABLES:
        _oracle_exec_ignoring(ddl, -955)

    # -955: name used; -1408: column list already indexed.
    for idx_sql in _INDEXES:
        _oracle_exec_ignoring(idx_sql, -955, -1408)

    # -955: name used; -1408: already indexed (fail loudly on ORA-29855 so missing Oracle Text is never silent).
    _oracle_exec_ignoring(_TEXT_INDEX, -955, -1408)
    op.get_bind().exec_driver_sql(_GRANT_KB_TABLES_PLSQL)


def _oracle_downgrade() -> None:
    for tbl in (
        "kb_staged_passage_vectors",
        "kb_staged_passages",
        "kb_passage_vectors",
        "kb_passages",
        "kb_documents",
    ):
        _oracle_exec_ignoring(f"DROP TABLE {tbl} CASCADE CONSTRAINTS PURGE", -942)


def upgrade() -> None:
    run_for_dialect(oracle=_oracle_upgrade)  # PG slot intentionally absent


def downgrade() -> None:
    run_for_dialect(oracle=_oracle_downgrade)
