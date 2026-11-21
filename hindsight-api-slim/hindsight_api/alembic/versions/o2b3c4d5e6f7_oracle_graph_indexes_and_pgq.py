"""Add Oracle memory_links composite indexes and native SQL/PGQ property graph

Revision ID: o2b3c4d5e6f7
Revises: kbtok1a2b3c4
Create Date: 2026-11-21

Optimizes Oracle 26ai link expansion and enables native SQL/PGQ graph pattern
matching over the Hindsight relational/vector schema:

1. Adds composite indexes on ``memory_links`` (``idx_ml_from_type_weight`` and
   ``idx_ml_to_type_weight``) matching PostgreSQL's index coverage so semantic
   and causal expansion scan only the targeted link_type rows ordered by weight.
2. Creates the zero-copy SQL/PGQ property graph ``hindsight_memory_graph`` over
   ``memory_units`` and ``entities`` (vertices) and ``memory_links``,
   ``unit_entities``, ``entity_cooccurrences``, and ``observation_sources``
   (edges) with ``TRUSTED MODE``.
3. Grants ``READ ON PROPERTY GRAPH hindsight_memory_graph`` automatically to
   every runtime user that holds ``SELECT`` on ``memory_units`` (e.g.
   ``HS26_RUNTIME``).

Oracle-only: PostgreSQL skips both steps cleanly.
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "o2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "kbtok1a2b3c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LINK_INDEXES = (
    "CREATE INDEX idx_ml_from_type_weight ON memory_links (from_unit_id, link_type, weight DESC)",
    "CREATE INDEX idx_ml_to_type_weight ON memory_links (to_unit_id, link_type, weight DESC)",
)

_CREATE_PROPERTY_GRAPH_SQL = """
CREATE PROPERTY GRAPH hindsight_memory_graph
  VERTEX TABLES (
    memory_units
      KEY (id)
      LABEL memory_unit
      PROPERTIES (id, bank_id, fact_type, event_date, occurred_start, occurred_end, mentioned_at, updated_at, proof_count),
    entities
      KEY (id)
      LABEL entity
      PROPERTIES (id, bank_id, canonical_name, mention_count)
  )
  EDGE TABLES (
    memory_links
      KEY (from_unit_id, to_unit_id, link_type)
      SOURCE KEY (from_unit_id) REFERENCES memory_units (id)
      DESTINATION KEY (to_unit_id) REFERENCES memory_units (id)
      LABEL memory_link
      PROPERTIES (link_type, weight, bank_id, created_at),
    unit_entities
      KEY (unit_id, entity_id)
      SOURCE KEY (unit_id) REFERENCES memory_units (id)
      DESTINATION KEY (entity_id) REFERENCES entities (id)
      LABEL mentions
      NO PROPERTIES,
    entity_cooccurrences
      KEY (entity_id_1, entity_id_2)
      SOURCE KEY (entity_id_1) REFERENCES entities (id)
      DESTINATION KEY (entity_id_2) REFERENCES entities (id)
      LABEL cooccurs_with
      PROPERTIES (cooccurrence_count, last_cooccurred),
    observation_sources
      KEY (observation_id, source_id)
      SOURCE KEY (observation_id) REFERENCES memory_units (id)
      DESTINATION KEY (source_id) REFERENCES memory_units (id)
      LABEL derived_from
      NO PROPERTIES
  )
  OPTIONS (TRUSTED MODE)
"""

_GRANT_PROPERTY_GRAPH_PLSQL = """
BEGIN
  FOR r IN (
    SELECT DISTINCT grantee
    FROM user_tab_privs_made
    WHERE table_name = 'MEMORY_UNITS'
      AND privilege = 'SELECT'
      AND grantee NOT IN ('PUBLIC')
  ) LOOP
    BEGIN
      EXECUTE IMMEDIATE
        'GRANT READ ON PROPERTY GRAPH hindsight_memory_graph TO "' || REPLACE(r.grantee, '"', '""') || '"';
    EXCEPTION
      WHEN OTHERS THEN
        IF SQLCODE IN (-1031, -1917, -42404, -905, -922) THEN
          NULL;
        ELSE
          RAISE;
        END IF;
    END;
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
    # -955: name already used; -1408: column list already indexed.
    for sql in _LINK_INDEXES:
        _oracle_exec_ignoring(sql, -955, -1408)

    # -955: name used; -42415: graph already exists; -1031: missing CREATE PROPERTY GRAPH priv;
    # -905 / -922 / -4044: Oracle versions prior to 23ai without SQL/PGQ syntax support.
    _oracle_exec_ignoring(_CREATE_PROPERTY_GRAPH_SQL, -955, -42415, -1031, -905, -922, -4044)
    op.get_bind().exec_driver_sql(_GRANT_PROPERTY_GRAPH_PLSQL)


def _oracle_downgrade() -> None:
    # -42404 / -942 / -905 / -922: property graph absent or unsupported.
    _oracle_exec_ignoring("DROP PROPERTY GRAPH hindsight_memory_graph", -42404, -942, -905, -922, -4044)
    for sql in _LINK_INDEXES:
        _oracle_exec_ignoring(f"DROP INDEX {sql.split()[2]}", -1418)


def upgrade() -> None:
    run_for_dialect(oracle=_oracle_upgrade)  # PG slot intentionally absent, see module docstring


def downgrade() -> None:
    run_for_dialect(oracle=_oracle_downgrade)
