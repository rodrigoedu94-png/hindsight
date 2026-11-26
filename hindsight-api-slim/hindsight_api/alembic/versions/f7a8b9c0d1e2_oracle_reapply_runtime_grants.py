"""Re-apply the Oracle runtime grants with the CURRENT_SCHEMA-scoped grantee lookup.

Revision ID: f7a8b9c0d1e2
Revises: e3c4d5e6f7a8
Create Date: 2026-11-26

``o2b3c4d5e6f7`` (property graph) and ``e3c4d5e6f7a8`` (knowledge-bank tables) forward
their privileges to the runtime users holding privileges on ``MEMORY_UNITS``. They first
looked those users up in ``user_tab_privs_made``, which lists the *session* user's grants:
when ``HINDSIGHT_API_MIGRATION_DATABASE_URL`` connects as a user other than the schema
owner, it found no grantee, and the runtime user was left without ``READ`` on
``hindsight_memory_graph`` and without DML on the ``kb_*`` tables. The lookup is now fixed
in both revisions, but a schema that already recorded them never runs them again, so this
revision runs both grant blocks once more. Both are idempotent: re-granting a held privilege
is a no-op, and a missing property graph is tolerated.

Oracle-only: PostgreSQL skips both upgrade and downgrade. Downgrade revokes nothing, since
the earlier revisions intend these grants to exist.
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.alembic._oracle_grants import GRANT_KB_TABLES_PLSQL, GRANT_PROPERTY_GRAPH_PLSQL

revision: str = "f7a8b9c0d1e2"
down_revision: str | Sequence[str] | None = "e3c4d5e6f7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _oracle_upgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql(GRANT_PROPERTY_GRAPH_PLSQL)
    bind.exec_driver_sql(GRANT_KB_TABLES_PLSQL)


def upgrade() -> None:
    run_for_dialect(oracle=_oracle_upgrade)  # PG slot intentionally absent


def downgrade() -> None:
    run_for_dialect()  # nothing to undo on either dialect
