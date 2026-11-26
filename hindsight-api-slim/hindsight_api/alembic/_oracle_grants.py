"""Runtime grants shared by the Oracle migrations.

The API may connect as a runtime user that holds only DML on the schema owner's tables
(``HINDSIGHT_API_MIGRATION_DATABASE_URL`` runs the DDL as the owner or another privileged
user). Every user that already holds the matching privilege on ``MEMORY_UNITS`` in the
target schema is treated as such a runtime user and receives the same access to the objects
a migration creates.

Grantees are read from ``dba_tab_privs`` (falling back to ``all_tab_privs`` when the
migration user cannot see the DBA views), scoped to ``CURRENT_SCHEMA`` rather than to the
session user: when the migration user is not the schema owner, ``user_tab_privs_made``
lists that user's own grants and misses the owner's runtime grantees. The session user and
the schema owner are excluded. Each block is idempotent, so a later migration can re-run it
to repair schemas migrated before the lookup was fixed (``f7a8b9c0d1e2``).
"""

GRANT_KB_TABLES_PLSQL = """
DECLARE
  l_rc SYS_REFCURSOR;
  l_grantee VARCHAR2(128);
BEGIN
  BEGIN
    OPEN l_rc FOR
      'SELECT DISTINCT grantee FROM dba_tab_privs '
      || 'WHERE owner = SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA'') '
      || 'AND table_name = ''MEMORY_UNITS'' AND privilege = ''INSERT'' '
      || 'AND grantee NOT IN (''PUBLIC'', SYS_CONTEXT(''USERENV'', ''SESSION_USER''), SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA''))';
  EXCEPTION
    WHEN OTHERS THEN
      IF SQLCODE IN (-942, -1031) THEN
        OPEN l_rc FOR
          'SELECT DISTINCT grantee FROM all_tab_privs '
          || 'WHERE table_schema = SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA'') '
          || 'AND table_name = ''MEMORY_UNITS'' AND privilege = ''INSERT'' '
          || 'AND grantee NOT IN (''PUBLIC'', SYS_CONTEXT(''USERENV'', ''SESSION_USER''), SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA''))';
      ELSE
        RAISE;
      END IF;
  END;
  LOOP
    FETCH l_rc INTO l_grantee;
    EXIT WHEN l_rc%NOTFOUND;
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
          'GRANT SELECT, INSERT, UPDATE, DELETE ON ' || t.tbl || ' TO "' || REPLACE(l_grantee, '"', '""') || '"';
      EXCEPTION
        WHEN OTHERS THEN
          IF SQLCODE IN (-1031, -1749, -1917, -942, -905, -922) THEN
            NULL;
          ELSE
            RAISE;
          END IF;
      END;
    END LOOP;
  END LOOP;
  CLOSE l_rc;
EXCEPTION
  WHEN OTHERS THEN
    IF l_rc%ISOPEN THEN
      CLOSE l_rc;
    END IF;
    RAISE;
END;
"""

GRANT_PROPERTY_GRAPH_PLSQL = """
DECLARE
  l_rc SYS_REFCURSOR;
  l_grantee VARCHAR2(128);
BEGIN
  BEGIN
    OPEN l_rc FOR
      'SELECT DISTINCT grantee FROM dba_tab_privs '
      || 'WHERE owner = SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA'') '
      || 'AND table_name = ''MEMORY_UNITS'' AND privilege = ''SELECT'' '
      || 'AND grantee NOT IN (''PUBLIC'', SYS_CONTEXT(''USERENV'', ''SESSION_USER''), SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA''))';
  EXCEPTION
    WHEN OTHERS THEN
      IF SQLCODE IN (-942, -1031) THEN
        OPEN l_rc FOR
          'SELECT DISTINCT grantee FROM all_tab_privs '
          || 'WHERE table_schema = SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA'') '
          || 'AND table_name = ''MEMORY_UNITS'' AND privilege = ''SELECT'' '
          || 'AND grantee NOT IN (''PUBLIC'', SYS_CONTEXT(''USERENV'', ''SESSION_USER''), SYS_CONTEXT(''USERENV'', ''CURRENT_SCHEMA''))';
      ELSE
        RAISE;
      END IF;
  END;
  LOOP
    FETCH l_rc INTO l_grantee;
    EXIT WHEN l_rc%NOTFOUND;
    BEGIN
      EXECUTE IMMEDIATE
        'GRANT READ ON PROPERTY GRAPH hindsight_memory_graph TO "' || REPLACE(l_grantee, '"', '""') || '"';
    EXCEPTION
      WHEN OTHERS THEN
        -- -42421: property graph does not exist (CREATE PROPERTY GRAPH was not granted, so o2b3c4d5e6f7 skipped it).
        IF SQLCODE IN (-1031, -1749, -1917, -42404, -42421, -905, -922) THEN
          NULL;
        ELSE
          RAISE;
        END IF;
    END;
  END LOOP;
  CLOSE l_rc;
EXCEPTION
  WHEN OTHERS THEN
    IF l_rc%ISOPEN THEN
      CLOSE l_rc;
    END IF;
    RAISE;
END;
"""
