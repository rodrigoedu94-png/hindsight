import unittest

from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle


class OracleDateTruncRewriteTests(unittest.TestCase):
    def test_simple_column_byte_identical(self):
        q = "SELECT date_trunc('day', created_at AT TIME ZONE 'UTC') AS bucket FROM memory_units"
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(created_at AS DATE), 'DD') AS bucket FROM memory_units",
        )

    def test_coalesce_mentioned_at_6787(self):
        q = "SELECT date_trunc('day', COALESCE(mentioned_at, created_at) AT TIME ZONE 'UTC') AS bucket FROM memory_units"
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(COALESCE(mentioned_at, created_at) AS DATE), 'DD') AS bucket FROM memory_units",
        )

    def test_nested_functions_and_case_insensitive(self):
        q = "SELECT DATE_TRUNC('hour', COALESCE(NULLIF(occurred_start, created_at), created_at)) AS b FROM memory_units"
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(COALESCE(NULLIF(occurred_start, created_at), created_at) AS DATE), 'HH24') AS b FROM memory_units",
        )

    def test_multiple_calls_in_one_query(self):
        q = (
            "SELECT date_trunc('day', COALESCE(mentioned_at, created_at)) AS d, "
            "date_trunc('month', created_at AT TIME ZONE 'UTC') AS m FROM memory_units"
        )
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(COALESCE(mentioned_at, created_at) AS DATE), 'DD') AS d, "
            "TRUNC(CAST(created_at AS DATE), 'MM') AS m FROM memory_units",
        )

    def test_quoted_parens_and_commas_ignored(self):
        q = "SELECT date_trunc('day', COALESCE(NULLIF(text, 'a),b'), created_at)) AS b FROM t"
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(COALESCE(NULLIF(text, 'a),b'), created_at) AS DATE), 'DD') AS b FROM t",
        )

    def test_unknown_interval_falls_back_to_dd(self):
        q = "SELECT date_trunc('quarter', created_at) AS b FROM t"
        self.assertEqual(
            _rewrite_pg_to_oracle(q).query,
            "SELECT TRUNC(CAST(created_at AS DATE), 'DD') AS b FROM t",
        )


if __name__ == "__main__":
    unittest.main()
