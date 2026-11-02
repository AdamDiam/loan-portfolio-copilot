"""The sandbox must allow analytics and block anything that writes or escapes."""
import pytest

from copilot.sql.safe_executor import QueryExecutionError, SafeExecutor, UnsafeQueryError, cte_names


@pytest.fixture()
def ex(small_db):
    return SafeExecutor(small_db, max_rows=50, timeout_s=2)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM loans",
        "```sql\nSELECT region, COUNT(*) FROM v_loan_overview GROUP BY region;\n```",
        "WITH x AS (SELECT loan_id FROM loans) SELECT COUNT(*) FROM x",
        "WITH a(n) AS (SELECT 1), b AS (SELECT n FROM a) SELECT COUNT(*) FROM b",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 5) SELECT x FROM c",
        "-- a comment\nSELECT strftime('%Y', origination_date) AS y, COUNT(*) FROM loans GROUP BY y",
    ],
)
def test_allows_read_only_analytics(ex, sql):
    assert ex.run(sql).df is not None


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM loans",
        "UPDATE loans SET is_npe = 0",
        "DROP TABLE loans",
        "INSERT INTO borrowers VALUES ('x','Individual','Attica',NULL,NULL,'2020-01-01')",
        "WITH x AS (SELECT 1) DELETE FROM loans",
        "SELECT 1; DROP TABLE loans",
        "SELECT * FROM sqlite_master",
        "WITH sqlite_master AS (SELECT 1) SELECT * FROM main.sqlite_master",
        "SELECT * FROM pragma_table_info('loans')",
        "ATTACH DATABASE 'other.db' AS o",
        "PRAGMA writable_schema = 1",
        "SELECT load_extension('evil')",
    ],
)
def test_blocks_writes_and_escapes(ex, sql):
    with pytest.raises(UnsafeQueryError):
        ex.run(sql)


def test_database_is_unchanged_after_attacks(ex):
    before = ex.run("SELECT COUNT(*) AS n FROM loans").df.n[0]
    for sql in ["DELETE FROM loans", "WITH x AS (SELECT 1) DELETE FROM loans"]:
        with pytest.raises(UnsafeQueryError):
            ex.run(sql)
    assert ex.run("SELECT COUNT(*) AS n FROM loans").df.n[0] == before


def test_row_cap_and_truncation_flag(ex):
    res = ex.run("SELECT loan_id FROM loans")
    assert len(res.df) == 50 and res.truncated


def test_runaway_query_times_out(ex):
    with pytest.raises(QueryExecutionError, match="time limit"):
        ex.run("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r")


def test_bad_column_is_an_execution_error(ex):
    with pytest.raises(QueryExecutionError, match="no such column"):
        ex.run("SELECT nonexistent FROM loans")


def test_cte_name_extraction():
    assert cte_names("WITH a AS (SELECT 1), b(x) AS (SELECT 2) SELECT * FROM a, b") == {"a", "b"}
    assert cte_names("WITH sqlite_master AS (SELECT 1) SELECT 1") == set()
