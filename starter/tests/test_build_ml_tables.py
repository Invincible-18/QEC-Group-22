import psycopg
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.stages.build_ml_tables import run_analyses


@pytest.fixture
def connection():
    try:
        connection = postgres_connection(Settings.from_environment())
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    connection.autocommit = True
    yield connection
    connection.close()


def test_every_query_file_becomes_a_csv(connection, tmp_path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "q9_example.sql").write_text("SELECT 1 AS answer, 'x' AS label")
    (sql_dir / "helper.sql").write_text("SELECT 2")  # not q*.sql, so not an analysis

    written = run_analyses(connection, tmp_path / "out", sql_dir=sql_dir)

    assert [path.name for path in written] == ["q9_example.csv"]
    assert (tmp_path / "out" / "q9_example.csv").read_text().splitlines() == ["answer,label", "1,x"]
