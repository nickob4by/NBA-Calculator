import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
import config

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

class Database:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with self.get_connection() as conn:
            self._migrate(conn)
            conn.executescript(schema_sql)

    def _migrate(self, conn):
        """Ensures any new multi-sport and player columns are added if tables already existed."""
        tables_to_migrate = [
            ("games", "sport", "TEXT DEFAULT 'nba'"),
            ("team_game_logs", "sport", "TEXT DEFAULT 'nba'"),
            ("team_game_logs", "runs", "INTEGER"),
            ("team_game_logs", "hits", "INTEGER"),
            ("team_game_logs", "errors", "INTEGER"),
            ("team_game_logs", "hr", "INTEGER"),
            ("team_game_logs", "rbi", "INTEGER"),
            ("team_game_logs", "bb", "INTEGER"),
            ("team_game_logs", "so", "INTEGER"),
            ("team_game_logs", "lob", "INTEGER"),
            ("team_game_logs", "ip", "REAL"),
            ("team_game_logs", "er", "INTEGER"),
            ("team_game_logs", "starting_pitcher_id", "INTEGER"),
            ("team_game_logs", "starting_pitcher_fip", "REAL"),
            ("team_advanced_stats", "sport", "TEXT DEFAULT 'nba'"),
            ("team_advanced_stats", "pythag_win_pct", "REAL"),
            ("team_advanced_stats", "obp", "REAL"),
            ("team_advanced_stats", "slg", "REAL"),
            ("team_advanced_stats", "ops", "REAL"),
            ("team_advanced_stats", "iso", "REAL"),
            ("team_advanced_stats", "woba_proxy", "REAL"),
            ("team_advanced_stats", "fip_proxy", "REAL"),
            ("team_advanced_stats", "whip", "REAL"),
            ("team_advanced_stats", "k_per_9", "REAL"),
            ("team_advanced_stats", "bb_per_9", "REAL"),
            ("odds", "sport", "TEXT DEFAULT 'nba'"),
            ("predictions", "sport", "TEXT DEFAULT 'nba'"),
            ("bets", "sport", "TEXT DEFAULT 'nba'")
        ]
        cursor = conn.cursor()
        for table, col, col_type in tables_to_migrate:
            try:
                cursor.execute(f"PRAGMA table_info({table});")
                existing_cols = [r["name"] for r in cursor.fetchall()]
                if existing_cols and col not in existing_cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};")
            except Exception:
                pass

    def execute(self, query: str, params: tuple = ()):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor

    def executemany(self, query: str, seq_of_params):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, seq_of_params)
            return cursor

    def fetch_all(self, query: str, params: tuple = ()):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def fetch_one(self, query: str, params: tuple = ()):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetch_df(self, query: str, params: tuple = ()) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def insert_df(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

db = Database()