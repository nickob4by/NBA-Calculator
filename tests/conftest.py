import pytest
import config
from src.db.database import Database
import src.db.database as db_mod
import src.betting.ledger as ledger_mod
import src.db.github_sync as sync_mod

@pytest.fixture(autouse=True, scope='session')
def isolate_test_database(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_dbs')
    test_db_file = temp_dir / 'test_app.db'
    
    test_db = Database(db_path=test_db_file)
    original_db = db_mod.db
    
    db_mod.db = test_db
    ledger_mod.db = test_db
    sync_mod.db = test_db
    
    with test_db.get_connection() as conn:
        with original_db.get_connection() as orig_conn:
            games_df = orig_conn.execute('SELECT * FROM games LIMIT 200').fetchall()
            logs_df = orig_conn.execute('SELECT * FROM team_game_logs LIMIT 400').fetchall()
            if games_df:
                cols = games_df[0].keys()
                placeholders = ', '.join(['?'] * len(cols))
                col_names = ', '.join(cols)
                conn.executemany(f'INSERT OR IGNORE INTO games ({col_names}) VALUES ({placeholders})', [tuple(r) for r in games_df])
            if logs_df:
                cols = logs_df[0].keys()
                placeholders = ', '.join(['?'] * len(cols))
                col_names = ', '.join(cols)
                conn.executemany(f'INSERT OR IGNORE INTO team_game_logs ({col_names}) VALUES ({placeholders})', [tuple(r) for r in logs_df])
                
    yield
    
    db_mod.db = original_db
    ledger_mod.db = original_db
    sync_mod.db = original_db
