import pytest
from src.db.database import db
from src.db.github_sync import GitHubDataSync

def test_github_sync_export_and_import():
    db.set_setting('sync_test_key', 'test_value')
    counts = GitHubDataSync.export_data_snapshot()
    assert 'settings' in counts
    assert 'transactions' in counts
    assert 'bets' in counts

    sync_dir = GitHubDataSync.get_sync_dir()
    assert (sync_dir / 'app_settings.json').exists()
    assert (sync_dir / 'bankroll_transactions.json').exists()
    assert (sync_dir / 'simulation_bets.json').exists()

    restored = GitHubDataSync.import_data_snapshot_if_exists()
    assert restored is True
    assert db.get_setting('sync_test_key') == 'test_value'
