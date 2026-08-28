import pytest
import sqlite3
import pandas as pd
from src.db.database import Database

def test_database_initialization(tmp_path):
    test_db_path = tmp_path / "test.db"
    test_db = Database(db_path=test_db_path)
    
    tables = [r["name"] for r in test_db.fetch_all("SELECT name FROM sqlite_master WHERE type=?", ("table",))]
    assert "games" in tables
    assert "team_game_logs" in tables
    assert "team_advanced_stats" in tables
    assert "odds" in tables

def test_database_crud(tmp_path):
    test_db_path = tmp_path / "test_crud.db"
    test_db = Database(db_path=test_db_path)
    
    test_db.execute(
        "INSERT INTO games (game_id, season, game_date, home_team_id, away_team_id, home_pts, away_pts, point_margin, total_points, home_win) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("001", "2024-25", "2025-01-01", 1610612738, 1610612747, 110, 100, 10, 210, 1)
    )
    
    row = test_db.fetch_one("SELECT * FROM games WHERE game_id=?", ("001",))
    assert row is not None
    assert row["home_pts"] == 110
    assert row["point_margin"] == 10