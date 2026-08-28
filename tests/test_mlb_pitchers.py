import pytest
import pandas as pd
import numpy as np
from src.features.mlb_pitcher_metrics import (
    calculate_pitcher_game_fip,
    calculate_sp_fip_differential,
    get_team_starters
)
from src.features.matchup_builder import build_upcoming_matchup
from src.db.database import db

def test_pitcher_fip_calculation():
    # 6 IP, 2 ER, 1 HR, 2 BB, 8 SO
    # FIP = ((13*1 + 3*2 - 2*8) / 6) + 3.15 = (3/6) + 3.15 = 3.65
    fip = calculate_pitcher_game_fip(ip=6.0, er=2, hr=1, bb=2, so=8)
    assert fip == pytest.approx(3.65, abs=0.05)

def test_sp_fip_differential():
    # Ace (2.80 FIP) vs Fifth starter (5.20 FIP)
    # Delta = Away FIP - Home FIP = 5.20 - 2.80 = +2.40 (Home has +2.40 run prevention advantage)
    delta = calculate_sp_fip_differential(home_sp_fip=2.80, away_sp_fip=5.20)
    assert delta == pytest.approx(2.40, abs=1e-2)

def test_get_team_starters():
    nyy_starters = get_team_starters(147)
    assert len(nyy_starters) >= 5
    assert nyy_starters[0]["name"] == "Gerrit Cole"
    assert nyy_starters[0]["fip"] < 3.0

def test_build_upcoming_matchup_with_pitchers():
    logs_df = db.fetch_df("SELECT * FROM team_game_logs WHERE sport='mlb' ORDER BY game_date, game_id")
    
    # Gerrit Cole (2.85 FIP) vs Taijuan Walker (5.20 FIP)
    matchup = build_upcoming_matchup(
        home_team_id=147,
        away_team_id=143,
        logs_df=logs_df,
        sport="mlb",
        home_sp_fip=2.85,
        away_sp_fip=5.20
    )
    
    assert "diff_sp_fip" in matchup.columns
    assert matchup["diff_sp_fip"].iloc[0] == pytest.approx(2.35, abs=1e-2)