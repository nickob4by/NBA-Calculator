import pytest
import pandas as pd
import numpy as np
from src.features.rolling_metrics import compute_rolling_team_features

def test_strict_zero_lookahead_lag():
    # Construct synthetic team game logs with known points
    df = pd.DataFrame([
        {"team_id": 1, "game_id": "G1", "game_date": "2024-11-01", "pts": 100, "plus_minus": 10, "wl": "W", "possessions": 100, "pace": 100, "off_rating": 100, "def_rating": 90, "net_rating": 10, "efg_pct": 0.5, "tov_pct": 0.1, "orb_pct": 0.25, "ftr": 0.2, "opp_efg_pct": 0.45, "opp_tov_pct": 0.15, "opp_orb_pct": 0.2, "opp_ftr": 0.18, "fg_pct": 0.45, "fg3_pct": 0.35, "ft_pct": 0.8, "ast": 25, "reb": 45, "tov": 10},
        {"team_id": 1, "game_id": "G2", "game_date": "2024-11-03", "pts": 120, "plus_minus": 20, "wl": "W", "possessions": 100, "pace": 100, "off_rating": 120, "def_rating": 100, "net_rating": 20, "efg_pct": 0.6, "tov_pct": 0.1, "orb_pct": 0.25, "ftr": 0.2, "opp_efg_pct": 0.45, "opp_tov_pct": 0.15, "opp_orb_pct": 0.2, "opp_ftr": 0.18, "fg_pct": 0.55, "fg3_pct": 0.40, "ft_pct": 0.8, "ast": 30, "reb": 50, "tov": 10},
        {"team_id": 1, "game_id": "G3", "game_date": "2024-11-05", "pts": 80,  "plus_minus": -15, "wl": "L", "possessions": 100, "pace": 100, "off_rating": 80, "def_rating": 95, "net_rating": -15, "efg_pct": 0.4, "tov_pct": 0.2, "orb_pct": 0.20, "ftr": 0.15, "opp_efg_pct": 0.50, "opp_tov_pct": 0.10, "opp_orb_pct": 0.3, "opp_ftr": 0.25, "fg_pct": 0.35, "fg3_pct": 0.25, "ft_pct": 0.7, "ast": 18, "reb": 38, "tov": 18}
    ])

    res = compute_rolling_team_features(df, windows=[5])
    
    # Game 1 (first game): lagged rolling points should be NaN (or expanding min_periods=1 of nothing -> NaN)
    assert pd.isna(res.iloc[0]["roll_pts_w5"])
    
    # Game 2: lagged rolling points should be exactly Game 1's points (100.0), NOT including Game 2's points (120)
    assert res.iloc[1]["roll_pts_w5"] == 100.0
    
    # Game 3: lagged rolling points should be average of Game 1 & Game 2 ((100 + 120) / 2 = 110.0), NOT including Game 3 (80)
    assert res.iloc[2]["roll_pts_w5"] == 110.0