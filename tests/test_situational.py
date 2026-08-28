import pytest
import pandas as pd
from src.features.situational import haversine_distance, compute_situational_features

def test_haversine_distance():
    # Boston TD Garden (42.3662, -71.0621) to NY MSG (40.7505, -73.9934) ~ 190 miles
    dist = haversine_distance(42.3662, -71.0621, 40.7505, -73.9934)
    assert 180 < dist < 210

def test_situational_features_rest_and_b2b():
    df = pd.DataFrame([
        {"team_id": 1610612738, "opponent_id": 1610612747, "game_date": "2024-11-01", "is_home": 1},
        {"team_id": 1610612738, "opponent_id": 1610612752, "game_date": "2024-11-02", "is_home": 0}, # B2B game
        {"team_id": 1610612738, "opponent_id": 1610612748, "game_date": "2024-11-05", "is_home": 1}  # 3 days rest
    ])
    
    res = compute_situational_features(df)
    assert res.iloc[0]["is_b2b"] == 0
    assert res.iloc[1]["is_b2b"] == 1
    assert res.iloc[1]["rest_days"] == 1.0
    assert res.iloc[2]["is_b2b"] == 0
    assert res.iloc[2]["rest_days"] == 3.0