import pytest
from src.features.four_factors import calculate_possessions, calculate_pace, calculate_four_factors

def test_possessions_calculation():
    # Standard values
    poss = calculate_possessions(
        fga=88, fta=22, oreb=10, tov=14,
        opp_fga=85, opp_fta=20, opp_oreb=9, opp_tov=13
    )
    # team_poss = 88 + 0.44*22 - 10 + 14 = 101.68
    # opp_poss = 85 + 0.44*20 - 9 + 13 = 97.8
    # poss = 0.5 * (101.68 + 97.8) = 99.74
    assert round(poss, 2) == 99.74

def test_pace_calculation():
    poss = 100.0
    pace = calculate_pace(poss, minutes=240.0)
    assert pace == 100.0

def test_four_factors_calculation():
    team = {
        "fgm": 42, "fga": 88, "fg3m": 14, "fta": 20,
        "oreb": 10, "dreb": 35, "tov": 12, "pts": 110, "min": 240
    }
    opp = {
        "fgm": 38, "fga": 85, "fg3m": 10, "fta": 18,
        "oreb": 8, "dreb": 32, "tov": 14, "pts": 98, "min": 240
    }
    
    factors = calculate_four_factors(team, opp)
    
    # eFG% = (42 + 0.5 * 14) / 88 = 49 / 88 = 0.5568
    assert factors["efg_pct"] == pytest.approx(0.5568, abs=1e-3)
    
    # TOV% = 12 / (88 + 0.44*20 + 12) = 12 / 108.8 = 0.1103
    assert factors["tov_pct"] == pytest.approx(0.1103, abs=1e-3)
    
    # OREB% = 10 / (10 + 32) = 10 / 42 = 0.2381
    assert factors["orb_pct"] == pytest.approx(0.2381, abs=1e-3)
    
    # FTR = 20 / 88 = 0.2273
    assert factors["ftr"] == pytest.approx(0.2273, abs=1e-3)
    
    # Net rating should be positive since PTS > Opp PTS
    assert factors["net_rating"] > 0