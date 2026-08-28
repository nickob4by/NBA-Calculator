import pytest
from src.features.mlb_sabermetrics import calculate_pythagorean_win_pct, calculate_sabermetrics

def test_pythagorean_win_pct():
    # 5 runs scored, 4 runs allowed -> expected win % ~ 60%
    pythag = calculate_pythagorean_win_pct(5.0, 4.0, exponent=1.83)
    assert 0.58 < pythag < 0.63

    # Equal runs -> 50%
    pythag_even = calculate_pythagorean_win_pct(4.5, 4.5)
    assert pythag_even == pytest.approx(0.50, abs=1e-3)

def test_calculate_sabermetrics():
    team = {"runs": 6, "hits": 10, "hr": 2, "bb": 4, "so": 6, "ip": 9.0}
    opp = {"runs": 3, "hits": 6, "hr": 1, "bb": 2, "so": 9, "ip": 9.0}

    stats = calculate_sabermetrics(team, opp)
    
    assert stats["runs"] == 6.0
    assert stats["opp_runs"] == 3.0
    assert stats["pythag_win_pct"] > 0.65
    assert stats["ops"] > 0.70
    assert stats["fip_proxy"] > 0
    assert stats["whip"] == pytest.approx((2 + 6) / 9.0, abs=1e-2)