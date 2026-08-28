import pytest
from src.betting.kelly import calculate_kelly_fractional, size_bet

def test_positive_ev_kelly_sizing():
    # 60% win prob with even money (decimal 2.00)
    # Full kelly: (1.0 * 0.6 - 0.4) / 1.0 = 0.20 (20%)
    # Fractional kelly (0.25): 0.20 * 0.25 = 0.05 (5.0%) -> capped at max_bankroll_pct=0.04 (4.0%)
    f_star = calculate_kelly_fractional(model_prob=0.60, decimal_odds=2.00, kelly_multiplier=0.25, max_bankroll_pct=0.04)
    assert f_star == 0.04

    # 15% Kelly with no capping
    f_star_15 = calculate_kelly_fractional(model_prob=0.60, decimal_odds=2.00, kelly_multiplier=0.15, max_bankroll_pct=0.10)
    assert f_star_15 == pytest.approx(0.03, abs=1e-4)

def test_negative_ev_kelly_zero():
    # 45% win prob with even money (2.00) -> Negative EV
    f_star = calculate_kelly_fractional(model_prob=0.45, decimal_odds=2.00)
    assert f_star == 0.0

def test_size_bet_dollars():
    res = size_bet(model_prob=0.60, decimal_odds=2.00, bankroll=10000.0, kelly_multiplier=0.15, max_bankroll_pct=0.05)
    assert res["is_actionable"] is True
    assert res["stake"] == 300.0 # 3.0% of $10,000