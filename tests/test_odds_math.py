import pytest
from src.betting.odds_math import american_to_decimal, decimal_to_american, remove_vig, remove_vig_shin, remove_vig_power

def test_american_to_decimal_conversion():
    assert american_to_decimal(100) == 2.00
    assert american_to_decimal(150) == 2.50
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-3)
    assert american_to_decimal(-200) == 1.50

def test_decimal_to_american_conversion():
    assert decimal_to_american(2.00) == 100
    assert decimal_to_american(2.50) == 150
    assert decimal_to_american(1.9091) == -110
    assert decimal_to_american(1.50) == -200

def test_vig_removal_sum_to_one():
    # Symmetric -110/-110 (1.91 / 1.91)
    p_h, p_a = remove_vig(1.9091, 1.9091, method="multiplicative")
    assert p_h == pytest.approx(0.5, abs=1e-3)
    assert p_a == pytest.approx(0.5, abs=1e-3)
    assert p_h + p_a == pytest.approx(1.0, abs=1e-6)

def test_vig_removal_shin_and_power():
    p_h_s, p_a_s = remove_vig_shin(1.50, 2.70)
    assert p_h_s + p_a_s == pytest.approx(1.0, abs=1e-4)
    assert p_h_s > p_a_s

    p_h_p, p_a_p = remove_vig_power(1.50, 2.70)
    assert p_h_p + p_a_p == pytest.approx(1.0, abs=1e-4)
    assert p_h_p > p_a_p