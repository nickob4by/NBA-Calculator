import pytest
from src.betting.ev_engine import calculate_expected_value, calculate_edge, evaluate_moneyline_market, evaluate_spread_market

def test_calculate_expected_value():
    # 55% win on 2.0 odds: 0.55 * 1.0 - 0.45 * 1.0 = +0.10
    ev = calculate_expected_value(0.55, 2.00)
    assert ev == pytest.approx(0.10, abs=1e-4)

    # 45% win on 2.0 odds: 0.45 * 1.0 - 0.55 * 1.0 = -0.10
    ev_neg = calculate_expected_value(0.45, 2.00)
    assert ev_neg == pytest.approx(-0.10, abs=1e-4)

def test_evaluate_moneyline_market():
    # Model gives Home 65%, Market offers Home @ 1.80, Away @ 2.10
    res = evaluate_moneyline_market(0.65, 1.80, 2.10, min_edge=0.02)
    assert len(res["opportunities"]) >= 1
    assert res["opportunities"][0]["side"] == "home"
    assert res["opportunities"][0]["ev"] > 0