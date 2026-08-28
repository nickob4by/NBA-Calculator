import pytest
from src.betting.ledger import BankrollLedger

def test_ledger_flow():
    # 1. Reset
    BankrollLedger.reset_bankroll(1200.0)
    assert BankrollLedger.get_current_balance() == 1200.0

    # 2. Deposit ₱500
    bal = BankrollLedger.deposit(500.0, "Top-up")
    assert bal == 1700.0

    # 3. Withdraw ₱200
    bal = BankrollLedger.withdraw(200.0, "Cashout")
    assert bal == 1500.0

    # 4. Record Win: Bet ₱50 at 2.00 odds -> Profit +₱50
    res_win = BankrollLedger.record_bet("nba", "BOS", 50.0, 2.00, is_win=True)
    assert res_win["pnl"] == 50.0
    assert BankrollLedger.get_current_balance() == 1550.0

    # 5. Record Loss: Bet ₱50 at 1.90 odds -> Loss -₱50
    res_loss = BankrollLedger.record_bet("mlb", "NYY", 50.0, 1.90, is_win=False)
    assert res_loss["pnl"] == -50.0
    assert BankrollLedger.get_current_balance() == 1500.0

    # 6. Verify metrics
    metrics = BankrollLedger.get_ledger_metrics()
    assert metrics["current_balance"] == 1500.0
    assert metrics["total_deposits"] == 500.0
    assert metrics["total_withdrawals"] == 200.0
    assert metrics["total_bets"] == 2
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == 50.0
    assert metrics["total_staked"] == 100.0
    assert metrics["net_betting_pnl"] == 0.0

    # Cleanup reset
    BankrollLedger.reset_bankroll(1200.0)