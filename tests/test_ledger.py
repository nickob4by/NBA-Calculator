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
    assert metrics["bankroll_roi_pct"] == 0.0
    assert metrics["yield_pct"] == 0.0

    # 7. Test set_starting_balance adjustment
    # Starting balance was 1200, balance was 1500 (with +300 deposit-withdraw).
    # If starting balance changes to 2000, new balance should be 2300.
    new_bal = BankrollLedger.set_starting_balance(2000.0)
    assert new_bal == 2300.0
    metrics_rebase = BankrollLedger.get_ledger_metrics()
    assert metrics_rebase["initial_balance"] == 2000.0
    assert metrics_rebase["current_balance"] == 2300.0

    # Cleanup reset
    BankrollLedger.reset_bankroll(1200.0)

def test_simulation_betting_lifecycle():
    BankrollLedger.reset_bankroll(1200.0)
    BankrollLedger.clear_all_simulation_bets()

    # 1. Place simulation bet
    bet_id1 = BankrollLedger.place_simulation_bet(
        sport="mlb",
        matchup="Chicago Cubs vs Cincinnati Reds",
        team="Cincinnati Reds",
        stake=50.0,
        odds=2.10,
        model_prob=0.55,
        edge_pct=0.074
    )
    assert bet_id1 is not None

    bet_id2 = BankrollLedger.place_simulation_bet(
        sport="nba",
        matchup="Boston Celtics vs New York Knicks",
        team="New York Knicks",
        stake=40.0,
        odds=2.50,
        model_prob=0.45,
        edge_pct=0.05
    )
    assert bet_id2 is not None

    # Check pending
    pending_df = BankrollLedger.get_pending_simulation_bets()
    assert len(pending_df) == 2
    assert BankrollLedger.get_pending_stakes_total() == 90.0
    assert BankrollLedger.get_current_balance() == 1110.0 # 1200 - 90
    assert BankrollLedger.get_settled_balance() == 1200.0

    # 2. Resolve bet1 as WON (+₱55 profit)
    res1 = BankrollLedger.resolve_simulation_bet(bet_id1, is_win=True)
    assert res1["status"] == "WON"
    assert res1["pnl"] == 55.0
    assert BankrollLedger.get_settled_balance() == 1255.0
    assert BankrollLedger.get_current_balance() == 1215.0 # 1255 - 40 pending

    # Check pending again
    pending_df2 = BankrollLedger.get_pending_simulation_bets()
    assert len(pending_df2) == 1
    assert pending_df2.iloc[0]["id"] == bet_id2

    # 3. Resolve bet2 as LOST (-₱40)
    res2 = BankrollLedger.resolve_simulation_bet(bet_id2, is_win=False)
    assert res2["status"] == "LOST"
    assert res2["pnl"] == -40.0
    assert BankrollLedger.get_settled_balance() == 1215.0
    assert BankrollLedger.get_current_balance() == 1215.0 # 1215 - 0 pending

    # 4. Check no pending bets left
    pending_df3 = BankrollLedger.get_pending_simulation_bets()
    assert len(pending_df3) == 0

    # 5. Place and void bet
    bet_id3 = BankrollLedger.place_simulation_bet(
        sport="nba",
        matchup="Lakers vs Warriors",
        team="Lakers",
        stake=30.0,
        odds=1.90
    )
    assert BankrollLedger.get_current_balance() == 1185.0 # 1215 - 30
    assert BankrollLedger.void_simulation_bet(bet_id3) is True
    assert BankrollLedger.get_current_balance() == 1215.0 # Released back to 1215
    assert len(BankrollLedger.get_pending_simulation_bets()) == 0

    # Cleanup
    BankrollLedger.reset_bankroll(1200.0)

def test_deletion_and_clear_methods():
    BankrollLedger.reset_bankroll(1000.0)

    # 1. Add deposit and bets
    tx_dep = BankrollLedger.deposit(500.0, "Test Deposit")
    assert BankrollLedger.get_current_balance() == 1500.0

    b_id = BankrollLedger.place_simulation_bet("nba", "BOS vs MIA", "BOS", 100.0, 2.00)
    BankrollLedger.resolve_simulation_bet(b_id, is_win=True)
    assert BankrollLedger.get_current_balance() == 1600.0

    # 2. Test delete_simulation_bet
    assert BankrollLedger.delete_simulation_bet(b_id) is True
    assert BankrollLedger.delete_simulation_bet(99999) is False

    # 3. Test clear_all_simulation_bets
    BankrollLedger.place_simulation_bet("mlb", "LAD vs SF", "LAD", 50.0, 1.80)
    cnt = BankrollLedger.clear_all_simulation_bets()
    assert cnt >= 1

    # 4. Test delete_transaction
    tx_df = BankrollLedger.get_ledger_history()
    dep_row = tx_df[tx_df["tx_type"] == "DEPOSIT"].iloc[0]
    dep_id = int(dep_row["id"])
    
    # After deleting deposit of 500, balance should drop by 500 (from 1600 to 1100)
    new_bal = BankrollLedger.delete_transaction(dep_id)
    assert new_bal == 1100.0

    # 5. Test clear_all_transactions
    bal_after_clear = BankrollLedger.clear_all_transactions(keep_initial_capital=True)
    assert bal_after_clear == 1000.0
    assert len(BankrollLedger.get_ledger_history()) == 1

    # Cleanup
    BankrollLedger.reset_bankroll(1200.0)