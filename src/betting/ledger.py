import pandas as pd
from typing import Dict, List, Optional
from src.db.database import db
import config

class BankrollLedger:
    """
    Manages user personal bankroll, real-time balance tracking, deposits, withdrawals,
    and individual wager win/loss grading in Philippine Pesos (PHP / ₱).
    """

    @staticmethod
    def get_current_balance() -> float:
        """
        Returns the latest bankroll balance.
        If no transactions exist, initializes with DEFAULT_STARTING_BANKROLL (₱1,200.00).
        """
        row = db.fetch_one("SELECT balance_after FROM bankroll_transactions ORDER BY id DESC LIMIT 1")
        if row is not None:
            return round(float(row["balance_after"]), 2)

        # Initialize default capital if ledger is empty
        initial_val = float(config.DEFAULT_STARTING_BANKROLL)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('INITIAL', ?, ?, 'general', 'Initial Starting Capital')
            """, (initial_val, initial_val))
        return initial_val

    @staticmethod
    def deposit(amount: float, note: str = "Deposit") -> float:
        """
        Adds funds to user bankroll.
        """
        amount = round(abs(float(amount)), 2)
        curr = BankrollLedger.get_current_balance()
        new_bal = round(curr + amount, 2)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('DEPOSIT', ?, ?, 'general', ?)
            """, (amount, new_bal, note or "Deposit"))
        return new_bal

    @staticmethod
    def withdraw(amount: float, note: str = "Withdrawal") -> float:
        """
        Deducts withdrawn funds from user bankroll.
        """
        amount = round(abs(float(amount)), 2)
        curr = BankrollLedger.get_current_balance()
        new_bal = round(max(curr - amount, 0.0), 2)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('WITHDRAWAL', ?, ?, 'general', ?)
            """, (-amount, new_bal, note or "Withdrawal"))
        return new_bal

    @staticmethod
    def record_bet(sport: str, team: str, stake: float, odds: float, is_win: bool, note: str = "") -> Dict:
        """
        Records a completed wager result (Win or Loss) and updates the bankroll.
        """
        stake = round(abs(float(stake)), 2)
        odds = round(float(odds), 2)
        curr = BankrollLedger.get_current_balance()

        if is_win:
            pnl = round(stake * (odds - 1.0), 2)
            tx_type = "BET_WIN"
            new_bal = round(curr + pnl, 2)
            amount = pnl
        else:
            pnl = round(-stake, 2)
            tx_type = "BET_LOSS"
            new_bal = round(max(curr - stake, 0.0), 2)
            amount = -stake

        desc = note or f"{sport.upper()} {team} @ {odds:.2f} ({'WON' if is_win else 'LOST'})"

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, stake, odds, team_selected, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (tx_type, amount, new_bal, sport.lower(), stake, odds, team, desc))

        return {
            "tx_type": tx_type,
            "stake": stake,
            "odds": odds,
            "pnl": pnl,
            "balance_after": new_bal,
            "is_win": is_win
        }

    @staticmethod
    def reset_bankroll(initial_amount: float = 1200.0) -> float:
        """
        Clears ledger history and resets starting capital.
        """
        initial_amount = round(abs(float(initial_amount)), 2)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bankroll_transactions")
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('INITIAL', ?, ?, 'general', 'Reset Starting Capital')
            """, (initial_amount, initial_amount))
        return initial_amount

    @staticmethod
    def get_ledger_history() -> pd.DataFrame:
        """
        Returns all transaction logs ordered chronologically.
        """
        return db.fetch_df("SELECT * FROM bankroll_transactions ORDER BY id DESC")

    @staticmethod
    def get_ledger_metrics() -> Dict:
        """
        Computes aggregate metrics from personal transaction history.
        """
        df = db.fetch_df("SELECT * FROM bankroll_transactions ORDER BY id ASC")
        if df.empty:
            curr = BankrollLedger.get_current_balance()
            return {
                "current_balance": curr,
                "initial_balance": curr,
                "total_deposits": 0.0,
                "total_withdrawals": 0.0,
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "net_betting_pnl": 0.0,
                "roi_pct": 0.0
            }

        init_row = df[df["tx_type"] == "INITIAL"]
        initial_bal = float(init_row.iloc[0]["amount"]) if not init_row.empty else float(config.DEFAULT_STARTING_BANKROLL)
        curr_bal = float(df.iloc[-1]["balance_after"])

        deposits = float(df[df["tx_type"] == "DEPOSIT"]["amount"].sum())
        withdrawals = float(abs(df[df["tx_type"] == "WITHDRAWAL"]["amount"].sum()))

        bets_df = df[df["tx_type"].isin(["BET_WIN", "BET_LOSS"])]
        total_bets = len(bets_df)
        wins = len(bets_df[bets_df["tx_type"] == "BET_WIN"])
        losses = len(bets_df[bets_df["tx_type"] == "BET_LOSS"])
        win_rate = round((wins / total_bets) * 100.0, 1) if total_bets > 0 else 0.0

        total_staked = float(bets_df["stake"].sum())
        net_betting_pnl = float(bets_df["amount"].sum())
        roi_pct = round((net_betting_pnl / total_staked) * 100.0, 2) if total_staked > 0 else 0.0

        return {
            "current_balance": curr_bal,
            "initial_balance": initial_bal,
            "total_deposits": deposits,
            "total_withdrawals": withdrawals,
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_staked": total_staked,
            "net_betting_pnl": net_betting_pnl,
            "roi_pct": roi_pct
        }