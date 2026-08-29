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
    def get_settled_balance() -> float:
        """
        Returns the settled bankroll balance from closed transactions.
        If no transactions exist, attempts auto-restoring from sync snapshot before initializing with DEFAULT_STARTING_BANKROLL.
        """
        row = db.fetch_one("SELECT balance_after FROM bankroll_transactions ORDER BY id DESC LIMIT 1")
        if row is not None:
            return round(float(row["balance_after"]), 2)

        # Attempt to auto-restore from sync snapshot first
        try:
            from src.db.github_sync import GitHubDataSync
            if GitHubDataSync.import_data_snapshot_if_exists():
                row = db.fetch_one("SELECT balance_after FROM bankroll_transactions ORDER BY id DESC LIMIT 1")
                if row is not None:
                    return round(float(row["balance_after"]), 2)
        except Exception:
            pass

        # Initialize default capital if ledger is empty
        stored_init = float(db.get_setting("starting_bankroll", str(config.DEFAULT_STARTING_BANKROLL)))
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('INITIAL', ?, ?, 'general', 'Initial Starting Capital')
            """, (stored_init, stored_init))
        return stored_init

    @staticmethod
    def get_pending_stakes_total() -> float:
        """
        Returns the total sum of stakes currently at risk across all active/pending simulation wagers.
        """
        row = db.fetch_one("SELECT COALESCE(SUM(stake), 0.0) as total FROM simulation_bets WHERE status = 'PENDING'")
        return round(float(row["total"]), 2) if row else 0.0

    @staticmethod
    def get_current_balance(include_pending_deduction: bool = True) -> float:
        """
        Returns the current available bankroll balance.
        Deducts active bets currently at risk when include_pending_deduction=True.
        """
        settled = BankrollLedger.get_settled_balance()
        if not include_pending_deduction:
            return settled
        pending = BankrollLedger.get_pending_stakes_total()
        return round(max(settled - pending, 0.0), 2)

    @staticmethod
    def deposit(amount: float, note: str = "Deposit") -> float:
        """
        Adds funds to user bankroll.
        """
        amount = round(abs(float(amount)), 2)
        curr = BankrollLedger.get_settled_balance()
        new_bal = round(curr + amount, 2)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('DEPOSIT', ?, ?, 'general', ?)
            """, (amount, new_bal, note or "Deposit"))
        return BankrollLedger.get_current_balance()

    @staticmethod
    def withdraw(amount: float, note: str = "Withdrawal") -> float:
        """
        Deducts withdrawn funds from user bankroll.
        """
        amount = round(abs(float(amount)), 2)
        curr = BankrollLedger.get_settled_balance()
        new_bal = round(max(curr - amount, 0.0), 2)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('WITHDRAWAL', ?, ?, 'general', ?)
            """, (-amount, new_bal, note or "Withdrawal"))
        return BankrollLedger.get_current_balance()

    @staticmethod
    def record_bet(sport: str, team: str, stake: float, odds: float, is_win: bool, note: str = "") -> Dict:
        """
        Records a completed wager result (Win or Loss) and updates the bankroll.
        """
        stake = round(abs(float(stake)), 2)
        odds = round(float(odds), 2)
        curr = BankrollLedger.get_settled_balance()

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
    def set_starting_balance(new_amount: float) -> float:
        """
        Updates the starting capital and recalculates all chronological balances,
        preserving existing transactions while remembering the new base capital.
        """
        new_amount = round(abs(float(new_amount)), 2)
        db.set_setting("starting_bankroll", str(new_amount))

        rows = db.fetch_all("SELECT * FROM bankroll_transactions ORDER BY id ASC")
        if not rows:
            return BankrollLedger.reset_bankroll(new_amount)

        # Check if INITIAL transaction exists
        has_initial = any(r["tx_type"] == "INITIAL" for r in rows)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if not has_initial:
                # Prepend an initial transaction
                cursor.execute("""
                    INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                    VALUES ('INITIAL', ?, ?, 'general', 'Starting Capital')
                """, (new_amount, new_amount))
            else:
                cursor.execute("UPDATE bankroll_transactions SET amount = ? WHERE tx_type = 'INITIAL'", (new_amount,))

            # Recompute running balance_after for all records
            all_txs = cursor.execute("SELECT id, tx_type, amount FROM bankroll_transactions ORDER BY id ASC").fetchall()
            running_bal = 0.0
            for tx in all_txs:
                if tx["tx_type"] == "INITIAL":
                    running_bal = float(tx["amount"])
                else:
                    running_bal += float(tx["amount"])
                running_bal = round(max(running_bal, 0.0), 2)
                cursor.execute("UPDATE bankroll_transactions SET balance_after = ? WHERE id = ?", (running_bal, tx["id"]))

        return BankrollLedger.get_current_balance()

    @staticmethod
    def reset_bankroll(initial_amount: Optional[float] = None) -> float:
        """
        Clears ledger history and resets starting capital.
        """
        if initial_amount is None:
            initial_amount = config.DEFAULT_STARTING_BANKROLL
        initial_amount = round(abs(float(initial_amount)), 2)
        db.set_setting("starting_bankroll", str(initial_amount))
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bankroll_transactions")
            cursor.execute("""
                INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                VALUES ('INITIAL', ?, ?, 'general', 'Starting Capital')
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
        stored_init = float(db.get_setting("starting_bankroll", str(config.DEFAULT_STARTING_BANKROLL)))
        pending_stakes = BankrollLedger.get_pending_stakes_total()
        pending_row = db.fetch_one("SELECT COUNT(*) as c FROM simulation_bets WHERE status = 'PENDING'")
        pending_count = int(pending_row["c"]) if pending_row else 0

        if df.empty:
            curr = BankrollLedger.get_current_balance()
            settled = BankrollLedger.get_settled_balance()
            return {
                "current_balance": curr,
                "available_balance": curr,
                "settled_balance": settled,
                "pending_stakes": pending_stakes,
                "pending_bets_count": pending_count,
                "initial_balance": stored_init,
                "total_deposits": 0.0,
                "total_withdrawals": 0.0,
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "net_betting_pnl": 0.0,
                "roi_pct": 0.0,
                "bankroll_roi_pct": 0.0,
                "yield_pct": 0.0,
                "turnover_roi_pct": 0.0
            }

        init_row = df[df["tx_type"] == "INITIAL"]
        initial_bal = float(init_row.iloc[0]["amount"]) if not init_row.empty else stored_init
        settled_bal = float(df.iloc[-1]["balance_after"])
        available_bal = round(max(settled_bal - pending_stakes, 0.0), 2)

        deposits = float(df[df["tx_type"] == "DEPOSIT"]["amount"].sum())
        withdrawals = float(abs(df[df["tx_type"] == "WITHDRAWAL"]["amount"].sum()))

        bets_df = df[df["tx_type"].isin(["BET_WIN", "BET_LOSS"])]
        total_bets = len(bets_df)
        wins = len(bets_df[bets_df["tx_type"] == "BET_WIN"])
        losses = len(bets_df[bets_df["tx_type"] == "BET_LOSS"])
        win_rate = round((wins / total_bets) * 100.0, 1) if total_bets > 0 else 0.0

        total_staked = float(bets_df["stake"].sum())
        net_betting_pnl = float(bets_df["amount"].sum())

        # 1. Bankroll ROI (Capital Growth %): Net PnL / (Starting Capital + Deposits)
        invested_capital = max(initial_bal + deposits, 1.0)
        bankroll_roi_pct = round((net_betting_pnl / invested_capital) * 100.0, 2)

        # 2. Betting Yield (Turnover ROI %): Net PnL / Total Staked Turnover
        yield_pct = round((net_betting_pnl / total_staked) * 100.0, 2) if total_staked > 0 else 0.0

        return {
            "current_balance": available_bal,
            "available_balance": available_bal,
            "settled_balance": settled_bal,
            "pending_stakes": pending_stakes,
            "pending_bets_count": pending_count,
            "initial_balance": initial_bal,
            "total_deposits": deposits,
            "total_withdrawals": withdrawals,
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_staked": total_staked,
            "net_betting_pnl": net_betting_pnl,
            "roi_pct": bankroll_roi_pct,
            "bankroll_roi_pct": bankroll_roi_pct,
            "yield_pct": yield_pct,
            "turnover_roi_pct": yield_pct
        }

    # ================= SIMULATION BETTING LIFECYCLE =================

    @staticmethod
    def place_simulation_bet(
        sport: str,
        matchup: str,
        team: str,
        stake: float,
        odds: float,
        model_prob: float = 0.0,
        edge_pct: float = 0.0,
        note: str = ""
    ) -> int:
        """
        Records an open pending simulation wager to track until game completion.
        """
        stake = round(abs(float(stake)), 2)
        odds = round(float(odds), 2)
        model_prob = round(float(model_prob), 4)
        edge_pct = round(float(edge_pct), 4)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulation_bets (sport, matchup, team_selected, stake, odds, model_prob, edge_pct, status, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (sport.lower(), matchup, team, stake, odds, model_prob, edge_pct, note))
            return cursor.lastrowid

    @staticmethod
    def resolve_simulation_bet(bet_id: int, is_win: bool) -> Optional[Dict]:
        """
        Grades an open pending simulation bet as WON or LOST,
        calculates payout, updates the bankroll ledger, and archives the bet.
        """
        bet = db.fetch_one("SELECT * FROM simulation_bets WHERE id = ?", (bet_id,))
        if not bet or bet["status"] != "PENDING":
            return None

        stake = float(bet["stake"])
        odds = float(bet["odds"])
        sport = str(bet["sport"])
        team = str(bet["team_selected"])
        matchup = str(bet["matchup"])

        # Record financial transaction in ledger
        tx_res = BankrollLedger.record_bet(
            sport=sport,
            team=team,
            stake=stake,
            odds=odds,
            is_win=is_win,
            note=f"Sim Bet #{bet_id}: {matchup} ({team})"
        )

        status_str = "WON" if is_win else "LOST"
        pnl = tx_res["pnl"]

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE simulation_bets
                SET status = ?, pnl = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status_str, pnl, bet_id))

        return {
            "bet_id": bet_id,
            "status": status_str,
            "pnl": pnl,
            "balance_after": tx_res["balance_after"]
        }

    @staticmethod
    def void_simulation_bet(bet_id: int) -> bool:
        """
        Cancels / voids a pending simulation bet without financial impact.
        """
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE simulation_bets
                SET status = 'VOID', resolved_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'PENDING'
            """, (bet_id,))
            return cursor.rowcount > 0

    @staticmethod
    def delete_transaction(tx_id: int) -> float:
        """
        Deletes a specific transaction from the ledger and recalculates
        the running balance_after for all remaining transactions.
        """
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bankroll_transactions WHERE id = ?", (tx_id,))
            
            # Recompute running balances
            rows = cursor.execute("SELECT id, tx_type, amount FROM bankroll_transactions ORDER BY id ASC").fetchall()
            stored_init = float(db.get_setting("starting_bankroll", str(config.DEFAULT_STARTING_BANKROLL)))
            
            if not rows:
                cursor.execute("""
                    INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                    VALUES ('INITIAL', ?, ?, 'general', 'Starting Capital')
                """, (stored_init, stored_init))
            else:
                running_bal = 0.0
                for tx in rows:
                    if tx["tx_type"] == "INITIAL":
                        running_bal = float(tx["amount"])
                    else:
                        running_bal += float(tx["amount"])
                    running_bal = round(max(running_bal, 0.0), 2)
                    cursor.execute("UPDATE bankroll_transactions SET balance_after = ? WHERE id = ?", (running_bal, tx["id"]))

        return BankrollLedger.get_current_balance()

    @staticmethod
    def clear_all_transactions(keep_initial_capital: bool = True) -> float:
        """
        Clears all transactions, optionally preserving the initial starting bankroll.
        """
        stored_init = float(db.get_setting("starting_bankroll", str(config.DEFAULT_STARTING_BANKROLL)))
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bankroll_transactions")
            if keep_initial_capital:
                cursor.execute("""
                    INSERT INTO bankroll_transactions (tx_type, amount, balance_after, sport, note)
                    VALUES ('INITIAL', ?, ?, 'general', 'Starting Capital')
                """, (stored_init, stored_init))
        return BankrollLedger.get_current_balance()

    @staticmethod
    def delete_simulation_bet(bet_id: int) -> bool:
        """
        Deletes a specific bet record from simulation_bets.
        """
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulation_bets WHERE id = ?", (bet_id,))
            return cursor.rowcount > 0

    @staticmethod
    def clear_all_simulation_bets() -> int:
        """
        Clears all bet records from simulation_bets.
        """
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulation_bets")
            return cursor.rowcount

    @staticmethod
    def get_pending_simulation_bets() -> pd.DataFrame:
        """
        Returns all active pending simulation bets awaiting resolution.
        """
        return db.fetch_df("SELECT * FROM simulation_bets WHERE status = 'PENDING' ORDER BY id DESC")

    @staticmethod
    def get_all_simulation_bets() -> pd.DataFrame:
        """
        Returns all simulation bets across all statuses.
        """
        return db.fetch_df("SELECT * FROM simulation_bets ORDER BY id DESC")