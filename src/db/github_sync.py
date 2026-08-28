import json
import subprocess
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from src.db.database import db
import config

SYNC_DIR = Path(__file__).parent.parent.parent / "data" / "sync"

class GitHubDataSync:
    @staticmethod
    def get_sync_dir() -> Path:
        SYNC_DIR.mkdir(parents=True, exist_ok=True)
        return SYNC_DIR

    @staticmethod
    def export_data_snapshot() -> Dict[str, int]:
        sync_dir = GitHubDataSync.get_sync_dir()
        counts = {}

        # 1. Export bankroll_transactions
        bt_df = db.fetch_df("SELECT * FROM bankroll_transactions ORDER BY id ASC")
        bt_path = sync_dir / "bankroll_transactions.json"
        with open(bt_path, "w", encoding="utf-8") as f:
            json.dump(bt_df.to_dict(orient="records"), f, indent=2, default=str)
        counts["transactions"] = len(bt_df)

        # 2. Export simulation_bets
        sb_df = db.fetch_df("SELECT * FROM simulation_bets ORDER BY id ASC")
        sb_path = sync_dir / "simulation_bets.json"
        with open(sb_path, "w", encoding="utf-8") as f:
            json.dump(sb_df.to_dict(orient="records"), f, indent=2, default=str)
        counts["bets"] = len(sb_df)

        # 3. Export app_settings
        st_df = db.fetch_df("SELECT * FROM app_settings ORDER BY key ASC")
        st_path = sync_dir / "app_settings.json"
        with open(st_path, "w", encoding="utf-8") as f:
            json.dump(st_df.to_dict(orient="records"), f, indent=2, default=str)
        counts["settings"] = len(st_df)

        return counts

    @staticmethod
    def import_data_snapshot_if_exists() -> bool:
        sync_dir = GitHubDataSync.get_sync_dir()
        bt_path = sync_dir / "bankroll_transactions.json"
        sb_path = sync_dir / "simulation_bets.json"
        st_path = sync_dir / "app_settings.json"

        restored_any = False

        # 1. Restore app_settings
        if st_path.exists():
            try:
                with open(st_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if records:
                    for rec in records:
                        db.set_setting(rec["key"], str(rec["value"]))
                    restored_any = True
            except Exception:
                pass

        # 2. Restore bankroll_transactions
        if bt_path.exists():
            try:
                with open(bt_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if records:
                    with db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("delete from bankroll_transactions")
                        for r in records:
                            cursor.execute("""
                                INSERT INTO bankroll_transactions (id, tx_type, amount, balance_after, sport, stake, odds, team_selected, note, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                r.get("id"),
                                r.get("tx_type", "INITIAL"),
                                float(r.get("amount", 0.0)),
                                float(r.get("balance_after", 0.0)),
                                r.get("sport", "general"),
                                float(r.get("stake", 0.0)),
                                float(r.get("odds", 1.0)),
                                r.get("team_selected", ""),
                                r.get("note", ""),
                                r.get("created_at")
                            ))
                    restored_any = True
            except Exception:
                pass

        # 3. Restore simulation_bets
        if sb_path.exists():
            try:
                with open(sb_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if records:
                    with db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("delete from simulation_bets")
                        for r in records:
                            cursor.execute("""
                                INSERT INTO simulation_bets (id, sport, matchup, team_selected, stake, odds, model_prob, edge_pct, status, pnl, placed_at, resolved_at, note)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                r.get("id"),
                                r.get("sport", "nba"),
                                r.get("matchup", ""),
                                r.get("team_selected", ""),
                                float(r.get("stake", 0.0)),
                                float(r.get("odds", 1.0)),
                                float(r.get("model_prob", 0.0)),
                                float(r.get("edge_pct", 0.0)),
                                r.get("status", "PENDING"),
                                float(r.get("pnl", 0.0)),
                                r.get("placed_at"),
                                r.get("resolved_at"),
                                r.get("note", "")
                            ))
                    restored_any = True
            except Exception:
                pass

        return restored_any

    @staticmethod
    def push_to_github(commit_msg: str = "sync: update website code, bankroll, and user settings", include_all_code: bool = True) -> Tuple[bool, str]:
        """
        Exports database snapshot and stages all website code + data changes,
        then commits and pushes to GitHub repository.
        """
        try:
            counts = GitHubDataSync.export_data_snapshot()
            project_dir = str(getattr(config, "PROJECT_ROOT", getattr(config, "BASE_DIR", Path(__file__).resolve().parent.parent.parent)))

            # Stage data sync and any modified website code files
            if include_all_code:
                subprocess.run(["git", "add", "."], cwd=project_dir, check=True, capture_output=True, text=True)
            else:
                subprocess.run(["git", "add", "data/sync/"], cwd=project_dir, check=True, capture_output=True, text=True)

            commit_res = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=project_dir,
                capture_output=True,
                text=True
            )

            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=project_dir,
                capture_output=True,
                text=True
            )

            if push_res.returncode != 0 and "Everything up-to-date" not in push_res.stderr and "nothing to commit" not in commit_res.stdout:
                return False, f"Git push failed: {push_res.stderr or push_res.stdout}"

            return True, f"Successfully pushed website updates and data to GitHub! ({counts.get('transactions', 0)} transactions, {counts.get('bets', 0)} bets synced)"

        except Exception as e:
            return False, f"Error during GitHub push: {str(e)}"

    @staticmethod
    def pull_from_github() -> Tuple[bool, str]:
        """
        Pulls latest website codebase updates and data snapshots from GitHub,
        then restores data into the local SQLite database.
        """
        try:
            project_dir = str(getattr(config, "PROJECT_ROOT", getattr(config, "BASE_DIR", Path(__file__).resolve().parent.parent.parent)))

            pull_res = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=project_dir,
                capture_output=True,
                text=True
            )

            if pull_res.returncode != 0:
                return False, f"Git pull failed: {pull_res.stderr or pull_res.stdout}"

            pull_output = (pull_res.stdout or "").strip()
            restored = GitHubDataSync.import_data_snapshot_if_exists()

            if "Already up to date." in pull_output:
                return True, "Website and data are already up-to-date with GitHub."
            else:
                return True, f"Website code and data updated successfully from GitHub! ({pull_output.splitlines()[-1] if pull_output else 'Fast-forward'})"
        except Exception as e:
            return False, f"Error during GitHub pull: {str(e)}"
