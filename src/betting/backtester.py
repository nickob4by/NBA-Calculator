import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.db.database import db
from src.features.matchup_builder import build_full_feature_dataset, get_feature_columns
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.betting.odds_math import remove_vig, american_to_decimal
from src.betting.ev_engine import evaluate_moneyline_market, evaluate_spread_market, evaluate_totals_market
from src.betting.kelly import calculate_kelly_fractional
import config

logger = logging.getLogger(__name__)

class HistoricalBacktester:
    """
    Simulates placing +EV wagers chronologically over historical NBA seasons
    with dynamic or flat bankroll updates, Fractional Kelly sizing, and CLV tracking.
    """
    def __init__(
        self,
        starting_bankroll: float = config.DEFAULT_STARTING_BANKROLL,
        kelly_fraction: float = config.DEFAULT_KELLY_FRACTION,
        min_edge: float = config.DEFAULT_MIN_EDGE,
        max_bankroll_pct: float = config.DEFAULT_MAX_BANKROLL_PCT,
        markets: List[str] = ["moneyline", "spread", "total"],
        compound_bankroll: bool = False
    ):
        self.starting_bankroll = starting_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bankroll_pct = max_bankroll_pct
        self.markets = markets
        self.compound_bankroll = compound_bankroll

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        season_filter: Optional[str] = None
    ) -> Dict:
        # 1. Load trained models
        margin_model = MarginPredictor.load()
        totals_model = TotalsPredictor.load()
        win_model = WinProbabilityModel.load()

        # 2. Load games, odds, and build features
        logs_df = db.fetch_df("SELECT * FROM team_game_logs ORDER BY game_date, game_id")
        matchups_df = build_full_feature_dataset(logs_df)

        odds_df = db.fetch_df("SELECT * FROM odds")
        if odds_df.empty:
            raise ValueError("No odds records found in database for backtesting.")

        odds_df = odds_df.drop_duplicates(subset=["game_id"], keep="first")

        # Merge matchups with odds
        merged = pd.merge(matchups_df, odds_df, on="game_id", how="inner")
        merged = merged.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)

        if start_date:
            merged = merged[merged["game_date"] >= start_date]
        if end_date:
            merged = merged[merged["game_date"] <= end_date]
        if season_filter:
            merged = merged[merged["season"] == season_filter]

        merged = merged.reset_index(drop=True)

        if merged.empty:
            return {
                "total_bets": 0, "win_rate": 0.0, "pnl": 0.0, "roi_pct": 0.0,
                "final_bankroll": self.starting_bankroll, "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0, "avg_clv_pct": 0.0, "equity_curve": [], "bets": []
            }

        feature_cols = get_feature_columns(merged)
        X = merged[feature_cols]

        pred_margins = margin_model.predict(X)
        pred_totals = totals_model.predict(X)
        pred_win_probs = win_model.predict_proba(predicted_margins=pred_margins, sigma=margin_model.residual_std)

        bankroll = self.starting_bankroll
        equity_curve = [{"game_date": merged.iloc[0]["game_date"], "bankroll": bankroll, "pnl": 0.0}]
        all_bets = []

        peak_bankroll = bankroll
        max_drawdown_dollars = 0.0
        max_drawdown_pct = 0.0

        for i, row in merged.iterrows():
            game_id = str(row["game_id"])
            game_date = str(row["game_date"])
            home_pts = row["home_pts"]
            away_pts = row["away_pts"]
            actual_margin = home_pts - away_pts
            actual_total = home_pts + away_pts

            p_home_win = float(pred_win_probs[i])
            m_margin = float(pred_margins[i])
            m_total = float(pred_totals[i])

            game_opportunities = []

            # 1. Moneyline evaluation
            if "moneyline" in self.markets and pd.notna(row.get("home_ml_close")) and pd.notna(row.get("away_ml_close")):
                h_ml = float(row["home_ml_close"])
                a_ml = float(row["away_ml_close"])
                ml_eval = evaluate_moneyline_market(p_home_win, h_ml, a_ml, min_edge=self.min_edge)
                
                for opp in ml_eval["opportunities"]:
                    side = opp["side"]
                    is_win = (actual_margin > 0) if side == "home" else (actual_margin < 0)
                    opp["actual_win"] = is_win
                    opp["open_odds"] = float(row.get("home_ml_open", h_ml)) if side == "home" else float(row.get("away_ml_open", a_ml))
                    game_opportunities.append(opp)

            # 2. Spread evaluation
            if "spread" in self.markets and pd.notna(row.get("spread_line")):
                spread_line = float(row["spread_line"])
                h_sp_odds = float(row.get("home_spread_odds", 1.91))
                a_sp_odds = float(row.get("away_spread_odds", 1.91))
                sp_eval = evaluate_spread_market(m_margin, spread_line, h_sp_odds, a_sp_odds, residual_std=margin_model.residual_std, min_edge=self.min_edge)

                for opp in sp_eval["opportunities"]:
                    side = opp["side"]
                    if side == "home":
                        diff = actual_margin + spread_line
                    else:
                        diff = -actual_margin - spread_line
                    
                    if diff == 0:
                        is_win = None # Push
                    else:
                        is_win = diff > 0

                    opp["actual_win"] = is_win
                    opp["open_odds"] = opp["decimal_odds"]
                    game_opportunities.append(opp)

            # 3. Totals evaluation
            if "total" in self.markets and pd.notna(row.get("total_line")):
                total_line = float(row["total_line"])
                o_odds = float(row.get("over_odds", 1.91))
                u_odds = float(row.get("under_odds", 1.91))
                tot_eval = evaluate_totals_market(m_total, total_line, o_odds, u_odds, total_residual_std=totals_model.total_residual_std, min_edge=self.min_edge)

                for opp in tot_eval["opportunities"]:
                    side = opp["side"]
                    if side == "over":
                        diff = actual_total - total_line
                    else:
                        diff = total_line - actual_total

                    if diff == 0:
                        is_win = None # Push
                    else:
                        is_win = diff > 0

                    opp["actual_win"] = is_win
                    opp["open_odds"] = opp["decimal_odds"]
                    game_opportunities.append(opp)

            # Execute bets with Fractional Kelly sizing
            effective_bankroll = bankroll if self.compound_bankroll else self.starting_bankroll

            for bet in game_opportunities:
                frac = calculate_kelly_fractional(
                    model_prob=bet["model_prob"],
                    decimal_odds=bet["decimal_odds"],
                    kelly_multiplier=self.kelly_fraction,
                    max_bankroll_pct=self.max_bankroll_pct
                )

                stake = round(frac * effective_bankroll, 2)
                if stake < 5.0:
                    continue

                is_win = bet["actual_win"]
                decimal_odds = bet["decimal_odds"]

                if is_win is True:
                    pnl = round(stake * (decimal_odds - 1.0), 2)
                    result = "WIN"
                elif is_win is False:
                    pnl = round(-stake, 2)
                    result = "LOSS"
                else:
                    pnl = 0.0
                    result = "PUSH"

                closing_odds = bet["decimal_odds"]
                open_odds = bet.get("open_odds", closing_odds)
                clv = round(((closing_odds / open_odds) - 1.0) * 100.0, 2) if open_odds > 0 else 0.0

                bankroll = round(bankroll + pnl, 2)
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll

                dd_dollars = peak_bankroll - bankroll
                dd_pct = (dd_dollars / peak_bankroll) * 100.0 if peak_bankroll > 0 else 0.0
                if dd_pct > max_drawdown_pct:
                    max_drawdown_pct = dd_pct
                    max_drawdown_dollars = dd_dollars

                all_bets.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "market_type": bet["market_type"],
                    "side": bet["side"],
                    "odds": decimal_odds,
                    "model_prob": bet["model_prob"],
                    "fair_implied_prob": bet["fair_implied_prob"],
                    "edge": bet["edge"],
                    "ev": bet["ev"],
                    "stake": stake,
                    "result": result,
                    "pnl": pnl,
                    "clv": clv,
                    "bankroll": bankroll
                })

                equity_curve.append({
                    "game_date": game_date,
                    "bankroll": bankroll,
                    "pnl": pnl
                })

        bets_df = pd.DataFrame(all_bets)
        if bets_df.empty:
            return {
                "total_bets": 0, "win_rate": 0.0, "pnl": 0.0, "roi_pct": 0.0,
                "final_bankroll": self.starting_bankroll, "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0, "avg_clv_pct": 0.0, "equity_curve": equity_curve, "bets": []
            }

        total_bets = len(bets_df)
        wins = int((bets_df["result"] == "WIN").sum())
        losses = int((bets_df["result"] == "LOSS").sum())
        pushes = int((bets_df["result"] == "PUSH").sum())
        win_rate = round((wins / (wins + losses)) * 100.0, 2) if (wins + losses) > 0 else 0.0

        total_staked = float(bets_df["stake"].sum())
        total_pnl = round(float(bets_df["pnl"].sum()), 2)
        roi_pct = round((total_pnl / total_staked) * 100.0, 2) if total_staked > 0 else 0.0

        returns = bets_df["pnl"] / bets_df["stake"]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = round(float((mean_ret / std_ret) * np.sqrt(len(returns))), 2) if std_ret > 0 else 0.0

        avg_clv = round(float(bets_df["clv"].mean()), 2)
        beat_closing = round(float((bets_df["clv"] >= 0).mean() * 100.0), 1)

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "total_staked": round(total_staked, 2),
            "pnl": total_pnl,
            "roi_pct": roi_pct,
            "starting_bankroll": self.starting_bankroll,
            "final_bankroll": bankroll,
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "max_drawdown_dollars": round(max_drawdown_dollars, 2),
            "sharpe_ratio": sharpe,
            "avg_clv_pct": avg_clv,
            "beat_closing_pct": beat_closing,
            "equity_curve": equity_curve,
            "bets": all_bets
        }