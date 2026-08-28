import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import pandas as pd
import numpy as np

import config
from src.db.database import db
from src.ingestion.pipeline import sync_season, generate_seed_dataset_if_empty
from src.ingestion.mlb_api_fetcher import generate_mlb_seed_dataset_if_empty
from src.models.train import train_all_models
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.models.mlb_models import MLBRunMarginPredictor, MLBTotalsPredictor, MLBWinProbabilityModel
from src.features.matchup_builder import build_full_feature_dataset, get_feature_columns
from src.betting.ev_engine import evaluate_moneyline_market, evaluate_spread_market, evaluate_totals_market
from src.betting.kelly import size_bet
from src.betting.backtester import HistoricalBacktester

console = Console()

def cmd_sync(args):
    sport = args.sport.lower()
    console.print(Panel.fit(f"[bold cyan]{sport.upper()} Ingestion Pipeline[/bold cyan]"))
    if sport == "mlb":
        generate_mlb_seed_dataset_if_empty()
        console.print("[bold green]MLB sync and seed verified![/bold green]")
    else:
        seasons = args.seasons.split(",") if args.seasons else ["2022-23", "2023-24", "2024-25"]
        for s in seasons:
            s = s.strip()
            with console.status(f"[yellow]Syncing season {s}...[/yellow]"):
                count = sync_season(s)
                if count == 0:
                    generate_seed_dataset_if_empty([s])
                else:
                    console.print(f"[green]Successfully synced {count} games for season {s}[/green]")
        console.print("[bold green]NBA Sync completed![/bold green]")

def cmd_train(args):
    sport = args.sport.lower()
    console.print(Panel.fit(f"[bold cyan]{sport.upper()} — Model Training & Calibration[/bold cyan]"))
    with console.status(f"[yellow]Running TimeSeriesSplit cross validation for {sport.upper()}...[/yellow]"):
        metrics = train_all_models(sport=sport, n_splits=args.splits)
    
    table = Table(title=f"{sport.upper()} Cross-Validation & Calibration Performance")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    unit = "runs" if sport == "mlb" else "pts"
    table.add_row("Evaluated Matches", str(metrics["num_games"]))
    table.add_row("Predictor Features", str(metrics["feature_count"]))
    table.add_row(f"Margin MAE", f"{metrics['margin_metrics']['mae']} {unit}")
    table.add_row(f"Margin RMSE", f"{metrics['margin_metrics']['rmse']} {unit}")
    table.add_row(f"Totals MAE", f"{metrics['totals_metrics']['mae']} {unit}")
    table.add_row("Win Prob Log Loss", str(metrics["win_probability_metrics"]["log_loss"]))
    table.add_row("Win Prob Brier Score", str(metrics["win_probability_metrics"]["brier_score"]))
    table.add_row("Win Accuracy", f"{metrics['win_probability_metrics']['accuracy'] * 100:.1f}%")
    table.add_row("Expected Calibration Error (ECE)", str(metrics["win_probability_metrics"]["ece"]))

    console.print(table)

def cmd_predict(args):
    sport = args.sport.lower()
    console.print(Panel.fit(f"[bold cyan]{sport.upper()} Matchup: {args.home} (Home) vs {args.away} (Away)[/bold cyan]"))
    
    home_id = config.get_team_id(args.home, sport=sport)
    away_id = config.get_team_id(args.away, sport=sport)

    if not home_id or not away_id:
        console.print(f"[bold red]Error: Invalid team identifier for {sport.upper()}.[/bold red]")
        sys.exit(1)

    # Load sport models
    if sport == "mlb":
        margin_model = MLBRunMarginPredictor.load()
        totals_model = MLBTotalsPredictor.load()
        win_model = MLBWinProbabilityModel.load()
    else:
        margin_model = MarginPredictor.load()
        totals_model = TotalsPredictor.load()
        win_model = WinProbabilityModel.load()

    logs_df = db.fetch_df("SELECT * FROM team_game_logs WHERE sport=? ORDER BY game_date, game_id", (sport,))
    matchups_df = build_full_feature_dataset(logs_df, sport=sport)

    recent_home = matchups_df[matchups_df["home_team_id"] == home_id].iloc[-1:]
    eval_row = recent_home.copy() if not recent_home.empty else matchups_df.iloc[-1:].copy()

    feature_cols = get_feature_columns(eval_row)
    X = eval_row[feature_cols]

    pred_margin = float(margin_model.predict(X)[0])
    pred_total = float(totals_model.predict(X)[0])
    pred_home_win = float(win_model.predict_proba(predicted_margins=np.array([pred_margin]), sigma=margin_model.residual_std)[0])
    pred_away_win = 1.0 - pred_home_win

    h_name = config.get_team_name(home_id, sport=sport)
    a_name = config.get_team_name(away_id, sport=sport)
    unit = "runs" if sport == "mlb" else "pts"

    table = Table(title=f"{sport.upper()} Forecast")
    table.add_column("Market", style="cyan")
    table.add_column("Forecast", style="bold green")
    table.add_column("Detail", style="yellow")

    table.add_row(f"{h_name} Win Prob", f"{pred_home_win * 100:.1f}%", f"Fair Odds: {1.0/pred_home_win:.2f}")
    table.add_row(f"{a_name} Win Prob", f"{pred_away_win * 100:.1f}%", f"Fair Odds: {1.0/pred_away_win:.2f}")
    table.add_row(f"Projected Score Margin", f"{pred_margin:+.2f} {unit}", f"{h_name if pred_margin > 0 else a_name} favored")
    table.add_row(f"Projected Total {unit.capitalize()}", f"{pred_total:.2f} {unit}", "Combined Score")

    console.print(table)

    if args.home_ml and args.away_ml:
        ml_eval = evaluate_moneyline_market(pred_home_win, args.home_ml, args.away_ml, min_edge=args.min_edge)
        console.print("\n[bold]Moneyline Value Analysis:[/bold]")
        for opp in ml_eval["opportunities"]:
            side_team = h_name if opp["side"] == "home" else a_name
            sizing = size_bet(opp["model_prob"], opp["decimal_odds"], args.bankroll, kelly_multiplier=args.kelly)
            console.print(f"[green]* +EV Opportunity: {side_team} @ {opp['decimal_odds']} | Edge: {opp['edge']*100:+.1f}% | EV: {opp['ev']:+.3f} | Stake: ${sizing['stake']} ({sizing['stake_pct']}%) [/green]")
        if not ml_eval["opportunities"]:
            console.print("[yellow]No edge exceeding minimum threshold found for this game.[/yellow]")

def cmd_backtest(args):
    sport = args.sport.lower()
    console.print(Panel.fit(f"[bold cyan]{sport.upper()} Historical Backtest ({args.season or 'All Seasons'})[/bold cyan]"))
    bt = HistoricalBacktester(
        sport=sport,
        starting_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        compound_bankroll=args.compound
    )
    with console.status(f"[yellow]Running chronological backtest for {sport.upper()}...[/yellow]"):
        res = bt.run_backtest(season_filter=args.season)

    table = Table(title=f"{sport.upper()} Backtest Summary Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Performance", style="bold green")

    table.add_row("Total Bets", str(res["total_bets"]))
    table.add_row("Record (W - L - P)", f"{res['wins']} - {res['losses']} - {res['pushes']}")
    table.add_row("Win Rate", f"{res['win_rate']}%")
    table.add_row("Total Staked", f"${res['total_staked']:,.2f}")
    table.add_row("Net Profit (PnL)", f"${res['pnl']:,.2f}")
    table.add_row("Return on Investment (ROI)", f"{res['roi_pct']:+.2f}%")
    table.add_row("Final Bankroll", f"${res['final_bankroll']:,.2f}")
    table.add_row("Max Drawdown", f"{res['max_drawdown_pct']:.2f}% (${res['max_drawdown_dollars']:,.2f})")
    table.add_row("Sharpe Ratio", str(res["sharpe_ratio"]))
    table.add_row("Avg Closing Line Value (CLV)", f"{res['avg_clv_pct']:+.2f}%")
    table.add_row("Beat Closing Line Rate", f"{res['beat_closing_pct']}%")

    console.print(table)

def cmd_stats(args):
    console.print(Panel.fit("[bold cyan]Multi-Sport Database Status[/bold cyan]"))
    nba_games = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='nba'")["c"]
    mlb_games = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='mlb'")["c"]
    total_logs = db.fetch_one("SELECT COUNT(*) as c FROM team_game_logs")["c"]
    total_odds = db.fetch_one("SELECT COUNT(*) as c FROM odds")["c"]

    table = Table(title="SQLite Database Overview")
    table.add_column("Sport / Entity", style="cyan")
    table.add_column("Count / Details", style="green")

    table.add_row("🏀 NBA Total Games", f"{nba_games:,}")
    table.add_row("⚾ MLB Total Games", f"{mlb_games:,}")
    table.add_row("Team Game Logs", f"{total_logs:,}")
    table.add_row("Odds Records", f"{total_odds:,}")

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Multi-Sport Quantitative Calculator CLI (NBA & MLB)")
    parser.add_argument("--sport", type=str, default="nba", choices=["nba", "mlb"], help="Sport selection (nba or mlb)")
    subparsers = parser.add_subparsers(dest="command")

    # sync
    p_sync = subparsers.add_parser("sync", help="Ingest game logs and odds")
    p_sync.add_argument("--seasons", type=str, default=None, help="Comma-separated seasons")

    # train
    p_train = subparsers.add_parser("train", help="Train and calibrate ML models")
    p_train.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict a matchup")
    p_pred.add_argument("--home", type=str, required=True, help="Home team (e.g. BOS or NYY)")
    p_pred.add_argument("--away", type=str, required=True, help="Away team (e.g. LAL or LAD)")
    p_pred.add_argument("--home-ml", type=float, default=None, help="Home decimal odds")
    p_pred.add_argument("--away-ml", type=float, default=None, help="Away decimal odds")
    p_pred.add_argument("--bankroll", type=float, default=10000.0, help="Current bankroll")
    p_pred.add_argument("--kelly", type=float, default=config.DEFAULT_KELLY_FRACTION, help="Kelly fraction")
    p_pred.add_argument("--min-edge", type=float, default=config.DEFAULT_MIN_EDGE, help="Minimum EV edge")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run historical backtesting")
    p_bt.add_argument("--season", type=str, default=None, help="Season filter")
    p_bt.add_argument("--bankroll", type=float, default=10000.0, help="Starting bankroll")
    p_bt.add_argument("--kelly", type=float, default=config.DEFAULT_KELLY_FRACTION, help="Kelly multiplier")
    p_bt.add_argument("--min-edge", type=float, default=config.DEFAULT_MIN_EDGE, help="Min edge")
    p_bt.add_argument("--compound", action="store_true", help="Dynamic compounding bankroll")

    # stats
    subparsers.add_parser("stats", help="Show database overview")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "stats":
        cmd_stats(args)

if __name__ == "__main__":
    main()