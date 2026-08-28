import time
import random
import logging
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import config
from src.db.database import db
from src.features.mlb_sabermetrics import compute_mlb_sabermetrics_dataframe
from src.ingestion.odds_fetcher import generate_synthetic_odds_for_historical_games

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_season_schedule(season: str) -> pd.DataFrame:
    """
    Fetches real MLB regular season schedule and game scores from official open MLB StatsAPI.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "hydrate": "linescore"
    }

    try:
        logger.info(f"Fetching MLB schedule for season {season} from {url}...")
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for date_obj in data.get("dates", []):
            game_date = date_obj.get("date")
            for g in date_obj.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue

                game_pk = str(g.get("gamePk"))
                teams = g.get("teams", {})
                home = teams.get("home", {})
                away = teams.get("away", {})

                home_id = home.get("team", {}).get("id")
                away_id = away.get("team", {}).get("id")

                home_score = home.get("score")
                away_score = away.get("score")

                if home_score is None or away_score is None:
                    continue

                records.append({
                    "game_id": f"mlb_{game_pk}",
                    "sport": "mlb",
                    "season": season,
                    "game_date": game_date,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_pts": int(home_score),
                    "away_pts": int(away_score),
                    "home_runs": int(home_score),
                    "away_runs": int(away_score),
                    "point_margin": int(home_score) - int(away_score),
                    "total_points": int(home_score) + int(away_score),
                    "home_win": 1 if int(home_score) > int(away_score) else 0,
                    "status": "Final"
                })

        df = pd.DataFrame(records)
        logger.info(f"Retrieved {len(df)} MLB completed games for season {season}.")
        return df
    except Exception as e:
        logger.warning(f"Error fetching from MLB StatsAPI for {season}: {e}")
        return pd.DataFrame()

DEFAULT_MLB_5YR_SEASONS = ["2020", "2021", "2022", "2023", "2024", "2025"]

def generate_mlb_seed_dataset_if_empty(seasons: List[str] = DEFAULT_MLB_5YR_SEASONS, force_refresh: bool = False) -> int:
    """
    Generates realistic 6-year historical MLB dataset seed across seasons (2020-2025)
    with realistic baseball runs distributions, pitcher/batter sabermetrics, and 162-game schedule.
    """
    count = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='mlb'")["c"]
    has_2025 = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='mlb' AND season='2025'")["c"] > 0
    if not force_refresh and count >= 14000 and has_2025:
        logger.info(f"Database already contains {count} MLB games including 2025. Skipping seed generation.")
        return count

    logger.info(f"Generating comprehensive realistic 5-season MLB seed dataset ({', '.join(seasons)})...")
    rng = np.random.RandomState(42)
    team_ids = list(config.MLB_TEAMS.keys())

    # True team offensive & pitching talent ratings
    off_ratings = {t: rng.normal(4.5, 0.45) for t in team_ids} # Expected runs scored per game
    def_ratings = {t: rng.normal(4.5, 0.45) for t in team_ids} # Expected runs allowed per game

    all_logs = []
    all_games = []
    game_counter = 7000001

    for season in seasons:
        start_date = pd.Timestamp(f"{season}-04-01")
        
        # Simulate ~180 days season with ~12-15 games per day (~2430 games per season)
        for day_offset in range(165):
            current_date = (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            shuffled_teams = rng.permutation(team_ids).tolist()
            
            while len(shuffled_teams) >= 2:
                home_team = shuffled_teams.pop()
                away_team = shuffled_teams.pop()
                
                game_id = f"mlb_{game_counter}"
                game_counter += 1

                park_factor = config.MLB_TEAMS[home_team].get("park_factor", 1.0)

                # Poisson / Neg-Binomial distribution for baseball runs with Home Field Advantage ~ 0.35 runs
                lambda_home = (off_ratings[home_team] * (def_ratings[away_team] / 4.5) * park_factor) + 0.35
                lambda_away = (off_ratings[away_team] * (def_ratings[home_team] / 4.5) * park_factor)

                home_runs = max(int(rng.poisson(lambda_home)), 0)
                away_runs = max(int(rng.poisson(lambda_away)), 0)
                if home_runs == away_runs:
                    home_runs += rng.choice([1, 2]) # Extra innings walk-off

                h_hits = max(int(round(home_runs + rng.normal(4.5, 2.0))), 2)
                h_hr = min(int(rng.poisson(1.1)), home_runs)
                h_bb = int(rng.poisson(3.2))
                h_so = int(rng.poisson(8.5))
                
                a_hits = max(int(round(away_runs + rng.normal(4.5, 2.0))), 2)
                a_hr = min(int(rng.poisson(1.1)), away_runs)
                a_bb = int(rng.poisson(3.2))
                a_so = int(rng.poisson(8.5))

                home_log = {
                    "game_id": game_id, "sport": "mlb", "team_id": home_team, "opponent_id": away_team,
                    "game_date": current_date, "season": season, "is_home": 1,
                    "wl": "W" if home_runs > away_runs else "L",
                    "min": 54, "pts": home_runs, "runs": home_runs, "hits": h_hits, "hr": h_hr,
                    "bb": h_bb, "so": h_so, "errors": rng.choice([0, 0, 1, 2]), "rbi": max(home_runs - 1, 0),
                    "plus_minus": home_runs - away_runs, "ip": 9.0, "er": away_runs
                }

                away_log = {
                    "game_id": game_id, "sport": "mlb", "team_id": away_team, "opponent_id": home_team,
                    "game_date": current_date, "season": season, "is_home": 0,
                    "wl": "W" if away_runs > home_runs else "L",
                    "min": 54, "pts": away_runs, "runs": away_runs, "hits": a_hits, "hr": a_hr,
                    "bb": a_bb, "so": a_so, "errors": rng.choice([0, 0, 1, 2]), "rbi": max(away_runs - 1, 0),
                    "plus_minus": away_runs - home_runs, "ip": 9.0, "er": home_runs
                }

                all_logs.append(home_log)
                all_logs.append(away_log)

                all_games.append({
                    "game_id": game_id, "sport": "mlb", "season": season, "game_date": current_date,
                    "home_team_id": home_team, "away_team_id": away_team,
                    "home_pts": home_runs, "away_pts": away_runs,
                    "point_margin": home_runs - away_runs,
                    "total_points": home_runs + away_runs,
                    "home_win": 1 if home_runs > away_runs else 0,
                    "is_playoff": 0, "status": "Final"
                })

    df_logs = pd.DataFrame(all_logs)
    df_games = pd.DataFrame(all_games)

    # Compute Sabermetrics
    adv_df = compute_mlb_sabermetrics_dataframe(df_logs)

    # Store into SQLite
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for _, g in df_games.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO games 
                (game_id, sport, season, game_date, home_team_id, away_team_id, home_pts, away_pts, point_margin, total_points, home_win, is_playoff, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(g['game_id']), str(g['sport']), str(g['season']), str(g['game_date']),
                int(g['home_team_id']), int(g['away_team_id']), int(g['home_pts']), int(g['away_pts']),
                int(g['point_margin']), int(g['total_points']), int(g['home_win']), int(g['is_playoff']), str(g['status'])
            ))

        log_cols = ["game_id", "sport", "team_id", "opponent_id", "game_date", "season", "is_home", "wl",
                    "pts", "runs", "hits", "hr", "bb", "so", "errors", "rbi", "plus_minus", "ip", "er"]
        for _, row in df_logs.iterrows():
            vals = [row.get(c, 0 if c not in ["game_id", "sport", "season", "game_date", "wl"] else "") for c in log_cols]
            cursor.execute(f"""
                INSERT OR REPLACE INTO team_game_logs ({','.join(log_cols)})
                VALUES ({','.join(['?'] * len(log_cols))})
            """, vals)

        if not adv_df.empty:
            adv_cols = ["game_id", "sport", "team_id", "opponent_id", "pythag_win_pct", "obp", "slg", "ops", "iso", "woba_proxy", "fip_proxy", "whip", "k_per_9", "bb_per_9"]
            for _, row in adv_df.iterrows():
                vals = [row.get(c, 0.0 if c not in ["game_id", "sport"] else "") for c in adv_cols]
                cursor.execute(f"""
                    INSERT OR REPLACE INTO team_advanced_stats ({','.join(adv_cols)})
                    VALUES ({','.join(['?'] * len(adv_cols))})
                """, vals)

    # Generate MLB Odds (Moneyline, Run Line -1.5/+1.5, Totals ~ 8.5)
    odds_records = []
    for _, row in df_games.iterrows():
        margin = row["point_margin"]
        from scipy.stats import norm
        # MLB run line is standard -1.5
        p_home_win = norm.cdf((margin + rng.normal(0, 1.5)) / 3.2)
        p_home_win = np.clip(p_home_win, 0.25, 0.75)
        p_away_win = 1.0 - p_home_win

        vig = 0.045
        home_ml_close = round(1.0 / (p_home_win * (1 + vig/2)), 2)
        away_ml_close = round(1.0 / (p_away_win * (1 + vig/2)), 2)

        odds_records.append({
            "game_id": str(row["game_id"]),
            "sport": "mlb",
            "bookmaker": "Pinnacle_Consensus",
            "home_ml_open": home_ml_close,
            "away_ml_open": away_ml_close,
            "home_ml_close": home_ml_close,
            "away_ml_close": away_ml_close,
            "spread_line": -1.5, # Run Line
            "home_spread_odds": 2.15,
            "away_spread_odds": 1.75,
            "total_line": 8.5,
            "over_odds": 1.91,
            "under_odds": 1.91
        })

    odds_df = pd.DataFrame(odds_records)
    if not odds_df.empty:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            for _, o in odds_df.iterrows():
                cursor.execute("""
                    INSERT INTO odds (game_id, sport, bookmaker, home_ml_open, away_ml_open, home_ml_close, away_ml_close, spread_line, home_spread_odds, away_spread_odds, total_line, over_odds, under_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(o['game_id']), str(o['sport']), str(o['bookmaker']), float(o['home_ml_open']), float(o['away_ml_open']),
                    float(o['home_ml_close']), float(o['away_ml_close']), float(o['spread_line']),
                    float(o['home_spread_odds']), float(o['away_spread_odds']), float(o['total_line']),
                    float(o['over_odds']), float(o['under_odds'])
                ))

    logger.info(f"Stored {len(df_games)} MLB games and {len(df_logs)} MLB game logs.")
    return len(df_games)