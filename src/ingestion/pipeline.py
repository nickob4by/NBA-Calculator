import logging
import pandas as pd
import numpy as np
from typing import List
from src.db.database import db
from src.ingestion.nba_api_fetcher import fetch_season_game_logs
from src.ingestion.odds_fetcher import generate_synthetic_odds_for_historical_games
from src.features.four_factors import compute_advanced_stats_dataframe
import config

logger = logging.getLogger(__name__)

from datetime import datetime

def get_dynamic_nba_seasons(start_year: int = 2020) -> List[str]:
    curr_year = datetime.now().year
    seasons = []
    for y in range(start_year, max(curr_year + 1, 2025)):
        next_y_short = str(y + 1)[-2:]
        seasons.append(f"{y}-{next_y_short}")
    return seasons

DEFAULT_NBA_5YR_SEASONS = get_dynamic_nba_seasons()

def sync_season(season: str) -> int:
    """
    Syncs game logs, games, and advanced stats for a single NBA season into SQLite.
    """
    logger.info(f"Syncing season {season}...")
    df_logs = fetch_season_game_logs(season)

    if df_logs.empty:
        logger.warning(f"No live data returned for season {season}. Attempting fallback seed generation.")
        return 0

    return process_and_store_season_data(df_logs, season)

def process_and_store_season_data(df_logs: pd.DataFrame, season: str) -> int:
    """
    Processes raw game logs, computes advanced stats, and stores all records in SQLite.
    """
    if df_logs.empty:
        return 0

    df_logs["sport"] = "nba"
    adv_df = compute_advanced_stats_dataframe(df_logs)

    home_logs = df_logs[df_logs["is_home"] == 1].copy()
    away_logs = df_logs[df_logs["is_home"] == 0].copy()

    games_merged = pd.merge(
        home_logs[["game_id", "season", "game_date", "team_id", "pts"]].rename(columns={"team_id": "home_team_id", "pts": "home_pts"}),
        away_logs[["game_id", "team_id", "pts"]].rename(columns={"team_id": "away_team_id", "pts": "away_pts"}),
        on="game_id",
        how="inner"
    )

    games_merged["sport"] = "nba"
    games_merged["point_margin"] = games_merged["home_pts"] - games_merged["away_pts"]
    games_merged["total_points"] = games_merged["home_pts"] + games_merged["away_pts"]
    games_merged["home_win"] = (games_merged["point_margin"] > 0).astype(int)
    games_merged["is_playoff"] = 0
    games_merged["status"] = "Final"

    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        for _, g in games_merged.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO games 
                (game_id, sport, season, game_date, home_team_id, away_team_id, home_pts, away_pts, point_margin, total_points, home_win, is_playoff, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(g['game_id']), str(g['sport']), str(g['season']), str(g['game_date']), int(g['home_team_id']), int(g['away_team_id']),
                int(g['home_pts']), int(g['away_pts']), int(g['point_margin']), int(g['total_points']),
                int(g['home_win']), int(g['is_playoff']), str(g['status'])
            ))

        log_cols = ["game_id", "sport", "team_id", "opponent_id", "game_date", "season", "is_home", "wl", "min",
                    "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct",
                    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts", "plus_minus"]
        
        for _, row in df_logs.iterrows():
            vals = [row.get(c, 0 if c not in ["game_id", "sport", "season", "game_date", "wl"] else "") for c in log_cols]
            cursor.execute(f"""
                INSERT OR REPLACE INTO team_game_logs ({','.join(log_cols)})
                VALUES ({','.join(['?'] * len(log_cols))})
            """, vals)

        if not adv_df.empty:
            adv_cols = ["game_id", "sport", "team_id", "opponent_id", "possessions", "pace", "off_rating", "def_rating", "net_rating",
                        "efg_pct", "tov_pct", "orb_pct", "ftr", "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ftr"]
            for _, row in adv_df.iterrows():
                vals = [row.get(c, 0.0 if c not in ["game_id", "sport"] else "") for c in adv_cols]
                cursor.execute(f"""
                    INSERT OR REPLACE INTO team_advanced_stats ({','.join(adv_cols)})
                    VALUES ({','.join(['?'] * len(adv_cols))})
                """, vals)

    odds_df = generate_synthetic_odds_for_historical_games(games_merged)
    if not odds_df.empty:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            for _, o in odds_df.iterrows():
                cursor.execute("""
                    INSERT INTO odds (game_id, sport, bookmaker, home_ml_open, away_ml_open, home_ml_close, away_ml_close, spread_line, home_spread_odds, away_spread_odds, total_line, over_odds, under_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(o['game_id']), "nba", str(o['bookmaker']), float(o['home_ml_open']), float(o['away_ml_open']),
                    float(o['home_ml_close']), float(o['away_ml_close']), float(o['spread_line']),
                    float(o['home_spread_odds']), float(o['away_spread_odds']), float(o['total_line']),
                    float(o['over_odds']), float(o['under_odds'])
                ))

    logger.info(f"Stored {len(games_merged)} NBA games and {len(df_logs)} game logs for season {season}.")
    return len(games_merged)

from typing import Optional

def generate_seed_dataset_if_empty(seasons: Optional[List[str]] = None, force_refresh: bool = False) -> int:
    """
    Generates realistic historical NBA dataset seed dynamically covering seasons up to the current calendar year.
    Automatically discovers and adds upcoming seasons (e.g. 2025-26, 2026-27).
    """
    if seasons is None:
        seasons = get_dynamic_nba_seasons()

    latest_target_season = seasons[-1]
    count = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='nba'")["c"]
    has_latest = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='nba' AND season=?", (latest_target_season,))["c"] > 0

    if not force_refresh and count >= (len(seasons) * 1000) and has_latest:
        return count

    logger.info(f"Generating comprehensive NBA historical dataset up to {latest_target_season} ({', '.join(seasons)})...")
    all_logs = []
    rng = np.random.RandomState(42)
    team_ids = list(config.NBA_TEAMS.keys())
    team_ratings = {t: rng.normal(0, 3.5) for t in team_ids}

    game_counter = 22000001

    for season_idx, season in enumerate(seasons):
        year_start = 2020 + season_idx
        start_date = pd.Timestamp(f"{year_start}-10-20")
        
        for day_offset in range(170):
            current_date = (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            shuffled_teams = rng.permutation(team_ids).tolist()
            num_games = rng.randint(4, 9)
            
            for g in range(num_games):
                if len(shuffled_teams) < 2:
                    break
                home_team = shuffled_teams.pop()
                away_team = shuffled_teams.pop()
                
                game_id = f"nba_{game_counter}"
                game_counter += 1

                poss = int(round(rng.normal(100.0, 3.5)))
                h_base_eff = 1.14 + (team_ratings[home_team] + 2.8) / 100.0 + rng.normal(0, 0.08)
                a_base_eff = 1.14 + (team_ratings[away_team] - 1.0) / 100.0 + rng.normal(0, 0.08)

                home_pts = max(int(round(poss * h_base_eff)), 80)
                away_pts = max(int(round(poss * a_base_eff)), 80)
                if home_pts == away_pts:
                    home_pts += rng.choice([1, 2, 3])

                h_fga = rng.randint(84, 96)
                h_fgm = int(round(h_fga * rng.uniform(0.44, 0.52)))
                h_fg3a = rng.randint(28, 42)
                h_fg3m = int(round(h_fg3a * rng.uniform(0.32, 0.40)))
                h_fta = rng.randint(16, 26)
                h_ftm = int(round(h_fta * rng.uniform(0.72, 0.84)))
                h_pts = (h_fgm - h_fg3m) * 2 + h_fg3m * 3 + h_ftm
                
                a_fga = rng.randint(84, 96)
                a_fgm = int(round(a_fga * rng.uniform(0.43, 0.51)))
                a_fg3a = rng.randint(28, 42)
                a_fg3m = int(round(a_fg3a * rng.uniform(0.31, 0.39)))
                a_fta = rng.randint(16, 26)
                a_ftm = int(round(a_fta * rng.uniform(0.72, 0.84)))
                a_pts = (a_fgm - a_fg3m) * 2 + a_fg3m * 3 + a_ftm
                if h_pts == a_pts:
                    h_pts += 2
                    h_fgm += 1

                home_log = {
                    'game_id': game_id, 'sport': 'nba', 'team_id': home_team, 'opponent_id': away_team,
                    'game_date': current_date, 'season': season, 'is_home': 1,
                    'wl': 'W' if h_pts > a_pts else 'L', 'min': 240,
                    'fgm': h_fgm, 'fga': h_fga, 'fg_pct': round(h_fgm / h_fga, 3),
                    'fg3m': h_fg3m, 'fg3a': h_fg3a, 'fg3_pct': round(h_fg3m / h_fg3a, 3),
                    'ftm': h_ftm, 'fta': h_fta, 'ft_pct': round(h_ftm / h_fta, 3),
                    'oreb': rng.randint(8, 14), 'dreb': rng.randint(30, 38),
                    'reb': rng.randint(38, 52), 'ast': rng.randint(22, 32),
                    'stl': rng.randint(5, 11), 'blk': rng.randint(3, 8),
                    'tov': rng.randint(10, 18), 'pf': rng.randint(16, 24),
                    'pts': h_pts, 'plus_minus': h_pts - a_pts
                }

                away_log = {
                    'game_id': game_id, 'sport': 'nba', 'team_id': away_team, 'opponent_id': home_team,
                    'game_date': current_date, 'season': season, 'is_home': 0,
                    'wl': 'W' if a_pts > h_pts else 'L', 'min': 240,
                    'fgm': a_fgm, 'fga': a_fga, 'fg_pct': round(a_fgm / a_fga, 3),
                    'fg3m': a_fg3m, 'fg3a': a_fg3a, 'fg3_pct': round(a_fg3m / a_fg3a, 3),
                    'ftm': a_ftm, 'fta': a_fta, 'ft_pct': round(a_ftm / a_fta, 3),
                    'oreb': rng.randint(8, 14), 'dreb': rng.randint(30, 38),
                    'reb': rng.randint(38, 52), 'ast': rng.randint(20, 30),
                    'stl': rng.randint(5, 11), 'blk': rng.randint(3, 8),
                    'tov': rng.randint(10, 18), 'pf': rng.randint(16, 24),
                    'pts': a_pts, 'plus_minus': a_pts - h_pts
                }

                all_logs.append(home_log)
                all_logs.append(away_log)

    df_all = pd.DataFrame(all_logs)
    total_stored = 0
    for s in seasons:
        s_df = df_all[df_all['season'] == s]
        total_stored += process_and_store_season_data(s_df, s)

    return total_stored