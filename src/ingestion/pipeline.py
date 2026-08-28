import logging
import pandas as pd
import numpy as np
from typing import List
from src.db.database import db
from src.ingestion.nba_api_fetcher import fetch_season_game_logs
from src.ingestion.odds_fetcher import generate_synthetic_odds_for_historical_games, fetch_live_odds_the_odds_api, parse_the_odds_api_response
from src.features.four_factors import compute_advanced_stats_dataframe
import config

logger = logging.getLogger(__name__)

def sync_season(season: str) -> int:
    """
    Syncs game logs, games, and advanced stats for a single season into SQLite.
    Returns number of games inserted.
    """
    logger.info(f"Syncing season {season}...")
    df_logs = fetch_season_game_logs(season)

    if df_logs.empty:
        logger.warning(f"No live data returned for season {season}. Attempting to check existing or generate fallback seed.")
        return 0

    return process_and_store_season_data(df_logs, season)

def process_and_store_season_data(df_logs: pd.DataFrame, season: str) -> int:
    """
    Processes raw game logs, computes advanced stats, and stores all records in SQLite.
    """
    if df_logs.empty:
        return 0

    # Ensure required columns exist
    required_cols = ['game_id', 'team_id', 'opponent_id', 'game_date', 'season', 'is_home']
    for col in required_cols:
        if col not in df_logs.columns:
            logger.error(f"Missing required column {col} in game logs.")
            return 0

    # 1. Compute Advanced Stats (Four Factors, Pace, Ratings)
    adv_df = compute_advanced_stats_dataframe(df_logs)

    # 2. Extract unique games table (1 row per game)
    home_logs = df_logs[df_logs['is_home'] == 1].copy()
    away_logs = df_logs[df_logs['is_home'] == 0].copy()

    games_merged = pd.merge(
        home_logs[['game_id', 'season', 'game_date', 'team_id', 'pts']].rename(columns={'team_id': 'home_team_id', 'pts': 'home_pts'}),
        away_logs[['game_id', 'team_id', 'pts']].rename(columns={'team_id': 'away_team_id', 'pts': 'away_pts'}),
        on='game_id',
        how='inner'
    )

    games_merged['point_margin'] = games_merged['home_pts'] - games_merged['away_pts']
    games_merged['total_points'] = games_merged['home_pts'] + games_merged['away_pts']
    games_merged['home_win'] = (games_merged['point_margin'] > 0).astype(int)
    games_merged['is_playoff'] = 0
    games_merged['status'] = 'Final'

    # 3. Store into SQLite with upsert/replace logic
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Insert or replace games
        for _, g in games_merged.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO games 
                (game_id, season, game_date, home_team_id, away_team_id, home_pts, away_pts, point_margin, total_points, home_win, is_playoff, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(g['game_id']), str(g['season']), str(g['game_date']), int(g['home_team_id']), int(g['away_team_id']),
                int(g['home_pts']), int(g['away_pts']), int(g['point_margin']), int(g['total_points']),
                int(g['home_win']), int(g['is_playoff']), str(g['status'])
            ))

        # Insert or replace team game logs
        log_cols = ['game_id', 'team_id', 'opponent_id', 'game_date', 'season', 'is_home', 'wl', 'min',
                    'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct',
                    'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus']
        
        for _, row in df_logs.iterrows():
            vals = [row.get(c, 0 if c not in ['game_id', 'season', 'game_date', 'wl'] else '') for c in log_cols]
            cursor.execute(f"""
                INSERT OR REPLACE INTO team_game_logs ({','.join(log_cols)})
                VALUES ({','.join(['?'] * len(log_cols))})
            """, vals)

        # Insert or replace advanced stats
        if not adv_df.empty:
            adv_cols = ['game_id', 'team_id', 'opponent_id', 'possessions', 'pace', 'off_rating', 'def_rating', 'net_rating',
                        'efg_pct', 'tov_pct', 'orb_pct', 'ftr', 'opp_efg_pct', 'opp_tov_pct', 'opp_orb_pct', 'opp_ftr']
            for _, row in adv_df.iterrows():
                vals = [row.get(c, 0.0 if c not in ['game_id'] else '') for c in adv_cols]
                cursor.execute(f"""
                    INSERT OR REPLACE INTO team_advanced_stats ({','.join(adv_cols)})
                    VALUES ({','.join(['?'] * len(adv_cols))})
                """, vals)

    # 4. Generate/sync odds for games
    odds_df = generate_synthetic_odds_for_historical_games(games_merged)
    if not odds_df.empty:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            for _, o in odds_df.iterrows():
                cursor.execute("""
                    INSERT INTO odds (game_id, bookmaker, home_ml_open, away_ml_open, home_ml_close, away_ml_close, spread_line, home_spread_odds, away_spread_odds, total_line, over_odds, under_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(o['game_id']), str(o['bookmaker']), float(o['home_ml_open']), float(o['away_ml_open']),
                    float(o['home_ml_close']), float(o['away_ml_close']), float(o['spread_line']),
                    float(o['home_spread_odds']), float(o['away_spread_odds']), float(o['total_line']),
                    float(o['over_odds']), float(o['under_odds'])
                ))

    logger.info(f"Stored {len(games_merged)} games and {len(df_logs)} game logs for season {season}.")
    return len(games_merged)

def generate_seed_dataset_if_empty(seasons: List[str] = ["2022-23", "2023-24", "2024-25"]) -> int:
    """
    Generates realistic historical NBA dataset seed across multiple seasons
    for instant local training and offline development.
    """
    count = db.fetch_one("SELECT COUNT(*) as c FROM games")['c']
    if count > 500:
        logger.info(f"Database already contains {count} games. Skipping seed generation.")
        return count

    logger.info("Database empty or sparse. Generating comprehensive realistic multi-season NBA seed dataset...")
    all_logs = []
    rng = np.random.RandomState(42)
    team_ids = list(config.NBA_TEAMS.keys())

    # Generate synthetic team baseline power ratings
    team_ratings = {t: rng.normal(0, 3.5) for t in team_ids}

    game_counter = 22200001

    for season_idx, season in enumerate(seasons):
        year_start = 2022 + season_idx
        start_date = pd.Timestamp(f"{year_start}-10-20")
        
        # Schedule ~1230 games per regular season
        for day_offset in range(170):
            current_date = (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            
            # Select random pairs playing on this day (4 to 10 games per day)
            shuffled_teams = rng.permutation(team_ids).tolist()
            num_games = rng.randint(4, 9)
            
            for g in range(num_games):
                if len(shuffled_teams) < 2:
                    break
                home_team = shuffled_teams.pop()
                away_team = shuffled_teams.pop()
                
                game_id = f"00{game_counter}"
                game_counter += 1

                # Possessions ~ 100 +- 4
                poss = int(round(rng.normal(100.0, 3.5)))
                
                # Base scoring efficiency: Offense Rating ~ 114, HCA +3.0 pts
                h_base_eff = 1.14 + (team_ratings[home_team] + 2.8) / 100.0 + rng.normal(0, 0.08)
                a_base_eff = 1.14 + (team_ratings[away_team] - 1.0) / 100.0 + rng.normal(0, 0.08)

                home_pts = max(int(round(poss * h_base_eff)), 80)
                away_pts = max(int(round(poss * a_base_eff)), 80)
                if home_pts == away_pts:
                    home_pts += rng.choice([1, 2, 3]) # No ties in NBA

                # Detailed Box Scores
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
                    'game_id': game_id, 'team_id': home_team, 'opponent_id': away_team,
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
                    'game_id': game_id, 'team_id': away_team, 'opponent_id': home_team,
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
