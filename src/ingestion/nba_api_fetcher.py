import time
import random
import logging
import pandas as pd
from typing import List, Optional
import config

try:
    from nba_api.stats.endpoints import leaguegamelog, scoreboardv2, leaguegamefinder
    from nba_api.stats.static import teams
except ImportError:
    leaguegamelog = None
    scoreboardv2 = None
    leaguegamefinder = None
    teams = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CUSTOM_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

def throttle_request():
    delay = config.NBA_API_REQUEST_DELAY + random.uniform(0.1, 0.4)
    time.sleep(delay)

def fetch_season_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch all team game logs for a given NBA season (e.g. '2023-24') with exponential backoff.
    """
    if leaguegamelog is None:
        logger.error("nba_api is not installed.")
        return pd.DataFrame()

    logger.info(f"Fetching NBA game logs for season {season} ({season_type})...")
    
    for attempt in range(1, config.NBA_API_MAX_RETRIES + 1):
        try:
            throttle_request()
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation='T',
                headers=CUSTOM_HEADERS,
                timeout=25
            )
            df = log.get_data_frames()[0]
            logger.info(f"Successfully fetched {len(df)} game log records for {season}.")
            return clean_game_logs_df(df, season)
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
            logger.warning(f"Attempt {attempt}/{config.NBA_API_MAX_RETRIES} failed for season {season}: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    logger.error(f"Failed to fetch game logs for season {season} after {config.NBA_API_MAX_RETRIES} retries.")
    return pd.DataFrame()

def clean_game_logs_df(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Standardizes column names and formats for SQLite storage.
    """
    if df.empty:
        return df

    cleaned = df.copy()
    # Normalize column names to lowercase
    cleaned.columns = [c.lower() for c in cleaned.columns]

    # Map Matchup to is_home and opponent
    if 'matchup' in cleaned.columns:
        cleaned['is_home'] = cleaned['matchup'].apply(lambda x: 1 if ' vs. ' in str(x) else 0)
        
        # Extract opponent abbreviation
        def parse_opp(matchup_str):
            if ' vs. ' in str(matchup_str):
                return matchup_str.split(' vs. ')[-1].strip()
            elif ' @ ' in str(matchup_str):
                return matchup_str.split(' @ ')[-1].strip()
            return ''
            
        cleaned['opp_abbrev'] = cleaned['matchup'].apply(parse_opp)
        cleaned['opponent_id'] = cleaned['opp_abbrev'].apply(config.get_team_id)

    # Format date
    if 'game_date' in cleaned.columns:
        cleaned['game_date'] = pd.to_datetime(cleaned['game_date']).dt.strftime('%Y-%m-%d')

    cleaned['season'] = season
    
    # Fill missing numeric values with 0
    numeric_cols = ['fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct',
                    'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus', 'min']
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').fillna(0)

    return cleaned

def fetch_scoreboard_for_date(game_date_str: str) -> pd.DataFrame:
    """
    Fetches scheduled games and line scores for a specific date (YYYY-MM-DD).
    """
    if scoreboardv2 is None:
        return pd.DataFrame()

    logger.info(f"Fetching NBA scoreboard for date {game_date_str}...")
    try:
        throttle_request()
        # scoreboardv2 expects game_date in 'YYYY-MM-DD' or 'MM/DD/YYYY'
        sb = scoreboardv2.ScoreboardV2(game_date=game_date_str, headers=CUSTOM_HEADERS, timeout=20)
        game_headers = sb.game_header.get_data_frame()
        return game_headers
    except Exception as e:
        logger.error(f"Scoreboard fetch failed for {game_date_str}: {e}")
        return pd.DataFrame()
