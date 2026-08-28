import requests
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import config

logger = logging.getLogger(__name__)

def fetch_live_odds_the_odds_api(api_key: Optional[str] = None) -> List[Dict]:
    """
    Fetches live NBA odds from The-Odds-API.
    """
    key = api_key or config.ODDS_API_KEY
    if not key:
        logger.warning("No ODDS_API_KEY configured. Returning empty live odds.")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{config.ODDS_API_SPORT}/odds/"
    params = {
        'apiKey': key,
        'regions': config.ODDS_API_REGIONS,
        'markets': config.ODDS_API_MARKETS,
        'oddsFormat': 'decimal'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Retrieved live odds for {len(data)} events from The-Odds-API.")
        return data
    except Exception as e:
        logger.error(f"Error fetching from The-Odds-API: {e}")
        return []

def parse_the_odds_api_response(events: List[Dict]) -> pd.DataFrame:
    """
    Parses The-Odds-API raw JSON response into a normalized DataFrame.
    """
    records = []
    for event in events:
        home_team_name = event.get('home_team')
        away_team_name = event.get('away_team')
        commence_time = event.get('commence_time')
        bookmakers = event.get('bookmakers', [])

        home_id = config.get_team_id(home_team_name)
        away_id = config.get_team_id(away_team_name)

        if not bookmakers:
            continue

        # Use first available major bookmaker or average
        for bm in bookmakers:
            bm_title = bm.get('title', 'Unknown')
            markets = {m['key']: m['outcomes'] for m in bm.get('markets', [])}

            home_ml = None
            away_ml = None
            spread_line = None
            home_spread_odds = -110
            away_spread_odds = -110
            total_line = None
            over_odds = -110
            under_odds = -110

            # Moneyline (h2h)
            if 'h2h' in markets:
                for outcome in markets['h2h']:
                    if outcome.get('name') == home_team_name:
                        home_ml = outcome.get('price')
                    elif outcome.get('name') == away_team_name:
                        away_ml = outcome.get('price')

            # Spreads
            if 'spreads' in markets:
                for outcome in markets['spreads']:
                    if outcome.get('name') == home_team_name:
                        spread_line = outcome.get('point') # e.g. -4.5
                        home_spread_odds = outcome.get('price')
                    elif outcome.get('name') == away_team_name:
                        away_spread_odds = outcome.get('price')

            # Totals
            if 'totals' in markets:
                for outcome in markets['totals']:
                    if outcome.get('name') == 'Over':
                        total_line = outcome.get('point')
                        over_odds = outcome.get('price')
                    elif outcome.get('name') == 'Under':
                        under_odds = outcome.get('price')

            records.append({
                'home_team_id': home_id,
                'away_team_id': away_id,
                'bookmaker': bm_title,
                'commence_time': commence_time,
                'home_ml_close': home_ml,
                'away_ml_close': away_ml,
                'spread_line': spread_line,
                'home_spread_odds': home_spread_odds,
                'away_spread_odds': away_spread_odds,
                'total_line': total_line,
                'over_odds': over_odds,
                'under_odds': under_odds
            })

    return pd.DataFrame(records)

def generate_synthetic_odds_for_historical_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates realistic synthetic opening & closing market odds for historical games
    where external betting API data is unavailable, reflecting realistic market efficiency
    with 4.5% standard bookmaker vig.
    """
    if games_df.empty:
        return pd.DataFrame()

    odds_records = []
    rng = np.random.RandomState(42)

    for _, row in games_df.iterrows():
        margin = row.get('point_margin', 0)
        # True market spread estimation with noise
        market_spread = round((-(margin + rng.normal(0, 4.0)) / 0.5)) * 0.5
        market_spread = np.clip(market_spread, -16.5, 16.5)

        # Implied win prob from spread via normal CDF (sigma=13.5 pts)
        from scipy.stats import norm
        implied_p_home = norm.cdf(-market_spread / 13.5)
        implied_p_away = 1.0 - implied_p_home

        # Add 4.5% bookmaker vig
        vig = 0.045
        home_vig_prob = implied_p_home * (1 + vig / 2)
        away_vig_prob = implied_p_away * (1 + vig / 2)

        home_ml_close = round(1.0 / home_vig_prob, 2)
        away_ml_close = round(1.0 / away_vig_prob, 2)

        # Opening line with slight market drift
        drift = rng.normal(0, 0.05)
        home_ml_open = round(1.0 / (np.clip(home_vig_prob + drift, 0.05, 0.95)), 2)
        away_ml_open = round(1.0 / (np.clip(away_vig_prob - drift, 0.05, 0.95)), 2)

        total_line = round((224.0 + rng.normal(0, 6.0)) / 0.5) * 0.5

        odds_records.append({
            'game_id': str(row['game_id']),
            'bookmaker': 'Pinnacle_Consensus',
            'home_ml_open': home_ml_open,
            'away_ml_open': away_ml_open,
            'home_ml_close': home_ml_close,
            'away_ml_close': away_ml_close,
            'spread_line': market_spread,
            'home_spread_odds': 1.91,
            'away_spread_odds': 1.91,
            'total_line': total_line,
            'over_odds': 1.91,
            'under_odds': 1.91
        })

    return pd.DataFrame(odds_records)
