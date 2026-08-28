import requests
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import config

logger = logging.getLogger(__name__)

def fetch_live_odds_the_odds_api(api_key: Optional[str] = None) -> List[Dict]:
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

            if 'h2h' in markets:
                for outcome in markets['h2h']:
                    if outcome.get('name') == home_team_name:
                        home_ml = outcome.get('price')
                    elif outcome.get('name') == away_team_name:
                        away_ml = outcome.get('price')

            if 'spreads' in markets:
                for outcome in markets['spreads']:
                    if outcome.get('name') == home_team_name:
                        spread_line = outcome.get('point')
                        home_spread_odds = outcome.get('price')
                    elif outcome.get('name') == away_team_name:
                        away_spread_odds = outcome.get('price')

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

def generate_synthetic_odds_for_historical_games(games_df: pd.DataFrame, sport: str = "nba") -> pd.DataFrame:
    """
    Generates realistic pre-game market opening & closing odds based on team power ratings
    and realistic market efficiency, with 4.5% standard bookmaker vigorish.
    """
    if games_df.empty:
        return pd.DataFrame()

    odds_records = []
    rng = np.random.RandomState(42)
    from scipy.stats import norm

    # Generate persistent team baseline power ratings for pre-game pricing
    all_teams = np.unique(np.concatenate([games_df['home_team_id'].unique(), games_df['away_team_id'].unique()]))
    team_ratings = {t: rng.normal(0.0, 3.5 if sport == "nba" else 0.4) for t in all_teams}

    for _, row in games_df.iterrows():
        h_id = row['home_team_id']
        a_id = row['away_team_id']

        h_rat = team_ratings.get(h_id, 0.0)
        a_rat = team_ratings.get(a_id, 0.0)

        if sport == "mlb":
            # MLB Pre-Game Expected Margin: Home Rating - Away Rating + HFA (0.30 runs) + Market Noise
            exp_margin = (h_rat - a_rat) + 0.30 + rng.normal(0, 0.35)
            implied_p_home = float(norm.cdf(exp_margin / 3.2))
            implied_p_home = np.clip(implied_p_home, 0.32, 0.68)
            implied_p_away = 1.0 - implied_p_home
            spread_line = -1.5
            total_line = round((8.5 + (h_rat + a_rat)*0.2 + rng.normal(0, 0.5)) * 2.0) / 2.0
            h_sp_odds = 2.15
            a_sp_odds = 1.75
        else:
            # NBA Pre-Game Expected Margin: Home Rating - Away Rating + HFA (2.5 pts) + Market Noise
            exp_margin = (h_rat - a_rat) + 2.5 + rng.normal(0, 2.0)
            market_spread = round((-exp_margin) * 2.0) / 2.0
            market_spread = np.clip(market_spread, -14.5, 14.5)
            implied_p_home = float(norm.cdf(-market_spread / 12.0))
            implied_p_home = np.clip(implied_p_home, 0.15, 0.85)
            implied_p_away = 1.0 - implied_p_home
            spread_line = market_spread
            total_line = round((224.0 + (h_rat + a_rat)*0.5 + rng.normal(0, 4.0)) * 2.0) / 2.0
            h_sp_odds = 1.91
            a_sp_odds = 1.91

        # Add 4.5% bookmaker vig
        vig = 0.045
        home_vig_prob = implied_p_home * (1 + vig / 2)
        away_vig_prob = implied_p_away * (1 + vig / 2)

        home_ml_close = round(1.0 / home_vig_prob, 2)
        away_ml_close = round(1.0 / away_vig_prob, 2)

        # Opening line with realistic market line drift
        drift = rng.normal(0, 0.03)
        home_ml_open = round(1.0 / (np.clip(home_vig_prob + drift, 0.10, 0.90)), 2)
        away_ml_open = round(1.0 / (np.clip(away_vig_prob - drift, 0.10, 0.90)), 2)

        odds_records.append({
            'game_id': str(row['game_id']),
            'sport': sport,
            'bookmaker': 'Pinnacle_Consensus',
            'home_ml_open': home_ml_open,
            'away_ml_open': away_ml_open,
            'home_ml_close': home_ml_close,
            'away_ml_close': away_ml_close,
            'spread_line': spread_line,
            'home_spread_odds': h_sp_odds,
            'away_spread_odds': a_sp_odds,
            'total_line': total_line,
            'over_odds': 1.91,
            'under_odds': 1.91
        })

    return pd.DataFrame(odds_records)