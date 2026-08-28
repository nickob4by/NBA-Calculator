import pandas as pd
import numpy as np

def calculate_possessions(fga, fta, oreb, tov, opp_fga, opp_fta, opp_oreb, opp_tov) -> float:
    """
    Standard Dean Oliver formula for game possessions.
    """
    team_poss = fga + 0.44 * fta - oreb + tov
    opp_poss = opp_fga + 0.44 * opp_fta - opp_oreb + opp_tov
    poss = 0.5 * (team_poss + opp_poss)
    return max(poss, 1.0)

def calculate_pace(possessions: float, minutes: float = 240.0) -> float:
    """
    Pace normalized to 48 minutes (standard NBA regulation game total team minutes = 240, 5 players x 48).
    """
    if minutes <= 0:
        minutes = 240.0
    return (possessions / (minutes / 5.0)) * 48.0 if minutes > 0 else possessions

def calculate_four_factors(team_stats: dict, opp_stats: dict) -> dict:
    """
    Calculate Dean Oliver's Four Factors for both offense and defense.
    """
    fga = team_stats.get('fga', 0)
    fgm = team_stats.get('fgm', 0)
    fg3m = team_stats.get('fg3m', 0)
    fta = team_stats.get('fta', 0)
    oreb = team_stats.get('oreb', 0)
    dreb = team_stats.get('dreb', 0)
    tov = team_stats.get('tov', 0)
    pts = team_stats.get('pts', 0)

    opp_fga = opp_stats.get('fga', 0)
    opp_fgm = opp_stats.get('fgm', 0)
    opp_fg3m = opp_stats.get('fg3m', 0)
    opp_fta = opp_stats.get('fta', 0)
    opp_oreb = opp_stats.get('oreb', 0)
    opp_dreb = opp_stats.get('dreb', 0)
    opp_tov = opp_stats.get('tov', 0)
    opp_pts = opp_stats.get('pts', 0)

    minutes = team_stats.get('min', 240)

    # 1. Effective Field Goal Percentage (eFG%)
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
    opp_efg_pct = (opp_fgm + 0.5 * opp_fg3m) / opp_fga if opp_fga > 0 else 0.0

    # 2. Turnover Percentage (TOV%)
    tov_denom = (fga + 0.44 * fta + tov)
    tov_pct = tov / tov_denom if tov_denom > 0 else 0.0
    opp_tov_denom = (opp_fga + 0.44 * opp_fta + opp_tov)
    opp_tov_pct = opp_tov / opp_tov_denom if opp_tov_denom > 0 else 0.0

    # 3. Offensive Rebound Percentage (OREB%)
    orb_denom = (oreb + opp_dreb)
    orb_pct = oreb / orb_denom if orb_denom > 0 else 0.0
    opp_orb_denom = (opp_oreb + dreb)
    opp_orb_pct = opp_oreb / opp_orb_denom if opp_orb_denom > 0 else 0.0

    # 4. Free Throw Rate (FTR)
    ftr = fta / fga if fga > 0 else 0.0
    opp_ftr = opp_fta / opp_fga if opp_fga > 0 else 0.0

    # Possessions and Pace
    poss = calculate_possessions(fga, fta, oreb, tov, opp_fga, opp_fta, opp_oreb, opp_tov)
    pace = calculate_pace(poss, minutes)

    # Ratings per 100 possessions
    off_rating = (pts / poss) * 100.0 if poss > 0 else 0.0
    def_rating = (opp_pts / poss) * 100.0 if poss > 0 else 0.0
    net_rating = off_rating - def_rating

    return {
        'possessions': round(poss, 2),
        'pace': round(pace, 2),
        'off_rating': round(off_rating, 2),
        'def_rating': round(def_rating, 2),
        'net_rating': round(net_rating, 2),
        'efg_pct': round(efg_pct, 4),
        'tov_pct': round(tov_pct, 4),
        'orb_pct': round(orb_pct, 4),
        'ftr': round(ftr, 4),
        'opp_efg_pct': round(opp_efg_pct, 4),
        'opp_tov_pct': round(opp_tov_pct, 4),
        'opp_orb_pct': round(opp_orb_pct, 4),
        'opp_ftr': round(opp_ftr, 4)
    }

def compute_advanced_stats_dataframe(game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of game logs (where each game has two rows: team and opponent)
    and computes advanced stats for each team game.
    """
    if game_logs_df.empty:
        return pd.DataFrame()

    records = []
    # Group by game_id
    grouped = game_logs_df.groupby('game_id')
    for game_id, group in grouped:
        if len(group) != 2:
            continue
        row_a = group.iloc[0].to_dict()
        row_b = group.iloc[1].to_dict()

        stats_a = calculate_four_factors(row_a, row_b)
        stats_a['game_id'] = game_id
        stats_a['team_id'] = row_a['team_id']
        stats_a['opponent_id'] = row_b['team_id']

        stats_b = calculate_four_factors(row_b, row_a)
        stats_b['game_id'] = game_id
        stats_b['team_id'] = row_b['team_id']
        stats_b['opponent_id'] = row_a['team_id']

        records.append(stats_a)
        records.append(stats_b)

    return pd.DataFrame(records)
