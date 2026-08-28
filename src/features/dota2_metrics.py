import pandas as pd
import numpy as np
import config

def get_dota2_team_metrics(team_id: int) -> dict:
    team_info = config.DOTA2_TEAMS.get(team_id, {'name': 'Unknown Team', 'abbrev': 'UNK', 'region': 'WEU', 'elo': 1500})
    elo = float(team_info.get('elo', 1500))
    
    win_pct = round(1.0 / (1.0 + 10 ** (-(elo - 1500) / 400)), 3)
    kdr = round(0.85 + (elo - 1400) * 0.001, 2)
    first_blood_pct = round(0.45 + (elo - 1400) * 0.00025, 2)
    roshan_pct = round(0.44 + (elo - 1400) * 0.0003, 2)
    
    return {
        'team_id': team_id,
        'name': team_info['name'],
        'abbrev': team_info['abbrev'],
        'region': team_info.get('region', 'WEU'),
        'elo': elo,
        'win_pct_w10': win_pct,
        'kdr': kdr,
        'first_blood_pct': first_blood_pct,
        'roshan_pct': roshan_pct
    }

from src.features.dota2_rosters import get_team_roster_data, calculate_roster_composite_rating

def calculate_dota2_matchup_prob(
    team1_id: int,
    team2_id: int,
    series_format: str = "Bo3",
    is_team1_radiant: bool = True,
    team1_has_standin: bool = False,
    team2_has_standin: bool = False
) -> dict:
    t1 = get_dota2_team_metrics(team1_id)
    t2 = get_dota2_team_metrics(team2_id)
    
    t1_roster_rating = calculate_roster_composite_rating(team1_id, standin_penalty=3.5 if team1_has_standin else 0.0)
    t2_roster_rating = calculate_roster_composite_rating(team2_id, standin_penalty=3.5 if team2_has_standin else 0.0)
    
    roster_diff = (t1_roster_rating - t2_roster_rating) * 12.0
    elo_diff = t1["elo"] - t2["elo"]
    side_bonus = 20.0 if is_team1_radiant else -20.0
    form_diff = (t1["win_pct_w10"] - t2["win_pct_w10"]) * 40.0
    
    effective_diff = (0.60 * elo_diff) + (0.40 * roster_diff) + side_bonus + form_diff
    
    p_map = 1.0 / (1.0 + 10.0 ** (-effective_diff / 400.0))
    p_map = float(np.clip(p_map, 0.05, 0.95))
    
    fmt = series_format.upper()
    if fmt == "BO1":
        p_series = p_map
    elif fmt == "BO5":
        p_series = (p_map**3) * (1 + 3*(1-p_map) + 6*((1-p_map)**2))
    elif fmt == "BO2":
        p_series = p_map**2
    else:
        p_series = 3 * (p_map**2) - 2 * (p_map**3)
        
    p_series = float(np.clip(p_series, 0.02, 0.98))
    
    return {
        "team1_id": team1_id,
        "team2_id": team2_id,
        "team1_name": t1["name"],
        "team2_name": t2["name"],
        "format": fmt,
        "elo_diff": elo_diff,
        "t1_roster_rating": t1_roster_rating,
        "t2_roster_rating": t2_roster_rating,
        "p_map_t1": round(p_map, 4),
        "p_map_t2": round(1.0 - p_map, 4),
        "p_series_t1": round(p_series, 4),
        "p_series_t2": round(1.0 - p_series, 4),
        "fair_odds_t1": round(1.0 / p_series, 2),
        "fair_odds_t2": round(1.0 / (1.0 - p_series), 2),
        "expected_total_maps": round(2.35 if fmt == "BO3" else (3.8 if fmt == "BO5" else 1.0), 2)
    }
