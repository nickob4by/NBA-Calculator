import pandas as pd
import numpy as np
from typing import Dict

def calculate_pythagorean_win_pct(runs_scored: float, runs_allowed: float, exponent: float = 1.83) -> float:
    """
    Bill James' Pythagorean Expectation for Baseball.
    Win% = RS^1.83 / (RS^1.83 + RA^1.83)
    """
    rs = max(float(runs_scored), 0.1)
    ra = max(float(runs_allowed), 0.1)
    
    rs_exp = rs ** exponent
    ra_exp = ra ** exponent
    
    return round(rs_exp / (rs_exp + ra_exp), 4)

def calculate_sabermetrics(team_box: dict, opp_box: dict) -> Dict:
    """
    Computes offensive sabermetrics and pitching run-prevention metrics for an MLB game.
    """
    runs = float(team_box.get("runs") or team_box.get("pts") or 0)
    hits = float(team_box.get("hits") or 8)
    hr = float(team_box.get("hr") or 1)
    bb = float(team_box.get("bb") or 3)
    so = float(team_box.get("so") or 8)
    
    opp_runs = float(opp_box.get("runs") or opp_box.get("pts") or 0)
    opp_hits = float(opp_box.get("hits") or 8)
    opp_hr = float(opp_box.get("hr") or 1)
    opp_bb = float(opp_box.get("bb") or 3)
    opp_so = float(opp_box.get("so") or 8)
    
    ip = float(team_box.get("ip") or 9.0)
    if ip <= 0:
        ip = 9.0
        
    ab = float(team_box.get("fga") or max(hits + so + 18, 30))
    
    # 1. Offensive metrics
    ba = hits / ab if ab > 0 else 0.250
    pa = ab + bb
    obp = (hits + bb) / pa if pa > 0 else 0.315
    
    singles = max(hits - hr, 0)
    slg = (singles + 4 * hr) / ab if ab > 0 else 0.400
    ops = obp + slg
    iso = max(slg - ba, 0.0)
    
    # wOBA proxy
    woba_proxy = (0.69 * bb + 0.89 * singles + 2.10 * hr) / pa if pa > 0 else 0.315
    
    # 2. Pitching & Run Prevention
    fip_constant = 3.15
    fip_proxy = ((13.0 * opp_hr + 3.0 * opp_bb - 2.0 * opp_so) / ip) + fip_constant
    fip_proxy = float(np.clip(fip_proxy, 1.5, 9.0))
    
    whip = (opp_bb + opp_hits) / ip if ip > 0 else 1.30
    whip = float(np.clip(whip, 0.5, 3.0))
    
    k_per_9 = (opp_so / ip) * 9.0
    bb_per_9 = (opp_bb / ip) * 9.0
    
    pythag = calculate_pythagorean_win_pct(runs, opp_runs)
    
    return {
        "runs": runs,
        "opp_runs": opp_runs,
        "pythag_win_pct": pythag,
        "obp": round(obp, 3),
        "slg": round(slg, 3),
        "ops": round(ops, 3),
        "iso": round(iso, 3),
        "woba_proxy": round(woba_proxy, 3),
        "fip_proxy": round(fip_proxy, 2),
        "whip": round(whip, 2),
        "k_per_9": round(k_per_9, 2),
        "bb_per_9": round(bb_per_9, 2)
    }

def compute_mlb_sabermetrics_dataframe(game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sabermetric stats for all team game logs in MLB.
    """
    if game_logs_df.empty:
        return pd.DataFrame()

    records = []
    grouped = game_logs_df.groupby("game_id")
    for game_id, group in grouped:
        if len(group) != 2:
            continue
        row_a = group.iloc[0].to_dict()
        row_b = group.iloc[1].to_dict()

        stats_a = calculate_sabermetrics(row_a, row_b)
        stats_a["game_id"] = str(game_id)
        stats_a["team_id"] = row_a["team_id"]
        stats_a["opponent_id"] = row_b["team_id"]
        stats_a["sport"] = "mlb"

        stats_b = calculate_sabermetrics(row_b, row_a)
        stats_b["game_id"] = str(game_id)
        stats_b["team_id"] = row_b["team_id"]
        stats_b["opponent_id"] = row_a["team_id"]
        stats_b["sport"] = "mlb"

        records.append(stats_a)
        records.append(stats_b)

    return pd.DataFrame(records)