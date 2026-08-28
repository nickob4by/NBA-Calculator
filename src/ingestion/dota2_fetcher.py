import pandas as pd
import numpy as np
from src.db.database import db
import config

def generate_dota2_seed_dataset_if_empty():
    count_row = db.fetch_one("SELECT COUNT(game_id) as c FROM games WHERE sport='dota2'")
    if count_row and count_row['c'] > 0:
        return

    games_rows = []
    logs_rows = []
    
    team_ids = list(config.DOTA2_TEAMS.keys())
    game_id = 30001

    for i in range(150):
        h1 = team_ids[i % len(team_ids)]
        a1 = team_ids[(i + 1) % len(team_ids)]
        
        t1_elo = config.DOTA2_TEAMS[h1]['elo']
        t2_elo = config.DOTA2_TEAMS[a1]['elo']
        
        p_win = 1.0 / (1.0 + 10 ** (-(t1_elo - t2_elo) / 400))
        h1_won = 1 if np.random.random() < p_win else 0
        h1_score = 2 if h1_won else (1 if np.random.random() > 0.5 else 0)
        a1_score = 2 if not h1_won else (1 if np.random.random() > 0.5 else 0)
        
        day = 1 + (i % 28)
        date_str = f"2024-10-{day:02d}"
        
        games_rows.append((
            game_id,
            'dota2',
            date_str,
            'EPL Masters II',
            h1,
            a1,
            h1_score,
            a1_score,
            'FINAL'
        ))
        
        logs_rows.append((
            str(game_id),
            'dota2',
            h1,
            a1,
            date_str,
            'EPL Masters II',
            1,
            'W' if h1_won else 'L',
            h1_score,
            h1_score - a1_score
        ))
        
        logs_rows.append((
            str(game_id),
            'dota2',
            a1,
            h1,
            date_str,
            'EPL Masters II',
            0,
            'L' if h1_won else 'W',
            a1_score,
            a1_score - h1_score
        ))
        
        game_id += 1

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO games 
            (game_id, sport, game_date, season, home_team_id, away_team_id, home_pts, away_pts, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', games_rows)

        cursor.executemany('''
            INSERT OR IGNORE INTO team_game_logs
            (game_id, sport, team_id, opponent_id, game_date, season, is_home, wl, pts, plus_minus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', logs_rows)
