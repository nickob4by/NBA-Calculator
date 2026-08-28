import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import config
from src.db.database import db

MLB_ROTATIONS = {
    # AL East
    147: [ # NYY
        {"id": 543037, "name": "Gerrit Cole", "fip": 2.85, "whip": 0.98, "k9": 9.8, "throws": "R"},
        {"id": 607074, "name": "Carlos Rodón", "fip": 3.65, "whip": 1.22, "k9": 10.0, "throws": "L"},
        {"id": 573186, "name": "Marcus Stroman", "fip": 4.10, "whip": 1.35, "k9": 6.8, "throws": "R"},
        {"id": 669456, "name": "Clarke Schmidt", "fip": 3.80, "whip": 1.18, "k9": 9.5, "throws": "R"},
        {"id": 669203, "name": "Nestor Cortes", "fip": 3.95, "whip": 1.15, "k9": 8.4, "throws": "L"}
    ],
    110: [ # BAL
        {"id": 669203, "name": "Corbin Burnes", "fip": 2.92, "whip": 1.09, "k9": 8.4, "throws": "R"},
        {"id": 680694, "name": "Grayson Rodriguez", "fip": 3.55, "whip": 1.24, "k9": 10.0, "throws": "R"},
        {"id": 669330, "name": "Dean Kremer", "fip": 4.25, "whip": 1.25, "k9": 8.5, "throws": "R"},
        {"id": 502043, "name": "Kyle Gibson", "fip": 4.40, "whip": 1.32, "k9": 7.1, "throws": "R"},
        {"id": 666142, "name": "Cole Irvin", "fip": 4.60, "whip": 1.40, "k9": 6.5, "throws": "L"}
    ],
    111: [ # BOS
        {"id": 678394, "name": "Brayan Bello", "fip": 3.90, "whip": 1.30, "k9": 8.6, "throws": "R"},
        {"id": 656302, "name": "Kutter Crawford", "fip": 3.80, "whip": 1.10, "k9": 9.3, "throws": "R"},
        {"id": 677944, "name": "Tanner Houck", "fip": 3.20, "whip": 1.14, "k9": 8.9, "throws": "R"},
        {"id": 601713, "name": "Nick Pivetta", "fip": 3.75, "whip": 1.12, "k9": 10.6, "throws": "R"},
        {"id": 656303, "name": "Cooper Criswell", "fip": 4.30, "whip": 1.34, "k9": 7.2, "throws": "R"}
    ],
    139: [ # TB
        {"id": 668984, "name": "Shane Baz", "fip": 3.60, "whip": 1.15, "k9": 8.8, "throws": "R"},
        {"id": 663556, "name": "Taj Bradley", "fip": 3.50, "whip": 1.18, "k9": 10.2, "throws": "R"},
        {"id": 669212, "name": "Ryan Pepiot", "fip": 3.75, "whip": 1.15, "k9": 9.4, "throws": "R"},
        {"id": 622065, "name": "Zack Littell", "fip": 4.10, "whip": 1.25, "k9": 7.9, "throws": "R"},
        {"id": 642232, "name": "Jeffrey Springs", "fip": 3.70, "whip": 1.20, "k9": 9.2, "throws": "L"}
    ],
    141: [ # TOR
        {"id": 592332, "name": "Kevin Gausman", "fip": 3.45, "whip": 1.20, "k9": 9.3, "throws": "R"},
        {"id": 669203, "name": "José Berríos", "fip": 4.15, "whip": 1.15, "k9": 7.2, "throws": "R"},
        {"id": 592791, "name": "Chris Bassitt", "fip": 3.90, "whip": 1.30, "k9": 8.8, "throws": "R"},
        {"id": 668909, "name": "Bowden Francis", "fip": 3.40, "whip": 0.95, "k9": 9.0, "throws": "R"},
        {"id": 605483, "name": "Yariel Rodríguez", "fip": 4.30, "whip": 1.35, "k9": 9.1, "throws": "R"}
    ],
    # NL East
    143: [ # PHI
        {"id": 554430, "name": "Zack Wheeler", "fip": 2.80, "whip": 0.96, "k9": 10.1, "throws": "R"},
        {"id": 605400, "name": "Aaron Nola", "fip": 3.60, "whip": 1.15, "k9": 8.9, "throws": "R"},
        {"id": 672282, "name": "Ranger Suárez", "fip": 3.25, "whip": 1.18, "k9": 8.6, "throws": "L"},
        {"id": 675911, "name": "Cristopher Sánchez", "fip": 3.10, "whip": 1.24, "k9": 7.6, "throws": "L"},
        {"id": 592836, "name": "Taijuan Walker", "fip": 5.20, "whip": 1.55, "k9": 6.1, "throws": "R"}
    ],
    144: [ # ATL
        {"id": 543037, "name": "Chris Sale", "fip": 2.65, "whip": 1.01, "k9": 11.4, "throws": "L"},
        {"id": 668678, "name": "Max Fried", "fip": 3.30, "whip": 1.16, "k9": 8.6, "throws": "L"},
        {"id": 663556, "name": "Reynaldo López", "fip": 3.20, "whip": 1.11, "k9": 9.6, "throws": "R"},
        {"id": 542881, "name": "Charlie Morton", "fip": 4.10, "whip": 1.32, "k9": 9.2, "throws": "R"},
        {"id": 676664, "name": "Spencer Schwellenbach", "fip": 3.35, "whip": 1.04, "k9": 9.6, "throws": "R"}
    ],
    121: [ # NYM
        {"id": 656945, "name": "Kodai Senga", "fip": 3.25, "whip": 1.15, "k9": 10.5, "throws": "R"},
        {"id": 668984, "name": "Sean Manaea", "fip": 3.70, "whip": 1.08, "k9": 9.1, "throws": "L"},
        {"id": 605483, "name": "Luis Severino", "fip": 3.90, "whip": 1.24, "k9": 8.0, "throws": "R"},
        {"id": 668909, "name": "David Peterson", "fip": 3.85, "whip": 1.29, "k9": 7.4, "throws": "L"},
        {"id": 592836, "name": "Jose Quintana", "fip": 4.20, "whip": 1.25, "k9": 7.1, "throws": "L"}
    ],
    # NL Central
    112: [ # CHC
        {"id": 669203, "name": "Justin Steele", "fip": 3.20, "whip": 1.10, "k9": 9.1, "throws": "L"},
        {"id": 680694, "name": "Shota Imanaga", "fip": 3.40, "whip": 1.02, "k9": 9.1, "throws": "L"},
        {"id": 669330, "name": "Javier Assad", "fip": 4.10, "whip": 1.35, "k9": 7.8, "throws": "R"},
        {"id": 502043, "name": "Jameson Taillon", "fip": 4.05, "whip": 1.16, "k9": 7.1, "throws": "R"},
        {"id": 666142, "name": "Kyle Hendricks", "fip": 4.80, "whip": 1.45, "k9": 6.2, "throws": "R"}
    ],
    113: [ # CIN
        {"id": 669456, "name": "Hunter Greene", "fip": 3.15, "whip": 1.02, "k9": 10.2, "throws": "R"},
        {"id": 607074, "name": "Nick Lodolo", "fip": 3.80, "whip": 1.20, "k9": 9.6, "throws": "L"},
        {"id": 669203, "name": "Andrew Abbott", "fip": 4.25, "whip": 1.30, "k9": 7.7, "throws": "L"},
        {"id": 573186, "name": "Nick Martinez", "fip": 3.70, "whip": 1.15, "k9": 7.5, "throws": "R"},
        {"id": 669203, "name": "Frankie Montas", "fip": 4.70, "whip": 1.42, "k9": 7.5, "throws": "R"}
    ],
    # NL West
    119: [ # LAD
        {"id": 607192, "name": "Tyler Glasnow", "fip": 2.90, "whip": 0.95, "k9": 11.3, "throws": "R"},
        {"id": 694973, "name": "Yoshinobu Yamamoto", "fip": 3.10, "whip": 1.11, "k9": 10.5, "throws": "R"},
        {"id": 477132, "name": "Clayton Kershaw", "fip": 3.50, "whip": 1.15, "k9": 8.5, "throws": "L"},
        {"id": 663556, "name": "Jack Flaherty", "fip": 3.35, "whip": 1.07, "k9": 10.8, "throws": "R"},
        {"id": 676664, "name": "Gavin Stone", "fip": 3.90, "whip": 1.21, "k9": 7.2, "throws": "R"}
    ],
    # AL Central
    116: [ # DET
        {"id": 669373, "name": "Tarik Skubal", "fip": 2.50, "whip": 0.92, "k9": 10.7, "throws": "L"},
        {"id": 656302, "name": "Reese Olson", "fip": 3.65, "whip": 1.18, "k9": 8.3, "throws": "R"},
        {"id": 677944, "name": "Casey Mize", "fip": 4.10, "whip": 1.30, "k9": 7.1, "throws": "R"},
        {"id": 601713, "name": "Keider Montero", "fip": 4.50, "whip": 1.33, "k9": 7.2, "throws": "R"},
        {"id": 656303, "name": "Matt Manning", "fip": 4.40, "whip": 1.35, "k9": 7.5, "throws": "R"}
    ],
    # AL West
    117: [ # HOU
        {"id": 621121, "name": "Framber Valdez", "fip": 3.25, "whip": 1.11, "k9": 8.6, "throws": "L"},
        {"id": 668678, "name": "Hunter Brown", "fip": 3.45, "whip": 1.25, "k9": 9.5, "throws": "R"},
        {"id": 663556, "name": "Ronel Blanco", "fip": 3.60, "whip": 1.09, "k9": 8.9, "throws": "R"},
        {"id": 542881, "name": "Justin Verlander", "fip": 4.30, "whip": 1.38, "k9": 7.6, "throws": "R"},
        {"id": 676664, "name": "Spencer Arrighetti", "fip": 3.95, "whip": 1.41, "k9": 10.4, "throws": "R"}
    ]
}

def get_team_starters(team_id: int) -> List[Dict]:
    """
    Returns the list of starting pitchers for team_id.
    """
    if team_id in MLB_ROTATIONS:
        return MLB_ROTATIONS[team_id]
    
    # Generic realistic 5-man rotation for unlisted teams
    t_name = config.get_team_name(team_id, sport="mlb")
    return [
        {"id": team_id * 100 + 1, "name": f"{t_name} Ace", "fip": 3.20, "whip": 1.08, "k9": 9.5, "throws": "R"},
        {"id": team_id * 100 + 2, "name": f"{t_name} #2 Starter", "fip": 3.75, "whip": 1.20, "k9": 8.6, "throws": "R"},
        {"id": team_id * 100 + 3, "name": f"{t_name} #3 Starter", "fip": 4.10, "whip": 1.28, "k9": 8.0, "throws": "L"},
        {"id": team_id * 100 + 4, "name": f"{t_name} #4 Starter", "fip": 4.45, "whip": 1.35, "k9": 7.4, "throws": "R"},
        {"id": team_id * 100 + 5, "name": f"{t_name} #5 Starter", "fip": 4.90, "whip": 1.45, "k9": 6.8, "throws": "R"}
    ]

def calculate_sp_fip_differential(home_sp_fip: float, away_sp_fip: float) -> float:
    """
    Computes delta SP quality in terms of run prevention:
    Positive delta means Home starting pitcher is superior (lower FIP) than Away starting pitcher.
    Delta = Away SP FIP - Home SP FIP
    """
    return round(float(away_sp_fip) - float(home_sp_fip), 3)

def calculate_pitcher_game_fip(ip: float, er: int, hr: int, bb: int, so: int) -> float:
    """
    Calculates Fielding Independent Pitching (FIP) for a game.
    FIP = ((13*HR + 3*BB - 2*SO) / IP) + 3.15
    """
    ip = max(float(ip), 0.33)
    fip = ((13.0 * hr + 3.0 * bb - 2.0 * so) / ip) + 3.15
    return round(float(np.clip(fip, 1.20, 9.50)), 2)