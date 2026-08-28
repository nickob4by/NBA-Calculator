import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Database
DB_PATH = DATA_DIR / "nba_calculator.db"

# API settings
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_REGIONS = "us"
ODDS_API_MARKETS = "h2h,spreads,totals"

# Rate limiting for stats.nba.com
NBA_API_REQUEST_DELAY = float(os.getenv("NBA_API_REQUEST_DELAY", "0.8"))
NBA_API_MAX_RETRIES = int(os.getenv("NBA_API_MAX_RETRIES", "4"))

# Feature Engineering Defaults
ROLLING_WINDOWS = [5, 10, 20]
EWMA_SPANS = [5, 10]

# Currency Settings
DEFAULT_CURRENCY = "₱"
DEFAULT_CURRENCY_CODE = "PHP"

# Betting Engine Defaults
DEFAULT_KELLY_FRACTION = 0.15     # 15% Fractional Kelly
DEFAULT_MIN_EDGE = 0.025          # 2.5% minimum edge
DEFAULT_MAX_BANKROLL_PCT = 0.04   # 4.0% max risk per bet
DEFAULT_STARTING_BANKROLL = 1200.0 # ₱1,200 PHP
MIN_BET_AMOUNT = 1.0              # ₱1.00 min micro-stake

# Supported Sports
SPORTS = ["nba", "mlb", "dota2"]

# ================= 30 NBA TEAMS =================
NBA_TEAMS = {
    1610612737: {"abbrev": "ATL", "name": "Atlanta Hawks", "city": "Atlanta", "conf": "East", "div": "Southeast", "lat": 33.7573, "lon": -84.3963, "arena": "State Farm Arena"},
    1610612738: {"abbrev": "BOS", "name": "Boston Celtics", "city": "Boston", "conf": "East", "div": "Atlantic", "lat": 42.3662, "lon": -71.0621, "arena": "TD Garden"},
    1610612751: {"abbrev": "BKN", "name": "Brooklyn Nets", "city": "Brooklyn", "conf": "East", "div": "Atlantic", "lat": 40.6826, "lon": -73.9754, "arena": "Barclays Center"},
    1610612766: {"abbrev": "CHA", "name": "Charlotte Hornets", "city": "Charlotte", "conf": "East", "div": "Southeast", "lat": 35.2251, "lon": -80.8392, "arena": "Spectrum Center"},
    1610612741: {"abbrev": "CHI", "name": "Chicago Bulls", "city": "Chicago", "conf": "East", "div": "Central", "lat": 41.8807, "lon": -87.6742, "arena": "United Center"},
    1610612739: {"abbrev": "CLE", "name": "Cleveland Cavaliers", "city": "Cleveland", "conf": "East", "div": "Central", "lat": 41.4965, "lon": -81.6882, "arena": "Rocket Mortgage FieldHouse"},
    1610612742: {"abbrev": "DAL", "name": "Dallas Mavericks", "city": "Dallas", "conf": "West", "div": "Southwest", "lat": 32.7905, "lon": -96.8103, "arena": "American Airlines Center"},
    1610612743: {"abbrev": "DEN", "name": "Denver Nuggets", "city": "Denver", "conf": "West", "div": "Northwest", "lat": 39.7487, "lon": -105.0076, "arena": "Ball Arena"},
    1610612765: {"abbrev": "DET", "name": "Detroit Pistons", "city": "Detroit", "conf": "East", "div": "Central", "lat": 42.3411, "lon": -83.0553, "arena": "Little Caesars Arena"},
    1610612744: {"abbrev": "GSW", "name": "Golden State Warriors", "city": "San Francisco", "conf": "West", "div": "Pacific", "lat": 37.7680, "lon": -122.3877, "arena": "Chase Center"},
    1610612745: {"abbrev": "HOU", "name": "Houston Rockets", "city": "Houston", "conf": "West", "div": "Southwest", "lat": 29.7508, "lon": -95.3621, "arena": "Toyota Center"},
    1610612754: {"abbrev": "IND", "name": "Indiana Pacers", "city": "Indianapolis", "conf": "East", "div": "Central", "lat": 39.7640, "lon": -86.1555, "arena": "Gainbridge Fieldhouse"},
    1610612746: {"abbrev": "LAC", "name": "LA Clippers", "city": "Los Angeles", "conf": "West", "div": "Pacific", "lat": 33.9583, "lon": -118.3418, "arena": "Intuit Dome"},
    1610612747: {"abbrev": "LAL", "name": "Los Angeles Lakers", "city": "Los Angeles", "conf": "West", "div": "Pacific", "lat": 34.0430, "lon": -118.2673, "arena": "Crypto.com Arena"},
    1610612763: {"abbrev": "MEM", "name": "Memphis Grizzlies", "city": "Memphis", "conf": "West", "div": "Southwest", "lat": 35.1382, "lon": -90.0506, "arena": "FedExForum"},
    1610612748: {"abbrev": "MIA", "name": "Miami Heat", "city": "Miami", "conf": "East", "div": "Southeast", "lat": 25.7814, "lon": -80.1870, "arena": "Kaseya Center"},
    1610612749: {"abbrev": "MIL", "name": "Milwaukee Bucks", "city": "Milwaukee", "conf": "East", "div": "Central", "lat": 43.0451, "lon": -87.9172, "arena": "Fiserv Forum"},
    1610612750: {"abbrev": "MIN", "name": "Minnesota Timberwolves", "city": "Minneapolis", "conf": "West", "div": "Northwest", "lat": 44.9795, "lon": -93.2761, "arena": "Target Center"},
    1610612740: {"abbrev": "NOP", "name": "New Orleans Pelicans", "city": "New Orleans", "conf": "West", "div": "Southwest", "lat": 29.9490, "lon": -90.0821, "arena": "Smoothie King Center"},
    1610612752: {"abbrev": "NYK", "name": "New York Knicks", "city": "New York", "conf": "East", "div": "Atlantic", "lat": 40.7505, "lon": -73.9934, "arena": "Madison Square Garden"},
    1610612760: {"abbrev": "OKC", "name": "Oklahoma City Thunder", "city": "Oklahoma City", "conf": "West", "div": "Northwest", "lat": 35.4634, "lon": -97.5151, "arena": "Paycom Center"},
    1610612753: {"abbrev": "ORL", "name": "Orlando Magic", "city": "Orlando", "conf": "East", "div": "Southeast", "lat": 28.5392, "lon": -81.3839, "arena": "Kia Center"},
    1610612755: {"abbrev": "PHI", "name": "Philadelphia 76ers", "city": "Philadelphia", "conf": "East", "div": "Atlantic", "lat": 39.9012, "lon": -75.1720, "arena": "Wells Fargo Center"},
    1610612756: {"abbrev": "PHX", "name": "Phoenix Suns", "city": "Phoenix", "conf": "West", "div": "Pacific", "lat": 33.4457, "lon": -112.0712, "arena": "Footprint Center"},
    1610612757: {"abbrev": "POR", "name": "Portland Trail Blazers", "city": "Portland", "conf": "West", "div": "Northwest", "lat": 45.5316, "lon": -122.6668, "arena": "Moda Center"},
    1610612758: {"abbrev": "SAC", "name": "Sacramento Kings", "city": "Sacramento", "conf": "West", "div": "Pacific", "lat": 38.5802, "lon": -121.4997, "arena": "Golden 1 Center"},
    1610612759: {"abbrev": "SAS", "name": "San Antonio Spurs", "city": "San Antonio", "conf": "West", "div": "Southwest", "lat": 29.4270, "lon": -98.4375, "arena": "Frost Bank Center"},
    1610612761: {"abbrev": "TOR", "name": "Toronto Raptors", "city": "Toronto", "conf": "East", "div": "Atlantic", "lat": 43.6435, "lon": -79.3791, "arena": "Scotiabank Arena"},
    1610612762: {"abbrev": "UTA", "name": "Utah Jazz", "city": "Salt Lake City", "conf": "West", "div": "Northwest", "lat": 40.7683, "lon": -111.9011, "arena": "Delta Center"},
    1610612764: {"abbrev": "WAS", "name": "Washington Wizards", "city": "Washington", "conf": "East", "div": "Southeast", "lat": 38.8982, "lon": -77.0209, "arena": "Capital One Arena"}
}

# ================= 30 MLB TEAMS =================
MLB_TEAMS = {
    108: {"abbrev": "LAA", "name": "Los Angeles Angels", "city": "Anaheim", "league": "AL", "div": "West", "lat": 33.8003, "lon": -117.8827, "stadium": "Angel Stadium", "park_factor": 1.00},
    109: {"abbrev": "ARI", "name": "Arizona Diamondbacks", "city": "Phoenix", "league": "NL", "div": "West", "lat": 33.4455, "lon": -112.0667, "stadium": "Chase Field", "park_factor": 1.05},
    110: {"abbrev": "BAL", "name": "Baltimore Orioles", "city": "Baltimore", "league": "AL", "div": "East", "lat": 39.2839, "lon": -76.6216, "stadium": "Oriole Park at Camden Yards", "park_factor": 0.98},
    111: {"abbrev": "BOS", "name": "Boston Red Sox", "city": "Boston", "league": "AL", "div": "East", "lat": 42.3467, "lon": -71.0972, "stadium": "Fenway Park", "park_factor": 1.08},
    112: {"abbrev": "CHC", "name": "Chicago Cubs", "city": "Chicago", "league": "NL", "div": "Central", "lat": 41.9484, "lon": -87.6553, "stadium": "Wrigley Field", "park_factor": 1.03},
    113: {"abbrev": "CIN", "name": "Cincinnati Reds", "city": "Cincinnati", "league": "NL", "div": "Central", "lat": 39.0979, "lon": -84.5082, "stadium": "Great American Ball Park", "park_factor": 1.12},
    114: {"abbrev": "CLE", "name": "Cleveland Guardians", "city": "Cleveland", "league": "AL", "div": "Central", "lat": 41.4962, "lon": -81.6852, "stadium": "Progressive Field", "park_factor": 0.97},
    115: {"abbrev": "COL", "name": "Colorado Rockies", "city": "Denver", "league": "NL", "div": "West", "lat": 39.7559, "lon": -104.9942, "stadium": "Coors Field", "park_factor": 1.25},
    116: {"abbrev": "DET", "name": "Detroit Tigers", "city": "Detroit", "league": "AL", "div": "Central", "lat": 42.3390, "lon": -83.0485, "stadium": "Comerica Park", "park_factor": 0.96},
    117: {"abbrev": "HOU", "name": "Houston Astros", "city": "Houston", "league": "AL", "div": "West", "lat": 29.7573, "lon": -95.3555, "stadium": "Daikin Park", "park_factor": 1.02},
    118: {"abbrev": "KC",  "name": "Kansas City Royals", "city": "Kansas City", "league": "AL", "div": "Central", "lat": 39.0517, "lon": -94.4803, "stadium": "Kauffman Stadium", "park_factor": 1.01},
    119: {"abbrev": "LAD", "name": "Los Angeles Dodgers", "city": "Los Angeles", "league": "NL", "div": "West", "lat": 34.0739, "lon": -118.2400, "stadium": "Dodger Stadium", "park_factor": 1.02},
    120: {"abbrev": "WSH", "name": "Washington Nationals", "city": "Washington", "league": "NL", "div": "East", "lat": 38.8730, "lon": -77.0074, "stadium": "Nationals Park", "park_factor": 1.00},
    121: {"abbrev": "NYM", "name": "New York Mets", "city": "New York", "league": "NL", "div": "East", "lat": 40.7571, "lon": -73.8458, "stadium": "Citi Field", "park_factor": 0.95},
    133: {"abbrev": "OAK", "name": "Oakland Athletics", "city": "Oakland", "league": "AL", "div": "West", "lat": 37.7516, "lon": -122.2005, "stadium": "Sutter Health Park", "park_factor": 0.96},
    134: {"abbrev": "PIT", "name": "Pittsburgh Pirates", "city": "Pittsburgh", "league": "NL", "div": "Central", "lat": 40.4469, "lon": -80.0057, "stadium": "PNC Park", "park_factor": 0.98},
    135: {"abbrev": "SD",  "name": "San Diego Padres", "city": "San Diego", "league": "NL", "div": "West", "lat": 32.7076, "lon": -117.1570, "stadium": "Petco Park", "park_factor": 0.96},
    136: {"abbrev": "SEA", "name": "Seattle Mariners", "city": "Seattle", "league": "AL", "div": "West", "lat": 47.5914, "lon": -122.3325, "stadium": "T-Mobile Park", "park_factor": 0.92},
    137: {"abbrev": "SF",  "name": "San Francisco Giants", "city": "San Francisco", "league": "NL", "div": "West", "lat": 37.7786, "lon": -122.3893, "stadium": "Oracle Park", "park_factor": 0.93},
    138: {"abbrev": "STL", "name": "St. Louis Cardinals", "city": "St. Louis", "league": "NL", "div": "Central", "lat": 38.6226, "lon": -90.1928, "stadium": "Busch Stadium", "park_factor": 0.97},
    139: {"abbrev": "TB",  "name": "Tampa Bay Rays", "city": "St. Petersburg", "league": "AL", "div": "East", "lat": 27.7682, "lon": -82.6534, "stadium": "Tropicana Field", "park_factor": 0.96},
    140: {"abbrev": "TEX", "name": "Texas Rangers", "city": "Arlington", "league": "AL", "div": "West", "lat": 32.7473, "lon": -97.0845, "stadium": "Globe Life Field", "park_factor": 1.01},
    141: {"abbrev": "TOR", "name": "Toronto Blue Jays", "city": "Toronto", "league": "AL", "div": "East", "lat": 43.6414, "lon": -79.3894, "stadium": "Rogers Centre", "park_factor": 1.02},
    142: {"abbrev": "MIN", "name": "Minnesota Twins", "city": "Minneapolis", "league": "AL", "div": "Central", "lat": 44.9817, "lon": -93.2778, "stadium": "Target Field", "park_factor": 0.99},
    143: {"abbrev": "PHI", "name": "Philadelphia Phillies", "city": "Philadelphia", "league": "NL", "div": "East", "lat": 39.9061, "lon": -75.1665, "stadium": "Citizens Bank Park", "park_factor": 1.06},
    144: {"abbrev": "ATL", "name": "Atlanta Braves", "city": "Atlanta", "league": "NL", "div": "East", "lat": 33.8908, "lon": -84.4678, "stadium": "Truist Park", "park_factor": 1.04},
    145: {"abbrev": "CWS", "name": "Chicago White Sox", "city": "Chicago", "league": "AL", "div": "Central", "lat": 41.8300, "lon": -87.6339, "stadium": "Guaranteed Rate Field", "park_factor": 1.03},
    146: {"abbrev": "MIA", "name": "Miami Marlins", "city": "Miami", "league": "NL", "div": "East", "lat": 25.7783, "lon": -80.2196, "stadium": "loanDepot park", "park_factor": 0.94},
    147: {"abbrev": "NYY", "name": "New York Yankees", "city": "New York", "league": "AL", "div": "East", "lat": 40.8296, "lon": -73.9262, "stadium": "Yankee Stadium", "park_factor": 1.04},
    158: {"abbrev": "MIL", "name": "Milwaukee Brewers", "city": "Milwaukee", "league": "NL", "div": "Central", "lat": 43.0280, "lon": -87.9712, "stadium": "American Family Field", "park_factor": 1.03}
}

# ================= 30 PRO DOTA 2 TEAMS (EPL Masters II & Tier 1/2) =================
DOTA2_TEAMS = {
    1001: {"abbrev": "NAVI", "name": "Natus Vincere", "region": "WEU/EEU", "elo": 1560},
    1002: {"abbrev": "NAV.J", "name": "NAVI Junior", "region": "EEU", "elo": 1510},
    1003: {"abbrev": "MOUZ", "name": "MOUZ", "region": "WEU", "elo": 1540},
    1004: {"abbrev": "1W", "name": "1win (1W Team)", "region": "EEU", "elo": 1580},
    1005: {"abbrev": "ALL", "name": "Alliance", "region": "WEU", "elo": 1490},
    1006: {"abbrev": "SEC", "name": "Team Secret", "region": "WEU", "elo": 1550},
    1007: {"abbrev": "OG", "name": "OG", "region": "WEU", "elo": 1620},
    1008: {"abbrev": "TUN", "name": "Tundra Esports", "region": "WEU", "elo": 1720},
    1009: {"abbrev": "FLC", "name": "Team Falcons", "region": "MENA", "elo": 1810},
    1010: {"abbrev": "GG", "name": "Gaimin Gladiators", "region": "WEU", "elo": 1790},
    1011: {"abbrev": "TS", "name": "Team Spirit", "region": "EEU", "elo": 1780},
    1012: {"abbrev": "TL", "name": "Team Liquid", "region": "WEU", "elo": 1800},
    1013: {"abbrev": "BB", "name": "BetBoom Team", "region": "EEU", "elo": 1740},
    1014: {"abbrev": "VP", "name": "Virtus.pro", "region": "EEU", "elo": 1590},
    1015: {"abbrev": "AUR", "name": "Aurora", "region": "SEA", "elo": 1610},
    1016: {"abbrev": "TAL", "name": "Talon Esports", "region": "SEA", "elo": 1580},
    1017: {"abbrev": "XG", "name": "Xtreme Gaming", "region": "CN", "elo": 1750},
    1018: {"abbrev": "NP", "name": "Night Pulse", "region": "EEU", "elo": 1480},
    1019: {"abbrev": "YS", "name": "Yellow Submarine", "region": "EEU", "elo": 1530},
    1020: {"abbrev": "NEM", "name": "Nemiga Gaming", "region": "EEU", "elo": 1470},
    1021: {"abbrev": "RF", "name": "Rest Farmers", "region": "WEU", "elo": 1460},
    1022: {"abbrev": "LVL", "name": "Level UP", "region": "WEU", "elo": 1450},
    1023: {"abbrev": "ASA", "name": "ASAKURA", "region": "EEU", "elo": 1460},
    1024: {"abbrev": "PST", "name": "PSG Quest", "region": "MENA", "elo": 1640},
    1025: {"abbrev": "C9", "name": "Cloud9 (Entity)", "region": "WEU", "elo": 1690},
    1026: {"abbrev": "HER", "name": "Heroic", "region": "SA", "elo": 1590},
    1027: {"abbrev": "BOM", "name": "BOOM Esports", "region": "SEA", "elo": 1560},
    1028: {"abbrev": "BLD", "name": "Bleed Esports", "region": "SEA", "elo": 1530},
    1029: {"abbrev": "LGD", "name": "LGD Gaming", "region": "CN", "elo": 1600},
    1030: {"abbrev": "MNT", "name": "Monte", "region": "EEU", "elo": 1450}
}

def get_teams_for_sport(sport: str = "nba"):
    s = sport.lower()
    if s == "mlb":
        return MLB_TEAMS
    elif s == "dota2":
        return DOTA2_TEAMS
    return NBA_TEAMS

def get_team_id(identifier, sport: str = "nba"):
    teams_dict = get_teams_for_sport(sport)
    if isinstance(identifier, int) and identifier in teams_dict:
        return identifier
    if isinstance(identifier, str):
        ident = identifier.strip().upper()
        for k, v in teams_dict.items():
            if v["abbrev"].upper() == ident:
                return k
        ident_lower = identifier.strip().lower()
        for k, v in teams_dict.items():
            if v["name"].lower() == ident_lower:
                return k
    return None

def get_team_abbrev(team_id, sport: str = "nba"):
    teams_dict = get_teams_for_sport(sport)
    if team_id in teams_dict:
        return teams_dict[team_id]["abbrev"]
    return str(team_id)

def get_team_name(team_id, sport: str = "nba"):
    teams_dict = get_teams_for_sport(sport)
    if team_id in teams_dict:
        return teams_dict[team_id]["name"]
    return f"Team {team_id}"