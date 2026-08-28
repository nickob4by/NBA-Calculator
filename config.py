import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
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
ODDS_API_SPORT = "basketball_nba"
ODDS_API_REGIONS = "us"
ODDS_API_MARKETS = "h2h,spreads,totals"

# Rate limiting for stats.nba.com
NBA_API_REQUEST_DELAY = float(os.getenv("NBA_API_REQUEST_DELAY", "0.8"))
NBA_API_MAX_RETRIES = int(os.getenv("NBA_API_MAX_RETRIES", "4"))

# Feature Engineering Defaults
ROLLING_WINDOWS = [5, 10, 20]
EWMA_SPANS = [5, 10]

# Betting Engine Defaults
DEFAULT_KELLY_FRACTION = 0.15     # 15% Fractional Kelly (conservative bankroll growth)
DEFAULT_MIN_EDGE = 0.025          # 2.5% minimum edge over fair implied probability
DEFAULT_MAX_BANKROLL_PCT = 0.04   # 4.0% maximum allocation on any single game
DEFAULT_STARTING_BANKROLL = 10000.0

# 30 NBA Teams Metadata: ID, Name, Arena Coordinates (Lat, Lon) for Haversine travel distance
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

# Lookup dictionaries
ABBREV_TO_ID = {v["abbrev"]: k for k, v in NBA_TEAMS.items()}
NAME_TO_ID = {v["name"].lower(): k for k, v in NBA_TEAMS.items()}
CITY_TO_ID = {v["city"].lower(): k for k, v in NBA_TEAMS.items()}

# Helper functions
def get_team_id(identifier):
    if isinstance(identifier, int) and identifier in NBA_TEAMS:
        return identifier
    if isinstance(identifier, str):
        ident = identifier.strip().upper()
        if ident in ABBREV_TO_ID:
            return ABBREV_TO_ID[ident]
        ident_lower = identifier.strip().lower()
        if ident_lower in NAME_TO_ID:
            return NAME_TO_ID[ident_lower]
        if ident_lower in CITY_TO_ID:
            return CITY_TO_ID[ident_lower]
    return None

def get_team_abbrev(team_id):
    if team_id in NBA_TEAMS:
        return NBA_TEAMS[team_id]["abbrev"]
    return str(team_id)

def get_team_name(team_id):
    if team_id in NBA_TEAMS:
        return NBA_TEAMS[team_id]["name"]
    return f"Team {team_id}"
