-- NBA Calculator Database Schema

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season TEXT NOT NULL,
    game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_pts INTEGER,
    away_pts INTEGER,
    point_margin INTEGER,
    total_points INTEGER,
    home_win INTEGER,
    is_playoff INTEGER DEFAULT 0,
    status TEXT DEFAULT 'Final',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games (game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games (season);
CREATE INDEX IF NOT EXISTS idx_games_home ON games (home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away ON games (away_team_id);

CREATE TABLE IF NOT EXISTS team_game_logs (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    season TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    wl TEXT,
    min INTEGER DEFAULT 240,
    fgm INTEGER,
    fga INTEGER,
    fg_pct REAL,
    fg3m INTEGER,
    fg3a INTEGER,
    fg3_pct REAL,
    ftm INTEGER,
    fta INTEGER,
    ft_pct REAL,
    oreb INTEGER,
    dreb INTEGER,
    reb INTEGER,
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    pf INTEGER,
    pts INTEGER,
    plus_minus REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tgl_team_date ON team_game_logs (team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_tgl_game ON team_game_logs (game_id);

CREATE TABLE IF NOT EXISTS team_advanced_stats (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    possessions REAL,
    pace REAL,
    off_rating REAL,
    def_rating REAL,
    net_rating REAL,
    efg_pct REAL,
    tov_pct REAL,
    orb_pct REAL,
    ftr REAL,
    opp_efg_pct REAL,
    opp_tov_pct REAL,
    opp_orb_pct REAL,
    opp_ftr REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tas_team ON team_advanced_stats (team_id);

CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    bookmaker TEXT DEFAULT 'Consensus',
    home_ml_open REAL,
    away_ml_open REAL,
    home_ml_close REAL,
    away_ml_close REAL,
    spread_line REAL,
    home_spread_odds REAL DEFAULT -110,
    away_spread_odds REAL DEFAULT -110,
    total_line REAL,
    over_odds REAL DEFAULT -110,
    under_odds REAL DEFAULT -110,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON odds (game_id);

CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    pred_margin REAL,
    pred_home_win_prob REAL,
    pred_total REAL,
    pred_home_pts REAL,
    pred_away_pts REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, model_version),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market_type TEXT NOT NULL, -- 'moneyline', 'spread', 'total'
    side TEXT NOT NULL, -- 'home', 'away', 'over', 'under'
    odds REAL NOT NULL, -- Decimal odds
    american_odds INTEGER,
    stake REAL NOT NULL,
    stake_pct REAL,
    model_prob REAL NOT NULL,
    implied_prob REAL NOT NULL,
    edge REAL NOT NULL,
    ev REAL NOT NULL,
    closing_odds REAL,
    clv REAL,
    result TEXT, -- 'WIN', 'LOSS', 'PUSH', 'PENDING'
    pnl REAL,
    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bets_game ON bets (game_id);
CREATE INDEX IF NOT EXISTS idx_bets_placed ON bets (placed_at);
