# 🏀 NBA-Calculator: Quantitative Predictive Analytics & +EV Betting Engine

An end-to-end quantitative NBA modeling, probability calibration, +EV betting discovery, and bankroll management system with zero lookahead bias and rigorous backtesting.

---

## 🌟 Core Features

1. **Environment & Automated Ingestion:**
   - Python 3.12 environment with `nba_api`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `streamlit`, and `plotly`.
   - Thread-safe SQLite database with WAL mode and tables for `games`, `team_game_logs`, `team_advanced_stats`, `odds`, `predictions`, and `bets`.
   - Ingestion pipeline with adaptive throttling, exponential backoff retries, and offline seed datasets.
   - Odds integration tracking opening/closing Moneyline, Spreads, and Totals (Over/Under).

2. **Feature Engineering Pipeline (Strict Zero-Lookahead Bias):**
   - **Dean Oliver's Four Factors** for Offense and Defense:
     - Effective Field Goal % (eFG%)
     - Turnover % (TOV%)
     - Offensive Rebound % (OREB%)
     - Free Throw Rate (FTR)
     - Possessions, Pace, Offensive Rating (ORtg), Defensive Rating (DRtg), and Net Rating.
   - **Situational Dynamics**: Rest days differential (Home Rest - Away Rest), Back-to-Back (B2B) flags (0 days rest), 3-in-4 nights, and Great-Circle Haversine travel distances across all 30 NBA arenas.
   - **Lag-1 Shifting**: Strictly shifted rolling moving averages (5, 10, 20 games) and EWMA (5, 10 spans) ensuring Game N only uses information from Games 1..N-1.

3. **Predictive Modeling & Probability Calibration:**
   - Point Margin (Home PTS - Away PTS) ensemble model blending Ridge Regression, LightGBM, and XGBoost.
   - Total Points model for Over/Under markets.
   - Calibrated Win Probabilities via Gaussian CDF mapping and **Platt Scaling** (Logistic Sigmoid Calibration).
   - Evaluated using Time-Series Cross Validation (`TimeSeriesSplit`), Log Loss, Brier Score, and Reliability Diagrams (Expected Calibration Error).

4. **Decision & Bankroll Sizing Engine:**
   - Bookmaker Vig removal algorithms: **Multiplicative**, **Power**, and **Shin** methods.
   - Expected Value (+EV) calculation: EV = (P_model * Net_Profit) - ((1 - P_model) * Stake).
   - Edge thresholding (e.g. minimum 2.5% edge over fair implied odds).
   - **Fractional Kelly Criterion** position sizing (10% to 25% Kelly multiplier) with risk caps.

5. **Chronological Backtesting & Interactive Dashboard:**
   - Historical backtesting simulating bets over past seasons, tracking dynamic ROI %, PnL, Max Drawdown %, Sharpe Ratio, and **Closing Line Value (CLV %)**.
   - Full interactive **Streamlit Dashboard** and rich **CLI Utility**.

---

## 🚀 Quick Start

### 1. Setup & Activate Virtual Environment
```powershell
# Create venv and install dependencies
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
.\.venv\Scripts\pytest -v
```

### 3. Launch Interactive Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit run src/app/dashboard.py
```

---

## 💻 CLI Commands

The CLI provides commands to sync, train, predict, and backtest:

```powershell
# Ingest and sync database
python -m src.app.cli sync --seasons 2022-23,2023-24,2024-25

# Train models with 5-fold TimeSeriesSplit cross validation
python -m src.app.cli train --splits 5

# Predict a matchup with live odds and Kelly sizing
python -m src.app.cli predict --home BOS --away LAL --home-ml 1.75 --away-ml 2.20 --bankroll 10000 --kelly 0.15

# Run historical backtest over 2024-25 season
python -m src.app.cli backtest --season 2024-25 --kelly 0.15 --min-edge 0.025

# View SQLite database statistics
python -m src.app.cli stats
```

---

## 📁 Project Structure

```
NBA/
├── config.py                     # Central configuration, hyperparameters & arena geolocations
├── requirements.txt              # Project dependencies
├── pytest.ini                    # Pytest configuration
├── src/
│   ├── db/
│   │   ├── schema.sql            # SQLite database schema
│   │   └── database.py          # Database connection manager & query helpers
│   ├── ingestion/
│   │   ├── nba_api_fetcher.py   # Throttled stats.nba.com client with exponential backoff
│   │   ├── odds_fetcher.py      # The-Odds-API client & historical odds loader
│   │   └── pipeline.py          # Unified data sync & fallback seed generator
│   ├── features/
│   │   ├── four_factors.py      # Dean Oliver's Four Factors & Pace/Possessions
│   │   ├── situational.py       # Rest, B2B, 3-in-4 & Haversine arena travel distance
│   │   ├── rolling_metrics.py   # Zero-lookahead (shift=1) rolling & EWMA form metrics
│   │   └── matchup_builder.py   # Matchup differential feature compiler
│   ├── models/
│   │   ├── margin_model.py      # Ridge, LightGBM, XGBoost ensemble point margin regressor
│   │   ├── totals_model.py      # Total points Over/Under regressor
│   │   ├── win_prob_model.py    # Calibrated Gaussian CDF win probability model
│   │   ├── calibration.py       # Platt Scaling & Isotonic calibration, Brier & Log Loss
│   │   └── train.py             # TimeSeriesSplit cross validation & model serialization
│   ├── betting/
│   │   ├── odds_math.py         # American/Decimal converters & Vig removal (Shin/Power/Mult)
│   │   ├── ev_engine.py         # Expected Value (+EV) and Edge discovery
│   │   ├── kelly.py             # Fractional Kelly Criterion & position sizing
│   │   └── backtester.py        # Chronological backtesting engine (ROI, CLV, Drawdown)
│   └── app/
│       ├── cli.py               # Rich interactive command-line interface
│       └── dashboard.py         # Full-featured Streamlit web application
└── tests/                       # Automated test suite (19 unit & integration tests)
```
