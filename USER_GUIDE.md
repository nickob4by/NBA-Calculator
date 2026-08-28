# 📖 NBA-Calculator: Complete User & Strategy Guide

Welcome to the **NBA-Calculator** quantitative betting and analytics platform. This guide explains how to use the interactive Web Dashboard, run CLI commands, interpret model forecasts, and execute **+EV (Expected Value)** betting strategies using **Fractional Kelly Criterion** position sizing.

---

## 📑 Table of Contents
1. [Quick Start](#1-quick-start)
2. [Interactive Web Dashboard Guide](#2-interactive-web-dashboard-guide)
   - [Tab 1: Matchup Forecast & +EV Sizer](#tab-1-matchup-forecast--ev-sizer)
   - [Tab 2: Four Factors Explorer](#tab-2-four-factors-explorer)
   - [Tab 3: Historical Backtester](#tab-3-historical-backtester)
   - [Tab 4: Calibration & Model Lab](#tab-4-calibration--model-lab)
   - [Tab 5: Data & Settings](#tab-5-data--settings)
3. [CLI (Command Line Interface) Guide](#3-cli-command-line-interface-guide)
4. [Mathematical & Modeling Reference](#4-mathematical--modeling-reference)
   - [Zero Lookahead Lag Engine](#zero-lookahead-lag-engine)
   - [Dean Oliver's Four Factors](#dean-olivers-four-factors)
   - [Probability Calibration & Normal CDF](#probability-calibration--normal-cdf)
   - [Vig Removal (Multiplicative, Power, Shin)](#vig-removal)
   - [Expected Value (+EV) & Fractional Kelly](#expected-value-ev--fractional-kelly)

---

## 1. Quick Start

### Launching the Web Dashboard
From your terminal in `D:\NBA`:
```powershell
.\.venv\Scripts\streamlit run src/app/dashboard.py
```
Open your browser to: **[http://localhost:8501](http://localhost:8501)**

### Running CLI Commands
```powershell
# Predict a matchup directly from the terminal
.\.venv\Scripts\python -m src.app.cli predict --home BOS --away LAL --home-ml 1.75 --away-ml 2.20

# Run a historical backtest for the 2024-25 season
.\.venv\Scripts\python -m src.app.cli backtest --season 2024-25 --kelly 0.15
```

---

## 2. Interactive Web Dashboard Guide

### Sidebar Controls
- **Your Bankroll ($)**: Set your active trading/betting bankroll (default: `$10,000`). All recommended dollar stakes will be calculated dynamically from this figure.
- **Fractional Kelly Multiplier**: Sizing scale factor between `0.05` (5% Kelly, ultra-conservative) and `0.50` (50% Kelly, aggressive). Recommended: `0.15` (15% Fractional Kelly).
- **Min Edge Threshold (%)**: Minimum required statistical edge before the system recommends placing a wager (default: `2.5%`).

---

### Tab 1: Matchup Forecast & +EV Sizer

Use this tab to analyze upcoming games and discover positive EV bets.

#### 1. Select Matchup
- Choose **Home Team** (e.g. *Boston Celtics*) and **Away Team** (e.g. *Los Angeles Lakers*).

#### 2. Interpret Model Forecast Cards
- **Win Probability**: Calibrated probability of victory for both teams (e.g. *Boston: 58.5%*, *Lakers: 41.5%*) alongside the **Fair (No-Vig) Decimal Odds**.
- **Projected Point Margin (ΔPTS)**: Expected final score difference (e.g. `+1.9 pts` indicates Boston is favored by 1.9 points).
- **Projected Total Points**: Projected combined score of both teams (for Over/Under markets).

#### 3. Enter Sportsbook Lines
- **Moneyline**: Enter the Decimal odds offered by your bookmaker (e.g., `1.85` for Boston, `2.05` for Lakers).
- **Point Spread**: Enter the bookmaker's spread line (e.g. `-4.5` for Boston) and offered price (`1.91`).
- **Over / Under Total**: Enter the bookmaker's point total (e.g. `224.5`) and offered price (`1.91`).

#### 4. Actionable Bets Table
If the model finds bets where your expected value exceeds the **Min Edge Threshold**, it will render a recommendation card with:
- **Offered Odds vs Fair Implied Odds**
- **Statistical Edge (%)**: $P_{\text{model}} - P_{\text{fair}}$
- **Expected Value (EV)** per $1 staked
- **Kelly Position Size (%)**: Safe allocation percentage of your bankroll
- **Recommended Dollar Stake ($)**: Exact dollar amount to place

---

### Tab 2: Four Factors Explorer

Analyze the fundamental basketball drivers behind the prediction based on Dean Oliver's Four Factors over the last 10 games:

- **eFG% (Shooting Efficiency)**: Field goal percentage giving extra credit for 3-pointers.
- **TOV% (Ball Security)**: Percentage of possessions ending in a turnover (lower is better).
- **OREB% (Offensive Rebounding)**: Percentage of available offensive rebounds grabbed.
- **FTR (Free Throw Rate)**: Ability to draw fouls and get to the free-throw line.
- **Net Rating**: Points scored minus points conceded per 100 possessions.

The **Radar Chart** visually overlays both teams to instantly reveal matchup advantages (e.g. Elite Offense vs Weak Turnover Defense).

---

### Tab 3: Historical Backtester

Simulate how the model's betting strategies would have performed chronologically on historical NBA seasons.

#### Controls:
- **Season Filter**: Test across `2024-25`, `2023-24`, `2022-23`, or `All Seasons`.
- **Dynamic Compounding Bankroll**: Toggle between fixed base bankroll sizing vs exponential compounding growth.
- **Active Markets**: Select `Moneyline`, `Spread`, and/or `Total`.

#### Key Metrics:
- **ROI (%)**: Net Profit divided by Total Volume Staked.
- **Win Rate (%)**: Percentage of winning wagers.
- **Max Drawdown (%)**: Maximum peak-to-trough decline in bankroll.
- **Sharpe Ratio**: Risk-adjusted excess return per unit volatility.
- **Beat Closing Line Rate (%) & Avg CLV (%)**: The percentage of bets where you beat the closing market odds (the gold standard metric of sharp sports betting).
- **Equity Curve Chart**: Interactive Plotly time-series showing portfolio growth over time.
- **Bet Log Table**: Complete searchable transaction ledger of every simulated bet.

---

### Tab 4: Calibration & Model Lab

Evaluates model reliability and accuracy:
- **Reliability Diagram**: Compares predicted probabilities against empirical win frequencies. If the blue line aligns with the dotted 45° diagonal, the probabilities are mathematically calibrated.
- **Log Loss**: Measures the cross-entropy penalty of probability forecasts (lower is better).
- **Brier Score**: Mean squared error of probability predictions (0.0 is perfect).
- **Margin MAE & RMSE**: Average point margin error in regulation games.

---

### Tab 5: Data & Settings

- **SQLite Database Overview**: View total row counts for games, logs, and odds.
- **Data Ingestion Sync**: Trigger automatic multi-season data sync from `nba_api`.
- **API Key Setup**: Add your `ODDS_API_KEY` for live bookmaker line ingestion.

---

## 3. CLI (Command Line Interface) Guide

The system includes a CLI interface powered by `rich`:

| Command | Usage | Description |
| :--- | :--- | :--- |
| `stats` | `python -m src.app.cli stats` | Displays SQLite database tables and record counts |
| `predict` | `python -m src.app.cli predict --home BOS --away MIA --home-ml 1.65 --away-ml 2.35` | Predicts game margin, win probs, and sizes bets |
| `backtest` | `python -m src.app.cli backtest --season 2024-25 --kelly 0.15 --min-edge 0.025` | Runs historical backtesting and outputs summary table |
| `train` | `python -m src.app.cli train --splits 5` | Re-trains models with TimeSeriesSplit cross validation |
| `sync` | `python -m src.app.cli sync --seasons 2023-24,2024-25` | Downloads latest game logs into SQLite database |

---

## 4. Mathematical & Modeling Reference

### Zero Lookahead Lag Engine
To eliminate lookahead bias (data leakage), every rolling feature for Game $N$ is shifted by 1:
$$\text{Feature}_{\text{Game } N} = f(\text{Game } 1, \text{Game } 2, \dots, \text{Game } N-1)$$

### Dean Oliver's Four Factors
$$\text{Possessions} = 0.5 \times \left[ (\text{FGA} + 0.44 \times \text{FTA} - \text{OREB} + \text{TOV}) + (\text{Opp\_FGA} + 0.44 \times \text{Opp\_FTA} - \text{Opp\_OREB} + \text{Opp\_TOV}) \right]$$
$$\text{Pace} = 48 \times \frac{\text{Possessions}}{\text{Minutes}}$$
$$\text{eFG\%} = \frac{\text{FGM} + 0.5 \times \text{3PM}}{\text{FGA}}$$
$$\text{TOV\%} = \frac{\text{TOV}}{\text{FGA} + 0.44 \times \text{FTA} + \text{TOV}}$$
$$\text{OREB\%} = \frac{\text{OREB}}{\text{OREB} + \text{Opp\_DREB}}$$
$$\text{FTR} = \frac{\text{FTA}}{\text{FGA}}$$

### Probability Calibration & Normal CDF
Model point margin forecasts $\Delta \hat{y}$ are converted to Win Probability via the standard normal cumulative distribution function $\Phi$:
$$P(\text{Home Win}) = \Phi\left( \frac{\Delta \hat{y}}{\sigma_{\text{residuals}}} \right)$$
Then refined with **Platt Scaling** (Logistic Calibration):
$$P_{\text{calibrated}} = \frac{1}{1 + e^{-(A \cdot \text{logit}(P) + B)}}$$

### Vig Removal
Bookmakers build an overround into odds. We remove vig using the **Multiplicative**, **Power**, and **Shin** methods:
$$P_{\text{fair, home}} = \frac{\frac{1}{d_{\text{home}}}}{\frac{1}{d_{\text{home}}} + \frac{1}{d_{\text{away}}}}$$

### Expected Value (+EV) & Fractional Kelly
$$\text{EV} = \left( P_{\text{model}} \times (\text{Decimal Odds} - 1) \right) - \left( (1 - P_{\text{model}}) \times 1.0 \right)$$
$$\text{Edge} = P_{\text{model}} - P_{\text{fair}}$$
$$\text{Fractional Kelly } f^* = c \times \left( \frac{b \cdot p - q}{b} \right)$$
where:
- $c = \text{Kelly fraction (e.g. 0.15)}$
- $b = \text{Decimal Odds} - 1$
- $p = P_{\text{model}}$
- $q = 1 - p$
