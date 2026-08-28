# 📖 Multi-Sport Quantitative Betting & Analytics Guide (NBA & MLB)

Welcome to the **Quantitative Sports Betting & Analytics Platform**. This platform provides quantitative modeling, probability calibration, +EV betting discovery, and bankroll management for both **NBA (Basketball)** and **MLB (Baseball)** with zero lookahead bias and rigorous backtesting.

---

## 📑 Table of Contents
1. [Quick Start & Sport Selection](#1-quick-start--sport-selection)
2. [Interactive Web Dashboard Guide](#2-interactive-web-dashboard-guide)
   - [Sport Selector (NBA / MLB)](#sport-selector-nba--mlb)
   - [Tab 1: Matchup Forecast & +EV Sizer](#tab-1-matchup-forecast--ev-sizer)
   - [Tab 2: Analytics (Four Factors or Sabermetrics)](#tab-2-analytics-four-factors-or-sabermetrics)
   - [Tab 3: Historical Backtester](#tab-3-historical-backtester)
   - [Tab 4: Calibration & Model Lab](#tab-4-calibration--model-lab)
   - [Tab 5: Data & Settings](#tab-5-data--settings)
3. [CLI (Command Line Interface) Guide](#3-cli-command-line-interface-guide)
4. [Sport-Specific Mathematical Reference](#4-sport-specific-mathematical-reference)
   - [NBA: Dean Oliver's Four Factors](#nba-dean-olivers-four-factors)
   - [MLB: Sabermetrics & Bill James' Pythagorean Expectation](#mlb-sabermetrics--bill-james-pythagorean-expectation)
   - [Betting Math: Vig Removal, +EV & Fractional Kelly](#betting-math-vig-removal-ev--fractional-kelly)

---

## 1. Quick Start & Sport Selection

### Launching the Web Dashboard
From your terminal:
```powershell
.\.venv\Scripts\streamlit run src/app/dashboard.py
```
Open your browser to: **[http://localhost:8501](http://localhost:8501)**

### Running Multi-Sport CLI Commands
```powershell
# Predict an NBA game
.\.venv\Scripts\python -m src.app.cli --sport nba predict --home BOS --away LAL --home-ml 1.75 --away-ml 2.20

# Predict an MLB game
.\.venv\Scripts\python -m src.app.cli --sport mlb predict --home NYY --away BOS --home-ml 1.70 --away-ml 2.25

# Run MLB historical backtest
.\.venv\Scripts\python -m src.app.cli --sport mlb backtest --season 2024 --kelly 0.15
```

---

## 2. Interactive Web Dashboard Guide

### Sport Selector (NBA / MLB)
In the top-left sidebar, use the **Choose Sport** dropdown:
- **🏀 NBA (Basketball)**: Loads 30 NBA teams, point margin models, Four Factors, and Point Spread markets.
- **⚾ MLB (Baseball)**: Loads 30 MLB teams, run margin models, Sabermetrics (Pythagorean win %, OPS, wOBA, FIP, WHIP), and **Run Line (±1.5 runs)** markets.

---

### Tab 1: Matchup Forecast & +EV Sizer
1. **Select Home and Away Teams** from the chosen sport.
2. **Review Forecast Metrics**: Calibrated Win Probability, Fair Odds, Projected Margin (PTS or Runs), and Projected Combined Total.
3. **Enter Live Sportsbook Lines**:
   - Moneyline Decimal Odds
   - Spread / Run Line (±1.5 for MLB)
   - Over / Under Total (Points or Runs)
4. **Actionable Bets Table**: Instant +EV calculation and **Fractional Kelly Recommended Stakes** ($ and % of your bankroll).

---

### Tab 2: Analytics (Four Factors or Sabermetrics)
- **NBA**: Dean Oliver's Four Factors radar chart comparing `eFG%`, `TOV%`, `OREB%`, `FTR`, `Net Rating`, and `Pace`.
- **MLB**: Baseball Sabermetrics radar chart comparing `Pythagorean Win %`, `OPS`, `wOBA`, `Run Prevention (FIP)`, and `WHIP`.

---

### Tab 3: Historical Backtester
- Run chronological simulations for either sport.
- View interactive **Portfolio Equity Trajectory ($)** Plotly curves, **ROI %**, **Win Rate**, **Max Drawdown %**, and **Beat Closing Line Rate (CLV %)**.

---

## 3. Sport-Specific Mathematical Reference

### NBA: Dean Oliver's Four Factors
- **eFG%**: $(\text{FGM} + 0.5 \times \text{3PM}) / \text{FGA}$
- **TOV%**: $\text{TOV} / (\text{FGA} + 0.44 \times \text{FTA} + \text{TOV})$
- **OREB%**: $\text{OREB} / (\text{OREB} + \text{Opp\_DREB})$
- **FTR**: $\text{FTA} / \text{FGA}$

### MLB: Sabermetrics & Pythagorean Expectation
- **Bill James Pythagorean Win Expectation**:
  $$\text{Pythag Win \%} = \frac{\text{Runs Scored}^{1.83}}{\text{Runs Scored}^{1.83} + \text{Runs Allowed}^{1.83}}$$
- **OPS**: $\text{OBP} + \text{SLG}$
- **FIP Proxy**: $\frac{13 \times \text{HR} + 3 \times \text{BB} - 2 \times \text{SO}}{\text{IP}} + 3.15$
- **WHIP**: $(\text{BB} + \text{H}) / \text{IP}$
