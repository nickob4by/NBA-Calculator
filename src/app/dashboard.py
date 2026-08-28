import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

import config
from src.db.database import db
from src.features.matchup_builder import build_full_feature_dataset, get_feature_columns
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.betting.odds_math import american_to_decimal, decimal_to_american, remove_vig
from src.betting.ev_engine import evaluate_moneyline_market, evaluate_spread_market, evaluate_totals_market
from src.betting.kelly import size_bet, calculate_kelly_fractional
from src.betting.backtester import HistoricalBacktester
from src.ingestion.pipeline import generate_seed_dataset_if_empty, sync_season

st.set_page_config(
    page_title="NBA Calculator — Predictive Analytics & +EV Sizing",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-win {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-loss {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-ev {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    generate_seed_dataset_if_empty()
    margin_model = MarginPredictor.load()
    totals_model = TotalsPredictor.load()
    win_model = WinProbabilityModel.load()
    logs_df = db.fetch_df("SELECT * FROM team_game_logs ORDER BY game_date, game_id")
    matchups_df = build_full_feature_dataset(logs_df)
    
    metrics_path = config.MODELS_DIR / "model_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return margin_model, totals_model, win_model, matchups_df, metrics

margin_model, totals_model, win_model, matchups_df, model_metrics = load_models_and_data()

# Sidebar
st.sidebar.title("🏀 NBA Calculator")
st.sidebar.caption("Quantitative Betting Engine & Analytics")

bankroll_val = st.sidebar.number_input("Your Bankroll ($)", min_value=100.0, max_value=1000000.0, value=10000.0, step=500.0)
kelly_mult = st.sidebar.slider("Fractional Kelly Multiplier", min_value=0.05, max_value=0.50, value=0.15, step=0.01, help="0.15 = 15% Fractional Kelly")
min_edge_pct = st.sidebar.slider("Min Edge Threshold (%)", min_value=0.5, max_value=10.0, value=2.5, step=0.5) / 100.0

tab_predict, tab_four_factors, tab_backtest, tab_diagnostics, tab_data = st.tabs([
    "🎯 Matchup Forecast & +EV Sizer",
    "📊 Four Factors Explorer",
    "📈 Historical Backtest",
    "🔬 Calibration & Model Lab",
    "⚙️ Data & Settings"
])

# ================= TAB 1: PREDICTION & +EV SIZER =================
with tab_predict:
    st.subheader("Matchup Prediction & Positive EV (+EV) Discovery")
    
    col_h, col_a = st.columns(2)
    
    team_list = list(config.NBA_TEAMS.keys())
    team_options = {k: f"{v['name']} ({v['abbrev']})" for k, v in config.NBA_TEAMS.items()}

    with col_h:
        home_team_id = st.selectbox("Select Home Team", options=team_list, format_func=lambda x: team_options[x], index=1)
    with col_a:
        # Default to another team (e.g. LAL)
        away_team_id = st.selectbox("Select Away Team", options=team_list, format_func=lambda x: team_options[x], index=13)

    if home_team_id == away_team_id:
        st.warning("Please select two distinct teams for the matchup.")
    else:
        # Extract features
        recent_home = matchups_df[matchups_df["home_team_id"] == home_team_id].iloc[-1:]
        recent_away = matchups_df[matchups_df["away_team_id"] == away_team_id].iloc[-1:]

        eval_row = recent_home.copy() if not recent_home.empty else matchups_df.iloc[-1:].copy()
        feature_cols = get_feature_columns(eval_row)
        X = eval_row[feature_cols]

        pred_margin = float(margin_model.predict(X)[0])
        pred_total = float(totals_model.predict(X)[0])
        pred_p_home = float(win_model.predict_proba(predicted_margins=np.array([pred_margin]), sigma=margin_model.residual_std)[0])
        pred_p_away = 1.0 - pred_p_home

        h_name = config.get_team_name(home_team_id)
        a_name = config.get_team_name(away_team_id)

        # Forecast Metrics Display
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"🏠 {h_name} Win Prob", f"{pred_p_home*100:.1f}%", f"Fair: {1.0/pred_p_home:.2f}")
        m2.metric(f"✈️ {a_name} Win Prob", f"{pred_p_away*100:.1f}%", f"Fair: {1.0/pred_p_away:.2f}")
        m3.metric("Projected Point Margin", f"{pred_margin:+.1f} pts", f"{h_name if pred_margin > 0 else a_name} favored")
        m4.metric("Projected Total Points", f"{pred_total:.1f} pts", "Combined Score")

        # Market Odds Inputs
        st.markdown("### 🎲 Sportsbook Lines & Expected Value Engine")
        c_ml, c_sp, c_tot = st.columns(3)

        with c_ml:
            st.markdown("**Moneyline (Decimal / American)**")
            def_h_ml = round(1.0 / (pred_p_home * 1.025), 2)
            def_a_ml = round(1.0 / (pred_p_away * 1.025), 2)
            input_h_ml = st.number_input(f"{config.get_team_abbrev(home_team_id)} Odds", value=float(def_h_ml), step=0.05)
            input_a_ml = st.number_input(f"{config.get_team_abbrev(away_team_id)} Odds", value=float(def_a_ml), step=0.05)

        with c_sp:
            st.markdown("**Point Spread**")
            def_spread = round(-pred_margin * 2.0) / 2.0
            input_spread = st.number_input(f"Home Spread Line", value=float(def_spread), step=0.5)
            input_sp_odds = st.number_input("Spread Price (e.g. 1.91)", value=1.91, step=0.01)

        with c_tot:
            st.markdown("**Over / Under Total**")
            def_tot_line = round(pred_total * 2.0) / 2.0
            input_total = st.number_input("Total Line", value=float(def_tot_line), step=0.5)
            input_tot_odds = st.number_input("Totals Price (e.g. 1.91)", value=1.91, step=0.01)

        # Compute EV & Sizing
        ml_eval = evaluate_moneyline_market(pred_p_home, input_h_ml, input_a_ml, min_edge=min_edge_pct)
        sp_eval = evaluate_spread_market(pred_margin, input_spread, input_sp_odds, input_sp_odds, residual_std=margin_model.residual_std, min_edge=min_edge_pct)
        tot_eval = evaluate_totals_market(pred_total, input_total, input_tot_odds, input_tot_odds, total_residual_std=totals_model.total_residual_std, min_edge=min_edge_pct)

        all_opps = ml_eval["opportunities"] + sp_eval["opportunities"] + tot_eval["opportunities"]

        st.markdown("#### 💎 Recommended Actionable Bets (Fractional Kelly)")
        if not all_opps:
            st.info("No market bets currently satisfy the minimum edge criteria. Model recommends **NO BET**.")
        else:
            rec_rows = []
            for opp in all_opps:
                sizing = size_bet(opp["model_prob"], opp["decimal_odds"], bankroll_val, kelly_multiplier=kelly_mult)
                team_label = h_name if opp["side"] == "home" else (a_name if opp["side"] == "away" else opp["side"].upper())
                rec_rows.append({
                    "Market": opp["market_type"].upper(),
                    "Selection": f"{team_label}",
                    "Offered Odds": f"{opp['decimal_odds']:.2f}",
                    "Model Prob": f"{opp['model_prob']*100:.1f}%",
                    "Fair Implied": f"{opp['fair_implied_prob']*100:.1f}%",
                    "Edge (%)": f"{opp['edge']*100:+.2f}%",
                    "Expected Value": f"{opp['ev']:+.3f}",
                    "Kelly %": f"{sizing['stake_pct']:.2f}%",
                    "Recommended Stake ($)": f"${sizing['stake']:,.2f}"
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)

# ================= TAB 2: FOUR FACTORS EXPLORER =================
with tab_four_factors:
    st.subheader("Dean Oliver's Four Factors & Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        team_a_sel = st.selectbox("Team 1", options=team_list, format_func=lambda x: team_options[x], index=1, key="ff_t1")
    with col2:
        team_b_sel = st.selectbox("Team 2", options=team_list, format_func=lambda x: team_options[x], index=13, key="ff_t2")

    # Fetch team recent stats
    t1_logs = matchups_df[matchups_df["home_team_id"] == team_a_sel].iloc[-1:]
    t2_logs = matchups_df[matchups_df["home_team_id"] == team_b_sel].iloc[-1:]

    if not t1_logs.empty and not t2_logs.empty:
        t1_row = t1_logs.iloc[0]
        t2_row = t2_logs.iloc[0]

        # Radar Chart of Four Factors
        categories = ["eFG% (Shooting)", "TOV% (Care)", "OREB% (Glass)", "FTR (Free Throws)", "Net Rating"]
        
        # Normalized values for radar
        t1_vals = [
            float(t1_row.get("home_roll_efg_pct_w10", 0.52)) * 100,
            (1.0 - float(t1_row.get("home_roll_tov_pct_w10", 0.13))) * 100,
            float(t1_row.get("home_roll_orb_pct_w10", 0.25)) * 100,
            float(t1_row.get("home_roll_ftr_w10", 0.22)) * 100,
            float(t1_row.get("home_roll_net_rating_w10", 3.0)) + 50
        ]

        t2_vals = [
            float(t2_row.get("home_roll_efg_pct_w10", 0.52)) * 100,
            (1.0 - float(t2_row.get("home_roll_tov_pct_w10", 0.13))) * 100,
            float(t2_row.get("home_roll_orb_pct_w10", 0.25)) * 100,
            float(t2_row.get("home_roll_ftr_w10", 0.22)) * 100,
            float(t2_row.get("home_roll_net_rating_w10", 3.0)) + 50
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=t1_vals, theta=categories, fill='toself', name=config.get_team_name(team_a_sel)))
        fig.add_trace(go.Scatterpolar(r=t2_vals, theta=categories, fill='toself', name=config.get_team_name(team_b_sel)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Four Factors Radar Comparison (10-Game Form)")
        
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: HISTORICAL BACKTEST =================
with tab_backtest:
    st.subheader("Historical Simulation & Performance Metrics")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        bt_season = st.selectbox("Season Filter", options=["All Seasons", "2024-25", "2023-24", "2022-23"], index=1)
    with b_col2:
        bt_compound = st.checkbox("Dynamic Compounding Bankroll", value=False)
    with b_col3:
        bt_markets = st.multiselect("Active Markets", options=["moneyline", "spread", "total"], default=["moneyline", "spread", "total"])
    with b_col4:
        run_btn = st.button("Run Simulation", type="primary")

    season_arg = None if bt_season == "All Seasons" else bt_season

    bt = HistoricalBacktester(
        starting_bankroll=bankroll_val,
        kelly_fraction=kelly_mult,
        min_edge=min_edge_pct,
        compound_bankroll=bt_compound,
        markets=bt_markets
    )
    
    with st.spinner("Simulating historical wagers chronologically..."):
        res = bt.run_backtest(season_filter=season_arg)

    # Metrics row
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Total Wagers", f"{res['total_bets']:,}")
    r2.metric("Win Rate", f"{res['win_rate']}%", f"{res['wins']}W - {res['losses']}L")
    r3.metric("Net Profit ($)", f"${res['pnl']:,.2f}")
    r4.metric("ROI (%)", f"{res['roi_pct']:+.2f}%")
    r5.metric("Max Drawdown", f"{res['max_drawdown_pct']:.2f}%", f"${res['max_drawdown_dollars']:,.2f}")
    r6.metric("Beat Closing Line", f"{res['beat_closing_pct']}%", f"Avg CLV: {res['avg_clv_pct']:+.2f}%")

    # Equity Curve Plotly Chart
    if res["equity_curve"]:
        eq_df = pd.DataFrame(res["equity_curve"])
        fig_equity = px.line(
            eq_df, x="game_date", y="bankroll",
            title="Portfolio Equity Trajectory ($)",
            labels={"game_date": "Date", "bankroll": "Bankroll ($)"}
        )
        fig_equity.update_traces(line=dict(color="#10b981", width=2.5))
        st.plotly_chart(fig_equity, use_container_width=True)

    # Bets Table
    if res["bets"]:
        st.markdown("#### Chronological Bet Log")
        st.dataframe(pd.DataFrame(res["bets"]).tail(100), use_container_width=True)

# ================= TAB 4: CALIBRATION & MODEL LAB =================
with tab_diagnostics:
    st.subheader("Probability Calibration & Model Diagnostics")
    
    if model_metrics:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Log Loss", str(model_metrics.get("win_probability_metrics", {}).get("log_loss", "N/A")))
        d2.metric("Brier Score", str(model_metrics.get("win_probability_metrics", {}).get("brier_score", "N/A")))
        d3.metric("Accuracy", f"{model_metrics.get('win_probability_metrics', {}).get('accuracy', 0)*100:.1f}%")
        d4.metric("Margin MAE", f"{model_metrics.get('margin_metrics', {}).get('mae', 'N/A')} pts")

        # Reliability Diagram
        rel_bins = model_metrics.get("win_probability_metrics", {}).get("reliability_bins", [])
        if rel_bins:
            b_df = pd.DataFrame(rel_bins)
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
            fig_rel.add_trace(go.Scatter(x=b_df["confidence"], y=b_df["accuracy"], mode="lines+markers", name="Model Calibration", line=dict(color="#3b82f6", width=3)))
            fig_rel.update_layout(
                title="Reliability Diagram (Predicted Probability vs True Win Rate)",
                xaxis_title="Forecast Confidence (Predicted Probability)",
                yaxis_title="Empirical Win Frequency",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_rel, use_container_width=True)

# ================= TAB 5: DATA & SETTINGS =================
with tab_data:
    st.subheader("Database & Ingestion Settings")
    games_cnt = db.fetch_one("SELECT COUNT(*) as c FROM games")["c"]
    logs_cnt = db.fetch_one("SELECT COUNT(*) as c FROM team_game_logs")["c"]
    odds_cnt = db.fetch_one("SELECT COUNT(*) as c FROM odds")["c"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### 📦 SQLite Status")
        st.write(f"- **Games Table**: {games_cnt:,} records")
        st.write(f"- **Team Logs Table**: {logs_cnt:,} records")
        st.write(f"- **Odds Table**: {odds_cnt:,} records")
        st.write(f"- **Database Path**: `{config.DB_PATH}`")

    with col_s2:
        st.markdown("### 🔄 Sync Seasons")
        sync_season_input = st.text_input("Season to sync (e.g. 2024-25)", value="2024-25")
        if st.button("Trigger Sync"):
            with st.spinner("Syncing..."):
                cnt = sync_season(sync_season_input)
                st.success(f"Synced {cnt} games!")