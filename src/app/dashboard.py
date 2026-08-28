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
from src.models.mlb_models import MLBRunMarginPredictor, MLBTotalsPredictor, MLBWinProbabilityModel
from src.betting.odds_math import american_to_decimal, decimal_to_american, remove_vig
from src.betting.ev_engine import evaluate_moneyline_market, evaluate_spread_market, evaluate_totals_market
from src.betting.kelly import size_bet, calculate_kelly_fractional
from src.betting.backtester import HistoricalBacktester
from src.ingestion.pipeline import generate_seed_dataset_if_empty, sync_season
from src.ingestion.mlb_api_fetcher import generate_mlb_seed_dataset_if_empty

st.set_page_config(
    page_title="Multi-Sport Quantitative Calculator (NBA & MLB)",
    page_icon="🏆",
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
    .stSelectbox label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: Sport Selector
st.sidebar.title("🏆 Sportsbook Analytics")
sport_choice = st.sidebar.selectbox(
    "Choose Sport",
    options=["🏀 NBA (Basketball)", "⚾ MLB (Baseball)"],
    index=0
)
sport = "mlb" if "MLB" in sport_choice else "nba"
sport_label = "MLB (Baseball)" if sport == "mlb" else "NBA (Basketball)"

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Bankroll & Risk Settings")
bankroll_val = st.sidebar.number_input(f"Your Bankroll ({config.DEFAULT_CURRENCY})", min_value=50.0, max_value=10000000.0, value=float(config.DEFAULT_STARTING_BANKROLL), step=50.0)
kelly_mult = st.sidebar.slider("Fractional Kelly Multiplier", min_value=0.05, max_value=0.50, value=0.15, step=0.01, help="0.15 = 15% Fractional Kelly")
min_edge_pct = st.sidebar.slider("Min Edge Threshold (%)", min_value=0.5, max_value=10.0, value=2.5, step=0.5) / 100.0

@st.cache_resource
def load_sport_models_and_data(target_sport: str):
    if target_sport == "mlb":
        generate_mlb_seed_dataset_if_empty()
        margin_model = MLBRunMarginPredictor.load()
        totals_model = MLBTotalsPredictor.load()
        win_model = MLBWinProbabilityModel.load()
        metrics_path = config.MODELS_DIR / "mlb_model_metrics.json"
    else:
        generate_seed_dataset_if_empty()
        margin_model = MarginPredictor.load()
        totals_model = TotalsPredictor.load()
        win_model = WinProbabilityModel.load()
        metrics_path = config.MODELS_DIR / "model_metrics.json"

    logs_df = db.fetch_df("SELECT * FROM team_game_logs WHERE sport=? ORDER BY game_date, game_id", (target_sport,))
    matchups_df = build_full_feature_dataset(logs_df, sport=target_sport)
    
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return margin_model, totals_model, win_model, matchups_df, logs_df, metrics

margin_model, totals_model, win_model, matchups_df, logs_df, model_metrics = load_sport_models_and_data(sport)

# Navigation Tabs
score_unit = "Runs" if sport == "mlb" else "Points"
spread_title = "Run Line (±1.5)" if sport == "mlb" else "Point Spread"
adv_tab_title = "⚾ Sabermetrics Explorer" if sport == "mlb" else "🏀 Four Factors Explorer"

tab_predict, tab_analytics, tab_backtest, tab_diagnostics, tab_data = st.tabs([
    f"🎯 {sport.upper()} Matchup & +EV Sizer",
    adv_tab_title,
    "📈 Historical Backtest",
    "🔬 Calibration & Model Lab",
    "⚙️ Data & Settings"
])

# ================= TAB 1: PREDICTION & +EV SIZER =================
with tab_predict:
    st.subheader(f"{sport_label} — Matchup Prediction & Positive EV (+EV) Discovery")
    
    col_h, col_a = st.columns(2)
    teams_dict = config.get_teams_for_sport(sport)
    team_list = list(teams_dict.keys())
    team_options = {k: f"{v['name']} ({v['abbrev']})" for k, v in teams_dict.items()}

    default_home_idx = 28 if sport == "mlb" else 1 # NYY for MLB, BOS for NBA
    default_away_idx = 3 if sport == "mlb" else 13  # BOS for MLB, LAL for NBA

    with col_h:
        home_team_id = st.selectbox(f"Select Home Team ({sport.upper()})", options=team_list, format_func=lambda x: team_options[x], index=min(default_home_idx, len(team_list)-1))
    with col_a:
        away_team_id = st.selectbox(f"Select Away Team ({sport.upper()})", options=team_list, format_func=lambda x: team_options[x], index=min(default_away_idx, len(team_list)-1))

    if home_team_id == away_team_id:
        st.warning("Please select two distinct teams for the matchup.")
    else:
        home_sp_fip = None
        away_sp_fip = None
        if sport == "mlb":
            from src.features.mlb_pitcher_metrics import get_team_starters
            st.markdown("#### ⚾ Starting Pitcher Duel")
            sp_col1, sp_col2 = st.columns(2)
            
            h_starters = get_team_starters(home_team_id)
            a_starters = get_team_starters(away_team_id)
            
            with sp_col1:
                h_sp_idx = st.selectbox(
                    f"🏠 {config.get_team_abbrev(home_team_id, sport='mlb')} Starting Pitcher",
                    options=range(len(h_starters)),
                    format_func=lambda i: f"{h_starters[i]['name']} ({h_starters[i]['fip']:.2f} FIP | {h_starters[i]['whip']:.2f} WHIP | {h_starters[i]['k9']:.1f} K/9)",
                    key="home_sp_select"
                )
                h_selected_sp = h_starters[h_sp_idx]
                home_sp_fip = h_selected_sp["fip"]

            with sp_col2:
                a_sp_idx = st.selectbox(
                    f"✈️ {config.get_team_abbrev(away_team_id, sport='mlb')} Starting Pitcher",
                    options=range(len(a_starters)),
                    format_func=lambda i: f"{a_starters[i]['name']} ({a_starters[i]['fip']:.2f} FIP | {a_starters[i]['whip']:.2f} WHIP | {a_starters[i]['k9']:.1f} K/9)",
                    key="away_sp_select"
                )
                a_selected_sp = a_starters[a_sp_idx]
                away_sp_fip = a_selected_sp["fip"]

        from src.features.matchup_builder import build_upcoming_matchup
        eval_row = build_upcoming_matchup(home_team_id, away_team_id, logs_df, sport=sport, home_sp_fip=home_sp_fip, away_sp_fip=away_sp_fip)
        feature_cols = margin_model.feature_names
        X = eval_row[feature_cols]

        pred_margin = float(margin_model.predict(X)[0])
        pred_total = float(totals_model.predict(X)[0])
        pred_p_home = float(win_model.predict_proba(predicted_margins=np.array([pred_margin]), sigma=margin_model.residual_std)[0])
        pred_p_away = 1.0 - pred_p_home

        h_name = config.get_team_name(home_team_id, sport=sport)
        a_name = config.get_team_name(away_team_id, sport=sport)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"🏠 {h_name} Win Prob", f"{pred_p_home*100:.1f}%", f"Fair: {1.0/pred_p_home:.2f}")
        m2.metric(f"✈️ {a_name} Win Prob", f"{pred_p_away*100:.1f}%", f"Fair: {1.0/pred_p_away:.2f}")
        m3.metric(f"Projected {score_unit} Margin", f"{pred_margin:+.2f} {score_unit.lower()}", f"{h_name if pred_margin > 0 else a_name} favored")
        m4.metric(f"Projected Total {score_unit}", f"{pred_total:.2f} {score_unit.lower()}", "Combined Score")

        st.markdown("### 🎲 Sportsbook Lines & Expected Value Engine")
        c_ml, c_sp, c_tot = st.columns(3)

        with c_ml:
            st.markdown("**Moneyline (Decimal Odds)**")
            def_h_ml = round(1.0 / (pred_p_home * 1.025), 2)
            def_a_ml = round(1.0 / (pred_p_away * 1.025), 2)
            input_h_ml = st.number_input(f"{config.get_team_abbrev(home_team_id, sport=sport)} Odds", value=float(def_h_ml), step=0.05)
            input_a_ml = st.number_input(f"{config.get_team_abbrev(away_team_id, sport=sport)} Odds", value=float(def_a_ml), step=0.05)

        with c_sp:
            st.markdown(f"**{spread_title}**")
            def_spread = -1.5 if sport == "mlb" else (round(-pred_margin * 2.0) / 2.0)
            input_spread = st.number_input("Home Line", value=float(def_spread), step=0.5)
            input_sp_odds = st.number_input("Home Price (e.g. 2.10 or 1.91)", value=2.10 if sport=="mlb" else 1.91, step=0.01)

        with c_tot:
            st.markdown(f"**Over / Under {score_unit}**")
            def_tot_line = 8.5 if sport == "mlb" else (round(pred_total * 2.0) / 2.0)
            input_total = st.number_input("Total Line", value=float(def_tot_line), step=0.5)
            input_tot_odds = st.number_input("Totals Price (e.g. 1.91)", value=1.91, step=0.01)

        ml_eval = evaluate_moneyline_market(pred_p_home, input_h_ml, input_a_ml, min_edge=min_edge_pct)
        sp_eval = evaluate_spread_market(pred_margin, input_spread, input_sp_odds, 1.91, residual_std=margin_model.residual_std, min_edge=min_edge_pct)
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
                market_display = "RUN LINE" if (sport=="mlb" and opp["market_type"]=="spread") else opp["market_type"].upper()
                rec_rows.append({
                    "Market": market_display,
                    "Selection": f"{team_label}",
                    "Offered Odds": f"{opp['decimal_odds']:.2f}",
                    "Model Prob": f"{opp['model_prob']*100:.1f}%",
                    "Fair Implied": f"{opp['fair_implied_prob']*100:.1f}%",
                    "Edge (%)": f"{opp['edge']*100:+.2f}%",
                    "Expected Value": f"{opp['ev']:+.3f}",
                    "Kelly %": f"{sizing['stake_pct']:.2f}%",
                    f"Recommended Stake ({config.DEFAULT_CURRENCY})": f"{config.DEFAULT_CURRENCY}{sizing['stake']:,.2f}"
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)

# ================= TAB 2: ANALYTICS (FOUR FACTORS OR SABERMETRICS) =================
with tab_analytics:
    if sport == "mlb":
        st.subheader("⚾ Baseball Sabermetrics & Starting Rotation Explorer")
        col1, col2 = st.columns(2)
        with col1:
            t1_sel = st.selectbox("Team 1", options=team_list, format_func=lambda x: team_options[x], index=default_home_idx, key="mlb_t1")
        with col2:
            t2_sel = st.selectbox("Team 2", options=team_list, format_func=lambda x: team_options[x], index=default_away_idx, key="mlb_t2")

        from src.features.mlb_pitcher_metrics import get_team_starters
        t1_rot = get_team_starters(t1_sel)
        t2_rot = get_team_starters(t2_sel)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**{config.get_team_name(t1_sel, sport='mlb')} Starting Rotation**")
            st.dataframe(pd.DataFrame(t1_rot)[["name", "fip", "whip", "k9", "throws"]].rename(columns={"name": "Pitcher", "fip": "FIP", "whip": "WHIP", "k9": "K/9", "throws": "Hand"}), use_container_width=True)
        with p_col2:
            st.markdown(f"**{config.get_team_name(t2_sel, sport='mlb')} Starting Rotation**")
            st.dataframe(pd.DataFrame(t2_rot)[["name", "fip", "whip", "k9", "throws"]].rename(columns={"name": "Pitcher", "fip": "FIP", "whip": "WHIP", "k9": "K/9", "throws": "Hand"}), use_container_width=True)

        t1_logs = matchups_df[matchups_df["home_team_id"] == t1_sel].iloc[-1:]
        t2_logs = matchups_df[matchups_df["home_team_id"] == t2_sel].iloc[-1:]

        if not t1_logs.empty and not t2_logs.empty:
            t1_r = t1_logs.iloc[0]
            t2_r = t2_logs.iloc[0]

            categories = ["Pythagorean Win %", "OPS (x100)", "wOBA (x100)", "Run Prevention (FIP Inv)", "WHIP Inv"]
            
            t1_vals = [
                float(t1_r.get("home_roll_pythag_win_pct_w10", 0.50)) * 100,
                float(t1_r.get("home_roll_ops_w10", 0.75)) * 100,
                float(t1_r.get("home_roll_woba_proxy_w10", 0.32)) * 200,
                (10.0 - float(t1_r.get("home_roll_fip_proxy_w10", 4.0))) * 10,
                (3.0 - float(t1_r.get("home_roll_whip_w10", 1.30))) * 40
            ]
            t2_vals = [
                float(t2_r.get("home_roll_pythag_win_pct_w10", 0.50)) * 100,
                float(t2_r.get("home_roll_ops_w10", 0.75)) * 100,
                float(t2_r.get("home_roll_woba_proxy_w10", 0.32)) * 200,
                (10.0 - float(t2_r.get("home_roll_fip_proxy_w10", 4.0))) * 10,
                (3.0 - float(t2_r.get("home_roll_whip_w10", 1.30))) * 40
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=t1_vals, theta=categories, fill='toself', name=config.get_team_name(t1_sel, sport="mlb")))
            fig.add_trace(go.Scatterpolar(r=t2_vals, theta=categories, fill='toself', name=config.get_team_name(t2_sel, sport="mlb")))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="MLB Sabermetric Radar Comparison (10-Game Lagged Form)")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("🏀 Dean Oliver's Four Factors & Head-to-Head Comparison")
        col1, col2 = st.columns(2)
        with col1:
            t1_sel = st.selectbox("Team 1", options=team_list, format_func=lambda x: team_options[x], index=default_home_idx, key="nba_t1")
        with col2:
            t2_sel = st.selectbox("Team 2", options=team_list, format_func=lambda x: team_options[x], index=default_away_idx, key="nba_t2")

        t1_logs = matchups_df[matchups_df["home_team_id"] == t1_sel].iloc[-1:]
        t2_logs = matchups_df[matchups_df["home_team_id"] == t2_sel].iloc[-1:]

        if not t1_logs.empty and not t2_logs.empty:
            t1_row = t1_logs.iloc[0]
            t2_row = t2_logs.iloc[0]

            categories = ["eFG% (Shooting)", "TOV% (Care)", "OREB% (Glass)", "FTR (Free Throws)", "Net Rating"]
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
            fig.add_trace(go.Scatterpolar(r=t1_vals, theta=categories, fill='toself', name=config.get_team_name(t1_sel, sport="nba")))
            fig.add_trace(go.Scatterpolar(r=t2_vals, theta=categories, fill='toself', name=config.get_team_name(t2_sel, sport="nba")))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="NBA Four Factors Radar Comparison (10-Game Form)")
            st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: HISTORICAL BACKTEST =================
with tab_backtest:
    st.subheader(f"📈 {sport.upper()} Historical Simulation & Performance Metrics")
    
    season_options = ["All Seasons", "2024", "2023", "2022", "2021", "2020"] if sport == "mlb" else ["All Seasons", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
    market_options = ["moneyline", "spread", "total"]

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        bt_season = st.selectbox("Season Filter", options=season_options, index=1, key=f"bt_season_filter_{sport}")
    with b_col2:
        bt_compound = st.checkbox("Dynamic Compounding Bankroll", value=False, key=f"bt_compound_{sport}")
    with b_col3:
        bt_markets = st.multiselect("Active Markets", options=market_options, default=market_options, key=f"bt_markets_{sport}")
    with b_col4:
        run_btn = st.button("Run Simulation", type="primary", key=f"bt_run_btn_{sport}")

    season_arg = None if bt_season == "All Seasons" else bt_season

    bt = HistoricalBacktester(
        sport=sport,
        starting_bankroll=bankroll_val,
        kelly_fraction=kelly_mult,
        min_edge=min_edge_pct,
        compound_bankroll=bt_compound,
        markets=bt_markets
    )
    
    with st.spinner(f"Simulating {sport.upper()} historical wagers chronologically..."):
        res = bt.run_backtest(season_filter=season_arg)

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Total Wagers", f"{res.get('total_bets', 0):,}")
    r2.metric("Win Rate", f"{res.get('win_rate', 0.0)}%", f"{res.get('wins', 0)}W - {res.get('losses', 0)}L")
    r3.metric(f"Net Profit ({config.DEFAULT_CURRENCY})", f"{config.DEFAULT_CURRENCY}{res.get('pnl', 0.0):,.2f}")
    r4.metric("ROI (%)", f"{res.get('roi_pct', 0.0):+.2f}%")
    r5.metric("Max Drawdown", f"{res.get('max_drawdown_pct', 0.0):.2f}%", f"{config.DEFAULT_CURRENCY}{res.get('max_drawdown_dollars', 0.0):,.2f}")
    r6.metric("Beat Closing Line", f"{res.get('beat_closing_pct', 0.0)}%", f"Avg CLV: {res.get('avg_clv_pct', 0.0):+.2f}%")

    if res["equity_curve"]:
        eq_df = pd.DataFrame(res["equity_curve"])
        fig_equity = px.line(
            eq_df, x="game_date", y="bankroll",
            title=f"{sport.upper()} Portfolio Equity Trajectory ({config.DEFAULT_CURRENCY})",
            labels={"game_date": "Date", "bankroll": f"Bankroll ({config.DEFAULT_CURRENCY})"}
        )
        fig_equity.update_traces(line=dict(color="#10b981", width=2.5))
        st.plotly_chart(fig_equity, use_container_width=True)

    if res["bets"]:
        st.markdown(f"#### {sport.upper()} Chronological Bet Log")
        st.dataframe(pd.DataFrame(res["bets"]).tail(100), use_container_width=True)

# ================= TAB 4: CALIBRATION & MODEL LAB =================
with tab_diagnostics:
    st.subheader(f"🔬 {sport.upper()} Probability Calibration & Model Diagnostics")
    
    if model_metrics:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Log Loss", str(model_metrics.get("win_probability_metrics", {}).get("log_loss", "N/A")))
        d2.metric("Brier Score", str(model_metrics.get("win_probability_metrics", {}).get("brier_score", "N/A")))
        d3.metric("Accuracy", f"{model_metrics.get('win_probability_metrics', {}).get('accuracy', 0)*100:.1f}%")
        d4.metric(f"Margin MAE", f"{model_metrics.get('margin_metrics', {}).get('mae', 'N/A')} {score_unit.lower()}")

        rel_bins = model_metrics.get("win_probability_metrics", {}).get("reliability_bins", [])
        if rel_bins:
            b_df = pd.DataFrame(rel_bins)
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
            fig_rel.add_trace(go.Scatter(x=b_df["confidence"], y=b_df["accuracy"], mode="lines+markers", name=f"{sport.upper()} Model Calibration", line=dict(color="#3b82f6", width=3)))
            fig_rel.update_layout(
                title=f"{sport.upper()} Reliability Diagram (Predicted Probability vs True Win Rate)",
                xaxis_title="Forecast Confidence (Predicted Probability)",
                yaxis_title="Empirical Win Frequency",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_rel, use_container_width=True)

# ================= TAB 5: DATA & SETTINGS =================
with tab_data:
    st.subheader("⚙️ Multi-Sport Database & Ingestion Settings")
    nba_cnt = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='nba'")["c"]
    mlb_cnt = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='mlb'")["c"]
    logs_cnt = db.fetch_one("SELECT COUNT(*) as c FROM team_game_logs")["c"]
    odds_cnt = db.fetch_one("SELECT COUNT(*) as c FROM odds")["c"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### 📦 SQLite Multi-Sport Status")
        st.write(f"- **🏀 NBA Games**: {nba_cnt:,} records")
        st.write(f"- **⚾ MLB Games**: {mlb_cnt:,} records")
        st.write(f"- **Team Logs Table**: {logs_cnt:,} records")
        st.write(f"- **Odds Table**: {odds_cnt:,} records")
        st.write(f"- **Database Path**: `{config.DB_PATH}`")

    with col_s2:
        st.markdown("### 🔄 Sync Seasons")
        sync_sport = st.selectbox("Sport to Sync", options=["NBA", "MLB"], index=0 if sport=="nba" else 1)
        sync_season_input = st.text_input("Season to sync", value="2024" if sync_sport=="MLB" else "2024-25")
        if st.button("Trigger Sync"):
            with st.spinner("Syncing..."):
                if sync_sport == "MLB":
                    generate_mlb_seed_dataset_if_empty()
                    st.success("MLB data synchronized!")
                else:
                    cnt = sync_season(sync_season_input)
                    st.success(f"Synced {cnt} NBA games!")