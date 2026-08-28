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
    
    # Pre-calculate rolled features once and cache in memory
    from src.features.four_factors import compute_advanced_stats_dataframe
    from src.features.mlb_sabermetrics import compute_mlb_sabermetrics_dataframe
    from src.features.situational import compute_situational_features
    from src.features.rolling_features import compute_rolling_team_features
    
    if target_sport == "mlb":
        adv_df = compute_mlb_sabermetrics_dataframe(logs_df)
        adv_cols = [c for c in adv_df.columns if c not in logs_df.columns or c in ["game_id", "team_id", "opponent_id", "sport"]]
        rolled_logs = logs_df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id", "sport"], how="left")
    else:
        adv_df = compute_advanced_stats_dataframe(logs_df)
        adv_cols = [c for c in adv_df.columns if c not in logs_df.columns or c in ["game_id", "team_id", "opponent_id"]]
        rolled_logs = logs_df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id"], how="left")

    rolled_logs = compute_situational_features(rolled_logs)
    rolled_logs = compute_rolling_team_features(rolled_logs)
    matchups_df = build_full_feature_dataset(logs_df, sport=target_sport)
    
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return margin_model, totals_model, win_model, matchups_df, rolled_logs, metrics

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
        # Winner Forecast Metrics
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            h_is_fav = pred_p_home >= 0.50
            h_badge = "🔥 PREDICTED WINNER" if h_is_fav else "UNDERDOG"
            st.markdown(f"""
            <div style="background-color: {'#132e24' if h_is_fav else '#1e2530'}; border: 1px solid {'#10b981' if h_is_fav else '#2d3748'}; border-radius: 10px; padding: 18px;">
                <span style="font-size: 13px; color: {'#34d399' if h_is_fav else '#9ca3af'}; font-weight: bold;">{h_badge}</span>
                <h2 style="margin: 4px 0; color: white;">🏠 {h_name}</h2>
                <h1 style="margin: 0; color: {'#10b981' if h_is_fav else '#f3f4f6'}; font-size: 42px;">{pred_p_home*100:.1f}%</h1>
                <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 14px;">Fair True Price: <b>{1.0/pred_p_home:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with w_col2:
            a_is_fav = pred_p_away > 0.50
            a_badge = "🔥 PREDICTED WINNER" if a_is_fav else "UNDERDOG"
            st.markdown(f"""
            <div style="background-color: {'#132e24' if a_is_fav else '#1e2530'}; border: 1px solid {'#10b981' if a_is_fav else '#2d3748'}; border-radius: 10px; padding: 18px;">
                <span style="font-size: 13px; color: {'#34d399' if a_is_fav else '#9ca3af'}; font-weight: bold;">{a_badge}</span>
                <h2 style="margin: 4px 0; color: white;">✈️ {a_name}</h2>
                <h1 style="margin: 0; color: {'#10b981' if a_is_fav else '#f3f4f6'}; font-size: 42px;">{pred_p_away*100:.1f}%</h1>
                <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 14px;">Fair True Price: <b>{1.0/pred_p_away:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎲 Enter Sportsbook Odds (To Win)")
        c_odd1, c_odd2 = st.columns(2)

        with c_odd1:
            def_h_ml = round(1.0 / (pred_p_home * 1.025), 2)
            input_h_ml = st.number_input(f"{h_name} Offered Odds (Decimal)", value=float(def_h_ml), step=0.05, key="input_h_ml")

        with c_odd2:
            def_a_ml = round(1.0 / (pred_p_away * 1.025), 2)
            input_a_ml = st.number_input(f"{a_name} Offered Odds (Decimal)", value=float(def_a_ml), step=0.05, key="input_a_ml")

        # Evaluate Outright Winner (Moneyline)
        ml_eval = evaluate_moneyline_market(pred_p_home, input_h_ml, input_a_ml, min_edge=min_edge_pct)
        opps = ml_eval["opportunities"]

        st.markdown("### 💡 Recommended Bet")
        if not opps:
            st.info(f"🚫 **NO BET RECOMMENDED**: Neither team offers positive EV value (>{min_edge_pct*100:.1f}% edge) against current bookmaker prices. Save your capital for a higher value opportunity.")
        else:
            for opp in opps:
                sizing = size_bet(opp["model_prob"], opp["decimal_odds"], bankroll_val, kelly_multiplier=kelly_mult)
                bet_team = h_name if opp["side"] == "home" else a_name
                stake_amt = sizing["stake"]
                payout = round(stake_amt * opp["decimal_odds"], 2)
                profit = round(payout - stake_amt, 2)

                st.markdown(f"""
                <div style="background-color: #064e3b; border: 2px solid #10b981; border-radius: 12px; padding: 22px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="color: #ecfdf5; margin: 0;">✅ BET ON: <span style="color: #6ee7b7;">{bet_team.upper()} TO WIN</span></h2>
                        <span style="background-color: #047857; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px;">+{opp['edge']*100:.1f}% VALUE EDGE</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #047857; margin: 15px 0;">
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;">
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <div style="font-size: 13px; color: #a7f3d0;">RECOMMENDED STAKE</div>
                            <div style="font-size: 26px; font-weight: bold; color: #ffffff;">{config.DEFAULT_CURRENCY}{stake_amt:,.2f}</div>
                            <div style="font-size: 12px; color: #6ee7b7;">({sizing['stake_pct']:.1f}% of ₱{bankroll_val:,.0f})</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <div style="font-size: 13px; color: #a7f3d0;">BOOKMAKER ODDS</div>
                            <div style="font-size: 26px; font-weight: bold; color: #ffffff;">{opp['decimal_odds']:.2f}</div>
                            <div style="font-size: 12px; color: #6ee7b7;">Fair: {1.0/opp['model_prob']:.2f}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <div style="font-size: 13px; color: #a7f3d0;">ESTIMATED WIN PROB</div>
                            <div style="font-size: 26px; font-weight: bold; color: #ffffff;">{opp['model_prob']*100:.1f}%</div>
                            <div style="font-size: 12px; color: #6ee7b7;">Market Implied: {opp['fair_implied_prob']*100:.1f}%</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <div style="font-size: 13px; color: #a7f3d0;">POTENTIAL PROFIT</div>
                            <div style="font-size: 26px; font-weight: bold; color: #34d399;">+{config.DEFAULT_CURRENCY}{profit:,.2f}</div>
                            <div style="font-size: 12px; color: #a7f3d0;">Total Return: {config.DEFAULT_CURRENCY}{payout:,.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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

@st.cache_data(show_spinner=False)
def run_cached_backtest(target_sport, bankroll, kelly, min_edge, compound, markets_tuple, season_val):
    bt = HistoricalBacktester(
        sport=target_sport,
        starting_bankroll=bankroll,
        kelly_fraction=kelly,
        min_edge=min_edge,
        compound_bankroll=compound,
        markets=list(markets_tuple)
    )
    return bt.run_backtest(season_filter=season_val)

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
        bt_markets = st.multiselect("Active Markets", options=market_options, default=["moneyline"], key=f"bt_markets_{sport}")
    with b_col4:
        run_btn = st.button("Run Simulation", type="primary", key=f"bt_run_btn_{sport}")

    season_arg = None if bt_season == "All Seasons" else bt_season
    res = run_cached_backtest(sport, bankroll_val, kelly_mult, min_edge_pct, bt_compound, tuple(bt_markets), season_arg)

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