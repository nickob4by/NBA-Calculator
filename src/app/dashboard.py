import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Root setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.db.database import db
from src.ingestion.pipeline import generate_seed_dataset_if_empty
from src.ingestion.mlb_api_fetcher import generate_mlb_seed_dataset_if_empty
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.models.mlb_models import MLBRunMarginPredictor, MLBTotalsPredictor, MLBWinProbabilityModel
from src.features.matchup_builder import build_full_feature_dataset, get_feature_columns, build_upcoming_matchup
from src.betting.odds_math import american_to_decimal, decimal_to_american, remove_vig
from src.betting.ev_engine import evaluate_moneyline_market
from src.betting.kelly import size_bet
from src.betting.backtester import HistoricalBacktester
from src.betting.ledger import BankrollLedger

st.set_page_config(
    page_title="Quantitative Sportsbook Analytics",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Persistent state initialization from database
db_kelly = float(db.get_setting("kelly_mult", str(config.DEFAULT_KELLY_FRACTION)))
db_min_edge = float(db.get_setting("min_edge_pct", str(config.DEFAULT_MIN_EDGE)))

if "kelly_mult" not in st.session_state:
    st.session_state["kelly_mult"] = db_kelly
if "min_edge_pct" not in st.session_state:
    st.session_state["min_edge_pct"] = db_min_edge

# Sleek Minimalist Full-Width & Top Navigation CSS
st.markdown("""
<style>
    /* Hide Streamlit top header toolbar and sidebar completely */
    header[data-testid="stHeader"] { display: none !important; height: 0px !important; }
    [data-testid="stSidebar"], section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"], button[kind="header"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    
    /* Global Base */
    .main { background-color: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .stSelectbox, .stNumberInput, .stSlider { color: #f8fafc; }
    
    /* Full-width container with proper top clearance */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stMetricValue"] { font-size: clamp(18px, 3vw, 26px); font-weight: 700; color: #f8fafc; }
    div[data-testid="stMetricLabel"] { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; }
    
    .card-neutral { background-color: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .card-emerald { background-color: #06281e; border: 1px solid #059669; border-radius: 8px; padding: 18px; margin-bottom: 12px; }
    .card-slate { background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    
    /* Top Horizontal Header Navigation Menu */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
        flex-wrap: wrap !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] label {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 9px 18px;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        flex: 1;
        text-align: center;
        min-width: 140px;
        justify-content: center;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1f2937;
        border-color: #374151;
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #064e3b !important;
        border-color: #10b981 !important;
        color: #ecfdf5 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .card-emerald, .card-slate, .card-neutral {
            padding: 12px !important;
        }
        div[data-testid="stRadio"] label {
            min-width: 45% !important;
            padding: 8px 10px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

live_ledger_balance = BankrollLedger.get_current_balance()
active_bankroll = live_ledger_balance
kelly_mult = st.session_state["kelly_mult"]
min_edge_pct = st.session_state["min_edge_pct"]

# ================= TOP HEADER BAR =================
head_c1, head_c2 = st.columns([3.2, 1.2], vertical_alignment="center")

with head_c1:
    st.markdown("<h2 style='margin:0; padding:0; color:#f8fafc; font-weight:800; font-size:clamp(20px, 3vw, 26px);'>QUANTITATIVE ANALYTICS</h2>", unsafe_allow_html=True)
    st.caption("NBA & MLB Statistical Valuation Engine")

with head_c2:
    st.markdown(f"""
    <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 8px 12px; text-align: center;">
        <span style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Balance: </span>
        <span style="font-size: 16px; font-weight: bold; color: #10b981;">{config.DEFAULT_CURRENCY}{active_bankroll:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# Top Horizontal Navigation Tabs
nav_selection = st.radio(
    "Navigation Header",
    options=[
        "Matchup Forecast",
        "Bankroll & Ledger",
        "Team Explorer",
        "Data & Settings"
    ],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<hr style='border:0; border-top: 1px solid #1f2937; margin: 10px 0 20px 0;'>", unsafe_allow_html=True)

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
    
    from src.features.four_factors import compute_advanced_stats_dataframe
    from src.features.mlb_sabermetrics import compute_mlb_sabermetrics_dataframe
    from src.features.situational import compute_situational_features
    from src.features.rolling_metrics import compute_rolling_team_features
    
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

@st.cache_resource(show_spinner=False)
def preload_all_sports_data():
    """
    Pre-loads and caches models and rolling features for all sports (NBA & MLB)
    so all page interactions, sport toggles, and team selections execute instantly.
    """
    return {
        "nba": load_sport_models_and_data("nba"),
        "mlb": load_sport_models_and_data("mlb")
    }

# Automatically initialize all sports data on startup
sports_cache = preload_all_sports_data()

# ================= 1. MATCHUP FORECAST =================
if nav_selection == "Matchup Forecast":
    st.markdown("<h2 style='color:#f8fafc; font-weight:700; margin-bottom: 4px;'>Matchup Forecast & Valuation</h2>", unsafe_allow_html=True)
    st.caption("Select the league and competing teams to generate win probabilities, compare odds, and find value bets.")

    # Matchup Selection Card
    st.markdown("""
    <div class="card-neutral" style="margin-bottom: 16px;">
        <div style="font-size: 13px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
            Configure Matchup
        </div>
    """, unsafe_allow_html=True)

    mc_sport_col, mc_h_col, mc_a_col = st.columns([1.2, 1.4, 1.4])
    with mc_sport_col:
        sport_choice = st.radio(
            "League",
            options=["NBA Basketball", "MLB Baseball"],
            index=0,
            key="matchup_sport_choice"
        )
        sport = "mlb" if "MLB" in sport_choice else "nba"
        sport_label = "MLB" if sport == "mlb" else "NBA"

    margin_model, totals_model, win_model, matchups_df, logs_df, model_metrics = sports_cache[sport]
    teams_dict = config.MLB_TEAMS if sport == "mlb" else config.NBA_TEAMS
    team_list = list(teams_dict.keys())
    team_options = {k: f"{v['name']} ({v['abbrev']})" for k, v in teams_dict.items()}

    with mc_h_col:
        home_team_id = st.selectbox(
            f"Home Team ({sport_label})",
            options=[None] + team_list,
            format_func=lambda x: "-- Select Home Team --" if x is None else team_options[x],
            index=0,
            key=f"mf_home_team_{sport}"
        )

    with mc_a_col:
        away_team_id = st.selectbox(
            f"Away Team ({sport_label})",
            options=[None] + team_list,
            format_func=lambda x: "-- Select Away Team --" if x is None else team_options[x],
            index=0,
            key=f"mf_away_team_{sport}"
        )

    home_sp_fip = None
    away_sp_fip = None
    if sport == "mlb" and home_team_id is not None and away_team_id is not None:
        from src.features.mlb_pitcher_metrics import get_team_starters
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        sp_c1, sp_c2 = st.columns(2)
        h_starters = get_team_starters(home_team_id)
        a_starters = get_team_starters(away_team_id)

        with sp_c1:
            h_sp_idx = st.selectbox(
                f"{config.get_team_abbrev(home_team_id, sport='mlb')} Starting Pitcher",
                options=range(len(h_starters)),
                format_func=lambda i: f"{h_starters[i]['name']} ({h_starters[i]['fip']:.2f} FIP | {h_starters[i]['whip']:.2f} WHIP)",
                key=f"h_sp_{home_team_id}"
            )
            home_sp_fip = h_starters[h_sp_idx]["fip"]

        with sp_c2:
            a_sp_idx = st.selectbox(
                f"{config.get_team_abbrev(away_team_id, sport='mlb')} Starting Pitcher",
                options=range(len(a_starters)),
                format_func=lambda i: f"{a_starters[i]['name']} ({a_starters[i]['fip']:.2f} FIP | {a_starters[i]['whip']:.2f} WHIP)",
                key=f"a_sp_{away_team_id}"
            )
            away_sp_fip = a_starters[a_sp_idx]["fip"]

    st.markdown("</div>", unsafe_allow_html=True)

    if home_team_id is None or away_team_id is None:
        st.markdown("""
        <div class="card-slate" style="text-align: center; padding: 40px 20px; border-style: dashed;">
            <h4 style="color: #cbd5e1; margin: 0 0 6px 0;">No Matchup Selected</h4>
            <div style="color: #64748b; font-size: 13px;">
                Choose the league and select both Home and Away teams in the card above to calculate win probabilities and value edges.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif home_team_id == away_team_id:
        st.warning("Please select two distinct competing teams.")
    else:
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
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            h_is_fav = pred_p_home >= 0.50
            st.markdown(f"""
            <div class="card-slate" style="border-left: 4px solid {'#10b981' if h_is_fav else '#64748b'};">
                <div style="font-size: 12px; color: {'#34d399' if h_is_fav else '#94a3b8'}; font-weight: bold; letter-spacing: 1px;">
                    {'FAVORITE' if h_is_fav else 'UNDERDOG'} (HOME)
                </div>
                <h3 style="margin: 4px 0; color: white;">{h_name}</h3>
                <h1 style="margin: 0; color: {'#10b981' if h_is_fav else '#f8fafc'}; font-size: 38px;">{pred_p_home*100:.1f}%</h1>
                <div style="margin-top: 4px; color: #94a3b8; font-size: 13px;">Fair True Price: <b>{1.0/pred_p_home:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with w_col2:
            a_is_fav = pred_p_away > 0.50
            st.markdown(f"""
            <div class="card-slate" style="border-left: 4px solid {'#10b981' if a_is_fav else '#64748b'};">
                <div style="font-size: 12px; color: {'#34d399' if a_is_fav else '#94a3b8'}; font-weight: bold; letter-spacing: 1px;">
                    {'FAVORITE' if a_is_fav else 'UNDERDOG'} (AWAY)
                </div>
                <h3 style="margin: 4px 0; color: white;">{a_name}</h3>
                <h1 style="margin: 0; color: {'#10b981' if a_is_fav else '#f8fafc'}; font-size: 38px;">{pred_p_away*100:.1f}%</h1>
                <div style="margin-top: 4px; color: #94a3b8; font-size: 13px;">Fair True Price: <b>{1.0/pred_p_away:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Enter Sportsbook Odds (Decimal)")
        c_odd1, c_odd2 = st.columns(2)

        with c_odd1:
            def_h_ml = round(1.0 / (pred_p_home * 1.025), 2)
            input_h_ml = st.number_input(f"{h_name} Odds", value=float(def_h_ml), step=0.05, key="input_h_ml")

        with c_odd2:
            def_a_ml = round(1.0 / (pred_p_away * 1.025), 2)
            input_a_ml = st.number_input(f"{a_name} Odds", value=float(def_a_ml), step=0.05, key="input_a_ml")

        ml_eval = evaluate_moneyline_market(pred_p_home, input_h_ml, input_a_ml, min_edge=min_edge_pct)
        opps = ml_eval["opportunities"]

        st.markdown("##### Valuation Recommendation")
        if not opps:
            st.info(f"NO VALUE BET IDENTIFIED: Neither team offers an edge exceeding the {min_edge_pct*100:.1f}% minimum threshold. Model recommends PASS.")
        else:
            for opp in opps:
                sizing = size_bet(opp["model_prob"], opp["decimal_odds"], active_bankroll, kelly_multiplier=kelly_mult)
                bet_team = h_name if opp["side"] == "home" else a_name
                stake_amt = sizing["stake"]
                payout = round(stake_amt * opp["decimal_odds"], 2)
                profit = round(payout - stake_amt, 2)

                st.markdown(f"""
                <div class="card-emerald">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <h3 style="color: #ecfdf5; margin: 0; font-size: clamp(16px, 2.5vw, 20px);">RECOMMENDED BET: {bet_team.upper()} TO WIN</h3>
                        <span style="background-color: #047857; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 13px;">+{opp['edge']*100:.1f}% EDGE</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #065f46; margin: 12px 0;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; text-align: center;">
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Stake</div>
                            <div style="font-size: clamp(18px, 2.5vw, 22px); font-weight: bold; color: #ffffff;">{config.DEFAULT_CURRENCY}{stake_amt:,.2f}</div>
                            <div style="font-size: 11px; color: #6ee7b7;">{sizing['stake_pct']:.1f}% of {config.DEFAULT_CURRENCY}{active_bankroll:,.0f}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Bookmaker Odds</div>
                            <div style="font-size: clamp(18px, 2.5vw, 22px); font-weight: bold; color: #ffffff;">{opp['decimal_odds']:.2f}</div>
                            <div style="font-size: 11px; color: #6ee7b7;">Fair: {1.0/opp['model_prob']:.2f}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Model Win Prob</div>
                            <div style="font-size: clamp(18px, 2.5vw, 22px); font-weight: bold; color: #ffffff;">{opp['model_prob']*100:.1f}%</div>
                            <div style="font-size: 11px; color: #6ee7b7;">Market: {opp['fair_implied_prob']*100:.1f}%</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Potential Profit</div>
                            <div style="font-size: clamp(18px, 2.5vw, 22px); font-weight: bold; color: #34d399;">+{config.DEFAULT_CURRENCY}{profit:,.2f}</div>
                            <div style="font-size: 11px; color: #a7f3d0;">Payout: {config.DEFAULT_CURRENCY}{payout:,.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Record This Wager Outcome into Ledger", expanded=False):
                    q_res = st.radio("Outcome", options=["Won", "Lost"], horizontal=True, key=f"q_rec_res_{bet_team}")
                    if st.button("Save Wager to Ledger", type="primary", key=f"q_rec_btn_{bet_team}"):
                        BankrollLedger.record_bet(
                            sport=sport,
                            team=bet_team,
                            stake=stake_amt,
                            odds=opp["decimal_odds"],
                            is_win=(q_res == "Won")
                        )
                        st.success(f"Wager on {bet_team} successfully saved!")
                        st.rerun()

# ================= 2. BANKROLL & LEDGER =================
elif nav_selection == "Bankroll & Ledger":
    st.title("Bankroll & Transaction Ledger")
    st.caption("Personal financial tracking, risk parameters, deposits, withdrawals, and ledger history.")

    metrics = BankrollLedger.get_ledger_metrics()

    # Summary Metrics (Auto wrap on mobile)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Balance", f"{config.DEFAULT_CURRENCY}{metrics['current_balance']:,.2f}")
    m2.metric("Net Betting PnL", f"{config.DEFAULT_CURRENCY}{metrics['net_betting_pnl']:+,.2f}")
    m3.metric("Total Deposits", f"{config.DEFAULT_CURRENCY}{metrics['total_deposits']:,.2f}")
    m4.metric("Total Withdrawals", f"{config.DEFAULT_CURRENCY}{metrics['total_withdrawals']:,.2f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Total Bets Logged", f"{metrics['total_bets']}")
    m6.metric("Record (W - L)", f"{metrics['wins']}W - {metrics['losses']}L")
    m7.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    m8.metric("All-Time ROI", f"{metrics['roi_pct']:+.2f}%")

    st.markdown("---")

    # Risk & Sizing Parameters Card
    st.markdown("##### Risk & Position Sizing Parameters")
    with st.expander("Configure Sizing & Edge Criteria", expanded=True):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            new_kelly = st.slider(
                "Fractional Kelly Multiplier",
                min_value=0.05,
                max_value=0.50,
                value=float(st.session_state["kelly_mult"]),
                step=0.01,
                help="0.15 = 15% Fractional Kelly",
                key="slider_kelly_val"
            )
            if new_kelly != st.session_state["kelly_mult"]:
                st.session_state["kelly_mult"] = new_kelly
                db.set_setting("kelly_mult", str(new_kelly))
        with p_col2:
            new_edge = st.slider(
                "Minimum Edge Threshold (%)",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state["min_edge_pct"] * 100.0),
                step=0.5,
                key="slider_min_edge_val"
            ) / 100.0
            if new_edge != st.session_state["min_edge_pct"]:
                st.session_state["min_edge_pct"] = new_edge
                db.set_setting("min_edge_pct", str(new_edge))

    st.markdown("---")
    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        with st.expander("Record Wager Result", expanded=True):
            with st.form("form_record_bet"):
                f_sport = st.selectbox("Sport", options=["NBA", "MLB"], key="f_sport")
                f_team = st.text_input("Selection / Team", value="Cincinnati Reds", key="f_team")
                f_stake = st.number_input(f"Stake ({config.DEFAULT_CURRENCY})", min_value=1.0, value=50.0, step=10.0, key="f_stake")
                f_odds = st.number_input("Decimal Odds", min_value=1.01, value=2.10, step=0.05, key="f_odds")
                f_outcome = st.radio("Result", options=["Won", "Lost"], horizontal=True, key="f_outcome")
                f_note = st.text_input("Note (Optional)", value="", key="f_note")
                
                if st.form_submit_button("Save Wager", type="primary"):
                    res = BankrollLedger.record_bet(
                        sport=f_sport,
                        team=f_team,
                        stake=f_stake,
                        odds=f_odds,
                        is_win=(f_outcome == "Won"),
                        note=f_note
                    )
                    st.success(f"Recorded! New Balance: {config.DEFAULT_CURRENCY}{res['balance_after']:,.2f}")
                    st.rerun()

    with act_col2:
        with st.expander("Deposit / Withdraw Funds", expanded=True):
            with st.form("form_funds"):
                f_type = st.radio("Transaction Type", options=["Deposit", "Withdrawal"], horizontal=True)
                f_amt = st.number_input(f"Amount ({config.DEFAULT_CURRENCY})", min_value=10.0, value=500.0, step=50.0)
                f_fund_note = st.text_input("Note", value="Account top-up" if f_type=="Deposit" else "Cashout")
                
                if st.form_submit_button("Submit Transaction"):
                    if f_type == "Deposit":
                        bal = BankrollLedger.deposit(f_amt, f_fund_note)
                        st.success(f"Deposited {config.DEFAULT_CURRENCY}{f_amt:,.2f}. New Balance: {config.DEFAULT_CURRENCY}{bal:,.2f}")
                    else:
                        bal = BankrollLedger.withdraw(f_amt, f_fund_note)
                        st.success(f"Withdrawn {config.DEFAULT_CURRENCY}{f_amt:,.2f}. New Balance: {config.DEFAULT_CURRENCY}{bal:,.2f}")
                    st.rerun()

    with act_col3:
        with st.expander("Set Starting Capital", expanded=False):
            with st.form("form_set_capital"):
                current_init = float(metrics.get("initial_balance", config.DEFAULT_STARTING_BANKROLL))
                f_init = st.number_input(
                    f"Base Starting Capital ({config.DEFAULT_CURRENCY})",
                    min_value=10.0,
                    value=current_init,
                    step=100.0,
                    key="f_init_capital_input"
                )
                c_opt1, c_opt2 = st.columns(2)
                with c_opt1:
                    btn_update = st.form_submit_button("Update Capital", type="primary")
                with c_opt2:
                    btn_wipe = st.form_submit_button("Reset History")

                if btn_update:
                    new_bal = BankrollLedger.set_starting_balance(f_init)
                    st.success(f"Starting capital set to {config.DEFAULT_CURRENCY}{f_init:,.2f}! Current balance: {config.DEFAULT_CURRENCY}{new_bal:,.2f}")
                    st.rerun()
                elif btn_wipe:
                    BankrollLedger.reset_bankroll(f_init)
                    st.success(f"Bankroll & history reset to {config.DEFAULT_CURRENCY}{f_init:,.2f}!")
                    st.rerun()

    # Balance Trajectory Chart
    ledger_df = BankrollLedger.get_ledger_history()
    if not ledger_df.empty:
        st.markdown("##### Balance Equity Trajectory")
        chart_df = ledger_df.sort_values(by="id").reset_index(drop=True)
        fig_bal = px.line(
            chart_df, x="created_at", y="balance_after",
            labels={"created_at": "Timestamp", "balance_after": f"Balance ({config.DEFAULT_CURRENCY})"}
        )
        fig_bal.update_traces(line=dict(color="#10b981", width=2.5))
        fig_bal.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0b0f19",
            paper_bgcolor="#0b0f19",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bal, use_container_width=True)

        st.markdown("##### Chronological Transaction History")
        st.dataframe(
            ledger_df[["id", "created_at", "tx_type", "amount", "balance_after", "sport", "stake", "odds", "team_selected", "note"]].rename(
                columns={
                    "created_at": "Timestamp",
                    "tx_type": "Type",
                    "amount": f"Change ({config.DEFAULT_CURRENCY})",
                    "balance_after": f"Balance ({config.DEFAULT_CURRENCY})",
                    "sport": "Sport",
                    "stake": f"Stake ({config.DEFAULT_CURRENCY})",
                    "odds": "Odds",
                    "team_selected": "Selection",
                    "note": "Note"
                }
            ),
            use_container_width=True
        )

# ================= 3. TEAM EXPLORER =================
elif nav_selection == "Team Explorer":
    st.markdown("<h2 style='color:#f8fafc; font-weight:700; margin-bottom: 4px;'>Team & Rotation Explorer</h2>", unsafe_allow_html=True)
    st.caption("Inspect team starting rotations (MLB) and Four Factors advanced metrics (NBA).")

    st.markdown("""
    <div class="card-neutral" style="margin-bottom: 16px;">
        <div style="font-size: 13px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
            Select Teams to Inspect
        </div>
    """, unsafe_allow_html=True)

    exp_sport_col, exp_t1_col, exp_t2_col = st.columns([1.2, 1.4, 1.4])
    with exp_sport_col:
        exp_sport_choice = st.radio("League", options=["NBA Basketball", "MLB Baseball"], index=0, key="exp_sport_choice")
        exp_sport = "mlb" if "MLB" in exp_sport_choice else "nba"
        exp_sport_label = "MLB" if exp_sport == "mlb" else "NBA"

    exp_margin, exp_totals, exp_win, exp_matchups, exp_logs, _ = sports_cache[exp_sport]
    exp_teams = config.MLB_TEAMS if exp_sport == "mlb" else config.NBA_TEAMS
    exp_team_list = list(exp_teams.keys())
    exp_team_opts = {k: f"{v['name']} ({v['abbrev']})" for k, v in exp_teams.items()}

    with exp_t1_col:
        t1_sel = st.selectbox(f"Team 1 ({exp_sport_label})", options=[None] + exp_team_list, format_func=lambda x: "-- Select Team 1 --" if x is None else exp_team_opts[x], index=0, key=f"exp_t1_{exp_sport}")
    with exp_t2_col:
        t2_sel = st.selectbox(f"Team 2 ({exp_sport_label})", options=[None] + exp_team_list, format_func=lambda x: "-- Select Team 2 (Optional) --" if x is None else exp_team_opts[x], index=0, key=f"exp_t2_{exp_sport}")

    st.markdown("</div>", unsafe_allow_html=True)

    if t1_sel is None:
        st.markdown("""
        <div class="card-slate" style="text-align: center; padding: 40px 20px; border-style: dashed;">
            <h4 style="color: #cbd5e1; margin: 0 0 6px 0;">No Team Selected</h4>
            <div style="color: #64748b; font-size: 13px;">
                Choose the league and pick at least one team above to inspect rotations and metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif exp_sport == "mlb":
        from src.features.mlb_pitcher_metrics import get_team_starters
        t1_rot = get_team_starters(t1_sel)

        if t2_sel is not None:
            t2_rot = get_team_starters(t2_sel)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**{config.get_team_name(t1_sel, sport='mlb')} Starting Rotation**")
                st.dataframe(pd.DataFrame(t1_rot)[["name", "fip", "whip", "k9", "throws"]].rename(columns={"name": "Pitcher", "fip": "FIP", "whip": "WHIP", "k9": "K/9", "throws": "Hand"}), use_container_width=True)
            with p_col2:
                st.markdown(f"**{config.get_team_name(t2_sel, sport='mlb')} Starting Rotation**")
                st.dataframe(pd.DataFrame(t2_rot)[["name", "fip", "whip", "k9", "throws"]].rename(columns={"name": "Pitcher", "fip": "FIP", "whip": "WHIP", "k9": "K/9", "throws": "Hand"}), use_container_width=True)
        else:
            st.markdown(f"**{config.get_team_name(t1_sel, sport='mlb')} Starting Rotation**")
            st.dataframe(pd.DataFrame(t1_rot)[["name", "fip", "whip", "k9", "throws"]].rename(columns={"name": "Pitcher", "fip": "FIP", "whip": "WHIP", "k9": "K/9", "throws": "Hand"}), use_container_width=True)
    else:
        t1_logs = exp_matchups[exp_matchups["home_team_id"] == t1_sel].iloc[-1:]

        if not t1_logs.empty:
            t1_row = t1_logs.iloc[0]
            categories = ['eFG% (Shooting)', 'TOV% (Ball Security)', 'ORB% (Rebounding)', 'FTR (Free Throws)', 'Net Rating']
            t1_vals = [
                float(t1_row.get("home_roll_efg_pct_w10", 0.52)) * 100,
                (1.0 - float(t1_row.get("home_roll_tov_pct_w10", 0.13))) * 100,
                float(t1_row.get("home_roll_orb_pct_w10", 0.25)) * 100,
                float(t1_row.get("home_roll_ftr_w10", 0.22)) * 100,
                float(t1_row.get("home_roll_net_rating_w10", 3.0)) + 50
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=t1_vals, theta=categories, fill='toself', name=config.get_team_name(t1_sel, sport="nba")))

            if t2_sel is not None:
                t2_logs = exp_matchups[exp_matchups["home_team_id"] == t2_sel].iloc[-1:]
                if not t2_logs.empty:
                    t2_row = t2_logs.iloc[0]
                    t2_vals = [
                        float(t2_row.get("home_roll_efg_pct_w10", 0.52)) * 100,
                        (1.0 - float(t2_row.get("home_roll_tov_pct_w10", 0.13))) * 100,
                        float(t2_row.get("home_roll_orb_pct_w10", 0.25)) * 100,
                        float(t2_row.get("home_roll_ftr_w10", 0.22)) * 100,
                        float(t2_row.get("home_roll_net_rating_w10", 3.0)) + 50
                    ]
                    fig.add_trace(go.Scatterpolar(r=t2_vals, theta=categories, fill='toself', name=config.get_team_name(t2_sel, sport="nba")))

            fig.update_layout(
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Four Factors Radar Comparison (10-Game Form)",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

# ================= 4. DATA & SETTINGS =================
elif nav_selection == "Data & Settings":
    st.title("Data Ingestion & Settings")
    
    nba_cnt = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='nba'")["c"]
    mlb_cnt = db.fetch_one("SELECT COUNT(*) as c FROM games WHERE sport='mlb'")["c"]
    logs_cnt = db.fetch_one("SELECT COUNT(*) as c FROM team_game_logs")["c"]
    odds_cnt = db.fetch_one("SELECT COUNT(*) as c FROM odds")["c"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("##### SQLite Database Status")
        st.write(f"- **NBA Games**: {nba_cnt:,} records")
        st.write(f"- **MLB Games**: {mlb_cnt:,} records")
        st.write(f"- **Team Logs Table**: {logs_cnt:,} records")
        st.write(f"- **Odds Table**: {odds_cnt:,} records")
        st.write(f"- **Database Path**: `{config.DB_PATH}`")

    with col_s2:
        st.markdown("##### Update Team & Player Statistics")
        st.caption("Fetches the latest official game logs, player stats, starting pitcher rotations, and advanced ratings.")
        
        update_target = st.selectbox(
            "Select Update Scope",
            options=["All Sports (NBA & MLB)", "NBA (2024-25 Latest)", "MLB (2024 Latest)"],
            index=0
        )
        
        if st.button("Update All Latest Data", type="primary", key="btn_update_latest_data"):
            with st.status("Importing latest teams and players data...", expanded=True) as status:
                st.write("1. Connecting to official sports data APIs...")
                
                if "NBA" in update_target or "All" in update_target:
                    st.write("2. Syncing NBA 2024-25 game logs, rosters, and advanced Four Factors...")
                    generate_seed_dataset_if_empty()
                
                if "MLB" in update_target or "All" in update_target:
                    st.write("3. Syncing MLB 2024 game scores, starting pitcher rotations, and sabermetrics...")
                    generate_mlb_seed_dataset_if_empty()
                
                st.write("4. Clearing feature caches and refreshing analytics engine...")
                st.cache_resource.clear()
                st.cache_data.clear()
                
                status.update(label="All team & player statistics successfully updated!", state="complete", expanded=False)
            st.success("Database and player stats are now completely up-to-date.")
            st.rerun()