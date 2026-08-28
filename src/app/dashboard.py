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
    initial_sidebar_state="expanded"
)

# Minimalist Quantitative Dark CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stSelectbox, .stNumberInput, .stSlider { color: #f8fafc; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; color: #f8fafc; }
    div[data-testid="stMetricLabel"] { font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; }
    .card-neutral { background-color: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 18px; }
    .card-emerald { background-color: #06281e; border: 1px solid #059669; border-radius: 8px; padding: 20px; }
    .card-slate { background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 18px; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR NAVIGATION =================
st.sidebar.markdown("### QUANTITATIVE ANALYTICS")
st.sidebar.caption("NBA & MLB Statistical Valuation Engine")

sport_choice = st.sidebar.selectbox(
    "Sport",
    options=["NBA (Basketball)", "MLB (Baseball)"],
    index=0
)
sport = "mlb" if "MLB" in sport_choice else "nba"
sport_label = "MLB" if sport == "mlb" else "NBA"

nav_selection = st.sidebar.radio(
    "Navigation",
    options=[
        "Matchup Forecast",
        "Bankroll & Ledger",
        "Team Explorer",
        "Backtest Simulation",
        "Model Diagnostics",
        "Data & Settings"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk & Sizing Parameters")

# Pull live balance from ledger
current_ledger_balance = BankrollLedger.get_current_balance()
bankroll_val = st.sidebar.number_input(
    f"Active Bankroll ({config.DEFAULT_CURRENCY})",
    min_value=10.0,
    max_value=10000000.0,
    value=float(current_ledger_balance),
    step=50.0,
    help="Live balance sourced directly from your ledger."
)
kelly_mult = st.sidebar.slider("Fractional Kelly", min_value=0.05, max_value=0.50, value=0.15, step=0.01, help="0.15 = 15% Kelly")
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
teams_dict = config.MLB_TEAMS if sport == "mlb" else config.NBA_TEAMS
team_list = list(teams_dict.keys())
team_options = {k: f"{v['name']} ({v['abbrev']})" for k, v in teams_dict.items()}

default_home_idx = 28 if sport == "mlb" else 1
default_away_idx = 3 if sport == "mlb" else 13

# ================= 1. MATCHUP FORECAST =================
if nav_selection == "Matchup Forecast":
    st.title(f"{sport_label} Matchup Forecast & Valuation")
    
    col_h, col_a = st.columns(2)
    with col_h:
        home_team_id = st.selectbox(f"Home Team ({sport_label})", options=team_list, format_func=lambda x: team_options[x], index=min(default_home_idx, len(team_list)-1))
    with col_a:
        away_team_id = st.selectbox(f"Away Team ({sport_label})", options=team_list, format_func=lambda x: team_options[x], index=min(default_away_idx, len(team_list)-1))

    if home_team_id == away_team_id:
        st.warning("Please select two distinct teams for the matchup.")
    else:
        home_sp_fip = None
        away_sp_fip = None
        if sport == "mlb":
            from src.features.mlb_pitcher_metrics import get_team_starters
            st.markdown("##### Starting Pitchers")
            sp_col1, sp_col2 = st.columns(2)
            
            h_starters = get_team_starters(home_team_id)
            a_starters = get_team_starters(away_team_id)
            
            with sp_col1:
                h_sp_idx = st.selectbox(
                    f"{config.get_team_abbrev(home_team_id, sport='mlb')} Starter",
                    options=range(len(h_starters)),
                    format_func=lambda i: f"{h_starters[i]['name']} ({h_starters[i]['fip']:.2f} FIP | {h_starters[i]['whip']:.2f} WHIP)",
                    key="home_sp_select"
                )
                h_selected_sp = h_starters[h_sp_idx]
                home_sp_fip = h_selected_sp["fip"]

            with sp_col2:
                a_sp_idx = st.selectbox(
                    f"{config.get_team_abbrev(away_team_id, sport='mlb')} Starter",
                    options=range(len(a_starters)),
                    format_func=lambda i: f"{a_starters[i]['name']} ({a_starters[i]['fip']:.2f} FIP | {a_starters[i]['whip']:.2f} WHIP)",
                    key="away_sp_select"
                )
                a_selected_sp = a_starters[a_sp_idx]
                away_sp_fip = a_selected_sp["fip"]

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
                sizing = size_bet(opp["model_prob"], opp["decimal_odds"], bankroll_val, kelly_multiplier=kelly_mult)
                bet_team = h_name if opp["side"] == "home" else a_name
                stake_amt = sizing["stake"]
                payout = round(stake_amt * opp["decimal_odds"], 2)
                profit = round(payout - stake_amt, 2)

                st.markdown(f"""
                <div class="card-emerald">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #ecfdf5; margin: 0;">RECOMMENDED BET: {bet_team.upper()} TO WIN</h3>
                        <span style="background-color: #047857; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 13px;">+{opp['edge']*100:.1f}% EDGE</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #065f46; margin: 12px 0;">
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; text-align: center;">
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Stake</div>
                            <div style="font-size: 22px; font-weight: bold; color: #ffffff;">{config.DEFAULT_CURRENCY}{stake_amt:,.2f}</div>
                            <div style="font-size: 11px; color: #6ee7b7;">{sizing['stake_pct']:.1f}% of bankroll</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Bookmaker Odds</div>
                            <div style="font-size: 22px; font-weight: bold; color: #ffffff;">{opp['decimal_odds']:.2f}</div>
                            <div style="font-size: 11px; color: #6ee7b7;">Fair: {1.0/opp['model_prob']:.2f}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Model Win Prob</div>
                            <div style="font-size: 22px; font-weight: bold; color: #ffffff;">{opp['model_prob']*100:.1f}%</div>
                            <div style="font-size: 11px; color: #6ee7b7;">Market: {opp['fair_implied_prob']*100:.1f}%</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 11px; color: #a7f3d0; text-transform: uppercase;">Potential Profit</div>
                            <div style="font-size: 22px; font-weight: bold; color: #34d399;">+{config.DEFAULT_CURRENCY}{profit:,.2f}</div>
                            <div style="font-size: 11px; color: #a7f3d0;">Payout: {config.DEFAULT_CURRENCY}{payout:,.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Quick Log Button
                st.write("")
                btn_c1, btn_c2 = st.columns([1, 4])
                with btn_c1:
                    if st.button("Log Wager Result", key="log_wager_modal_btn"):
                        st.session_state["pending_log_team"] = bet_team
                        st.session_state["pending_log_stake"] = stake_amt
                        st.session_state["pending_log_odds"] = opp["decimal_odds"]
                        st.session_state["pending_log_sport"] = sport
                        st.info("Fill result in the Bankroll & Ledger tab or quick record below.")

                if st.session_state.get("pending_log_team") == bet_team:
                    with st.expander("Record This Wager Outcome", expanded=True):
                        rec_res = st.radio("Outcome", options=["Won", "Lost"], horizontal=True, key="quick_rec_radio")
                        if st.button("Confirm & Save to Ledger", type="primary", key="quick_rec_confirm"):
                            BankrollLedger.record_bet(
                                sport=sport,
                                team=bet_team,
                                stake=stake_amt,
                                odds=opp["decimal_odds"],
                                is_win=(rec_res == "Won")
                            )
                            st.success(f"Wager on {bet_team} successfully recorded into ledger!")
                            st.session_state.pop("pending_log_team", None)
                            st.rerun()

# ================= 2. BANKROLL & LEDGER =================
elif nav_selection == "Bankroll & Ledger":
    st.title("Bankroll & Transaction Ledger")
    st.caption("Personal balance tracking, deposits, withdrawals, and wager history in Philippine Pesos (₱).")

    metrics = BankrollLedger.get_ledger_metrics()

    # Summary Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Current Balance", f"{config.DEFAULT_CURRENCY}{metrics['current_balance']:,.2f}")
    m2.metric("Net Betting PnL", f"{config.DEFAULT_CURRENCY}{metrics['net_betting_pnl']:+,.2f}")
    m3.metric("Total Deposits", f"{config.DEFAULT_CURRENCY}{metrics['total_deposits']:,.2f}")
    m4.metric("Total Withdrawals", f"{config.DEFAULT_CURRENCY}{metrics['total_withdrawals']:,.2f}")
    m5.metric("Win Rate", f"{metrics['win_rate']:.1f}%", f"{metrics['wins']}W - {metrics['losses']}L")
    m6.metric("Betting ROI", f"{metrics['roi_pct']:+.2f}%")

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
                
                if st.form_submit_button("Record Bet", type="primary"):
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
        with st.expander("Reset / Adjust Capital", expanded=False):
            with st.form("form_reset"):
                f_init = st.number_input(f"New Starting Capital ({config.DEFAULT_CURRENCY})", min_value=50.0, value=1200.0, step=100.0)
                if st.form_submit_button("Reset Starting Capital"):
                    BankrollLedger.reset_bankroll(f_init)
                    st.success(f"Bankroll reset to {config.DEFAULT_CURRENCY}{f_init:,.2f}!")
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
        fig_bal.update_layout(template="plotly_dark", plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19")
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
    st.title(f"{sport_label} Team & Rotation Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        t1_sel = st.selectbox("Team 1", options=team_list, format_func=lambda x: team_options[x], index=default_home_idx, key="exp_t1")
    with col2:
        t2_sel = st.selectbox("Team 2", options=team_list, format_func=lambda x: team_options[x], index=default_away_idx, key="exp_t2")

    if sport == "mlb":
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
    else:
        t1_logs = matchups_df[matchups_df["home_team_id"] == t1_sel].iloc[-1:]
        t2_logs = matchups_df[matchups_df["home_team_id"] == t2_sel].iloc[-1:]

        if not t1_logs.empty and not t2_logs.empty:
            t1_row = t1_logs.iloc[0]
            t2_row = t2_logs.iloc[0]
            categories = ['eFG% (Shooting)', 'TOV% (Ball Security)', 'ORB% (Rebounding)', 'FTR (Free Throws)', 'Net Rating']
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
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Four Factors Radar Comparison (10-Game Form)")
            st.plotly_chart(fig, use_container_width=True)

# ================= 4. BACKTEST SIMULATION =================
elif nav_selection == "Backtest Simulation":
    st.title(f"{sport_label} Historical Backtest Simulation")
    
    season_options = ["All Seasons", "2024", "2023", "2022", "2021", "2020"] if sport == "mlb" else ["All Seasons", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
    market_options = ["moneyline", "spread", "total"]

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        bt_season = st.selectbox("Season Filter", options=season_options, index=1, key=f"bt_season_filter_{sport}")
    with b_col2:
        bt_compound = st.checkbox("Dynamic Compounding", value=False, key=f"bt_compound_{sport}")
    with b_col3:
        bt_markets = st.multiselect("Markets", options=market_options, default=["moneyline"], key=f"bt_markets_{sport}")
    with b_col4:
        run_btn = st.button("Run Simulation", type="primary", key=f"bt_run_btn_{sport}")

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
            labels={"game_date": "Date", "bankroll": f"Bankroll ({config.DEFAULT_CURRENCY})"}
        )
        fig_equity.update_traces(line=dict(color="#10b981", width=2.5))
        fig_equity.update_layout(template="plotly_dark", plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19")
        st.plotly_chart(fig_equity, use_container_width=True)

    if res["bets"]:
        st.markdown("##### Chronological Simulated Bet Log")
        st.dataframe(pd.DataFrame(res["bets"]).tail(100), use_container_width=True)

# ================= 5. MODEL DIAGNOSTICS =================
elif nav_selection == "Model Diagnostics":
    st.title(f"{sport_label} Calibration & Model Diagnostics")
    
    if model_metrics:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Log Loss", str(model_metrics.get("win_probability_metrics", {}).get("log_loss", "N/A")))
        d2.metric("Brier Score", str(model_metrics.get("win_probability_metrics", {}).get("brier_score", "N/A")))
        d3.metric("Accuracy", f"{model_metrics.get('win_probability_metrics', {}).get('accuracy', 0)*100:.1f}%")
        d4.metric("Margin MAE", f"{model_metrics.get('margin_metrics', {}).get('mae', 'N/A')}")

        rel_bins = model_metrics.get("win_probability_metrics", {}).get("reliability_bins", [])
        if rel_bins:
            b_df = pd.DataFrame(rel_bins)
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
            fig_rel.add_trace(go.Scatter(x=b_df["confidence"], y=b_df["accuracy"], mode="lines+markers", name=f"{sport.upper()} Model Calibration", line=dict(color="#10b981", width=3)))
            fig_rel.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0b0f19",
                paper_bgcolor="#0b0f19",
                xaxis_title="Forecast Confidence (Predicted Probability)",
                yaxis_title="Empirical Win Frequency",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_rel, use_container_width=True)

# ================= 6. DATA & SETTINGS =================
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
        st.markdown("##### Synchronize Season Data")
        sync_sport = st.selectbox("Target Sport", options=["NBA", "MLB"], index=0 if sport=="nba" else 1)
        sync_season_input = st.text_input("Season Identifier", value="2024" if sync_sport=="MLB" else "2024-25")
        if st.button("Trigger Sync"):
            with st.spinner("Synchronizing data..."):
                if sync_sport == "MLB":
                    generate_mlb_seed_dataset_if_empty()
                    st.success("MLB data synchronized successfully.")
                else:
                    generate_seed_dataset_if_empty()
                    st.success("NBA data synchronized successfully.")