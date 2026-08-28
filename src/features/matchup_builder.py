import pandas as pd
import numpy as np
from src.features.four_factors import compute_advanced_stats_dataframe
from src.features.mlb_sabermetrics import compute_mlb_sabermetrics_dataframe
from src.features.situational import compute_situational_features, haversine_distance, get_arena_coordinates
from src.features.rolling_metrics import compute_rolling_team_features
import config

CORE_DIFF_STATS = [
    # NBA
    "net_rating", "off_rating", "def_rating",
    "efg_pct", "tov_pct", "orb_pct", "ftr",
    "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ftr",
    "pts", "plus_minus", "pace",
    # MLB
    "pythag_win_pct", "obp", "slg", "ops", "iso", "woba_proxy",
    "fip_proxy", "whip", "k_per_9", "bb_per_9", "runs", "hits", "hr",
    "win_numeric"
]

def build_full_feature_dataset(game_logs_df: pd.DataFrame, games_df: pd.DataFrame = None, sport: str = "nba") -> pd.DataFrame:
    """
    Builds the complete single-row-per-game matchup dataset from raw game logs with zero lookahead bias.
    """
    if game_logs_df.empty:
        return pd.DataFrame()

    df = game_logs_df.copy()
    if "sport" in df.columns and not df.empty:
        sport = str(df.iloc[0]["sport"]).lower()

    # 1. Compute advanced stats (NBA Four Factors or MLB Sabermetrics) if missing
    if sport == "mlb":
        if "pythag_win_pct" not in df.columns or "ops" not in df.columns:
            adv_df = compute_mlb_sabermetrics_dataframe(df)
            if not adv_df.empty:
                adv_cols = [c for c in adv_df.columns if c not in df.columns or c in ["game_id", "team_id", "opponent_id", "sport"]]
                df = df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id", "sport"], how="left")
    else:
        if "net_rating" not in df.columns or "efg_pct" not in df.columns:
            adv_df = compute_advanced_stats_dataframe(df)
            if not adv_df.empty:
                adv_cols = [c for c in adv_df.columns if c not in df.columns or c in ["game_id", "team_id", "opponent_id"]]
                df = df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id"], how="left")

    # 2. Add situational features (Rest, B2B, 3-in-4, Travel Distance)
    df = compute_situational_features(df)

    # 3. Add zero-lookahead rolling features
    df = compute_rolling_team_features(df)

    # 4. Split into Home and Away sets and merge by game_id
    home_df = df[df["is_home"] == 1].copy()
    away_df = df[df["is_home"] == 0].copy()

    home_ignore_cols = ["game_id", "game_date", "season", "sport"]
    away_ignore_cols = ["game_id", "game_date", "season", "sport", "opponent_id"]

    home_rename = {c: f"home_{c}" for c in home_df.columns if c not in home_ignore_cols}
    away_rename = {c: f"away_{c}" for c in away_df.columns if c not in away_ignore_cols}

    home_df = home_df.rename(columns=home_rename)
    away_df = away_df.rename(columns=away_rename)

    matchups = pd.merge(home_df, away_df, on=["game_id", "game_date", "season", "sport"], how="inner")

    diff_features = {}

    # Windows
    for w in [5, 10, 20]:
        for stat in CORE_DIFF_STATS:
            h_col = f"home_roll_{stat}_w{w}"
            a_col = f"away_roll_{stat}_w{w}"
            if h_col in matchups.columns and a_col in matchups.columns:
                diff_features[f"diff_roll_{stat}_w{w}"] = matchups[h_col] - matchups[a_col]

    # EWMA Diffs
    for span in [5, 10]:
        for stat in CORE_DIFF_STATS:
            h_col = f"home_ewma_{stat}_s{span}"
            a_col = f"away_ewma_{stat}_s{span}"
            if h_col in matchups.columns and a_col in matchups.columns:
                diff_features[f"diff_ewma_{stat}_s{span}"] = matchups[h_col] - matchups[a_col]

    # Season Average Diffs
    for stat in CORE_DIFF_STATS:
        h_col = f"home_season_avg_{stat}"
        a_col = f"away_season_avg_{stat}"
        if h_col in matchups.columns and a_col in matchups.columns:
            diff_features[f"diff_season_avg_{stat}"] = matchups[h_col] - matchups[a_col]

    # Situational Diffs
    diff_features["diff_rest_days"] = matchups["home_rest_days"] - matchups["away_rest_days"]
    diff_features["diff_travel_distance"] = matchups["home_travel_distance"] - matchups["away_travel_distance"]
    diff_features["is_b2b_diff"] = matchups["home_is_b2b"] - matchups["away_is_b2b"]
    
    if "home_roll_pace_w10" in matchups.columns and "away_roll_pace_w10" in matchups.columns:
        diff_features["expected_pace_w10"] = 0.5 * (matchups["home_roll_pace_w10"] + matchups["away_roll_pace_w10"])

    # Targets (PTS for NBA, Runs for MLB)
    h_score = matchups["home_runs"] if ("home_runs" in matchups.columns and sport == "mlb") else matchups.get("home_pts", 0)
    a_score = matchups["away_runs"] if ("away_runs" in matchups.columns and sport == "mlb") else matchups.get("away_pts", 0)

    if h_score is not None and a_score is not None:
        diff_features["point_margin"] = h_score - a_score
        diff_features["total_points"] = h_score + a_score
        diff_features["home_win"] = (diff_features["point_margin"] > 0).astype(int)

    diff_df = pd.DataFrame(diff_features, index=matchups.index)
    matchups = pd.concat([matchups, diff_df], axis=1)

    matchups = matchups.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)
    return matchups

def build_upcoming_matchup(home_team_id: int, away_team_id: int, logs_df: pd.DataFrame, sport: str = "nba") -> pd.DataFrame:
    """
    Constructs a true synthesized forward-looking feature vector comparing
    home_team's latest lagged form against away_team's latest lagged form directly from logs.
    """
    if "roll_pts_w5" not in logs_df.columns and "roll_runs_w5" not in logs_df.columns:
        # Precompute rolled stats if raw logs passed
        if sport == "mlb":
            adv_df = compute_mlb_sabermetrics_dataframe(logs_df)
            adv_cols = [c for c in adv_df.columns if c not in logs_df.columns or c in ["game_id", "team_id", "opponent_id", "sport"]]
            logs_df = logs_df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id", "sport"], how="left")
        else:
            adv_df = compute_advanced_stats_dataframe(logs_df)
            adv_cols = [c for c in adv_df.columns if c not in logs_df.columns or c in ["game_id", "team_id", "opponent_id"]]
            logs_df = logs_df.merge(adv_df[adv_cols], on=["game_id", "team_id", "opponent_id"], how="left")
        
        logs_df = compute_situational_features(logs_df)
        logs_df = compute_rolling_team_features(logs_df)

    h_logs = logs_df[logs_df["team_id"] == home_team_id]
    a_logs = logs_df[logs_df["team_id"] == away_team_id]

    h_last = h_logs.iloc[-1] if not h_logs.empty else logs_df.iloc[-1]
    a_last = a_logs.iloc[-1] if not a_logs.empty else logs_df.iloc[-1]

    matchup_dict = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "sport": sport
    }

    # Home team lagged stats
    for col in h_last.index:
        if col.startswith("roll_") or col.startswith("ewma_") or col.startswith("season_avg_"):
            matchup_dict[f"home_{col}"] = h_last[col]

    # Away team lagged stats
    for col in a_last.index:
        if col.startswith("roll_") or col.startswith("ewma_") or col.startswith("season_avg_"):
            matchup_dict[f"away_{col}"] = a_last[col]

    # Differentials (Home - Away)
    for w in [5, 10, 20]:
        for stat in CORE_DIFF_STATS:
            h_val = h_last.get(f"roll_{stat}_w{w}", np.nan)
            a_val = a_last.get(f"roll_{stat}_w{w}", np.nan)
            if pd.notna(h_val) and pd.notna(a_val):
                matchup_dict[f"diff_roll_{stat}_w{w}"] = h_val - a_val

    for span in [5, 10]:
        for stat in CORE_DIFF_STATS:
            h_val = h_last.get(f"ewma_{stat}_s{span}", np.nan)
            a_val = a_last.get(f"ewma_{stat}_s{span}", np.nan)
            if pd.notna(h_val) and pd.notna(a_val):
                matchup_dict[f"diff_ewma_{stat}_s{span}"] = h_val - a_val

    for stat in CORE_DIFF_STATS:
        h_val = h_last.get(f"season_avg_{stat}", np.nan)
        a_val = a_last.get(f"season_avg_{stat}", np.nan)
        if pd.notna(h_val) and pd.notna(a_val):
            matchup_dict[f"diff_season_avg_{stat}"] = h_val - a_val

    # Situational features
    h_lat, h_lon = get_arena_coordinates(home_team_id)
    a_lat, a_lon = get_arena_coordinates(away_team_id)
    travel_miles = haversine_distance(a_lat, a_lon, h_lat, h_lon)

    matchup_dict["home_rest_days"] = 1.0
    matchup_dict["away_rest_days"] = 1.0
    matchup_dict["diff_rest_days"] = 0.0
    matchup_dict["home_travel_distance"] = 0.0
    matchup_dict["away_travel_distance"] = travel_miles
    matchup_dict["diff_travel_distance"] = -travel_miles
    matchup_dict["home_is_b2b"] = 0.0
    matchup_dict["away_is_b2b"] = 0.0
    matchup_dict["is_b2b_diff"] = 0.0

    if f"home_roll_pace_w10" in matchup_dict and f"away_roll_pace_w10" in matchup_dict:
        matchup_dict["expected_pace_w10"] = 0.5 * (matchup_dict["home_roll_pace_w10"] + matchup_dict["away_roll_pace_w10"])

    return pd.DataFrame([matchup_dict])

def get_feature_columns(matchup_df: pd.DataFrame) -> list:
    """
    Returns strictly valid lagged predictor features (whitelisting legitimate rolling & differential features).
    Guarantees ZERO target or raw boxscore leakage.
    """
    valid_prefixes = (
        "home_roll_", "away_roll_", "diff_roll_",
        "home_ewma_", "away_ewma_", "diff_ewma_",
        "home_season_avg_", "away_season_avg_", "diff_season_avg_",
        "diff_rest_days", "diff_travel_distance", "is_b2b_diff",
        "home_rest_days", "away_rest_days", "home_travel_distance", "away_travel_distance",
        "home_is_b2b", "away_is_b2b", "expected_pace_w10"
    )

    numeric_cols = matchup_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c.startswith(valid_prefixes)]
    # Filter out columns that have no observed non-NaN values in this sport
    features = [c for c in features if not matchup_df[c].isna().all()]
    return sorted(features)