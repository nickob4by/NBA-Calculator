import pandas as pd
import numpy as np
from src.features.four_factors import compute_advanced_stats_dataframe
from src.features.mlb_sabermetrics import compute_mlb_sabermetrics_dataframe
from src.features.situational import compute_situational_features
from src.features.rolling_metrics import compute_rolling_team_features

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
                df = df.merge(adv_df, on=["game_id", "team_id", "opponent_id", "sport"], how="left")
    else:
        if "net_rating" not in df.columns or "efg_pct" not in df.columns:
            adv_df = compute_advanced_stats_dataframe(df)
            if not adv_df.empty:
                df = df.merge(adv_df, on=["game_id", "team_id", "opponent_id"], how="left")

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

def get_feature_columns(matchup_df: pd.DataFrame) -> list:
    """
    Returns the list of clean predictor feature column names (strictly numeric, non-target, non-metadata).
    """
    exclude_cols = {
        "game_id", "game_date", "season", "sport", "home_team_id", "away_team_id",
        "home_opponent_id", "away_opponent_id", "home_wl", "away_wl",
        "home_created_at", "away_created_at", "home_opp_abbrev", "away_opp_abbrev",
        "home_pts", "away_pts", "home_plus_minus", "away_plus_minus",
        "home_runs", "away_runs", "home_hits", "away_hits", "home_errors", "away_errors",
        "home_hr", "away_hr", "home_rbi", "away_rbi", "home_bb", "away_bb",
        "home_so", "away_so", "home_lob", "away_lob", "home_ip", "away_ip", "home_er", "away_er",
        "home_fgm", "away_fgm", "home_fga", "away_fga",
        "home_fg3m", "away_fg3m", "home_fg3a", "away_fg3a",
        "home_ftm", "away_ftm", "home_fta", "away_fta",
        "home_oreb", "away_oreb", "home_dreb", "away_dreb",
        "home_reb", "away_reb", "home_ast", "away_ast",
        "home_stl", "away_stl", "home_blk", "away_blk",
        "home_tov", "away_tov", "home_pf", "away_pf",
        "home_min", "away_min", "home_fg_pct", "away_fg_pct",
        "home_fg3_pct", "away_fg3_pct", "home_ft_pct", "away_ft_pct",
        "home_possessions", "away_possessions", "home_pace", "away_pace",
        "home_off_rating", "away_off_rating", "home_def_rating", "away_def_rating",
        "home_net_rating", "away_net_rating", "home_efg_pct", "away_efg_pct",
        "home_tov_pct", "away_tov_pct", "home_orb_pct", "away_orb_pct",
        "home_ftr", "away_ftr", "home_opp_efg_pct", "away_opp_efg_pct",
        "home_opp_tov_pct", "away_opp_tov_pct", "home_opp_orb_pct", "away_opp_orb_pct",
        "home_opp_ftr", "away_opp_ftr", "home_arena_lat", "home_arena_lon",
        "away_arena_lat", "away_arena_lon", "home_is_home", "away_is_home",
        "home_pythag_win_pct", "away_pythag_win_pct", "home_obp", "away_obp",
        "home_slg", "away_slg", "home_ops", "away_ops", "home_iso", "away_iso",
        "home_woba_proxy", "away_woba_proxy", "home_fip_proxy", "away_fip_proxy",
        "home_whip", "away_whip", "home_k_per_9", "away_k_per_9", "home_bb_per_9", "away_bb_per_9",
        "point_margin", "total_points", "home_win", "created_at", "win_numeric",
        "home_win_numeric", "away_win_numeric"
    }

    numeric_cols = matchup_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude_cols]
    # Filter out columns that have no observed non-NaN values in this sport
    features = [c for c in features if not matchup_df[c].isna().all()]
    return sorted(features)