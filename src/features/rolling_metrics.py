import pandas as pd
import numpy as np
import config

STAT_COLUMNS_TO_ROLL = [
    # NBA Box score core
    "pts", "plus_minus", "fg_pct", "fg3_pct", "ft_pct", "ast", "reb", "tov",
    # NBA Advanced ratings & pace
    "possessions", "pace", "off_rating", "def_rating", "net_rating",
    # NBA Four Factors Offense
    "efg_pct", "tov_pct", "orb_pct", "ftr",
    # NBA Four Factors Defense
    "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ftr",
    # MLB Box score & sabermetrics
    "runs", "hits", "hr", "rbi", "bb", "so",
    "pythag_win_pct", "obp", "slg", "ops", "iso", "woba_proxy",
    "fip_proxy", "whip", "k_per_9", "bb_per_9"
]

def compute_rolling_team_features(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Computes strictly lagged (shift=1) rolling moving averages and EWMA for each team.
    Guarantees ZERO lookahead bias: row i for team T only uses data from matches 0..i-1.
    """
    if df.empty:
        return df

    windows = windows or config.ROLLING_WINDOWS
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(by=["team_id", "game_date"]).reset_index(drop=True)

    if "wl" in df.columns:
        df["win_numeric"] = df["wl"].apply(lambda x: 1 if str(x).upper() == "W" else 0)
    else:
        df["win_numeric"] = 0

    cols_to_roll = [c for c in STAT_COLUMNS_TO_ROLL if c in df.columns] + ["win_numeric"]

    transformed_groups = []

    for team_id, group in df.groupby("team_id", sort=False):
        group = group.copy().reset_index(drop=True)
        lagged_stats = group[cols_to_roll].shift(1)

        new_features = {}

        # 1. Rolling Moving Averages
        for w in windows:
            rolled = lagged_stats.rolling(window=w, min_periods=1).mean()
            for col in cols_to_roll:
                new_features[f"roll_{col}_w{w}"] = rolled[col]

        # 2. EWMA
        for span in config.EWMA_SPANS:
            ewma = lagged_stats.ewm(span=span, min_periods=1).mean()
            for col in cols_to_roll:
                new_features[f"ewma_{col}_s{span}"] = ewma[col]

        # 3. Expanding season average
        expanding_avg = lagged_stats.expanding(min_periods=1).mean()
        for col in cols_to_roll:
            new_features[f"season_avg_{col}"] = expanding_avg[col]

        new_df = pd.DataFrame(new_features, index=group.index)
        combined = pd.concat([group, new_df], axis=1)
        transformed_groups.append(combined)

    result_df = pd.concat(transformed_groups, axis=0).sort_values(by=["game_date", "game_id"]).reset_index(drop=True)
    return result_df