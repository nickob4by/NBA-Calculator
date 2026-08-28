import math
import pandas as pd
import numpy as np
import config

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth (in miles).
    """
    # Radius of earth in miles
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def get_arena_coordinates(team_id: int):
    """
    Returns (lat, lon) for the home arena of team_id.
    """
    team = config.NBA_TEAMS.get(team_id)
    if team:
        return team["lat"], team["lon"]
    return 39.8283, -98.5795 # Default geographic center of US

def compute_situational_features(game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rest days, B2B flags, 3-in-4 nights, and travel distance per team game log.
    Ensures calculations are performed chronologically per team.
    """
    if game_logs_df.empty:
        return game_logs_df

    df = game_logs_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(by=["team_id", "game_date"]).reset_index(drop=True)

    # Arena location of the current game:
    # If is_home == 1, current game arena is team's arena.
    # If is_home == 0, current game arena is opponent's arena.
    def get_game_location(row):
        arena_team_id = row["team_id"] if row["is_home"] == 1 else row["opponent_id"]
        return get_arena_coordinates(arena_team_id)

    coords = [get_game_location(row) for _, row in df.iterrows()]
    df["arena_lat"] = [c[0] for c in coords]
    df["arena_lon"] = [c[1] for c in coords]

    # Calculate lag situational metrics per team
    rest_days_list = []
    is_b2b_list = []
    is_3_in_4_list = []
    travel_dist_list = []

    for team_id, group in df.groupby("team_id", sort=False):
        dates = group["game_date"].tolist()
        lats = group["arena_lat"].tolist()
        lons = group["arena_lon"].tolist()

        for i in range(len(dates)):
            if i == 0:
                # First game of season: assume well-rested (e.g. 5 days rest) and 0 travel
                rest_days_list.append(5.0)
                is_b2b_list.append(0)
                is_3_in_4_list.append(0)
                travel_dist_list.append(0.0)
            else:
                days_diff = (dates[i] - dates[i-1]).days
                # Cap rest days at 7 to prevent long-break skew
                rest_days = min(float(days_diff), 7.0)
                rest_days_list.append(rest_days)

                # B2B: played yesterday (1 calendar day difference)
                b2b = 1 if days_diff == 1 else 0
                is_b2b_list.append(b2b)

                # 3-in-4 nights: played 3 games within 4 days (dates[i] - dates[i-2] <= 3 days)
                three_in_four = 1 if (i >= 2 and (dates[i] - dates[i-2]).days <= 3) else 0
                is_3_in_4_list.append(three_in_four)

                # Travel distance from previous game arena to current game arena
                dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
                travel_dist_list.append(round(dist, 1))

    df["rest_days"] = rest_days_list
    df["is_b2b"] = is_b2b_list
    df["is_3_in_4"] = is_3_in_4_list
    df["travel_distance"] = travel_dist_list

    return df
