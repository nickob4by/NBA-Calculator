import numpy as np
import pandas as pd
from src.features.dota2_metrics import calculate_dota2_matchup_prob, get_dota2_team_metrics

class Dota2WinProbabilityModel:
    def __init__(self):
        self.feature_names = ["elo_diff", "win_pct_diff", "roshan_diff"]
        self.residual_std = 1.0

    @staticmethod
    def load():
        return Dota2WinProbabilityModel()

    def predict_proba(self, team1_id: int, team2_id: int, series_format: str = "Bo3", is_radiant: bool = True) -> dict:
        return calculate_dota2_matchup_prob(
            team1_id=team1_id,
            team2_id=team2_id,
            series_format=series_format,
            is_team1_radiant=is_radiant
        )

class Dota2SeriesPredictor:
    def __init__(self):
        self.feature_names = ["elo_diff"]
        self.residual_std = 0.85

    @staticmethod
    def load():
        return Dota2SeriesPredictor()

    def predict_maps(self, team1_id: int, team2_id: int) -> float:
        t1 = get_dota2_team_metrics(team1_id)
        t2 = get_dota2_team_metrics(team2_id)
        elo_diff = t1["elo"] - t2["elo"]
        # Map margin scaled
        return round(elo_diff / 200.0, 2)
