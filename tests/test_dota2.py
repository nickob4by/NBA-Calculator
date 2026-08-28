import pytest
import config
from src.features.dota2_metrics import get_dota2_team_metrics, calculate_dota2_matchup_prob
from src.models.dota2_models import Dota2SeriesPredictor, Dota2WinProbabilityModel
from src.ingestion.dota2_fetcher import generate_dota2_seed_dataset_if_empty
from src.db.database import db

def test_dota2_config():
    assert 'dota2' in config.SPORTS
    assert len(config.DOTA2_TEAMS) >= 30
    assert 1001 in config.DOTA2_TEAMS
    assert config.get_team_name(1001, sport='dota2') == 'Natus Vincere'

def test_dota2_metrics_and_prob():
    falcons_id = 1009
    alliance_id = 1005
    
    t1 = get_dota2_team_metrics(falcons_id)
    t2 = get_dota2_team_metrics(alliance_id)
    assert t1['elo'] > t2['elo']
    
    res_bo3 = calculate_dota2_matchup_prob(falcons_id, alliance_id, series_format='Bo3', is_team1_radiant=True)
    assert res_bo3['p_series_t1'] > 0.70
    assert res_bo3['p_series_t2'] < 0.30
    assert round(res_bo3['p_series_t1'] + res_bo3['p_series_t2'], 2) == 1.00
    
    res_bo1 = calculate_dota2_matchup_prob(falcons_id, alliance_id, series_format='Bo1', is_team1_radiant=True)
    assert res_bo1['p_series_t1'] > 0.60
    
    res_bo5 = calculate_dota2_matchup_prob(falcons_id, alliance_id, series_format='Bo5', is_team1_radiant=True)
    assert res_bo5['p_series_t1'] > res_bo3['p_series_t1']

def test_dota2_models():
    model = Dota2WinProbabilityModel.load()
    prob = model.predict_proba(1001, 1003, series_format='Bo3')
    assert 0.0 < prob['p_series_t1'] < 1.0
    
    predictor = Dota2SeriesPredictor.load()
    margin = predictor.predict_maps(1001, 1003)
    assert isinstance(margin, float)
