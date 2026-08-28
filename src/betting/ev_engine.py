import numpy as np
from typing import Dict, List, Optional
from scipy.stats import norm
from src.betting.odds_math import remove_vig, american_to_decimal
import config

def calculate_expected_value(model_prob: float, decimal_odds: float) -> float:
    """
    Calculates expected value per unit stake:
    EV = (P_model * (Decimal_Odds - 1)) - ((1 - P_model) * 1.0)
    """
    if decimal_odds <= 1.0:
        return -1.0
    net_profit = decimal_odds - 1.0
    loss_prob = 1.0 - model_prob
    ev = (model_prob * net_profit) - (loss_prob * 1.0)
    return round(float(ev), 4)

def calculate_edge(model_prob: float, fair_implied_prob: float) -> float:
    """
    Edge = Model Probability - Fair Market Implied Probability
    """
    return round(float(model_prob - fair_implied_prob), 4)

def evaluate_moneyline_market(
    pred_home_win_prob: float,
    home_ml_decimal: float,
    away_ml_decimal: float,
    min_edge: float = config.DEFAULT_MIN_EDGE,
    vig_method: str = "multiplicative"
) -> Dict:
    """
    Evaluates moneyline market for both Home and Away sides, determining +EV opportunities.
    """
    fair_p_home, fair_p_away = remove_vig(home_ml_decimal, away_ml_decimal, method=vig_method)
    pred_away_win_prob = 1.0 - pred_home_win_prob

    # Home side
    ev_home = calculate_expected_value(pred_home_win_prob, home_ml_decimal)
    edge_home = calculate_edge(pred_home_win_prob, fair_p_home)

    # Away side
    ev_away = calculate_expected_value(pred_away_win_prob, away_ml_decimal)
    edge_away = calculate_edge(pred_away_win_prob, fair_p_away)

    opportunities = []

    if edge_home >= min_edge and ev_home > 0:
        opportunities.append({
            "market_type": "moneyline",
            "side": "home",
            "decimal_odds": home_ml_decimal,
            "model_prob": round(pred_home_win_prob, 4),
            "fair_implied_prob": round(fair_p_home, 4),
            "edge": edge_home,
            "ev": ev_home
        })

    if edge_away >= min_edge and ev_away > 0:
        opportunities.append({
            "market_type": "moneyline",
            "side": "away",
            "decimal_odds": away_ml_decimal,
            "model_prob": round(pred_away_win_prob, 4),
            "fair_implied_prob": round(fair_p_away, 4),
            "edge": edge_away,
            "ev": ev_away
        })

    return {
        "home": {
            "decimal_odds": home_ml_decimal,
            "model_prob": round(pred_home_win_prob, 4),
            "fair_implied_prob": round(fair_p_home, 4),
            "edge": edge_home,
            "ev": ev_home
        },
        "away": {
            "decimal_odds": away_ml_decimal,
            "model_prob": round(pred_away_win_prob, 4),
            "fair_implied_prob": round(fair_p_away, 4),
            "edge": edge_away,
            "ev": ev_away
        },
        "opportunities": opportunities
    }

def evaluate_spread_market(
    pred_margin: float,
    spread_line: float, # e.g. -4.5 (Home is 4.5 pt favorite)
    home_spread_decimal: float = 1.91,
    away_spread_decimal: float = 1.91,
    residual_std: float = 13.5,
    min_edge: float = config.DEFAULT_MIN_EDGE
) -> Dict:
    """
    Evaluates point spread market.
    Home covers if (Home PTS - Away PTS) + spread_line > 0  => Margin > -spread_line.
    """
    # Model probability that Home covers: P(Margin > -spread_line) = 1 - Phi((-spread_line - pred_margin) / sigma)
    # Equivalently: Phi((pred_margin + spread_line) / sigma)
    z_home = (pred_margin + spread_line) / residual_std
    p_home_cover = float(norm.cdf(z_home))
    p_away_cover = 1.0 - p_home_cover

    fair_p_home, fair_p_away = remove_vig(home_spread_decimal, away_spread_decimal)

    ev_home = calculate_expected_value(p_home_cover, home_spread_decimal)
    edge_home = calculate_edge(p_home_cover, fair_p_home)

    ev_away = calculate_expected_value(p_away_cover, away_spread_decimal)
    edge_away = calculate_edge(p_away_cover, fair_p_away)

    opportunities = []
    if edge_home >= min_edge and ev_home > 0:
        opportunities.append({
            "market_type": "spread",
            "side": "home",
            "spread_line": spread_line,
            "decimal_odds": home_spread_decimal,
            "model_prob": round(p_home_cover, 4),
            "fair_implied_prob": round(fair_p_home, 4),
            "edge": edge_home,
            "ev": ev_home
        })

    if edge_away >= min_edge and ev_away > 0:
        opportunities.append({
            "market_type": "spread",
            "side": "away",
            "spread_line": -spread_line,
            "decimal_odds": away_spread_decimal,
            "model_prob": round(p_away_cover, 4),
            "fair_implied_prob": round(fair_p_away, 4),
            "edge": edge_away,
            "ev": ev_away
        })

    return {
        "spread_line": spread_line,
        "p_home_cover": round(p_home_cover, 4),
        "p_away_cover": round(p_away_cover, 4),
        "home_ev": ev_home,
        "away_ev": ev_away,
        "home_edge": edge_home,
        "away_edge": edge_away,
        "opportunities": opportunities
    }

def evaluate_totals_market(
    pred_total: float,
    total_line: float,
    over_decimal: float = 1.91,
    under_decimal: float = 1.91,
    total_residual_std: float = 18.0,
    min_edge: float = config.DEFAULT_MIN_EDGE
) -> Dict:
    """
    Evaluates Over / Under game total points market.
    """
    z = (pred_total - total_line) / total_residual_std
    p_over = float(norm.cdf(z))
    p_under = 1.0 - p_over

    fair_p_over, fair_p_under = remove_vig(over_decimal, under_decimal)

    ev_over = calculate_expected_value(p_over, over_decimal)
    edge_over = calculate_edge(p_over, fair_p_over)

    ev_under = calculate_expected_value(p_under, under_decimal)
    edge_under = calculate_edge(p_under, fair_p_under)

    opportunities = []
    if edge_over >= min_edge and ev_over > 0:
        opportunities.append({
            "market_type": "total",
            "side": "over",
            "total_line": total_line,
            "decimal_odds": over_decimal,
            "model_prob": round(p_over, 4),
            "fair_implied_prob": round(fair_p_over, 4),
            "edge": edge_over,
            "ev": ev_over
        })

    if edge_under >= min_edge and ev_under > 0:
        opportunities.append({
            "market_type": "total",
            "side": "under",
            "total_line": total_line,
            "decimal_odds": under_decimal,
            "model_prob": round(p_under, 4),
            "fair_implied_prob": round(fair_p_under, 4),
            "edge": edge_under,
            "ev": ev_under
        })

    return {
        "total_line": total_line,
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "over_ev": ev_over,
        "under_ev": ev_under,
        "over_edge": edge_over,
        "under_edge": edge_under,
        "opportunities": opportunities
    }