import numpy as np
from typing import Dict, Optional
import config

def calculate_kelly_fractional(
    model_prob: float,
    decimal_odds: float,
    kelly_multiplier: float = config.DEFAULT_KELLY_FRACTION,
    max_bankroll_pct: float = config.DEFAULT_MAX_BANKROLL_PCT
) -> float:
    """
    Computes the Fractional Kelly Criterion position size as a fraction of current bankroll.
    f* = multiplier * ((b * p - q) / b)
    where:
      b = decimal_odds - 1 (net profit per 1 unit wagered)
      p = model win probability
      q = 1 - p (loss probability)
    """
    if decimal_odds <= 1.0 or model_prob <= 0.0 or model_prob >= 1.0:
        return 0.0

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    raw_kelly = (b * p - q) / b

    # If edge is zero or negative, do not bet
    if raw_kelly <= 0.0:
        return 0.0

    # Apply fractional Kelly scaling (e.g., 0.15 = 15% Kelly)
    fractional_kelly = raw_kelly * kelly_multiplier

    # Cap at maximum allowable risk per single wager
    return min(fractional_kelly, max_bankroll_pct)

def size_bet(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    kelly_multiplier: float = config.DEFAULT_KELLY_FRACTION,
    max_bankroll_pct: float = config.DEFAULT_MAX_BANKROLL_PCT,
    min_bet_amount: float = 5.0
) -> Dict:
    """
    Calculates exact dollar stake and bankroll percentage allocation for a given bet.
    """
    pct = calculate_kelly_fractional(
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        kelly_multiplier=kelly_multiplier,
        max_bankroll_pct=max_bankroll_pct
    )

    stake = round(pct * bankroll, 2)
    
    if stake < min_bet_amount or pct <= 0.0:
        return {
            "stake": 0.0,
            "stake_pct": 0.0,
            "is_actionable": False,
            "reason": "Stake below minimum or negative EV"
        }

    return {
        "stake": stake,
        "stake_pct": round(pct * 100.0, 2),
        "is_actionable": True,
        "reason": f"Allocating {round(pct * 100.0, 2)}% of bankroll ({kelly_multiplier*100:.0f}% Fractional Kelly)"
    }