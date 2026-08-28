import numpy as np
from typing import Tuple, Dict
from scipy.optimize import minimize_scalar

def american_to_decimal(american: float) -> float:
    """
    Converts American odds (e.g., -110, +150) to Decimal odds (e.g., 1.91, 2.50).
    """
    if american == 0:
        return 1.0
    if american > 0:
        return round(1.0 + (american / 100.0), 4)
    else:
        return round(1.0 + (100.0 / abs(american)), 4)

def decimal_to_american(decimal: float) -> int:
    """
    Converts Decimal odds to American odds integer.
    """
    if decimal <= 1.0:
        return 0
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    else:
        return int(round(-100.0 / (decimal - 1.0)))

def decimal_to_raw_implied_prob(decimal: float) -> float:
    """
    Returns raw implied probability with vig included.
    """
    if decimal <= 1.0:
        return 1.0
    return 1.0 / decimal

def remove_vig_multiplicative(decimal_home: float, decimal_away: float) -> Tuple[float, float]:
    """
    Standard multiplicative normalization to remove bookmaker margin / overround.
    """
    raw_h = 1.0 / decimal_home
    raw_a = 1.0 / decimal_away
    overround = raw_h + raw_a
    fair_h = raw_h / overround
    fair_a = raw_a / overround
    return fair_h, fair_a

def remove_vig_power(decimal_home: float, decimal_away: float) -> Tuple[float, float]:
    """
    Power method vig removal: finds exponent k such that (1/d_h)^k + (1/d_a)^k = 1.
    Better accounts for favorite-longshot bias.
    """
    raw_h = 1.0 / decimal_home
    raw_a = 1.0 / decimal_away

    def objective(k):
        return abs((raw_h ** k) + (raw_a ** k) - 1.0)

    res = minimize_scalar(objective, bounds=(0.5, 2.0), method="bounded")
    k = res.x
    fair_h = raw_h ** k
    fair_a = raw_a ** k
    total = fair_h + fair_a
    return fair_h / total, fair_a / total

def remove_vig_shin(decimal_home: float, decimal_away: float) -> Tuple[float, float]:
    """
    Shin method vig removal: models the proportion of informed insider trading 'z'.
    """
    raw_h = 1.0 / decimal_home
    raw_a = 1.0 / decimal_away
    overround = raw_h + raw_a
    
    if overround <= 1.0:
        return raw_h / overround, raw_a / overround

    def shin_objective(z):
        p_h = (np.sqrt(z**2 + 4 * (1 - z) * (raw_h**2) / overround) - z) / (2 * (1 - z))
        p_a = (np.sqrt(z**2 + 4 * (1 - z) * (raw_a**2) / overround) - z) / (2 * (1 - z))
        return abs((p_h + p_a) - 1.0)

    res = minimize_scalar(shin_objective, bounds=(0.0, 0.3), method="bounded")
    z = res.x
    p_h = (np.sqrt(z**2 + 4 * (1 - z) * (raw_h**2) / overround) - z) / (2 * (1 - z))
    p_a = (np.sqrt(z**2 + 4 * (1 - z) * (raw_a**2) / overround) - z) / (2 * (1 - z))
    s = p_h + p_a
    return p_h / s, p_a / s

def remove_vig(decimal_home: float, decimal_away: float, method: str = "multiplicative") -> Tuple[float, float]:
    if method == "shin":
        return remove_vig_shin(decimal_home, decimal_away)
    elif method == "power":
        return remove_vig_power(decimal_home, decimal_away)
    else:
        return remove_vig_multiplicative(decimal_home, decimal_away)