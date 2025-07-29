import numpy as np
from scipy.optimize import root
from scipy.stats import norm

N = norm.cdf


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
):
    """Calculates BS price for array of strikes and vols"""
    pc = 1 if is_call else -1
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return pc * S * N(pc * d1) - pc * K * np.exp(-r * T) * N(pc * d2)


def black76(f, K, T, r, sigma, is_call=True):
    return np.exp(-r * T) * black_scholes(f, K, T, 0, sigma, is_call)


def dK(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
):
    """Calculates BS price for array of strikes and vols"""
    d2 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (
        sigma * np.sqrt(T)
    ) - sigma * np.sqrt(T)
    return -np.exp(-r * T) * N(d2)


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
):
    """Delta of BS formula"""
    pc = 1 if is_call else -1
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return pc * N(pc * d1)


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
):
    """Vega of BS formula"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def imply_volatility(v, s, k, t, r, iv_init, isCall=True):
    """Implies Black-Scholes volatility given an option price using root finder"""
    option_price = lambda sigma: black_scholes(s, k, t, r, sigma, isCall)
    optimized = [
        root(
            lambda sigma: v[j] - option_price(sigma)[j], iv_init, options={"xtol": 1e-4}
        )
        for j in range(len(k))
    ]
    return np.array([optim.x if optim.success else np.nan for optim in optimized])
