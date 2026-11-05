"""Fórmulas cerradas: Black-Scholes y call down-and-out"""
#(Hull secc. 26.9).

import numpy as np
from scipy.stats import norm        # distribución normal


def call_bs(S, K, T, r, sigma, q=0.0):
    """Call europea de Black-Scholes."""
    sigma_raiz_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_raiz_T
    d2 = d1 - sigma_raiz_T
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def call_down_and_out(S, K, T, r, sigma, H, q=0.0):
    """Call europea down-and-out con barrera continua H < S y sin rebate.

    Para H <= K es la ec. (8) del paper, escrita en la forma de Hull: c_do = c - c_di.
    Para H > K se usa la fórmula de Hull para ese caso.
    """
    if S <= H:
        return 0.0
    sigma_raiz_T = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    desc_q, desc_r = np.exp(-q * T), np.exp(-r * T)

    if H <= K:
        y = np.log(H**2 / (S * K)) / sigma_raiz_T + lam * sigma_raiz_T
        c_di = (S * desc_q * (H / S) ** (2 * lam) * norm.cdf(y)
                - K * desc_r * (H / S) ** (2 * lam - 2) * norm.cdf(y - sigma_raiz_T))
        return call_bs(S, K, T, r, sigma, q) - c_di

    x1 = np.log(S / H) / sigma_raiz_T + lam * sigma_raiz_T
    y1 = np.log(H / S) / sigma_raiz_T + lam * sigma_raiz_T
    return (S * desc_q * norm.cdf(x1) - K * desc_r * norm.cdf(x1 - sigma_raiz_T)
            - S * desc_q * (H / S) ** (2 * lam) * norm.cdf(y1)
            + K * desc_r * (H / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma_raiz_T))
