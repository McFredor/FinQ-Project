"""Árboles trinomiales en ln(S) para la call down-and-out (Figlewski & Gao, secc. 4.1).

El árbol no se centra en la media: X(t) = ln S_t, el nodo del medio es "sin cambio" y
las probabilidades de la ec. (9) ajustan media y varianza a la distribución neutral al riesgo.
"""

import numpy as np


def probabilidades(h, k, r, sigma, q=0.0):
    """Probabilidades (p_u, p_m, p_d) de la ec. (9) para paso de precio h y de tiempo k."""
    a = r - q - 0.5 * sigma**2
    base = sigma**2 * k / h**2 + a**2 * k**2 / h**2
    pu = 0.5 * (base + a * k / h)
    pd = 0.5 * (base - a * k / h)
    return pu, 1.0 - pu - pd, pd


def paso_temporal(h, T, sigma, lam=3.0):
    """Paso temporal k = T / int(lam * sigma**2 * T / h**2) de la ec. (17):
    N entero y lambda => objetivo."""
    N = max(int(lam * sigma**2 * T / h**2), 1)
    return T / N, N


def trinomial_estandar_gao(S, K, T, r, sigma, H, N, q=0.0, lam=3.0):
    """Trinomial estándar con N pasos y h = sigma * sqrt(lam * k).
    La barrera no cae, en general, en un nodo."""
    k = T / N
    h = sigma * np.sqrt(lam * k)
    pu, pm, pd = probabilidades(h, k, r, sigma, q)
    desc = np.exp(-r * k)

    x = np.log(S) + h * np.arange(-N, N + 1)          # precios al vencimiento
    vivos = x > np.log(H)                             # nodos que no tocaron la barrera
    V = np.where(vivos, np.maximum(np.exp(x) - K, 0.0), 0.0)
    for _ in range(N):
        x, vivos = x[1:-1], vivos[1:-1]
        V = desc * (pu * V[2:] + pm * V[1:-1] + pd * V[:-2])
        V[~vivos] = 0.0
    return V[0]


def arbol_barrera(S, K, T, r, sigma, H, h, N, q=0.0, guardar_lineas=False):
    """Retícula con una capa de nodos exactamente sobre la barrera.

    Los nodos son x_i = ln H + i·h (i = 0 es la barrera) y S debe caer en un nodo:
    i0 = (ln S - ln H)/h entero. Devuelve el valor en S.  Si guardar_lineas=True devuelve
    también las líneas i = 1 e i = 2 en todos los pasos (las usa el AMM).
    """
    i0 = int(round((np.log(S) - np.log(H)) / h))
    k = T / N
    pu, pm, pd = probabilidades(h, k, r, sigma, q)
    desc = np.exp(-r * k)

    tope = i0 + N                                       # nodo más alto al vencimiento
    x = np.log(H) + h * np.arange(tope + 1)
    V = np.maximum(np.exp(x) - K, 0.0)
    V[0] = 0.0                                          # sobre la barrera: knock-out
    linea1 = np.full(N + 1, np.nan)
    linea2 = np.full(N + 1, np.nan)
    linea1[N], linea2[N] = V[1], V[2]

    for j in range(N - 1, -1, -1):
        V = np.concatenate(([0.0], desc * (pu * V[2:] + pm * V[1:-1] + pd * V[:-2])))
        linea1[j] = V[1]
        if V.size > 2:
            linea2[j] = V[2]

    valor = V[i0]
    return (valor, linea1, linea2) if guardar_lineas else valor


def rtm_ritchken(S, K, T, r, sigma, H, pasos_hasta_barrera=1, q=0.0, lam=3.0):
    """Restricted Trinomial Model Ritchken (1995):
    h = ln(S/H)/n y k según la ec. (17)."""
    h = (np.log(S) - np.log(H)) / pasos_hasta_barrera
    _, N = paso_temporal(h, T, sigma, lam)
    return arbol_barrera(S, K, T, r, sigma, H, h, N, q), N
