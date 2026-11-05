"""Adaptive Mesh Model para la call down-and-out (Figlewski & Gao, secc. 4.2-4.4).

Notación:
L_0 = ln H + h      es la línea de la malla gruesa justo encima de la barrera y
L_m = ln H + h/2^m  es la línea del medio del nivel m (paso h/2^m, tiempo k/4^m).
El valor buscado está en
                L_M, con L_M = ln S,    es decir   h = 2^M · (ln S - ln H).

Cada nivel m se construye en dos pasos:

1. Refinar en el tiempo la línea de arriba (L_{m-1}): los puntos que caen entre
   dos pasos del nivel anterior se calculan desde el paso siguiente de ese nivel
   con un paso efectivo de 3, 2 o 1 veces k/4^m.

2. Recorrer hacia atrás la nueva línea L_m con probabilidades (h/2^m, k/4^m),
   usando como vecinos la barrera (valor 0) y la línea refinada de arriba.
"""

import numpy as np

from .trinomial import arbol_barrera, probabilidades, paso_temporal


def _recursion_hacia_atras(a, b):
    """Resuelve V[n] = a·V[n+1] + b[n] hacia atrás, con V[N] = b[N]."""
    b = np.asarray(b, dtype=float)
    V = np.empty_like(b)
    V[-1] = b[-1]
    for n in range(len(b) - 2, -1, -1):
        V[n] = a * V[n + 1] + b[n]
    return V


def amm_gao(S, K, T, r, sigma, H, M=None, q=0.0, lam=3.0, h_grueso=np.log(92 / 90)):
    """Valor AMM de la call down-and-out con M niveles de malla fina.

    Si M es None se elige el menor M tal que el paso grueso h = 2^M·ln(S/H) llegue a h_grueso
    Devuelve (valor, M, N) con N los pasos de la malla gruesa.
    """
    d = np.log(S) - np.log(H)
    if M is None:
        M = max(int(np.ceil(np.log2(h_grueso / d) - 1e-12)), 0)
    h = 2**M * d
    k, N = paso_temporal(h, T, sigma, lam)

    # Malla gruesa (nivel A): líneas i = 1 (L_0) e i = 2 en cada paso k.
    valor, arriba, arriba2 = arbol_barrera(S * np.exp(h - d), K, T, r, sigma, H, h, N, q,
                                           guardar_lineas=True)
    if M == 0:
        return valor, M, N

    h_m, k_m = h, k
    for _ in range(M):
        # Paso 1: refinar la línea de arriba al tiempo k_m/4 (vecinos: barrera, arriba, arriba2).
        n_ant = arriba.size - 1
        refinada = np.empty(4 * n_ant + 1)
        refinada[::4] = arriba
        for m in (1, 2, 3):
            dt = (4 - m) * k_m / 4
            pu, pm, _ = probabilidades(h_m, dt, r, sigma, q)
            refinada[m::4][:n_ant] = np.exp(-r * dt) * (pu * arriba2[1:] + pm * arriba[1:])

        # Paso 2: línea del medio con paso (h_m/2, k_m/4); el vecino de abajo es la barrera (0).
        h_m, k_m = h_m / 2, k_m / 4
        pu, pm, _ = probabilidades(h_m, k_m, r, sigma, q)
        desc = np.exp(-r * k_m)
        x_medio = np.log(H) + h_m
        b = np.empty(refinada.size)
        b[:-1] = desc * pu * refinada[1:]
        b[-1] = max(np.exp(x_medio) - K, 0.0)          # pago al vencimiento
        media = _recursion_hacia_atras(desc * pm, b)

        arriba2, arriba = refinada, media

    return arriba[0], M, N
