# FinQ-Project — Método adaptativo (AMM) para opciones barrera

***TP de Modelos Matemáticos en Finanzas Cuantitativas 2026:** 
Valuación de una call down-and-out con el Adaptive Mesh Model de Figlewski & Gao (1999).*

Hecho por:
- Santino Quevedo (santino.quevedo@mi.unc.edu.ar)
- Wilfredo Avila (wilfredo.avila@mi.unc.edu.ar) 

Informe completo en [Google Docs](https://docs.google.com/document/d/1pRylztYm8HdsFG305bRPOCOdWMTbZVxj3M7m7XzfKOw/edit?usp=sharing)

## Estructura

- `notebooks/amm_barrera.ipynb` — el Análisis: teoría, gráficos y replicación de la Tabla 3 del paper.
- `src/amm/` — implementación:
  - `analitico.py`: Black-Scholes y fórmula cerrada de la down-and-out.
  - `trinomial.py`: Trinomial estándar y RTM (barrera sobre una capa de nodos).
  - `amm_barrera.py`: AMM con M niveles de malla fina.
- `tests/` — tests con pytest (compara con los valores teoricos de la Tabla 3).
- `res/` — enunciado y bibliografía.

## Uso (con uv)

> [Instalar uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync                      # crea .venv e instala dependencias
uv run jupyter lab           # abrir el notebook
uv run pytest                # tests
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/amm_barrera.ipynb
```
