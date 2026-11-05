import numpy as np
import pytest

from amm import amm_gao, call_down_and_out, rtm_ritchken

P = dict(K=100, T=1.0, r=0.10, sigma=0.25, H=90)


def test_analitico_coincide_con_paper():
    # Fig. 3 del paper: 11.323
    assert call_down_and_out(100, **P) == pytest.approx(11.323, abs=1e-3)


@pytest.mark.parametrize("S, esperado", [(92, 2.506), (91, 1.274), (90.5, 0.642),
                                         (90.25, 0.323), (90.125, 0.162)])
def test_amm_reproduce_tabla3(S, esperado):
    valor, _, _ = amm_gao(S, **P)
    assert valor == pytest.approx(esperado, abs=1.5e-3)
    assert valor == pytest.approx(call_down_and_out(S, **P), abs=1.5e-3)


def test_rtm_pasos_coinciden_con_tabla3():
    assert rtm_ritchken(92, **P)[1] == 388
    assert rtm_ritchken(91, **P)[1] == 1535


def test_amm_eleccion_de_nivel_coincide_con_tabla3():
    assert [amm_gao(S, **P)[1] for S in (92, 91, 90.5, 90.25, 90.125)] == [0, 1, 2, 3, 4]


def test_amm_preciso_para_cualquier_paso_grueso():
    # Cambiar el paso grueso cambia M y N, pero el error sigue chico.
    exacto = call_down_and_out(90.5, **P)
    errores = [abs(amm_gao(90.5, h_grueso=hg, **P)[0] - exacto) for hg in (0.08, 0.04, 0.02, 0.01)]
    assert np.all(np.array(errores) < 5e-4)
