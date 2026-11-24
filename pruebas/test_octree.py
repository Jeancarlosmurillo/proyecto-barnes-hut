
import numpy as np

from base.particula import Particula
from base.generador import generar_particulas
from base.octree import construir_octree
from base.gravedad import calcular_todas_aceleraciones
from base.barnes_hut import calcular_todas_barnes_hut


def test_particulas_coincidentes():
    particulas = [
        Particula(0, [0.5, 0.5, 0.5], [0, 0, 0], 0.5),
        Particula(1, [0.5, 0.5, 0.5], [0, 0, 0], 0.5)
    ]

    arbol = construir_octree(particulas)

    assert np.isclose(arbol.masa_total, 1.0)

    aceleraciones = calcular_todas_barnes_hut(
        particulas, arbol, theta=0.5
    )

    assert np.all(np.isfinite(aceleraciones))
    assert np.allclose(aceleraciones, 0.0)


def test_distribucion_concentrada():
    particulas = generar_particulas(100, semilla=42)

    for particula in particulas:
        particula.posicion *= 0.0001

    arbol = construir_octree(particulas)

    assert np.isclose(arbol.masa_total, 1.0)

    aceleraciones = calcular_todas_barnes_hut(
        particulas, arbol, theta=0.5
    )

    assert np.all(np.isfinite(aceleraciones))


def test_barnes_hut_theta_cero():
    particulas = generar_particulas(10, semilla=42)
    arbol = construir_octree(particulas)

    directas = calcular_todas_aceleraciones(particulas)

    barnes_hut = calcular_todas_barnes_hut(
        particulas, arbol, theta=0.0
    )

    assert np.allclose(
        directas,
        barnes_hut,
        rtol=1e-10,
        atol=1e-12
    )


def test_masa_y_centro_de_masa():
    particulas = generar_particulas(10, semilla=42)
    arbol = construir_octree(particulas)

    posiciones = np.array([
        p.posicion for p in particulas
    ])

    assert np.isclose(arbol.masa_total, 1.0)

    assert np.allclose(
        arbol.centro_masa,
        np.mean(posiciones, axis=0)
    )
