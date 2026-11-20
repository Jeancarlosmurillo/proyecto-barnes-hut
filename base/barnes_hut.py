import numpy as np

from base.gravedad import G, EPSILON


THETA = 0.5


def calcular_aceleracion_barnes_hut(particula, nodo, theta=THETA):
    if nodo is None or nodo.masa_total == 0:
        return np.zeros(3, dtype=float)

    # Si el nodo es una hoja, calcular la interacción directa.
    if nodo.es_hoja():
        if nodo.particula is particula:
            return np.zeros(3, dtype=float)

        if nodo.particula is None:
            return np.zeros(3, dtype=float)

        diferencia = nodo.particula.posicion - particula.posicion
        distancia2 = np.dot(diferencia, diferencia) + EPSILON**2

        return (
            G * nodo.particula.masa
            * diferencia / (distancia2**1.5)
        )

    diferencia = nodo.centro_masa - particula.posicion
    distancia = np.linalg.norm(diferencia)

    # El tamaño corresponde al lado completo del cubo.
    tamaño = 2.0 * nodo.mitad

    # Evitar aproximar nodos que contienen la propia partícula.
    contiene_particula = np.all(
        np.abs(particula.posicion - nodo.centro) <= nodo.mitad
    )

    if (
        theta > 0
        and distancia > 0
        and tamaño / distancia < theta
        and not contiene_particula
    ):
        distancia2 = np.dot(diferencia, diferencia) + EPSILON**2

        return (
            G * nodo.masa_total
            * diferencia / (distancia2**1.5)
        )

    aceleracion = np.zeros(3, dtype=float)

    for hijo in nodo.hijos:
        if hijo is not None:
            aceleracion += calcular_aceleracion_barnes_hut(
                particula, hijo, theta
            )

    return aceleracion


def calcular_todas_barnes_hut(particulas, arbol, theta=THETA):
    aceleraciones = []

    for particula in particulas:
        aceleracion = calcular_aceleracion_barnes_hut(
            particula, arbol, theta
        )
        aceleraciones.append(aceleracion)

    return np.array(aceleraciones)
