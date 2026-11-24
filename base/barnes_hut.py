
import numpy as np

from base.gravedad import G, EPSILON


THETA = 0.5


def calcular_interaccion(posicion, otra_posicion, masa):
    diferencia = otra_posicion - posicion

    distancia2 = (
        np.dot(diferencia, diferencia)
        + EPSILON ** 2
    )

    return (
        G * masa * diferencia
        / (distancia2 ** 1.5)
    )


def calcular_aceleracion_barnes_hut(
    particula, nodo, theta=THETA
):
    if nodo is None or nodo.masa_total == 0:
        return np.zeros(3, dtype=float)

    if nodo.es_hoja():
        aceleracion = np.zeros(3, dtype=float)

        for otra in nodo.particulas:
            if otra is particula:
                continue

            aceleracion += calcular_interaccion(
                particula.posicion,
                otra.posicion,
                otra.masa
            )

        return aceleracion

    diferencia = (
        nodo.centro_masa - particula.posicion
    )

    distancia = np.linalg.norm(diferencia)
    tamano = 2.0 * nodo.mitad

    if (
        theta > 0
        and distancia > 0
        and tamano / distancia < theta
        and not nodo.contiene(particula.posicion)
    ):
        return calcular_interaccion(
            particula.posicion,
            nodo.centro_masa,
            nodo.masa_total
        )

    aceleracion = np.zeros(3, dtype=float)

    for hijo in nodo.hijos:
        if hijo is not None:
            aceleracion += (
                calcular_aceleracion_barnes_hut(
                    particula, hijo, theta
                )
            )

    return aceleracion


def calcular_todas_barnes_hut(
    particulas, arbol, theta=THETA
):
    return np.array([
        calcular_aceleracion_barnes_hut(
            particula, arbol, theta
        )
        for particula in particulas
    ])
