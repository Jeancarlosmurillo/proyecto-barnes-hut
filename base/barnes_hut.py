
import math
import numpy as np

from base.gravedad import G, EPSILON


THETA = 0.5
EPSILON2 = EPSILON ** 2


def calcular_aceleracion_barnes_hut(
    particula, nodo, theta=THETA
):
    if theta < 0:
        raise ValueError("Theta no puede ser negativo")

    if nodo is None or nodo.masa_total == 0:
        return np.zeros(3, dtype=float)

    px, py, pz = particula.posicion
    ax = ay = az = 0.0

    # Recorrido iterativo del octree
    pendientes = [nodo]

    while pendientes:
        actual = pendientes.pop()

        if actual.masa_total == 0:
            continue

        # Calcular interacciones directas en las hojas
        if actual.es_hoja():
            for otra in actual.particulas:
                if otra is particula:
                    continue

                dx = otra.posicion[0] - px
                dy = otra.posicion[1] - py
                dz = otra.posicion[2] - pz

                distancia2 = (
                    dx * dx + dy * dy + dz * dz
                    + EPSILON2
                )

                factor = (
                    G * otra.masa
                    / (distancia2 * math.sqrt(distancia2))
                )

                ax += factor * dx
                ay += factor * dy
                az += factor * dz

            continue

        dx = actual.centro_masa[0] - px
        dy = actual.centro_masa[1] - py
        dz = actual.centro_masa[2] - pz

        distancia2 = dx * dx + dy * dy + dz * dz
        distancia = math.sqrt(distancia2)

        cx, cy, cz = actual.centro
        mitad = actual.mitad

        contiene = (
            abs(px - cx) <= mitad
            and abs(py - cy) <= mitad
            and abs(pz - cz) <= mitad
        )

        # Aproximación gravitacional de Barnes-Hut
        if (
            theta > 0
            and distancia > 0
            and (2.0 * mitad) / distancia < theta
            and not contiene
        ):
            distancia_suavizada2 = distancia2 + EPSILON2

            factor = (
                G * actual.masa_total
                / (
                    distancia_suavizada2
                    * math.sqrt(distancia_suavizada2)
                )
            )

            ax += factor * dx
            ay += factor * dy
            az += factor * dz

        else:
            # Examinar los nodos hijos
            for hijo in actual.hijos:
                if hijo is not None:
                    pendientes.append(hijo)

    return np.array([ax, ay, az], dtype=float)


def calcular_todas_barnes_hut(
    particulas, arbol, theta=THETA
):
    aceleraciones = np.empty(
        (len(particulas), 3),
        dtype=float
    )

    for i, particula in enumerate(particulas):
        aceleraciones[i] = calcular_aceleracion_barnes_hut(
            particula, arbol, theta
        )

    return aceleraciones
