import numpy as np

G = 1.0
EPSILON = 0.01


def calcular_aceleracion(particula, particulas):
    aceleracion = np.zeros(3, dtype=float)

    for otra in particulas:
        if particula is otra:
            continue

        diferencia = otra.posicion - particula.posicion

        distancia_cuadrada = (
            np.dot(diferencia, diferencia) + EPSILON ** 2
        )

        inverso_distancia_cubo = (
            distancia_cuadrada ** (-1.5)
        )

        aceleracion += (
            G * otra.masa
            * diferencia
            * inverso_distancia_cubo
        )

    return aceleracion


def calcular_todas_aceleraciones(particulas):
    aceleraciones = []

    for particula in particulas:
        aceleracion = calcular_aceleracion(
            particula,
            particulas
        )

        aceleraciones.append(aceleracion)

    return np.array(aceleraciones)
