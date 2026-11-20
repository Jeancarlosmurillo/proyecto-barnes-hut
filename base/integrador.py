import numpy as np

from base.octree import construir_octree
from base.barnes_hut import calcular_todas_barnes_hut


def calcular_aceleraciones(particulas, theta):
    arbol = construir_octree(particulas)

    return calcular_todas_barnes_hut(
        particulas,
        arbol,
        theta
    )


def paso_leapfrog(particulas, dt=0.001, theta=0.5):
    if dt <= 0:
        raise ValueError("El paso de tiempo debe ser positivo.")

    if not particulas:
        raise ValueError("La lista de partículas está vacía.")

    # Aceleraciones al inicio del paso
    aceleraciones = calcular_aceleraciones(
        particulas, theta
    )

    # Primera mitad de la actualización de velocidad
    for particula, aceleracion in zip(
        particulas, aceleraciones
    ):
        particula.velocidad += 0.5 * dt * aceleracion

    # Actualizar posiciones
    for particula in particulas:
        particula.posicion += dt * particula.velocidad

    # Reconstruir el octree y calcular nuevas aceleraciones
    nuevas_aceleraciones = calcular_aceleraciones(
        particulas, theta
    )

    # Segunda mitad de la actualización de velocidad
    for particula, aceleracion in zip(
        particulas, nuevas_aceleraciones
    ):
        particula.velocidad += 0.5 * dt * aceleracion

    return particulas
