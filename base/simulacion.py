
import numpy as np

from base.gravedad import G, EPSILON
from base.integrador import paso_leapfrog


def calcular_energia(particulas):
    energia_cinetica = 0.0
    energia_potencial = 0.0

    for particula in particulas:
        velocidad2 = np.dot(
            particula.velocidad,
            particula.velocidad
        )

        energia_cinetica += (
            0.5 * particula.masa * velocidad2
        )

    for i in range(len(particulas)):
        for j in range(i + 1, len(particulas)):
            diferencia = (
                particulas[j].posicion
                - particulas[i].posicion
            )

            distancia = np.sqrt(
                np.dot(diferencia, diferencia)
                + EPSILON ** 2
            )

            energia_potencial -= (
                G * particulas[i].masa
                * particulas[j].masa
                / distancia
            )

    return energia_cinetica + energia_potencial


def ejecutar_simulacion(particulas, pasos=100,
                       dt=0.001, theta=0.5):

    energia_inicial = calcular_energia(particulas)

    print(f"Energía inicial: {energia_inicial:.10f}")

    for paso in range(1, pasos + 1):
        paso_leapfrog(particulas, dt, theta)

        if paso % 10 == 0:
            print(f"Paso {paso}/{pasos}")

    energia_final = calcular_energia(particulas)

    error_relativo = (
        abs(energia_final - energia_inicial)
        / max(abs(energia_inicial), 1e-15)
    )

    print(f"\nEnergía final: {energia_final:.10f}")
    print(f"Error relativo de energía: {error_relativo:.8f}")
    print(f"Error porcentual: {error_relativo * 100:.6f}%")

    return error_relativo
