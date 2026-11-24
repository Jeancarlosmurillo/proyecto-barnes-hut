
import argparse
import time

import numpy as np
from multiprocessing import get_context

from base.generador import generar_particulas
from base.octree import construir_octree
from base.integrador import paso_leapfrog

from octree_plano import convertir_octree_plano
from barnes_hut_compartido import calcular_compartido


def calcular_aceleraciones(pool, particulas, procesos, theta):
    # Reconstruir el octree con las posiciones actuales.
    arbol = construir_octree(particulas)

    # Convertir el árbol a arreglos planos.
    datos = convertir_octree_plano(particulas, arbol)

    # Calcular aceleraciones mediante memoria compartida.
    return calcular_compartido(
        pool,
        datos,
        procesos,
        theta
    )


def simular_compartida(
    particulas,
    pasos=100,
    procesos=4,
    dt=0.001,
    theta=0.5
):
    if not particulas:
        raise ValueError("No hay partículas")

    if pasos < 1 or procesos < 1 or dt <= 0 or theta < 0:
        raise ValueError("Parámetros inválidos")

    contexto = get_context("fork")

    inicio_total = time.perf_counter()

    # Crear los procesos una sola vez.
    with contexto.Pool(processes=procesos) as pool:

        aceleraciones = calcular_aceleraciones(
            pool, particulas, procesos, theta
        )

        for paso in range(1, pasos + 1):

            # Primera mitad de la velocidad (kick).
            for particula, aceleracion in zip(
                particulas, aceleraciones
            ):
                particula.velocidad += (
                    0.5 * dt * aceleracion
                )

            # Actualizar posiciones (drift).
            for particula in particulas:
                particula.posicion += (
                    dt * particula.velocidad
                )

            # Reconstruir el árbol y calcular
            # las nuevas aceleraciones.
            nuevas_aceleraciones = calcular_aceleraciones(
                pool, particulas, procesos, theta
            )

            # Segunda mitad de la velocidad (kick).
            for particula, aceleracion in zip(
                particulas, nuevas_aceleraciones
            ):
                particula.velocidad += (
                    0.5 * dt * aceleracion
                )

            aceleraciones = nuevas_aceleraciones

            if paso % 10 == 0 or paso == pasos:
                print(
                    f"Paso {paso}/{pasos} completado"
                )

    tiempo_total = time.perf_counter() - inicio_total

    return tiempo_total


def obtener_estado(particulas):
    posiciones = np.array(
        [p.posicion for p in particulas]
    )

    velocidades = np.array(
        [p.velocidad for p in particulas]
    )

    return posiciones, velocidades


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas", type=int, default=100
    )

    parser.add_argument(
        "--procesos", type=int, default=4
    )

    parser.add_argument(
        "--pasos", type=int, default=100
    )

    parser.add_argument(
        "--dt", type=float, default=0.001
    )

    parser.add_argument(
        "--theta", type=float, default=0.5
    )

    args = parser.parse_args()

    if (
        args.particulas < 1
        or args.procesos < 1
        or args.pasos < 1
        or args.dt <= 0
        or args.theta < 0
    ):
        parser.error("Los parámetros no son válidos")

    print("=== BARNES-HUT MEMORIA COMPARTIDA + LEAPFROG ===")

    paralelas = generar_particulas(
        args.particulas, semilla=42
    )

    secuenciales = generar_particulas(
        args.particulas, semilla=42
    )

    print(f"Partículas: {args.particulas}")
    print(f"Procesos: {args.procesos}")
    print(f"Pasos: {args.pasos}")

    tiempo_mp = simular_compartida(
        paralelas,
        pasos=args.pasos,
        procesos=args.procesos,
        dt=args.dt,
        theta=args.theta
    )

    print("\nValidando contra la versión secuencial...")

    inicio_sec = time.perf_counter()

    for _ in range(args.pasos):
        paso_leapfrog(
            secuenciales,
            args.dt,
            args.theta
        )

    tiempo_sec = time.perf_counter() - inicio_sec

    pos_mp, vel_mp = obtener_estado(paralelas)
    pos_sec, vel_sec = obtener_estado(secuenciales)

    error_pos = np.max(
        np.abs(pos_mp - pos_sec)
    )

    error_vel = np.max(
        np.abs(vel_mp - vel_sec)
    )

    print("\n=== RESULTADOS ===")
    print(f"Tiempo paralelo: {tiempo_mp:.6f} s")
    print(f"Tiempo secuencial: {tiempo_sec:.6f} s")
    print(f"Error posiciones: {error_pos:.12e}")
    print(f"Error velocidades: {error_vel:.12e}")

    assert np.allclose(
        pos_mp, pos_sec,
        rtol=1e-9, atol=1e-11
    )

    assert np.allclose(
        vel_mp, vel_sec,
        rtol=1e-9, atol=1e-11
    )

    print("\nVALIDACION SIMULACION COMPARTIDA SUPERADA")


if __name__ == "__main__":
    main()
