
import argparse
import time
import numpy as np

from multiprocessing import get_context

from base.generador import generar_particulas
from base.octree import construir_octree
from base.barnes_hut import calcular_aceleracion_barnes_hut
from base.integrador import paso_leapfrog


def procesar_bloque(tarea):
    particulas, arbol, theta, inicio, fin = tarea

    aceleraciones = np.empty((fin - inicio, 3))

    for i in range(inicio, fin):
        aceleraciones[i - inicio] = (
            calcular_aceleracion_barnes_hut(
                particulas[i], arbol, theta
            )
        )

    return inicio, aceleraciones


def calcular_paralelo(pool, particulas, arbol,
                      procesos, theta):

    cantidad = len(particulas)

    tareas = []

    for i in range(procesos):
        inicio = i * cantidad // procesos
        fin = (i + 1) * cantidad // procesos

        if inicio < fin:
            tareas.append((
                particulas, arbol, theta, inicio, fin
            ))

    resultados = pool.map(procesar_bloque, tareas)

    aceleraciones = np.empty((cantidad, 3))

    for inicio, bloque in resultados:
        fin = inicio + len(bloque)
        aceleraciones[inicio:fin] = bloque

    return aceleraciones


def simular(particulas, pasos=100,
            procesos=4, dt=0.001, theta=0.5):

    if not particulas or pasos < 1 or procesos < 1:
        raise ValueError("Parámetros de simulación inválidos")

    if dt <= 0 or theta < 0:
        raise ValueError("dt o theta inválido")

    contexto = get_context("fork")

    inicio_total = time.perf_counter()

    # Los procesos se crean una sola vez.
    with contexto.Pool(procesos) as pool:

        arbol = construir_octree(particulas)

        aceleraciones = calcular_paralelo(
            pool, particulas, arbol, procesos, theta
        )

        for paso in range(1, pasos + 1):

            # Kick: primera mitad de velocidad
            for p, a in zip(particulas, aceleraciones):
                p.velocidad += 0.5 * dt * a

            # Drift: actualizar posiciones
            for p in particulas:
                p.posicion += dt * p.velocidad

            # Reconstruir octree
            arbol = construir_octree(particulas)

            # Calcular aceleraciones con procesos activos
            nuevas = calcular_paralelo(
                pool, particulas, arbol,
                procesos, theta
            )

            # Kick: segunda mitad de velocidad
            for p, a in zip(particulas, nuevas):
                p.velocidad += 0.5 * dt * a

            aceleraciones = nuevas

            if paso % 10 == 0:
                print(f"Paso {paso}/{pasos} completado")

    tiempo_total = time.perf_counter() - inicio_total

    return tiempo_total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--particulas", type=int, default=100)
    parser.add_argument("--procesos", type=int, default=4)
    parser.add_argument("--pasos", type=int, default=100)

    args = parser.parse_args()

    if min(args.particulas, args.procesos, args.pasos) < 1:
        parser.error("Los parámetros deben ser positivos")

    print("=== BARNES-HUT PROCESOS PERSISTENTES ===")

    paralelas = generar_particulas(
        args.particulas, semilla=42
    )

    secuenciales = generar_particulas(
        args.particulas, semilla=42
    )

    print(f"Partículas: {args.particulas}")
    print(f"Procesos: {args.procesos}")
    print(f"Pasos: {args.pasos}")

    tiempo = simular(
        paralelas,
        pasos=args.pasos,
        procesos=args.procesos
    )

    print("\nValidando contra versión secuencial...")

    for _ in range(args.pasos):
        paso_leapfrog(secuenciales, 0.001, 0.5)

    pos_mp = np.array([p.posicion for p in paralelas])
    pos_sec = np.array([p.posicion for p in secuenciales])

    vel_mp = np.array([p.velocidad for p in paralelas])
    vel_sec = np.array([p.velocidad for p in secuenciales])

    error_pos = np.max(np.abs(pos_mp - pos_sec))
    error_vel = np.max(np.abs(vel_mp - vel_sec))

    print("\n=== RESULTADOS ===")
    print(f"Tiempo paralelo: {tiempo:.6f} s")
    print(f"Error posiciones: {error_pos:.12e}")
    print(f"Error velocidades: {error_vel:.12e}")

    assert np.allclose(pos_mp, pos_sec, rtol=1e-9, atol=1e-11)
    assert np.allclose(vel_mp, vel_sec, rtol=1e-9, atol=1e-11)

    print("\nVALIDACION SUPERADA")


if __name__ == "__main__":
    main()
