
import argparse
import time
import numpy as np

from base.generador import generar_particulas
from base.octree import construir_octree
from barnes_hut_mp import calcular_paralelo


def ejecutar_simulacion_mp(
    particulas, pasos=10, dt=0.001,
    theta=0.5, procesos=4
):
    if pasos < 1 or dt <= 0:
        raise ValueError("Pasos y dt deben ser positivos.")

    inicio = time.perf_counter()

    arbol = construir_octree(particulas)
    aceleraciones = calcular_paralelo(
        particulas, arbol, procesos, theta
    )

    for paso in range(1, pasos + 1):
        # Primera mitad de velocidad
        for p, a in zip(particulas, aceleraciones):
            p.velocidad += 0.5 * dt * a

        # Actualización de posiciones
        for p in particulas:
            p.posicion += dt * p.velocidad

        # Reconstrucción del árbol
        arbol = construir_octree(particulas)

        # Nuevo cálculo paralelo de aceleraciones
        nuevas_aceleraciones = calcular_paralelo(
            particulas, arbol, procesos, theta
        )

        # Segunda mitad de velocidad
        for p, a in zip(particulas, nuevas_aceleraciones):
            p.velocidad += 0.5 * dt * a

        aceleraciones = nuevas_aceleraciones

        print(f"Paso {paso}/{pasos} completado")

    tiempo = time.perf_counter() - inicio

    print(f"\nTiempo total: {tiempo:.6f} segundos")

    return particulas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particulas", type=int, default=100)
    parser.add_argument("--procesos", type=int, default=4)
    parser.add_argument("--pasos", type=int, default=10)

    args = parser.parse_args()

    if min(args.particulas, args.procesos, args.pasos) < 1:
        parser.error("Los valores deben ser positivos.")

    print("=== SIMULACION MULTIPROCESSING ===")

    particulas = generar_particulas(
        args.particulas, semilla=42
    )

    posiciones_iniciales = np.array([
        p.posicion.copy() for p in particulas
    ])

    ejecutar_simulacion_mp(
        particulas,
        pasos=args.pasos,
        procesos=args.procesos
    )

    posiciones_finales = np.array([
        p.posicion for p in particulas
    ])

    desplazamiento = np.linalg.norm(
        posiciones_finales - posiciones_iniciales,
        axis=1
    )

    print(
        "Desplazamiento promedio:",
        f"{np.mean(desplazamiento):.8f}"
    )

    print("Simulación paralela finalizada.")


if __name__ == "__main__":
    main()
