
import argparse
import time
import numpy as np
from multiprocessing import get_context

from base.generador import generar_particulas
from base.octree import construir_octree
from base.barnes_hut import (
    calcular_aceleracion_barnes_hut,
    calcular_todas_barnes_hut
)


# Datos heredados por los procesos mediante fork.
_PARTICULAS = None
_ARBOL = None
_THETA = 0.5


def procesar_bloque(limites):
    inicio, fin = limites

    aceleraciones = np.empty(
        (fin - inicio, 3),
        dtype=float
    )

    for i in range(inicio, fin):
        aceleraciones[i - inicio] = (
            calcular_aceleracion_barnes_hut(
                _PARTICULAS[i],
                _ARBOL,
                _THETA
            )
        )

    return inicio, aceleraciones


def calcular_paralelo(particulas, arbol, procesos=4, theta=0.5):
    global _PARTICULAS, _ARBOL, _THETA

    if procesos < 1:
        raise ValueError("El número de procesos debe ser positivo.")

    if not particulas:
        raise ValueError("No hay partículas.")

    if procesos == 1:
        return calcular_todas_barnes_hut(
            particulas, arbol, theta
        )

    _PARTICULAS = particulas
    _ARBOL = arbol
    _THETA = theta

    cantidad = len(particulas)

    numero_bloques = min(cantidad, procesos * 4)
    limites = np.linspace(
        0, cantidad, numero_bloques + 1,
        dtype=int
    )

    bloques = [
        (int(limites[i]), int(limites[i + 1]))
        for i in range(numero_bloques)
    ]

    contexto = get_context("fork")

    try:
        with contexto.Pool(processes=procesos) as pool:
            resultados = pool.map(procesar_bloque, bloques)
    finally:
        _PARTICULAS = None
        _ARBOL = None

    aceleraciones = np.empty(
        (cantidad, 3),
        dtype=float
    )

    for inicio, bloque in resultados:
        fin = inicio + len(bloque)
        aceleraciones[inicio:fin] = bloque

    return aceleraciones


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas", type=int, default=1000
    )

    parser.add_argument(
        "--procesos", type=int, default=4
    )

    parser.add_argument(
        "--theta", type=float, default=0.5
    )

    args = parser.parse_args()

    if args.particulas < 1 or args.procesos < 1:
        parser.error("Partículas y procesos deben ser positivos.")

    if args.theta < 0:
        parser.error("Theta no puede ser negativo.")

    print("=== BARNES-HUT MULTIPROCESSING ===")
    print(f"Partículas: {args.particulas}")
    print(f"Procesos: {args.procesos}")

    particulas = generar_particulas(
        args.particulas, semilla=42
    )

    inicio_arbol = time.perf_counter()
    arbol = construir_octree(particulas)
    tiempo_arbol = time.perf_counter() - inicio_arbol

    inicio = time.perf_counter()

    aceleraciones = calcular_paralelo(
        particulas,
        arbol,
        args.procesos,
        args.theta
    )

    tiempo_fuerzas = time.perf_counter() - inicio

    print(f"Construcción: {tiempo_arbol:.6f} s")
    print(f"Aceleraciones: {tiempo_fuerzas:.6f} s")
    print(f"Tiempo total: {tiempo_arbol + tiempo_fuerzas:.6f} s")

    # Comprobar el resultado con el método secuencial.
    referencia = calcular_todas_barnes_hut(
        particulas, arbol, args.theta
    )

    error = np.linalg.norm(
        aceleraciones - referencia
    ) / max(np.linalg.norm(referencia), 1e-15)

    print(f"Error relativo: {error:.10e}")
    print("Prueba finalizada.")


if __name__ == "__main__":
    main()
