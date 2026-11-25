
import argparse
import time

import numpy as np
from mpi4py import MPI

from base.generador import generar_particulas
from base.octree import construir_octree
from base.barnes_hut import (
    calcular_aceleracion_barnes_hut,
    calcular_todas_barnes_hut
)


def calcular_bloque(particulas, arbol, inicio, fin, theta):
    aceleraciones = np.empty(
        (fin - inicio, 3),
        dtype=np.float64
    )

    for i in range(inicio, fin):
        aceleraciones[i - inicio] = (
            calcular_aceleracion_barnes_hut(
                particulas[i],
                arbol,
                theta
            )
        )

    return aceleraciones


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas",
        type=int,
        default=100
    )

    parser.add_argument(
        "--theta",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    if (
        args.particulas < 1
        or not np.isfinite(args.theta)
        or args.theta < 0
    ):
        parser.error("Parámetros inválidos")

    cantidad = args.particulas

    if rank == 0:
        print("=== BARNES-HUT CON MPI ===")
        print(f"Partículas: {cantidad}")
        print(f"Procesos MPI: {procesos}")
        print(f"Theta: {args.theta}")

    comm.Barrier()
    inicio_total = MPI.Wtime()

    if rank == 0:
        particulas = generar_particulas(
            cantidad,
            semilla=42
        )

        arbol = construir_octree(particulas)

        # Enviar ambos objetos juntos permite mantener
        # las referencias de las partículas del árbol.
        datos = (particulas, arbol)

    else:
        datos = None

    # Distribuir el estado inicial a todos los procesos
    particulas, arbol = comm.bcast(
        datos,
        root=0
    )

    # Cada rank recibe un rango distinto
    inicio = rank * cantidad // procesos
    fin = (rank + 1) * cantidad // procesos

    aceleraciones_locales = calcular_bloque(
        particulas,
        arbol,
        inicio,
        fin,
        args.theta
    )

    # Reunir los bloques en el proceso principal
    resultados = comm.gather(
        (inicio, aceleraciones_locales),
        root=0
    )

    if rank == 0:
        aceleraciones_mpi = np.empty(
            (cantidad, 3),
            dtype=np.float64
        )

        for indice, bloque in resultados:
            aceleraciones_mpi[
                indice:indice + len(bloque)
            ] = bloque

        tiempo_total = MPI.Wtime() - inicio_total

        print("\n=== RESULTADOS MPI ===")
        print(f"Tiempo MPI: {tiempo_total:.6f} s")

        print("\nValidando contra Barnes-Hut secuencial...")

        aceleraciones_sec = calcular_todas_barnes_hut(
            particulas,
            arbol,
            args.theta
        )

        error_maximo = np.max(
            np.abs(aceleraciones_mpi - aceleraciones_sec)
        )

        print(f"Error máximo: {error_maximo:.12e}")

        assert np.allclose(
            aceleraciones_mpi,
            aceleraciones_sec,
            rtol=1e-12,
            atol=1e-12
        )

        print("\nVALIDACION MPI SUPERADA")


if __name__ == "__main__":
    main()
