
import argparse
import time

import numpy as np
from mpi4py import MPI

from base.generador import generar_particulas
from base.octree import construir_octree

from barnes_hut_mpi_plano import (
    convertir_octree_plano,
    distribuir_octree,
    calcular_aceleraciones_plano,
    reunir_aceleraciones
)


def medir_fase(comm, funcion):
    # Todos comienzan la fase al mismo tiempo.
    comm.Barrier()
    inicio = MPI.Wtime()

    resultado = funcion()

    comm.Barrier()
    duracion = MPI.Wtime() - inicio

    # Usamos el mayor tiempo entre los procesos.
    duracion_maxima = comm.reduce(
        duracion, op=MPI.MAX, root=0
    )

    return resultado, duracion_maxima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particulas", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=0.5)
    args = parser.parse_args()

    if args.particulas < 1 or not np.isfinite(args.theta) or args.theta < 0:
        parser.error("Parámetros inválidos")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    particulas = None
    arbol = None
    datos_raiz = None

    if rank == 0:
        particulas = generar_particulas(
            args.particulas, semilla=42
        )

    def construir():
        return construir_octree(particulas) if rank == 0 else None

    arbol, tiempo_arbol = medir_fase(comm, construir)

    def convertir():
        if rank == 0:
            return convertir_octree_plano(particulas, arbol)
        return None

    datos_raiz, tiempo_conversion = medir_fase(comm, convertir)

    datos, tiempo_bcast = medir_fase(
        comm,
        lambda: distribuir_octree(comm, datos_raiz)
    )

    inicio_bloque = rank * args.particulas // procesos
    fin_bloque = (rank + 1) * args.particulas // procesos

    locales, tiempo_calculo = medir_fase(
        comm,
        lambda: calcular_aceleraciones_plano(
            datos,
            theta=args.theta,
            inicio=inicio_bloque,
            fin=fin_bloque
        )
    )

    resultado, tiempo_gather = medir_fase(
        comm,
        lambda: reunir_aceleraciones(
            comm, locales, args.particulas
        )
    )

    if rank == 0:
        tiempos = {
            "Construccion octree": tiempo_arbol,
            "Conversion a NumPy": tiempo_conversion,
            "Comunicacion Bcast": tiempo_bcast,
            "Calculo aceleraciones": tiempo_calculo,
            "Comunicacion Gatherv": tiempo_gather
        }

        print("\n=== PERFIL MPI PLANO ===")
        print(f"Particulas: {args.particulas}")
        print(f"Procesos: {procesos}")
        print(f"Aceleraciones reunidas: {len(resultado)}")

        total = sum(tiempos.values())

        for nombre, duracion in tiempos.items():
            porcentaje = 100 * duracion / total
            print(
                f"{nombre}: {duracion:.6f} s "
                f"({porcentaje:.2f}%)"
            )

        print(f"\nTiempo total medido: {total:.6f} s")


if __name__ == "__main__":
    main()
