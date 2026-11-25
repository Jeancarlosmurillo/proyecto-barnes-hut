
import argparse
import os
import sys
import time

import numpy as np
from mpi4py import MPI

from base.generador import generar_particulas
from base.octree import construir_octree

# Permitir importar el octree plano existente
RUTA_MP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "multiprocessing")
)

if RUTA_MP not in sys.path:
    sys.path.insert(0, RUTA_MP)

from octree_plano import (
    convertir_octree_plano,
    calcular_aceleraciones_plano
)


def obtener_tipo_mpi(arreglo):
    """Relacionar los tipos NumPy con tipos MPI."""

    tipo = arreglo.dtype

    if tipo == np.dtype("float64"):
        return MPI.DOUBLE

    if tipo == np.dtype("int64"):
        return MPI.INT64_T

    if tipo == np.dtype("uint8"):
        return MPI.UNSIGNED_CHAR

    raise TypeError(f"Tipo NumPy no soportado: {tipo}")


def distribuir_octree(comm, datos_raiz):
    """
    El proceso 0 distribuye los arreglos planos.
    Los demás procesos reservan sus propios arreglos.
    """
    rank = comm.Get_rank()

    if rank == 0:
        datos_raiz = {
            nombre: np.ascontiguousarray(
                arreglo.astype(np.uint8)
                if arreglo.dtype == np.dtype("bool")
                else arreglo
            )
            for nombre, arreglo in datos_raiz.items()
        }

        metadatos = {
            nombre: (arreglo.shape, arreglo.dtype.str)
            for nombre, arreglo in datos_raiz.items()
        }
    else:
        metadatos = None

    # Solo los metadatos pequeños utilizan bcast.
    metadatos = comm.bcast(metadatos, root=0)

    datos = {}

    for nombre, (forma, tipo) in metadatos.items():
        if rank == 0:
            arreglo = datos_raiz[nombre]
        else:
            arreglo = np.empty(
                forma,
                dtype=np.dtype(tipo)
            )

        # Transferencia directa de datos NumPy.
        comm.Bcast(
            [arreglo, obtener_tipo_mpi(arreglo)],
            root=0
        )

        datos[nombre] = arreglo

    return datos


def reunir_aceleraciones(comm, locales, cantidad):
    """Reunir bloques de aceleraciones mediante Gatherv."""

    rank = comm.Get_rank()
    procesos = comm.Get_size()

    # Cada partícula tiene tres aceleraciones.
    cantidades = [
        3 * (
            (r + 1) * cantidad // procesos
            - r * cantidad // procesos
        )
        for r in range(procesos)
    ]

    desplazamientos = [0]

    for cantidad_local in cantidades[:-1]:
        desplazamientos.append(
            desplazamientos[-1] + cantidad_local
        )

    enviar = np.ascontiguousarray(
        locales.ravel(),
        dtype=np.float64
    )

    if rank == 0:
        resultado = np.empty(
            (cantidad, 3),
            dtype=np.float64
        )

        recibir = [
            resultado.ravel(),
            cantidades,
            desplazamientos,
            MPI.DOUBLE
        ]
    else:
        resultado = None
        recibir = None

    comm.Gatherv(
        [enviar, MPI.DOUBLE],
        recibir,
        root=0
    )

    return resultado


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas", type=int, default=100
    )

    parser.add_argument(
        "--theta", type=float, default=0.5
    )

    parser.add_argument(
        "--sin-validacion",
        action="store_true"
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
        print("=== BARNES-HUT MPI CON OCTREE PLANO ===")
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

        datos_raiz = convertir_octree_plano(
            particulas,
            arbol
        )

        numero_nodos = len(datos_raiz["masas"])

    else:
        datos_raiz = None
        numero_nodos = None

    # Enviar las estructuras NumPy a todos los ranks.
    datos = distribuir_octree(
        comm,
        datos_raiz
    )

    # Distribuir la carga de trabajo.
    inicio = rank * cantidad // procesos
    fin = (rank + 1) * cantidad // procesos

    aceleraciones_locales = calcular_aceleraciones_plano(
        datos,
        theta=args.theta,
        inicio=inicio,
        fin=fin
    )

    # Reunir todas las aceleraciones en rank 0.
    aceleraciones_mpi = reunir_aceleraciones(
        comm,
        aceleraciones_locales,
        cantidad
    )

    comm.Barrier()
    tiempo_local = MPI.Wtime() - inicio_total

    tiempo_total = comm.reduce(
        tiempo_local,
        op=MPI.MAX,
        root=0
    )

    if rank == 0:
        print(f"Nodos del octree: {numero_nodos}")

        print("\n=== RESULTADOS ===")
        print(f"Tiempo MPI plano: {tiempo_total:.6f} s")

        if args.sin_validacion:
            print("Validación secuencial omitida")
            print("EJECUCION MPI PLANA FINALIZADA")
            return

        print("\nValidando contra cálculo secuencial...")

        aceleraciones_sec = calcular_aceleraciones_plano(
            datos_raiz,
            theta=args.theta
        )

        error = np.max(
            np.abs(aceleraciones_mpi - aceleraciones_sec)
        )

        print(f"Error máximo: {error:.12e}")

        assert np.allclose(
            aceleraciones_mpi,
            aceleraciones_sec,
            rtol=1e-12,
            atol=1e-12
        )

        print("\nVALIDACION MPI PLANO SUPERADA")


if __name__ == "__main__":
    main()
