
import argparse
import time

import numpy as np
from mpi4py import MPI

from base.generador import generar_particulas
from base.octree import construir_octree
from base.integrador import paso_leapfrog

from barnes_hut_mpi_plano import (
    distribuir_octree,
    reunir_aceleraciones,
    calcular_aceleraciones_plano,
    convertir_octree_plano
)


def calcular_aceleraciones_mpi_plano(
    comm, particulas, theta
):
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    if rank == 0:
        arbol = construir_octree(particulas)
        datos_raiz = convertir_octree_plano(
            particulas, arbol
        )
    else:
        datos_raiz = None

    # Distribuir las posiciones y el octree actualizados.
    datos = distribuir_octree(comm, datos_raiz)

    cantidad = len(datos["posiciones"])

    inicio = rank * cantidad // procesos
    fin = (rank + 1) * cantidad // procesos

    locales = calcular_aceleraciones_plano(
        datos,
        theta=theta,
        inicio=inicio,
        fin=fin
    )

    # En rank 0 queda el arreglo completo.
    aceleraciones = reunir_aceleraciones(
        comm, locales, cantidad
    )

    return aceleraciones


def simular_mpi_plano(
    comm, particulas, pasos=10,
    dt=0.001, theta=0.5
):
    rank = comm.Get_rank()

    comm.Barrier()
    inicio_total = MPI.Wtime()

    aceleraciones = calcular_aceleraciones_mpi_plano(
        comm, particulas, theta
    )

    for paso in range(1, pasos + 1):

        if rank == 0:
            # Kick: primera mitad de velocidad.
            for p, a in zip(particulas, aceleraciones):
                p.velocidad += 0.5 * dt * a

            # Drift: actualizar las posiciones.
            for p in particulas:
                p.posicion += dt * p.velocidad

        # Todos los procesos calculan fuerzas con
        # las posiciones nuevas.
        nuevas = calcular_aceleraciones_mpi_plano(
            comm, particulas, theta
        )

        if rank == 0:
            # Kick: segunda mitad de velocidad.
            for p, a in zip(particulas, nuevas):
                p.velocidad += 0.5 * dt * a

            aceleraciones = nuevas

            if paso % 10 == 0 or paso == pasos:
                print(f"Paso {paso}/{pasos} completado")

    comm.Barrier()

    tiempo_local = MPI.Wtime() - inicio_total

    tiempo_total = comm.reduce(
        tiempo_local,
        op=MPI.MAX,
        root=0
    )

    return tiempo_total


def obtener_estado(particulas):
    posiciones = np.array(
        [p.posicion for p in particulas],
        dtype=np.float64
    )

    velocidades = np.array(
        [p.velocidad for p in particulas],
        dtype=np.float64
    )

    return posiciones, velocidades


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas", type=int, default=100
    )

    parser.add_argument(
        "--pasos", type=int, default=10
    )

    parser.add_argument(
        "--dt", type=float, default=0.001
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
        or args.pasos < 1
        or not np.isfinite(args.dt)
        or args.dt <= 0
        or not np.isfinite(args.theta)
        or args.theta < 0
    ):
        parser.error("Parametros invalidos")

    if rank == 0:
        print("=== BARNES-HUT MPI PLANO + LEAPFROG ===")
        print(f"Particulas: {args.particulas}")
        print(f"Procesos MPI: {procesos}")
        print(f"Pasos: {args.pasos}")
        print(f"dt: {args.dt}")
        print(f"Theta: {args.theta}")

        particulas = generar_particulas(
            args.particulas, semilla=42
        )
    else:
        particulas = None

    tiempo_mpi = simular_mpi_plano(
        comm,
        particulas,
        pasos=args.pasos,
        dt=args.dt,
        theta=args.theta
    )

    if rank != 0:
        return

    print("\n=== RESULTADOS ===")
    print(f"Tiempo MPI plano: {tiempo_mpi:.6f} s")

    if args.sin_validacion:
        print("Validacion secuencial omitida")
        print("SIMULACION MPI PLANA FINALIZADA")
        return

    print("\nValidando contra version secuencial...")

    secuenciales = generar_particulas(
        args.particulas, semilla=42
    )

    inicio_sec = time.perf_counter()

    for _ in range(args.pasos):
        paso_leapfrog(
            secuenciales,
            args.dt,
            args.theta
        )

    tiempo_sec = time.perf_counter() - inicio_sec

    pos_mpi, vel_mpi = obtener_estado(particulas)
    pos_sec, vel_sec = obtener_estado(secuenciales)

    error_pos = np.max(
        np.abs(pos_mpi - pos_sec)
    )

    error_vel = np.max(
        np.abs(vel_mpi - vel_sec)
    )

    print(f"Tiempo secuencial: {tiempo_sec:.6f} s")
    print(f"Error posiciones: {error_pos:.12e}")
    print(f"Error velocidades: {error_vel:.12e}")

    assert np.allclose(
        pos_mpi, pos_sec,
        rtol=1e-9,
        atol=1e-11
    )

    assert np.allclose(
        vel_mpi, vel_sec,
        rtol=1e-9,
        atol=1e-11
    )

    print("\nVALIDACION SIMULACION MPI PLANA SUPERADA")


if __name__ == "__main__":
    main()
