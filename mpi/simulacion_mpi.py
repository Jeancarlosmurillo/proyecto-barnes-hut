
import argparse
import time

import numpy as np
from mpi4py import MPI

from base.generador import generar_particulas
from base.octree import construir_octree
from base.barnes_hut import calcular_aceleracion_barnes_hut
from base.integrador import paso_leapfrog


def calcular_aceleraciones_mpi(comm, particulas, theta):
    """
    El proceso 0 construye el árbol.
    Cada proceso calcula una parte de las aceleraciones.
    El resultado completo se reúne en el proceso 0.
    """
    rank = comm.Get_rank()
    procesos = comm.Get_size()

    if rank == 0:
        arbol = construir_octree(particulas)
        datos = (particulas, arbol)
    else:
        datos = None

    # Compartir estado y árbol actualizados.
    particulas_locales, arbol = comm.bcast(
        datos, root=0
    )

    cantidad = len(particulas_locales)

    inicio = rank * cantidad // procesos
    fin = (rank + 1) * cantidad // procesos

    aceleraciones_locales = np.empty(
        (fin - inicio, 3),
        dtype=np.float64
    )

    for i in range(inicio, fin):
        aceleraciones_locales[i - inicio] = (
            calcular_aceleracion_barnes_hut(
                particulas_locales[i],
                arbol,
                theta
            )
        )

    resultados = comm.gather(
        (inicio, aceleraciones_locales),
        root=0
    )

    if rank != 0:
        return None

    aceleraciones = np.empty(
        (cantidad, 3),
        dtype=np.float64
    )

    for inicio, bloque in resultados:
        fin = inicio + len(bloque)
        aceleraciones[inicio:fin] = bloque

    return aceleraciones


def simular_mpi(comm, particulas, pasos, dt, theta):
    rank = comm.Get_rank()

    # Sincronizar antes de medir.
    comm.Barrier()
    inicio_tiempo = MPI.Wtime()

    # Aceleraciones iniciales.
    aceleraciones = calcular_aceleraciones_mpi(
        comm, particulas, theta
    )

    for paso in range(1, pasos + 1):

        if rank == 0:
            # Primera mitad de velocidad (kick).
            for particula, aceleracion in zip(
                particulas, aceleraciones
            ):
                particula.velocidad += (
                    0.5 * dt * aceleracion
                )

            # Actualización de posición (drift).
            for particula in particulas:
                particula.posicion += (
                    dt * particula.velocidad
                )

        # Todos los procesos participan.
        # El proceso 0 reconstruye el árbol actualizado.
        nuevas_aceleraciones = calcular_aceleraciones_mpi(
            comm, particulas, theta
        )

        if rank == 0:
            # Segunda mitad de velocidad (kick).
            for particula, aceleracion in zip(
                particulas, nuevas_aceleraciones
            ):
                particula.velocidad += (
                    0.5 * dt * aceleracion
                )

            aceleraciones = nuevas_aceleraciones

            if paso % 10 == 0 or paso == pasos:
                print(f"Paso {paso}/{pasos} completado")

    comm.Barrier()
    tiempo_local = MPI.Wtime() - inicio_tiempo

    # Reportar el tiempo del proceso más lento.
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
        parser.error("Parámetros inválidos")

    if rank == 0:
        print("=== BARNES-HUT MPI + LEAPFROG ===")
        print(f"Partículas: {args.particulas}")
        print(f"Procesos MPI: {procesos}")
        print(f"Pasos: {args.pasos}")
        print(f"dt: {args.dt}")
        print(f"Theta: {args.theta}")

        particulas = generar_particulas(
            args.particulas, semilla=42
        )

    else:
        particulas = None

    tiempo_mpi = simular_mpi(
        comm,
        particulas,
        args.pasos,
        args.dt,
        args.theta
    )

    if rank == 0:
        print("\n=== RESULTADOS ===")
        print(f"Tiempo MPI: {tiempo_mpi:.6f} s")

        if args.sin_validacion:
            print("Validación secuencial omitida")
            print("SIMULACION MPI FINALIZADA")
            return

        print("\nValidando contra versión secuencial...")

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
            rtol=1e-9, atol=1e-11
        )

        assert np.allclose(
            vel_mpi, vel_sec,
            rtol=1e-9, atol=1e-11
        )

        print("\nVALIDACION SIMULACION MPI SUPERADA")


if __name__ == "__main__":
    main()
