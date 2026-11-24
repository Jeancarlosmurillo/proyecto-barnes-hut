
import argparse
import time
import numpy as np

from multiprocessing import get_context, shared_memory

from base.generador import generar_particulas
from base.octree import construir_octree
from octree_plano import (
    convertir_octree_plano,
    calcular_aceleraciones_plano
)


def trabajador(tarea):
    metadatos, inicio, fin, theta = tarea

    memorias = []
    datos = {}

    try:
        # Conectar con los arreglos compartidos.
        for nombre, (identificador, forma, tipo) in (
            metadatos.items()
        ):
            memoria = shared_memory.SharedMemory(
    name=identificador,
    track=False
)

            memorias.append(memoria)

            datos[nombre] = np.ndarray(
                shape=tuple(forma),
                dtype=np.dtype(tipo),
                buffer=memoria.buf
            )

        # Cada proceso calcula su propio bloque.
        aceleraciones = calcular_aceleraciones_plano(
            datos,
            theta=theta,
            inicio=inicio,
            fin=fin
        )

        return inicio, aceleraciones.copy()

    finally:
        for memoria in memorias:
            memoria.close()


def calcular_compartido(pool, datos, procesos, theta=0.5):
    cantidad = len(datos["posiciones"])

    if procesos < 1:
        raise ValueError("Procesos debe ser positivo")

    if cantidad < 1:
        raise ValueError("No hay partículas")

    memorias = []
    metadatos = {}

    try:
        # Crear los bloques de memoria compartida.
        for nombre, arreglo in datos.items():
            arreglo = np.ascontiguousarray(arreglo)

            memoria = shared_memory.SharedMemory(
                create=True,
                size=max(1, arreglo.nbytes)
            )

            memorias.append(memoria)

            compartido = np.ndarray(
                shape=arreglo.shape,
                dtype=arreglo.dtype,
                buffer=memoria.buf
            )

            compartido[:] = arreglo

            metadatos[nombre] = (
                memoria.name,
                arreglo.shape,
                arreglo.dtype.str
            )

        # Distribuir las partículas entre procesos.
        tareas = []

        for i in range(procesos):
            inicio = i * cantidad // procesos
            fin = (i + 1) * cantidad // procesos

            if inicio < fin:
                tareas.append(
                    (metadatos, inicio, fin, theta)
                )

        resultados = pool.map(trabajador, tareas)

        aceleraciones = np.empty(
            (cantidad, 3),
            dtype=np.float64
        )

        for inicio, bloque in resultados:
            fin = inicio + len(bloque)
            aceleraciones[inicio:fin] = bloque

        return aceleraciones

    finally:
        # Liberar memoria incluso si ocurre un error.
        for memoria in memorias:
            try:
                memoria.unlink()
            finally:
                memoria.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--particulas", type=int, default=100
    )

    parser.add_argument(
        "--procesos", type=int, default=4
    )

    parser.add_argument(
        "--theta", type=float, default=0.5
    )

    args = parser.parse_args()

    if args.particulas < 1 or args.procesos < 1:
        parser.error("Partículas y procesos deben ser positivos")

    if args.theta < 0:
        parser.error("Theta no puede ser negativo")

    print("=== BARNES-HUT MEMORIA COMPARTIDA ===")

    particulas = generar_particulas(
        args.particulas,
        semilla=42
    )

    arbol = construir_octree(particulas)
    datos = convertir_octree_plano(particulas, arbol)

    print(f"Partículas: {args.particulas}")
    print(f"Procesos: {args.procesos}")
    print(f"Nodos: {len(datos['masas'])}")

    contexto = get_context("fork")

    with contexto.Pool(args.procesos) as pool:
        inicio = time.perf_counter()

        aceleraciones_mp = calcular_compartido(
            pool,
            datos,
            args.procesos,
            args.theta
        )

        tiempo = time.perf_counter() - inicio

    # Calcular referencia sin paralelización.
    aceleraciones_sec = calcular_aceleraciones_plano(
        datos,
        theta=args.theta
    )

    error_maximo = np.max(
        np.abs(aceleraciones_mp - aceleraciones_sec)
    )

    print("\n=== RESULTADOS ===")
    print(f"Tiempo compartido: {tiempo:.6f} s")
    print(f"Error máximo: {error_maximo:.12e}")

    assert np.allclose(
        aceleraciones_mp,
        aceleraciones_sec,
        rtol=1e-12,
        atol=1e-12
    )

    print("\nVALIDACION MEMORIA COMPARTIDA SUPERADA")


if __name__ == "__main__":
    main()
