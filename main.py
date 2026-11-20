
import numpy as np

from base.generador import generar_particulas
from base.integrador import paso_leapfrog


def main():
    print("=== SIMULACIÓN BARNES-HUT ===")

    cantidad = 10
    pasos = 100
    dt = 0.001
    theta = 0.5

    particulas = generar_particulas(
        cantidad=cantidad,
        semilla=42
    )

    posiciones_iniciales = np.array([
        p.posicion.copy() for p in particulas
    ])

    print(f"\nPartículas: {cantidad}")
    print(f"Pasos: {pasos}")
    print(f"Delta tiempo: {dt}")
    print(f"Theta: {theta}")

    for paso in range(1, pasos + 1):
        paso_leapfrog(particulas, dt, theta)

        if paso % 10 == 0:
            print(f"Paso {paso}/{pasos} completado")

    posiciones_finales = np.array([
        p.posicion for p in particulas
    ])

    desplazamientos = np.linalg.norm(
        posiciones_finales - posiciones_iniciales,
        axis=1
    )

    print("\n=== RESULTADOS ===")

    for i, desplazamiento in enumerate(desplazamientos):
        print(
            f"Partícula {i}: "
            f"desplazamiento = {desplazamiento:.8f}"
        )

    print(
        "\nDesplazamiento promedio:",
        f"{np.mean(desplazamientos):.8f}"
    )

    print("\nSimulación finalizada correctamente.")


if __name__ == "__main__":
    main()
