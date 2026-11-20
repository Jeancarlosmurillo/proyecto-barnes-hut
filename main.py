
import numpy as np

from base.generador import generar_particulas
from base.gravedad import calcular_todas_aceleraciones


def main():
    print("=== PROYECTO BARNES-HUT ===")

    cantidad = 10
    particulas = generar_particulas(cantidad, semilla=42)

    aceleraciones = calcular_todas_aceleraciones(particulas)

    print(f"\nPartículas generadas: {cantidad}")
    print("\nAceleraciones gravitacionales:\n")

    for i, aceleracion in enumerate(aceleraciones):
        print(
            f"Partícula {i}: "
            f"ax={aceleracion[0]:.6f}, "
            f"ay={aceleracion[1]:.6f}, "
            f"az={aceleracion[2]:.6f}"
        )

    print("\nCálculo gravitacional completado correctamente.")


if __name__ == "__main__":
    main()
