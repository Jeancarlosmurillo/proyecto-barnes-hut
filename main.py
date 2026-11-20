
from pathlib import Path

from base.generador import generar_particulas, guardar_particulas


def main():
    print("=== PROYECTO BARNES-HUT ===")

    cantidad = 10

    particulas = generar_particulas(
        cantidad=cantidad,
        semilla=42
    )

    print(f"\nSe generaron {len(particulas)} partículas.\n")

    for particula in particulas:
        print(particula)

    ruta = Path("datos") / "particulas_iniciales.txt"

    guardar_particulas(particulas, ruta)


if __name__ == "__main__":
    main()