
from base.generador import generar_particulas
from base.simulacion import ejecutar_simulacion


def main():
    print("=== VALIDACIÓN BARNES-HUT ===")

    particulas = generar_particulas(
        cantidad=10,
        semilla=42
    )

    ejecutar_simulacion(
        particulas,
        pasos=100,
        dt=0.001,
        theta=0.5
    )


if __name__ == "__main__":
    main()
