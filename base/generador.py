import numpy as np

from base.particula import Particula


def generar_particulas(cantidad, semilla=42):
    if cantidad <= 0:
        raise ValueError("La cantidad de partículas debe ser positiva.")

    generador = np.random.default_rng(semilla)
    posiciones = generador.random((cantidad, 3))

    masa = 1.0 / cantidad
    particulas = []

    for i in range(cantidad):
        particula = Particula(
            identificador=i,
            posicion=posiciones[i],
            velocidad=[0.0, 0.0, 0.0],
            masa=masa
        )
        particulas.append(particula)

    return particulas


def guardar_particulas(particulas, ruta):
    datos = np.array([
        [
            *particula.posicion,
            *particula.velocidad,
            particula.masa
        ]
        for particula in particulas
    ])

    np.savetxt(ruta, datos, fmt="%.12f")
    print(f"Se guardaron {len(particulas)} partículas en {ruta}")
