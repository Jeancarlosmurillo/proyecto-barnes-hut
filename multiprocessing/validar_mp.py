
import numpy as np

from base.generador import generar_particulas
from base.integrador import paso_leapfrog
from base.simulacion import calcular_energia
from simulacion_mp import ejecutar_simulacion_mp


def obtener_estado(particulas):
    posiciones = np.array([p.posicion for p in particulas])
    velocidades = np.array([p.velocidad for p in particulas])
    return posiciones, velocidades


def main():
    cantidad = 100
    pasos = 100
    dt = 0.001
    theta = 0.5

    print("=== VALIDACION MULTIPROCESSING ===")

    secuenciales = generar_particulas(cantidad, semilla=42)
    paralelas = generar_particulas(cantidad, semilla=42)

    energia_inicial = calcular_energia(paralelas)

    print("\nEjecutando simulacion secuencial...")

    for _ in range(pasos):
        paso_leapfrog(secuenciales, dt, theta)

    print("\nEjecutando simulacion paralela...")

    ejecutar_simulacion_mp(
        paralelas,
        pasos=pasos,
        dt=dt,
        theta=theta,
        procesos=4
    )

    pos_sec, vel_sec = obtener_estado(secuenciales)
    pos_mp, vel_mp = obtener_estado(paralelas)

    error_pos = np.max(np.abs(pos_sec - pos_mp))
    error_vel = np.max(np.abs(vel_sec - vel_mp))

    energia_final = calcular_energia(paralelas)

    error_energia = (
        abs(energia_final - energia_inicial)
        / max(abs(energia_inicial), 1e-15)
    )

    print("\n=== RESULTADOS DE VALIDACION ===")
    print(f"Error maximo posiciones: {error_pos:.12e}")
    print(f"Error maximo velocidades: {error_vel:.12e}")
    print(f"Error relativo energia: {error_energia:.12e}")
    print(f"Error porcentual energia: {error_energia * 100:.8f}%")

    assert np.allclose(pos_sec, pos_mp, rtol=1e-9, atol=1e-11)
    assert np.allclose(vel_sec, vel_mp, rtol=1e-9, atol=1e-11)
    assert np.isfinite(error_energia)

    print("\nVALIDACION DE ESTADOS SUPERADA")


if __name__ == "__main__":
    main()
