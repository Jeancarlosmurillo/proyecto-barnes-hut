
import csv
import time
from pathlib import Path

from base.generador import generar_particulas
from base.octree import construir_octree
from base.barnes_hut import calcular_todas_barnes_hut


def medir_rendimiento(cantidad, theta=0.5):
    print(f"\nProbando con {cantidad} partículas...")

    particulas = generar_particulas(
        cantidad=cantidad,
        semilla=42
    )

    inicio = time.perf_counter()
    arbol = construir_octree(particulas)
    tiempo_arbol = time.perf_counter() - inicio

    inicio = time.perf_counter()
    aceleraciones = calcular_todas_barnes_hut(
        particulas, arbol, theta=theta
    )
    tiempo_fuerzas = time.perf_counter() - inicio

    tiempo_total = tiempo_arbol + tiempo_fuerzas

    print(f"Construcción del árbol: {tiempo_arbol:.6f} s")
    print(f"Cálculo de aceleraciones: {tiempo_fuerzas:.6f} s")
    print(f"Tiempo total: {tiempo_total:.6f} s")
    print(f"Aceleraciones calculadas: {len(aceleraciones)}")

    return {
        "particulas": cantidad,
        "theta": theta,
        "tiempo_arbol": tiempo_arbol,
        "tiempo_fuerzas": tiempo_fuerzas,
        "tiempo_total": tiempo_total
    }


def main():
    print("=== RENDIMIENTO BARNES-HUT ===")

    cantidades = [1000]
    resultados = []

    for cantidad in cantidades:
        resultado = medir_rendimiento(cantidad)
        resultados.append(resultado)

    ruta = Path("resultados")
    ruta.mkdir(exist_ok=True)

    archivo = ruta / "rendimiento_secuencial.csv"

    with archivo.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=list(resultados[0].keys())
        )

        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nResultados guardados en: {archivo}")
    print("\nPruebas finalizadas.")


if __name__ == "__main__":
    main()
