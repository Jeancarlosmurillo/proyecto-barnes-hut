
from base.generador import generar_particulas
from base.octree import construir_octree


def contar_nodos(nodo):
    if nodo is None:
        return 0

    total = 1

    for hijo in nodo.hijos:
        if hijo is not None:
            total += contar_nodos(hijo)

    return total


def main():
    print("=== PROYECTO BARNES-HUT ===")

    cantidad = 10

    particulas = generar_particulas(
        cantidad=cantidad,
        semilla=42
    )

    arbol = construir_octree(particulas)

    print(f"\nPartículas generadas: {cantidad}")
    print(f"Masa total del árbol: {arbol.masa_total:.6f}")
    print(f"Centro de masa: {arbol.centro_masa}")
    print(f"Número de nodos: {contar_nodos(arbol)}")

    print("\nOctree construido correctamente.")


if __name__ == "__main__":
    main()
