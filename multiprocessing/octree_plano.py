import math
import numpy as np

from base.generador import generar_particulas
from base.particula import Particula
from base.octree import construir_octree
from base.barnes_hut import calcular_todas_barnes_hut
from base.gravedad import G, EPSILON


def convertir_octree_plano(particulas, raiz):
    """
    Convierte el octree de objetos en arreglos NumPy.

    Cada nodo se identifica con un índice entero.
    El índice -1 representa un hijo inexistente.
    """
    indice_particulas = {
        id(particula): i
        for i, particula in enumerate(particulas)
    }

    nodos = [raiz]
    centros = []
    mitades = []
    masas = []
    centros_masa = []
    hijos = []
    es_hoja = []
    indices_hojas = []
    offsets_hojas = [0]

    # Recorrido por niveles.
    actual = 0
    while actual < len(nodos):
        nodo = nodos[actual]

        centros.append(nodo.centro)
        mitades.append(nodo.mitad)
        masas.append(nodo.masa_total)
        centros_masa.append(nodo.centro_masa)

        fila_hijos = [-1] * 8
        hoja = nodo.es_hoja()
        es_hoja.append(hoja)

        if hoja:
            for particula in nodo.particulas:
                indices_hojas.append(
                    indice_particulas[id(particula)]
                )
        else:
            for j, hijo in enumerate(nodo.hijos):
                if hijo is not None:
                    fila_hijos[j] = len(nodos)
                    nodos.append(hijo)

        hijos.append(fila_hijos)
        offsets_hojas.append(len(indices_hojas))
        actual += 1

    return {
        "posiciones": np.asarray(
            [p.posicion for p in particulas], dtype=np.float64
        ),
        "masas_particulas": np.asarray(
            [p.masa for p in particulas], dtype=np.float64
        ),
        "centros": np.asarray(centros, dtype=np.float64),
        "mitades": np.asarray(mitades, dtype=np.float64),
        "masas": np.asarray(masas, dtype=np.float64),
        "centros_masa": np.asarray(centros_masa, dtype=np.float64),
        "hijos": np.asarray(hijos, dtype=np.int64),
        "es_hoja": np.asarray(es_hoja, dtype=np.bool_),
        "indices_hojas": np.asarray(indices_hojas, dtype=np.int64),
        "offsets_hojas": np.asarray(offsets_hojas, dtype=np.int64),
    }


def calcular_aceleraciones_plano(datos, theta=0.5, inicio=0, fin=None):
    """
    Calcula las aceleraciones para las partículas del rango [inicio, fin).

    Si no se especifica un rango, calcula las de todas las partículas.
    Conserva el orden de recorrido y el criterio del Barnes-Hut original.
    """
    if theta < 0:
        raise ValueError("Theta no puede ser negativo")

    posiciones = datos["posiciones"]
    masas_particulas = datos["masas_particulas"]
    centros = datos["centros"]
    mitades = datos["mitades"]
    masas = datos["masas"]
    centros_masa = datos["centros_masa"]
    hijos = datos["hijos"]
    es_hoja = datos["es_hoja"]
    indices_hojas = datos["indices_hojas"]
    offsets_hojas = datos["offsets_hojas"]

    if fin is None:
        fin = len(posiciones)

    if not (0 <= inicio <= fin <= len(posiciones)):
        raise ValueError("Rango de partículas inválido")

    aceleraciones = np.zeros((fin - inicio, 3), dtype=np.float64)
    epsilon2 = EPSILON ** 2

    for i in range(inicio, fin):
        px, py, pz = posiciones[i]
        ax = ay = az = 0.0
        pendientes = [0]

        while pendientes:
            actual = pendientes.pop()

            if masas[actual] == 0:
                continue

            if es_hoja[actual]:
                comienzo = offsets_hojas[actual]
                termino = offsets_hojas[actual + 1]

                for k in range(comienzo, termino):
                    j = indices_hojas[k]
                    if i == j:
                        continue

                    dx = posiciones[j, 0] - px
                    dy = posiciones[j, 1] - py
                    dz = posiciones[j, 2] - pz

                    distancia2 = dx * dx + dy * dy + dz * dz + epsilon2
                    factor = (
                        G * masas_particulas[j]
                        / (distancia2 * math.sqrt(distancia2))
                    )

                    ax += factor * dx
                    ay += factor * dy
                    az += factor * dz

                continue

            dx = centros_masa[actual, 0] - px
            dy = centros_masa[actual, 1] - py
            dz = centros_masa[actual, 2] - pz

            distancia2 = dx * dx + dy * dy + dz * dz
            distancia = math.sqrt(distancia2)

            cx, cy, cz = centros[actual]
            mitad = mitades[actual]

            contiene = (
                abs(px - cx) <= mitad
                and abs(py - cy) <= mitad
                and abs(pz - cz) <= mitad
            )

            if (
                theta > 0
                and distancia > 0
                and (2.0 * mitad) / distancia < theta
                and not contiene
            ):
                distancia_suavizada2 = distancia2 + epsilon2
                factor = (
                    G * masas[actual]
                    / (
                        distancia_suavizada2
                        * math.sqrt(distancia_suavizada2)
                    )
                )

                ax += factor * dx
                ay += factor * dy
                az += factor * dz
            else:
                # Igual orden de recorrido que el Barnes-Hut original.
                for indice_hijo in hijos[actual]:
                    if indice_hijo != -1:
                        pendientes.append(int(indice_hijo))

        aceleraciones[i - inicio] = [ax, ay, az]

    return aceleraciones


def verificar_caso(nombre, particulas):
    arbol = construir_octree(particulas)
    datos = convertir_octree_plano(particulas, arbol)

    print(f"\nCaso: {nombre}")
    print(f"Particulas: {len(particulas)}")
    print(f"Nodos del octree: {len(datos['masas'])}")

    for theta in (0.0, 0.5, 1.0):
        referencia = calcular_todas_barnes_hut(particulas, arbol, theta)
        resultado = calcular_aceleraciones_plano(datos, theta)
        error_maximo = np.max(np.abs(referencia - resultado))

        assert np.all(np.isfinite(resultado))
        assert np.allclose(
            referencia, resultado, rtol=1e-12, atol=1e-12
        )

        # También comprobar que el cálculo por bloques es equivalente.
        corte = len(particulas) // 2
        por_bloques = np.vstack([
            calcular_aceleraciones_plano(datos, theta, 0, corte),
            calcular_aceleraciones_plano(
                datos, theta, corte, len(particulas)
            ),
        ])
        assert np.array_equal(resultado, por_bloques)

        print(
            f"Theta={theta:.1f} | "
            f"Error maximo={error_maximo:.12e} | OK"
        )


def main():
    print("=== VALIDACION OCTREE PLANO ===")

    verificar_caso(
        "Distribucion normal",
        generar_particulas(100, semilla=42),
    )

    # Incluir partículas coincidentes para comprobar las hojas múltiples.
    particulas_densas = [
        Particula(i, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
        for i in range(5)
    ]
    particulas_densas.append(
        Particula(5, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    )

    verificar_caso("Particulas coincidentes", particulas_densas)
    print("\nVALIDACION OCTREE PLANO SUPERADA")


if __name__ == "__main__":
    main()
