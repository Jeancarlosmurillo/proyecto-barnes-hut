
#include "octree.h"

#include <math.h>
#include <stdlib.h>


static NodoOctree *crear_nodo(
    const double centro[3],
    double mitad,
    int profundidad
) {
    NodoOctree *nodo = calloc(1, sizeof(NodoOctree));

    if (nodo == NULL) {
        return NULL;
    }

    for (int eje = 0; eje < 3; eje++) {
        nodo->centro[eje] = centro[eje];
    }

    nodo->mitad = mitad;
    nodo->profundidad = profundidad;

    return nodo;
}


static int es_hoja(const NodoOctree *nodo) {
    for (int i = 0; i < 8; i++) {
        if (nodo->hijos[i] != NULL) {
            return 0;
        }
    }

    return 1;
}


static int guardar_indice(NodoOctree *nodo, int indice) {
    if (nodo->cantidad == nodo->capacidad) {
        int nueva_capacidad = nodo->capacidad == 0
            ? 4
            : nodo->capacidad * 2;

        int *nuevos = realloc(
            nodo->indices,
            (size_t)nueva_capacidad * sizeof(int)
        );

        if (nuevos == NULL) {
            return 0;
        }

        nodo->indices = nuevos;
        nodo->capacidad = nueva_capacidad;
    }

    nodo->indices[nodo->cantidad++] = indice;
    return 1;
}


static int obtener_indice_hijo(
    const NodoOctree *nodo,
    const Particula *particula
) {
    int indice = 0;

    for (int eje = 0; eje < 3; eje++) {
        if (particula->posicion[eje] >= nodo->centro[eje]) {
            indice |= 1 << eje;
        }
    }

    return indice;
}


static int insertar(
    NodoOctree *nodo,
    const Particula *particulas,
    int indice
);


static int insertar_en_hijo(
    NodoOctree *nodo,
    const Particula *particulas,
    int indice_particula
) {
    const Particula *p = &particulas[indice_particula];

    int indice_hijo = obtener_indice_hijo(nodo, p);

    if (nodo->hijos[indice_hijo] == NULL) {
        double centro[3];
        double desplazamiento = nodo->mitad / 2.0;

        for (int eje = 0; eje < 3; eje++) {
            int positivo = (indice_hijo & (1 << eje)) != 0;

            centro[eje] = nodo->centro[eje]
                + (positivo ? desplazamiento : -desplazamiento);
        }

        nodo->hijos[indice_hijo] = crear_nodo(
            centro,
            desplazamiento,
            nodo->profundidad + 1
        );

        if (nodo->hijos[indice_hijo] == NULL) {
            return 0;
        }
    }

    return insertar(
        nodo->hijos[indice_hijo],
        particulas,
        indice_particula
    );
}


static int insertar(
    NodoOctree *nodo,
    const Particula *particulas,
    int indice
) {
    const Particula *p = &particulas[indice];

    double masa_anterior = nodo->masa_total;
    double masa_nueva = masa_anterior + p->masa;

    for (int eje = 0; eje < 3; eje++) {
        nodo->centro_masa[eje] = (
            nodo->centro_masa[eje] * masa_anterior
            + p->posicion[eje] * p->masa
        ) / masa_nueva;
    }

    nodo->masa_total = masa_nueva;

    if (es_hoja(nodo)) {
        if (
            nodo->profundidad >= MAX_PROFUNDIDAD
            || nodo->mitad <= MIN_MITAD
            || nodo->cantidad == 0
        ) {
            return guardar_indice(nodo, indice);
        }

        // Subdividir y trasladar las particulas
        // que estaban almacenadas en esta hoja.
        int cantidad_anterior = nodo->cantidad;

        for (int i = 0; i < cantidad_anterior; i++) {
            if (!insertar_en_hijo(
                nodo, particulas, nodo->indices[i]
            )) {
                return 0;
            }
        }

        free(nodo->indices);
        nodo->indices = NULL;
        nodo->cantidad = 0;
        nodo->capacidad = 0;
    }

    return insertar_en_hijo(nodo, particulas, indice);
}


void liberar_octree(NodoOctree *nodo) {
    if (nodo == NULL) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        liberar_octree(nodo->hijos[i]);
    }

    free(nodo->indices);
    free(nodo);
}


NodoOctree *construir_octree(
    const Particula *particulas,
    int cantidad
) {
    if (particulas == NULL || cantidad <= 0) {
        return NULL;
    }

    double minimo[3];
    double maximo[3];

    for (int eje = 0; eje < 3; eje++) {
        minimo[eje] = particulas[0].posicion[eje];
        maximo[eje] = particulas[0].posicion[eje];
    }

    for (int i = 0; i < cantidad; i++) {
        if (
            !isfinite(particulas[i].masa)
            || particulas[i].masa <= 0.0
        ) {
            return NULL;
        }

        for (int eje = 0; eje < 3; eje++) {
            double valor = particulas[i].posicion[eje];

            if (!isfinite(valor)) {
                return NULL;
            }

            if (valor < minimo[eje]) minimo[eje] = valor;
            if (valor > maximo[eje]) maximo[eje] = valor;
        }
    }

    double centro[3];
    double mitad = 1e-9;

    for (int eje = 0; eje < 3; eje++) {
        centro[eje] = (minimo[eje] + maximo[eje]) / 2.0;

        double extension = (
            maximo[eje] - minimo[eje]
        ) / 2.0;

        if (extension > mitad) {
            mitad = extension;
        }
    }

    NodoOctree *raiz = crear_nodo(
        centro,
        mitad * 1.01,
        0
    );

    if (raiz == NULL) {
        return NULL;
    }

    for (int i = 0; i < cantidad; i++) {
        if (!insertar(raiz, particulas, i)) {
            liberar_octree(raiz);
            return NULL;
        }
    }

    return raiz;
}


int contar_nodos(const NodoOctree *nodo) {
    if (nodo == NULL) {
        return 0;
    }

    int total = 1;

    for (int i = 0; i < 8; i++) {
        total += contar_nodos(nodo->hijos[i]);
    }

    return total;
}


int contar_particulas_hojas(const NodoOctree *nodo) {
    if (nodo == NULL) {
        return 0;
    }

    int total = nodo->cantidad;

    for (int i = 0; i < 8; i++) {
        total += contar_particulas_hojas(nodo->hijos[i]);
    }

    return total;
}
