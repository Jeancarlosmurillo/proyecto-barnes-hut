
#ifndef OCTREE_H
#define OCTREE_H

#include "particula.h"

#define MAX_PROFUNDIDAD 32
#define MIN_MITAD 1e-12

typedef struct NodoOctree {
    double centro[3];
    double mitad;

    double masa_total;
    double centro_masa[3];

    int profundidad;

    int *indices;
    int cantidad;
    int capacidad;

    struct NodoOctree *hijos[8];
} NodoOctree;

NodoOctree *construir_octree(
    const Particula *particulas,
    int cantidad
);

void liberar_octree(NodoOctree *nodo);

int contar_nodos(const NodoOctree *nodo);

int contar_particulas_hojas(const NodoOctree *nodo);

#endif
