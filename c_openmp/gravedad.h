
#ifndef GRAVEDAD_H
#define GRAVEDAD_H

#include "particula.h"
#include "octree.h"

void calcular_aceleracion_bh(
    const Particula *particulas,
    int indice,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    double aceleracion[3]
);

void calcular_todas_aceleraciones(
    const Particula *particulas,
    int cantidad,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    double (*aceleraciones)[3]
);

#endif
