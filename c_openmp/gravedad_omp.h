
#ifndef GRAVEDAD_OMP_H
#define GRAVEDAD_OMP_H

#include "particula.h"
#include "octree.h"

void calcular_todas_aceleraciones_omp(
    const Particula *particulas,
    int cantidad,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    int hilos,
    double (*aceleraciones)[3]
);

#endif
