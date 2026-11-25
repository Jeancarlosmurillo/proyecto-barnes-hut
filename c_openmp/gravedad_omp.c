
#include "gravedad_omp.h"
#include "gravedad.h"

#include <omp.h>
#include <stddef.h>

void calcular_todas_aceleraciones_omp(
    const Particula *particulas,
    int cantidad,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    int hilos,
    double (*aceleraciones)[3]
) {
    if (
        particulas == NULL
        || raiz == NULL
        || aceleraciones == NULL
        || cantidad <= 0
        || hilos <= 0
    ) {
        return;
    }

    #pragma omp parallel for num_threads(hilos) schedule(static)
    for (int i = 0; i < cantidad; i++) {
        calcular_aceleracion_bh(
            particulas,
            i,
            raiz,
            theta,
            G,
            epsilon,
            aceleraciones[i]
        );
    }
}
