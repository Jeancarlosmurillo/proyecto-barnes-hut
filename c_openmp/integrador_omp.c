
#include "integrador_omp.h"

#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "octree.h"
#include "gravedad_omp.h"


int paso_leapfrog_omp(
    Particula *particulas,
    int cantidad,
    double dt,
    double theta,
    double G,
    double epsilon,
    int hilos
) {
    if (
        particulas == NULL || cantidad <= 0 ||
        !isfinite(dt) || dt <= 0.0 ||
        !isfinite(theta) || theta < 0.0 ||
        !isfinite(G) ||
        !isfinite(epsilon) || epsilon <= 0.0 ||
        hilos <= 0
    ) {
        return 0;
    }

    double (*aceleraciones)[3] = malloc(
        (size_t)cantidad * sizeof(*aceleraciones)
    );

    if (aceleraciones == NULL) {
        return 0;
    }

    // Primera construccion del octree.
    NodoOctree *arbol = construir_octree(
        particulas, cantidad
    );

    if (arbol == NULL) {
        free(aceleraciones);
        return 0;
    }

    // Aceleraciones iniciales con OpenMP.
    calcular_todas_aceleraciones_omp(
        particulas, cantidad, arbol,
        theta, G, epsilon, hilos,
        aceleraciones
    );

    liberar_octree(arbol);

    // Kick y Drift.
    #pragma omp parallel for num_threads(hilos) schedule(static)
    for (int i = 0; i < cantidad; i++) {
        for (int eje = 0; eje < 3; eje++) {
            particulas[i].velocidad[eje] += (
                0.5 * dt * aceleraciones[i][eje]
            );

            particulas[i].posicion[eje] += (
                dt * particulas[i].velocidad[eje]
            );
        }
    }

    // Reconstruccion con posiciones actualizadas.
    arbol = construir_octree(particulas, cantidad);

    if (arbol == NULL) {
        free(aceleraciones);
        return 0;
    }

    // Nuevas aceleraciones con OpenMP.
    calcular_todas_aceleraciones_omp(
        particulas, cantidad, arbol,
        theta, G, epsilon, hilos,
        aceleraciones
    );

    liberar_octree(arbol);

    // Segundo Kick.
    #pragma omp parallel for num_threads(hilos) schedule(static)
    for (int i = 0; i < cantidad; i++) {
        for (int eje = 0; eje < 3; eje++) {
            particulas[i].velocidad[eje] += (
                0.5 * dt * aceleraciones[i][eje]
            );
        }
    }

    free(aceleraciones);
    return 1;
}
