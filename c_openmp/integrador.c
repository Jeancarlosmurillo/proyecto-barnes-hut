
#include "integrador.h"

#include <math.h>
#include <stdlib.h>

#include "octree.h"
#include "gravedad.h"


int paso_leapfrog_c(
    Particula *particulas,
    int cantidad,
    double dt,
    double theta,
    double G,
    double epsilon
) {
    if (
        particulas == NULL
        || cantidad <= 0
        || !isfinite(dt)
        || dt <= 0.0
        || !isfinite(theta)
        || theta < 0.0
        || !isfinite(G)
        || !isfinite(epsilon)
        || epsilon <= 0.0
    ) {
        return 0;
    }

    double (*aceleraciones)[3] = malloc(
        (size_t)cantidad * sizeof(*aceleraciones)
    );

    if (aceleraciones == NULL) {
        return 0;
    }

    // Calcular aceleraciones iniciales.
    NodoOctree *arbol = construir_octree(
        particulas, cantidad
    );

    if (arbol == NULL) {
        free(aceleraciones);
        return 0;
    }

    calcular_todas_aceleraciones(
        particulas,
        cantidad,
        arbol,
        theta,
        G,
        epsilon,
        aceleraciones
    );

    liberar_octree(arbol);

    // Kick: primera mitad de la velocidad.
    for (int i = 0; i < cantidad; i++) {
        for (int eje = 0; eje < 3; eje++) {
            particulas[i].velocidad[eje] += (
                0.5 * dt * aceleraciones[i][eje]
            );
        }
    }

    // Drift: actualizar posiciones.
    for (int i = 0; i < cantidad; i++) {
        for (int eje = 0; eje < 3; eje++) {
            particulas[i].posicion[eje] += (
                dt * particulas[i].velocidad[eje]
            );
        }
    }

    // Reconstruir el octree con las posiciones nuevas.
    arbol = construir_octree(particulas, cantidad);

    if (arbol == NULL) {
        free(aceleraciones);
        return 0;
    }

    // Calcular aceleraciones actualizadas.
    calcular_todas_aceleraciones(
        particulas,
        cantidad,
        arbol,
        theta,
        G,
        epsilon,
        aceleraciones
    );

    liberar_octree(arbol);

    // Kick: segunda mitad de la velocidad.
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
