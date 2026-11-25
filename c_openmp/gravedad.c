
#include "gravedad.h"

#include <math.h>
#include <stddef.h>


static int es_hoja(const NodoOctree *nodo) {
    for (int i = 0; i < 8; i++) {
        if (nodo->hijos[i] != NULL) {
            return 0;
        }
    }

    return 1;
}


static int contiene(
    const NodoOctree *nodo,
    const double posicion[3]
) {
    for (int eje = 0; eje < 3; eje++) {
        if (
            fabs(posicion[eje] - nodo->centro[eje])
            > nodo->mitad
        ) {
            return 0;
        }
    }

    return 1;
}


static void visitar_nodo(
    const Particula *particulas,
    int indice,
    const NodoOctree *nodo,
    double theta,
    double G,
    double epsilon2,
    double aceleracion[3]
) {
    if (nodo == NULL || nodo->masa_total == 0.0) {
        return;
    }

    const Particula *objetivo = &particulas[indice];

    if (es_hoja(nodo)) {
        // Interaccion directa con cada particula
        // almacenada en la hoja.
        for (int k = 0; k < nodo->cantidad; k++) {
            int j = nodo->indices[k];

            if (j == indice) {
                continue;
            }

            const Particula *otra = &particulas[j];

            double dx = (
                otra->posicion[0] - objetivo->posicion[0]
            );

            double dy = (
                otra->posicion[1] - objetivo->posicion[1]
            );

            double dz = (
                otra->posicion[2] - objetivo->posicion[2]
            );

            double distancia2 = (
                dx * dx + dy * dy + dz * dz
                + epsilon2
            );

            double factor = (
                G * otra->masa
                / (distancia2 * sqrt(distancia2))
            );

            aceleracion[0] += factor * dx;
            aceleracion[1] += factor * dy;
            aceleracion[2] += factor * dz;
        }

        return;
    }

    double dx = (
        nodo->centro_masa[0] - objetivo->posicion[0]
    );

    double dy = (
        nodo->centro_masa[1] - objetivo->posicion[1]
    );

    double dz = (
        nodo->centro_masa[2] - objetivo->posicion[2]
    );

    double distancia2 = dx * dx + dy * dy + dz * dz;
    double distancia = sqrt(distancia2);

    int usar_aproximacion = (
        theta > 0.0
        && distancia > 0.0
        && (2.0 * nodo->mitad) / distancia < theta
        && !contiene(nodo, objetivo->posicion)
    );

    if (usar_aproximacion) {
        double suavizada2 = distancia2 + epsilon2;

        double factor = (
            G * nodo->masa_total
            / (suavizada2 * sqrt(suavizada2))
        );

        aceleracion[0] += factor * dx;
        aceleracion[1] += factor * dy;
        aceleracion[2] += factor * dz;

        return;
    }

    // Recorrer los hijos del nodo.
    for (int hijo = 0; hijo < 8; hijo++) {
        if (nodo->hijos[hijo] != NULL) {
            visitar_nodo(
                particulas,
                indice,
                nodo->hijos[hijo],
                theta,
                G,
                epsilon2,
                aceleracion
            );
        }
    }
}


void calcular_aceleracion_bh(
    const Particula *particulas,
    int indice,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    double aceleracion[3]
) {
    aceleracion[0] = 0.0;
    aceleracion[1] = 0.0;
    aceleracion[2] = 0.0;

    if (
        particulas == NULL
        || raiz == NULL
        || aceleracion == NULL
        || indice < 0
        || theta < 0.0
        || epsilon <= 0.0
    ) {
        return;
    }

    visitar_nodo(
        particulas,
        indice,
        raiz,
        theta,
        G,
        epsilon * epsilon,
        aceleracion
    );
}


void calcular_todas_aceleraciones(
    const Particula *particulas,
    int cantidad,
    const NodoOctree *raiz,
    double theta,
    double G,
    double epsilon,
    double (*aceleraciones)[3]
) {
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
