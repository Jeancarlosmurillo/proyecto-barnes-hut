
#include <stdio.h>
#include <math.h>

#include "particula.h"
#include "octree.h"
#include "gravedad.h"


int main(void) {
    printf("=== PRUEBA GRAVEDAD BARNES-HUT EN C ===\n");

    Particula particulas[2] = {0};

    particulas[0].identificador = 0;
    particulas[0].posicion[0] = 0.0;
    particulas[0].masa = 1.0;

    particulas[1].identificador = 1;
    particulas[1].posicion[0] = 1.0;
    particulas[1].masa = 2.0;

    const double G = 1.0;
    const double epsilon = 0.01;
    const double theta = 0.5;

    NodoOctree *raiz = construir_octree(
        particulas, 2
    );

    if (raiz == NULL) {
        printf("ERROR construyendo octree\n");
        return 1;
    }

    double aceleraciones[2][3];

    calcular_todas_aceleraciones(
        particulas,
        2,
        raiz,
        theta,
        G,
        epsilon,
        aceleraciones
    );

    double d2 = 1.0 + epsilon * epsilon;
    double denominador = d2 * sqrt(d2);

    double esperado_0 = 2.0 / denominador;
    double esperado_1 = -1.0 / denominador;

    printf(
        "Aceleracion particula 0: %.12f\n",
        aceleraciones[0][0]
    );

    printf(
        "Aceleracion particula 1: %.12f\n",
        aceleraciones[1][0]
    );

    int correcto = (
        fabs(aceleraciones[0][0] - esperado_0) < 1e-12
        && fabs(aceleraciones[1][0] - esperado_1) < 1e-12
        && fabs(aceleraciones[0][1]) < 1e-12
        && fabs(aceleraciones[0][2]) < 1e-12
        && fabs(aceleraciones[1][1]) < 1e-12
        && fabs(aceleraciones[1][2]) < 1e-12
    );

    liberar_octree(raiz);

    if (!correcto) {
        printf("VALIDACION GRAVEDAD FALLIDA\n");
        return 1;
    }

    printf("VALIDACION GRAVEDAD EN C SUPERADA\n");

    return 0;
}
