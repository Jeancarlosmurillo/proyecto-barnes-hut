
#include <stdio.h>
#include <math.h>

#include "particula.h"
#include "octree.h"


int main(void) {
    printf("=== PRUEBA OCTREE 3D EN C ===\n");

    // Cinco particulas en la misma posicion.
    Particula particulas[6] = {0};

    for (int i = 0; i < 6; i++) {
        particulas[i].identificador = i;
        particulas[i].masa = 1.0;
    }

    // La sexta particula esta desplazada.
    particulas[5].posicion[0] = 1.0;

    NodoOctree *raiz = construir_octree(particulas, 6);

    if (raiz == NULL) {
        printf("ERROR: No se pudo construir el octree.\n");
        return 1;
    }

    int nodos = contar_nodos(raiz);
    int almacenadas = contar_particulas_hojas(raiz);

    printf("Particulas: 6\n");
    printf("Nodos: %d\n", nodos);
    printf("Particulas almacenadas: %d\n", almacenadas);
    printf("Masa total: %.6f\n", raiz->masa_total);

    printf(
        "Centro de masa: (%.6f, %.6f, %.6f)\n",
        raiz->centro_masa[0],
        raiz->centro_masa[1],
        raiz->centro_masa[2]
    );

    int correcto = (
        fabs(raiz->masa_total - 6.0) < 1e-12
        && fabs(raiz->centro_masa[0] - 1.0 / 6.0) < 1e-12
        && fabs(raiz->centro_masa[1]) < 1e-12
        && fabs(raiz->centro_masa[2]) < 1e-12
        && almacenadas == 6
    );

    liberar_octree(raiz);

    if (!correcto) {
        printf("VALIDACION OCTREE FALLIDA\n");
        return 1;
    }

    printf("VALIDACION OCTREE EN C SUPERADA\n");

    return 0;
}
