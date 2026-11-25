
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "particula.h"
#include "octree.h"
#include "gravedad.h"
#include "gravedad_omp.h"

#include <omp.h>

#define N 1000
#define G 1.0
#define EPSILON 0.01
#define THETA 0.5


int main(void) {
    printf("=== VALIDACION BARNES-HUT OPENMP ===\n");

    Particula *particulas = calloc(
        N, sizeof(Particula)
    );

    double (*secuenciales)[3] = calloc(
        N, sizeof(*secuenciales)
    );

    double (*paralelas)[3] = calloc(
        N, sizeof(*paralelas)
    );

    if (
        particulas == NULL
        || secuenciales == NULL
        || paralelas == NULL
    ) {
        printf("Error reservando memoria\n");
        free(particulas);
        free(secuenciales);
        free(paralelas);
        return 1;
    }

    // Distribución determinista para reproducir pruebas.
    for (int i = 0; i < N; i++) {
        particulas[i].identificador = i;
        particulas[i].masa = 1.0 + (i % 7) * 0.1;

        particulas[i].posicion[0] = (
            (i * 37 % 1009) / 1009.0
        );

        particulas[i].posicion[1] = (
            (i * 67 % 1013) / 1013.0
        );

        particulas[i].posicion[2] = (
            (i * 97 % 1019) / 1019.0
        );
    }

    NodoOctree *arbol = construir_octree(
        particulas, N
    );

    if (arbol == NULL) {
        printf("Error construyendo octree\n");
        free(particulas);
        free(secuenciales);
        free(paralelas);
        return 1;
    }

    double inicio = omp_get_wtime();

    calcular_todas_aceleraciones(
        particulas,
        N,
        arbol,
        THETA,
        G,
        EPSILON,
        secuenciales
    );

    double tiempo_secuencial = (
        omp_get_wtime() - inicio
    );

    printf("Particulas: %d\n", N);
    printf("Nodos: %d\n", contar_nodos(arbol));
    printf(
        "Tiempo secuencial: %.6f s\n",
        tiempo_secuencial
    );

    int configuraciones[] = {1, 2, 4, 8};
    int correcto = 1;

    for (int k = 0; k < 4; k++) {
        int hilos = configuraciones[k];

        inicio = omp_get_wtime();

        calcular_todas_aceleraciones_omp(
            particulas,
            N,
            arbol,
            THETA,
            G,
            EPSILON,
            hilos,
            paralelas
        );

        double tiempo = omp_get_wtime() - inicio;
        double error_maximo = 0.0;

        for (int i = 0; i < N; i++) {
            for (int eje = 0; eje < 3; eje++) {
                double diferencia = fabs(
                    paralelas[i][eje]
                    - secuenciales[i][eje]
                );

                if (!isfinite(diferencia)) {
                    correcto = 0;
                }

                if (diferencia > error_maximo) {
                    error_maximo = diferencia;
                }
            }
        }

        double speedup = (
            tiempo > 0.0
            ? tiempo_secuencial / tiempo
            : 0.0
        );

        printf(
            "%d hilos | Tiempo: %.6f s | "
            "Speedup: %.2fx | Error: %.12e\n",
            hilos,
            tiempo,
            speedup,
            error_maximo
        );

        if (error_maximo > 1e-10) {
            correcto = 0;
        }
    }

    liberar_octree(arbol);
    free(particulas);
    free(secuenciales);
    free(paralelas);

    if (!correcto) {
        printf("\nVALIDACION OPENMP FALLIDA\n");
        return 1;
    }

    printf("\nVALIDACION OPENMP SUPERADA\n");

    return 0;
}
