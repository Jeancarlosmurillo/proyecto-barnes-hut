
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "particula.h"
#include "integrador.h"
#include "integrador_omp.h"

#define N 20000
#define PASOS 1


int main(void) {
    const double dt = 0.001;
    const double theta = 0.5;
    const double G = 1.0;
    const double epsilon = 0.01;

    printf("=== VALIDACION LEAPFROG OPENMP ===\n");

    Particula *original = calloc(N, sizeof(Particula));
    Particula *secuencial = calloc(N, sizeof(Particula));
    Particula *paralela = calloc(N, sizeof(Particula));

    if (!original || !secuencial || !paralela) {
        printf("Error reservando memoria\n");
        free(original);
        free(secuencial);
        free(paralela);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        original[i].identificador = i;
        original[i].masa = 0.5 + (i % 7) * 0.1;

        original[i].posicion[0] =
            (i * 37 % 101) / 101.0;

        original[i].posicion[1] =
            (i * 67 % 103) / 103.0;

        original[i].posicion[2] =
            (i * 97 % 107) / 107.0;
    }

    memcpy(secuencial, original, N * sizeof(Particula));

    double inicio = omp_get_wtime();

    for (int paso = 0; paso < PASOS; paso++) {
        if (!paso_leapfrog_c(
            secuencial, N, dt, theta, G, epsilon
        )) {
            printf("Error en simulacion secuencial\n");
            free(original);
            free(secuencial);
            free(paralela);
            return 1;
        }
    }

    double tiempo_secuencial = omp_get_wtime() - inicio;

    printf("Particulas: %d\n", N);
    printf("Pasos: %d\n", PASOS);
    printf(
        "Tiempo secuencial: %.6f s\n",
        tiempo_secuencial
    );

    int configuraciones[] = {1, 2, 4, 8};
    int correcto = 1;

    for (int k = 0; k < 4; k++) {
        int hilos = configuraciones[k];

        memcpy(
            paralela, original, N * sizeof(Particula)
        );

        inicio = omp_get_wtime();

        int ejecucion_ok = 1;

        for (int paso = 0; paso < PASOS; paso++) {
            if (!paso_leapfrog_omp(
                paralela, N, dt, theta,
                G, epsilon, hilos
            )) {
                ejecucion_ok = 0;
                break;
            }
        }

        double tiempo = omp_get_wtime() - inicio;

        if (!ejecucion_ok) {
            printf("Error usando %d hilos\n", hilos);
            correcto = 0;
            continue;
        }

        double error_maximo = 0.0;

        for (int i = 0; i < N; i++) {
            for (int eje = 0; eje < 3; eje++) {
                double error_pos = fabs(
                    secuencial[i].posicion[eje]
                    - paralela[i].posicion[eje]
                );

                double error_vel = fabs(
                    secuencial[i].velocidad[eje]
                    - paralela[i].velocidad[eje]
                );

                if (!isfinite(error_pos) ||
                    !isfinite(error_vel)) {
                    correcto = 0;
                    continue;
                }

                if (error_pos > error_maximo) {
                    error_maximo = error_pos;
                }

                if (error_vel > error_maximo) {
                    error_maximo = error_vel;
                }
            }
        }

        printf(
            "%d hilos | Tiempo: %.6f s | "
            "Error maximo: %.12e\n",
            hilos, tiempo, error_maximo
        );

        if (error_maximo > 1e-10) {
            correcto = 0;
        }
    }

    free(original);
    free(secuencial);
    free(paralela);

    if (!correcto) {
        printf("VALIDACION LEAPFROG OPENMP FALLIDA\n");
        return 1;
    }

    printf("VALIDACION LEAPFROG OPENMP SUPERADA\n");
    return 0;
}
