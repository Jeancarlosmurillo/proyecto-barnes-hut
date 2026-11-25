
#include <stdio.h>
#include <math.h>

#include "particula.h"
#include "integrador.h"


int main(void) {
    const double G = 1.0;
    const double epsilon = 0.01;
    const double theta = 0.5;
    const double dt = 0.001;

    Particula particulas[2] = {0};

    particulas[0].identificador = 0;
    particulas[0].masa = 1.0;
    particulas[0].posicion[0] = -1.0;

    particulas[1].identificador = 1;
    particulas[1].masa = 1.0;
    particulas[1].posicion[0] = 1.0;

    printf("=== PRUEBA LEAPFROG EN C ===\n");

    printf(
        "Distancia inicial: %.12f\n",
        particulas[1].posicion[0]
            - particulas[0].posicion[0]
    );

    // Realizar 100 pasos de integración.
    for (int paso = 0; paso < 100; paso++) {
        if (!paso_leapfrog_c(
            particulas,
            2,
            dt,
            theta,
            G,
            epsilon
        )) {
            printf("ERROR en paso %d\n", paso + 1);
            return 1;
        }
    }

    double distancia_final = (
        particulas[1].posicion[0]
        - particulas[0].posicion[0]
    );

    double momento_total = (
        particulas[0].masa
            * particulas[0].velocidad[0]
        + particulas[1].masa
            * particulas[1].velocidad[0]
    );

    printf(
        "Distancia final: %.12f\n",
        distancia_final
    );

    printf(
        "Velocidad particula 0: %.12f\n",
        particulas[0].velocidad[0]
    );

    printf(
        "Velocidad particula 1: %.12f\n",
        particulas[1].velocidad[0]
    );

    printf(
        "Momento total X: %.12e\n",
        momento_total
    );

    int correcto = (
        isfinite(distancia_final)
        && distancia_final > 0.0
        && distancia_final < 2.0
        && particulas[0].velocidad[0] > 0.0
        && particulas[1].velocidad[0] < 0.0
        && fabs(momento_total) < 1e-10
        && fabs(
            particulas[0].posicion[0]
            + particulas[1].posicion[0]
        ) < 1e-10
    );

    if (!correcto) {
        printf("VALIDACION LEAPFROG EN C FALLIDA\n");
        return 1;
    }

    printf("VALIDACION LEAPFROG EN C SUPERADA\n");
    return 0;
}
