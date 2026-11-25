
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "particula.h"
#include "octree.h"
#include "gravedad.h"

#define G 1.0
#define EPSILON 0.01


void calcular_directo(
    const Particula *particulas,
    int cantidad,
    double (*aceleraciones)[3]
) {
    for (int i = 0; i < cantidad; i++) {
        for (int eje = 0; eje < 3; eje++) {
            aceleraciones[i][eje] = 0.0;
        }

        for (int j = 0; j < cantidad; j++) {
            if (i == j) {
                continue;
            }

            double dx = particulas[j].posicion[0]
                      - particulas[i].posicion[0];

            double dy = particulas[j].posicion[1]
                      - particulas[i].posicion[1];

            double dz = particulas[j].posicion[2]
                      - particulas[i].posicion[2];

            double distancia2 = (
                dx * dx + dy * dy + dz * dz
                + EPSILON * EPSILON
            );

            double factor = (
                G * particulas[j].masa
                / (distancia2 * sqrt(distancia2))
            );

            aceleraciones[i][0] += factor * dx;
            aceleraciones[i][1] += factor * dy;
            aceleraciones[i][2] += factor * dz;
        }
    }
}


int validar_caso(
    const char *nombre,
    Particula *particulas,
    int cantidad
) {
    NodoOctree *arbol = construir_octree(
        particulas, cantidad
    );

    if (arbol == NULL) {
        printf("ERROR al construir octree\n");
        return 0;
    }

    double (*directas)[3] = calloc(
        (size_t)cantidad, sizeof(*directas)
    );

    double (*barnes_hut)[3] = calloc(
        (size_t)cantidad, sizeof(*barnes_hut)
    );

    if (directas == NULL || barnes_hut == NULL) {
        free(directas);
        free(barnes_hut);
        liberar_octree(arbol);
        return 0;
    }

    calcular_directo(particulas, cantidad, directas);

    printf("\nCaso: %s\n", nombre);
    printf("Particulas: %d\n", cantidad);
    printf("Nodos: %d\n", contar_nodos(arbol));

    int correcto = 1;

    const double valores_theta[] = {
        0.0, 0.5, 1.0
    };

    for (int t = 0; t < 3; t++) {
        double theta = valores_theta[t];

        calcular_todas_aceleraciones(
            particulas,
            cantidad,
            arbol,
            theta,
            G,
            EPSILON,
            barnes_hut
        );

        double error_maximo = 0.0;
        double suma_error2 = 0.0;
        double suma_directa2 = 0.0;

        for (int i = 0; i < cantidad; i++) {
            for (int eje = 0; eje < 3; eje++) {
                double referencia = directas[i][eje];
                double resultado = barnes_hut[i][eje];

                if (!isfinite(resultado)) {
                    correcto = 0;
                    continue;
                }

                double diferencia = resultado - referencia;
                double error = fabs(diferencia);

                if (error > error_maximo) {
                    error_maximo = error;
                }

                suma_error2 += diferencia * diferencia;
                suma_directa2 += referencia * referencia;
            }
        }

        double error_relativo = (
            sqrt(suma_error2)
            / fmax(sqrt(suma_directa2), 1e-15)
        );

        printf(
            "Theta=%.1f | "
            "Error maximo=%.12e | "
            "Error relativo=%.8f%%\n",
            theta,
            error_maximo,
            error_relativo * 100.0
        );

        // Con theta 0, Barnes-Hut debe coincidir
        // con el calculo directo.
        if (
            theta == 0.0
            && error_maximo > 1e-9
        ) {
            correcto = 0;
        }
    }

    free(directas);
    free(barnes_hut);
    liberar_octree(arbol);

    return correcto;
}


int main(void) {
    printf("=== VALIDACION BARNES-HUT EN C ===\n");

    Particula normales[10] = {0};

    for (int i = 0; i < 10; i++) {
        normales[i].identificador = i;
        normales[i].masa = 1.0 + i * 0.1;

        normales[i].posicion[0] = i * 0.13;
        normales[i].posicion[1] = (i % 3) * 0.27;
        normales[i].posicion[2] = (i % 4) * 0.19;
    }

    Particula densas[6] = {0};

    for (int i = 0; i < 6; i++) {
        densas[i].identificador = i;
        densas[i].masa = 1.0;
    }

    // Cinco particulas coinciden en el origen.
    // La sexta esta separada.
    densas[5].posicion[0] = 1.0;

    int normal_ok = validar_caso(
        "Distribucion normal", normales, 10
    );

    int densa_ok = validar_caso(
        "Particulas coincidentes", densas, 6
    );

    if (normal_ok && densa_ok) {
        printf(
            "\nVALIDACION BARNES-HUT EN C SUPERADA\n"
        );
        return 0;
    }

    printf("\nVALIDACION BARNES-HUT EN C FALLIDA\n");

    return 1;
}
