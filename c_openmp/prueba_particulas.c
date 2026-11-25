
#include <stdio.h>
#include "particula.h"

int main(void) {
    Particula p = {
        .identificador = 1,
        .posicion = {1.0, 2.0, 3.0},
        .velocidad = {0.1, 0.2, 0.3},
        .masa = 5.0
    };

    printf("=== PRUEBA PARTICULAS EN C ===\n");
    printf("ID: %d\n", p.identificador);

    printf(
        "Posicion: (%.2f, %.2f, %.2f)\n",
        p.posicion[0],
        p.posicion[1],
        p.posicion[2]
    );

    printf(
        "Velocidad: (%.2f, %.2f, %.2f)\n",
        p.velocidad[0],
        p.velocidad[1],
        p.velocidad[2]
    );

    printf("Masa: %.2f\n", p.masa);
    printf("Prueba finalizada.\n");

    return 0;
}
