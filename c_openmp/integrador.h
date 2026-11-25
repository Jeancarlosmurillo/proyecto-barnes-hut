
#ifndef INTEGRADOR_H
#define INTEGRADOR_H

#include "particula.h"

// Devuelve 1 si el paso se completa,
// o 0 si ocurre un error.
int paso_leapfrog_c(
    Particula *particulas,
    int cantidad,
    double dt,
    double theta,
    double G,
    double epsilon
);

#endif
