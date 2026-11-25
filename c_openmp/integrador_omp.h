
#ifndef INTEGRADOR_OMP_H
#define INTEGRADOR_OMP_H

#include "particula.h"

int paso_leapfrog_omp(
    Particula *particulas,
    int cantidad,
    double dt,
    double theta,
    double G,
    double epsilon,
    int hilos
);

#endif
