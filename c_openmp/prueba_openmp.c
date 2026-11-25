
#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("=== PRUEBA OPENMP ===\n");

    #pragma omp parallel num_threads(4)
    {
        int hilo = omp_get_thread_num();
        int total = omp_get_num_threads();

        #pragma omp critical
        {
            printf(
                "Hola desde el hilo %d de %d\n",
                hilo,
                total
            );
        }
    }

    printf("Prueba finalizada.\n");

    return 0;
}
    