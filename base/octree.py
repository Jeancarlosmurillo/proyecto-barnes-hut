
import numpy as np


class NodoOctree:
    MAX_PROFUNDIDAD = 32
    MIN_MITAD = 1e-12

    def __init__(self, centro, mitad, profundidad=0):
        self.centro = np.array(centro, dtype=float)
        self.mitad = float(mitad)
        self.profundidad = profundidad

        self.masa_total = 0.0
        self.centro_masa = np.zeros(3, dtype=float)

        self.particulas = []
        self.hijos = [None] * 8

    def es_hoja(self):
        return all(hijo is None for hijo in self.hijos)

    def contiene(self, posicion):
        return np.all(
            np.abs(posicion - self.centro) <= self.mitad
        )

    def obtener_indice(self, posicion):
        indice = 0

        for eje in range(3):
            if posicion[eje] >= self.centro[eje]:
                indice |= (1 << eje)

        return indice

    def crear_hijo(self, indice):
        desplazamiento = self.mitad / 2.0

        direccion = np.array([
            1 if indice & 1 else -1,
            1 if indice & 2 else -1,
            1 if indice & 4 else -1
        ])

        centro_hijo = (
            self.centro + direccion * desplazamiento
        )

        hijo = NodoOctree(
            centro_hijo,
            desplazamiento,
            self.profundidad + 1
        )

        self.hijos[indice] = hijo
        return hijo

    def insertar_en_hijo(self, particula):
        indice = self.obtener_indice(
            particula.posicion
        )

        if self.hijos[indice] is None:
            self.crear_hijo(indice)

        self.hijos[indice].insertar(particula)

    def insertar(self, particula):
        nueva_masa = self.masa_total + particula.masa

        self.centro_masa = (
            self.centro_masa * self.masa_total
            + particula.posicion * particula.masa
        ) / nueva_masa

        self.masa_total = nueva_masa

        if self.es_hoja():
            # Detener la subdivisión cuando el
            # espacio sea demasiado pequeño.
            if (
                self.profundidad >= self.MAX_PROFUNDIDAD
                or self.mitad <= self.MIN_MITAD
            ):
                self.particulas.append(particula)
                return

            if not self.particulas:
                self.particulas.append(particula)
                return

            anteriores = self.particulas
            self.particulas = []

            for anterior in anteriores:
                self.insertar_en_hijo(anterior)

        self.insertar_en_hijo(particula)


def construir_octree(particulas):
    if not particulas:
        raise ValueError(
            "No hay partículas para construir el árbol."
        )

    posiciones = np.array([
        p.posicion for p in particulas
    ])

    if not np.all(np.isfinite(posiciones)):
        raise ValueError(
            "Las posiciones deben ser finitas."
        )

    minimo = posiciones.min(axis=0)
    maximo = posiciones.max(axis=0)

    centro = (minimo + maximo) / 2.0

    mitad = max(
        np.max(maximo - minimo) / 2.0,
        1e-9
    )

    raiz = NodoOctree(
        centro=centro,
        mitad=mitad * 1.01
    )

    for particula in particulas:
        raiz.insertar(particula)

    return raiz
