import numpy as np


class NodoOctree:
    def __init__(self, centro, mitad):
        self.centro = np.array(centro, dtype=float)
        self.mitad = float(mitad)

        self.masa_total = 0.0
        self.centro_masa = np.zeros(3, dtype=float)

        self.particula = None
        self.hijos = [None] * 8

    def es_hoja(self):
        return all(hijo is None for hijo in self.hijos)

    def obtener_indice(self, posicion):
        indice = 0

        if posicion[0] >= self.centro[0]:
            indice |= 1

        if posicion[1] >= self.centro[1]:
            indice |= 2

        if posicion[2] >= self.centro[2]:
            indice |= 4

        return indice

    def crear_hijo(self, indice):
        desplazamiento = self.mitad / 2

        direccion = np.array([
            1 if indice & 1 else -1,
            1 if indice & 2 else -1,
            1 if indice & 4 else -1
        ])

        centro_hijo = (
            self.centro + direccion * desplazamiento
        )

        hijo = NodoOctree(
            centro=centro_hijo,
            mitad=desplazamiento
        )

        self.hijos[indice] = hijo
        return hijo

    def insertar_en_hijo(self, particula):
        indice = self.obtener_indice(particula.posicion)

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
            if self.particula is None:
                self.particula = particula
                return

            anterior = self.particula
            self.particula = None

            self.insertar_en_hijo(anterior)

        self.insertar_en_hijo(particula)


def construir_octree(particulas):
    if not particulas:
        raise ValueError("No hay partículas para construir el árbol.")

    posiciones = np.array([
        p.posicion for p in particulas
    ])

    minimo = posiciones.min(axis=0)
    maximo = posiciones.max(axis=0)

    centro = (minimo + maximo) / 2
    mitad = max(np.max(maximo - minimo) / 2, 1e-9)

    # Margen para contener todas las partículas.
    mitad *= 1.01

    raiz = NodoOctree(centro, mitad)

    for particula in particulas:
        raiz.insertar(particula)

    return raiz
