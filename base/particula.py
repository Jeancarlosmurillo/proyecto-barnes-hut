import numpy as np


class Particula:
    def __init__(self, identificador, posicion, velocidad, masa):
        self.identificador = identificador
        self.posicion = np.array(posicion, dtype=float)
        self.velocidad = np.array(velocidad, dtype=float)
        self.masa = float(masa)

        if self.posicion.shape != (3,):
            raise ValueError("La posición debe tener 3 coordenadas.")

        if self.velocidad.shape != (3,):
            raise ValueError("La velocidad debe tener 3 componentes.")

        if self.masa <= 0:
            raise ValueError("La masa debe ser mayor que cero.")

    def __repr__(self):
        return (
            f"Particula(id={self.identificador}, "
            f"posicion={self.posicion.tolist()}, "
            f"velocidad={self.velocidad.tolist()}, "
            f"masa={self.masa})"
        )
