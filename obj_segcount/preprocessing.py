import numpy as np


def rgb_a_gris(imagen):
    # Formula estandar de luminancia: Y = 0.299R + 0.587G + 0.114B
    if imagen.ndim == 2:
        return imagen.astype(np.uint8)

    if imagen.ndim == 3 and imagen.shape[2] == 1:
        return imagen[:, :, 0].astype(np.uint8)

    pesos = np.array([0.299, 0.587, 0.114])
    gris = np.dot(imagen[:, :, :3].astype(np.float64), pesos)
    return np.clip(gris, 0, 255).astype(np.uint8)


def normalizar(imagen, nuevo_min=0, nuevo_max=255):
    # Ajuste lineal para llevar los valores al rango [nuevo_min, nuevo_max]
    img = imagen.astype(np.float64)
    viejo_min = img.min()
    viejo_max = img.max()

    if viejo_max - viejo_min == 0:
        return np.full_like(imagen, int(nuevo_min), dtype=np.uint8)

    resultado = (img - viejo_min) / (viejo_max - viejo_min) * (nuevo_max - nuevo_min) + nuevo_min
    return np.clip(resultado, nuevo_min, nuevo_max).astype(np.uint8)
