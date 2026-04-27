# Filtrado espacial

import numpy as np


def convolucion2d(imagen, kernel):
    """Convolucion 2D con zero-padding."""
    img = imagen.astype(np.float64)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    relleno = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant', constant_values=0)

    h, w = img.shape
    salida = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = relleno[i:i + kh, j:j + kw]
            salida[i, j] = np.sum(region * kernel)

    return salida


def kernel_gaussiano(tamano, sigma):
    """Genera un kernel gaussiano 2D normalizado."""
    mitad = tamano // 2
    eje = np.arange(-mitad, mitad + 1)
    xx, yy = np.meshgrid(eje, eje)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def filtro_promedio(imagen, tamano=3):
    """Filtro promedio (box filter)."""
    kernel = np.ones((tamano, tamano), dtype=np.float64) / (tamano * tamano)
    resultado = convolucion2d(imagen, kernel)
    return np.clip(resultado, 0, 255).astype(np.uint8)


def filtro_gaussiano(imagen, tamano=5, sigma=1.0):
    """Filtro gaussiano para suavizado."""
    kernel = kernel_gaussiano(tamano, sigma)
    resultado = convolucion2d(imagen, kernel)
    return np.clip(resultado, 0, 255).astype(np.uint8)


def filtro_mediana(imagen, tamano=3):
    """Filtro de mediana. Bueno para ruido sal y pimienta."""
    img = imagen.astype(np.float64)
    pad = tamano // 2
    relleno = np.pad(img, pad, mode='constant', constant_values=0)

    h, w = img.shape
    salida = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = relleno[i:i + tamano, j:j + tamano]
            salida[i, j] = np.median(region)

    return np.clip(salida, 0, 255).astype(np.uint8)
