# Umbralizacion

import numpy as np


def umbral_global(imagen, T=128):
    """Umbralizacion con valor fijo T. Pixeles >= T van a 255."""
    binaria = np.zeros_like(imagen, dtype=np.uint8)
    binaria[imagen >= T] = 255
    return binaria


def umbral_otsu(imagen):
    """Metodo de Otsu: encuentra el umbral que maximiza la varianza
    entre clases. Retorna (imagen_binaria, valor_umbral)."""

    # Histograma normalizado
    histograma = np.zeros(256, dtype=np.float64)
    for valor in imagen.ravel():
        histograma[valor] += 1
    histograma = histograma / imagen.size

    mejor_umbral = 0
    mejor_varianza = 0.0

    for T in range(256):
        # Peso de cada clase
        w0 = np.sum(histograma[:T + 1])
        w1 = np.sum(histograma[T + 1:])

        if w0 == 0 or w1 == 0:
            continue

        # Media de cada clase
        indices = np.arange(256)
        mu0 = np.sum(indices[:T + 1] * histograma[:T + 1]) / w0
        mu1 = np.sum(indices[T + 1:] * histograma[T + 1:]) / w1

        # Varianza entre clases
        varianza = w0 * w1 * (mu0 - mu1) ** 2

        if varianza > mejor_varianza:
            mejor_varianza = varianza
            mejor_umbral = T

    binaria = umbral_global(imagen, mejor_umbral)
    return binaria, mejor_umbral


def umbral_adaptativo(imagen, tamano_bloque=15, C=5):
    """Umbral adaptativo: el umbral de cada pixel es la media
    de su vecindario local menos C."""
    img = imagen.astype(np.float64)
    h, w = img.shape
    pad = tamano_bloque // 2

    relleno = np.pad(img, pad, mode='reflect')

    # Imagen integral para calcular sumas rapido
    integral = np.cumsum(np.cumsum(relleno, axis=0), axis=1)

    binaria = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            y1 = i
            y2 = i + tamano_bloque - 1
            x1 = j
            x2 = j + tamano_bloque - 1

            total = integral[y2, x2]
            if y1 > 0:
                total -= integral[y1 - 1, x2]
            if x1 > 0:
                total -= integral[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                total += integral[y1 - 1, x1 - 1]

            media_local = total / (tamano_bloque * tamano_bloque)

            if img[i, j] >= (media_local - C):
                binaria[i, j] = 255

    return binaria
