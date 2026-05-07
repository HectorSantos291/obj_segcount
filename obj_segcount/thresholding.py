import numpy as np


def umbral_global(imagen, T=128):
    # Pixeles con valor >= T se marcan como 255, el resto como 0
    binaria = np.zeros_like(imagen, dtype=np.uint8)
    binaria[imagen >= T] = 255
    return binaria


def umbral_otsu(imagen):
    # Encuentra el umbral que maximiza la varianza entre las dos clases (fondo y objeto)
    histograma = np.zeros(256, dtype=np.float64)
    for valor in imagen.ravel():
        histograma[valor] += 1
    histograma = histograma / imagen.size

    mejor_umbral = 0
    mejor_varianza = 0.0

    for T in range(256):
        w0 = np.sum(histograma[:T + 1])
        w1 = np.sum(histograma[T + 1:])

        if w0 == 0 or w1 == 0:
            continue

        indices = np.arange(256)
        mu0 = np.sum(indices[:T + 1] * histograma[:T + 1]) / w0
        mu1 = np.sum(indices[T + 1:] * histograma[T + 1:]) / w1

        varianza = w0 * w1 * (mu0 - mu1) ** 2

        if varianza > mejor_varianza:
            mejor_varianza = varianza
            mejor_umbral = T

    binaria = umbral_global(imagen, mejor_umbral)
    return binaria, mejor_umbral


def umbral_adaptativo(imagen, tamano_bloque=15, C=5):
    # El umbral de cada pixel es la media de su vecindario local menos C
    img = imagen.astype(np.float64)
    h, w = img.shape
    pad = tamano_bloque // 2

    relleno = np.pad(img, pad, mode='reflect')
    binaria = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            vecindario = relleno[i:i + tamano_bloque, j:j + tamano_bloque]
            media_local = np.mean(vecindario)

            if img[i, j] >= (media_local - C):
                binaria[i, j] = 255

    return binaria
