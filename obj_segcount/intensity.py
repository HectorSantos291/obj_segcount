# Transformaciones de intensidad

import numpy as np


def negativo(imagen):
    """Negativo: s = 255 - r"""
    return (255 - imagen).astype(np.uint8)


def transformacion_log(imagen, c=1.0):
    """Transformacion logaritmica: s = c * log(1 + r)"""
    img = imagen.astype(np.float64)
    resultado = c * np.log(1 + img)

    if resultado.max() - resultado.min() > 0:
        resultado = (resultado - resultado.min()) / (resultado.max() - resultado.min()) * 255

    return np.clip(resultado, 0, 255).astype(np.uint8)


def estiramiento_contraste(imagen, m=128, e=4):
    """Estiramiento de contraste sigmoidal: s = 1 / (1 + (m/r)^e)"""
    img = imagen.astype(np.float64)
    epsilon = 1e-6

    resultado = 1.0 / (1.0 + (m / (img + epsilon)) ** e)

    if resultado.max() - resultado.min() > 0:
        resultado = (resultado - resultado.min()) / (resultado.max() - resultado.min()) * 255

    return np.clip(resultado, 0, 255).astype(np.uint8)


def ajuste_lineal(imagen, c=0, d=255):
    """Ajuste lineal: mapea [a,b] de la imagen a [c,d]"""
    img = imagen.astype(np.float64)
    a = img.min()
    b = img.max()

    if b - a == 0:
        return np.full_like(imagen, int(c), dtype=np.uint8)

    resultado = ((d - c) / (b - a)) * (img - a) + c
    return np.clip(resultado, 0, 255).astype(np.uint8)


def ecualizacion_histograma(imagen):
    """Ecualizacion de histograma.
    Calcula histograma -> CDF -> normaliza -> aplica LUT."""

    histograma = np.zeros(256, dtype=np.int64)
    for valor in imagen.ravel():
        histograma[valor] += 1

    # CDF (histograma acumulado)
    cdf = np.cumsum(histograma)
    cdf_min = cdf[cdf > 0].min()
    total_pixeles = imagen.size

    denominador = total_pixeles - cdf_min
    if denominador == 0:
        return imagen.copy()

    lut = np.round((cdf - cdf_min) / denominador * 255).astype(np.uint8)
    ecualizada = lut[imagen]

    return ecualizada
