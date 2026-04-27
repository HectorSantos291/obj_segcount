# Preprocesamiento de imagenes

import numpy as np


def rgb_a_gris(imagen):
    """Convierte una imagen RGB a escala de grises.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B"""
    if imagen.ndim == 2:
        return imagen.astype(np.uint8)

    if imagen.ndim == 3 and imagen.shape[2] == 1:
        return imagen[:, :, 0].astype(np.uint8)

    pesos = np.array([0.299, 0.587, 0.114])
    gris = np.dot(imagen[:, :, :3].astype(np.float64), pesos)
    return np.clip(gris, 0, 255).astype(np.uint8)


def normalizar(imagen, nuevo_min=0, nuevo_max=255):
    """Normalizacion min-max al rango [nuevo_min, nuevo_max]."""
    img = imagen.astype(np.float64)
    viejo_min = img.min()
    viejo_max = img.max()

    if viejo_max - viejo_min == 0:
        return np.full_like(imagen, int(nuevo_min), dtype=np.uint8)

    resultado = (img - viejo_min) / (viejo_max - viejo_min) * (nuevo_max - nuevo_min) + nuevo_min
    return np.clip(resultado, nuevo_min, nuevo_max).astype(np.uint8)


def redimensionar(imagen, escala):
    """Redimensiona una imagen con interpolacion bilineal.
    escala > 1 agranda, escala < 1 reduce."""
    if imagen.ndim == 2:
        h, w = imagen.shape
        nuevo_h, nuevo_w = int(h * escala), int(w * escala)
        return _interpolacion_bilineal(imagen, nuevo_h, nuevo_w)
    else:
        h, w, c = imagen.shape
        nuevo_h, nuevo_w = int(h * escala), int(w * escala)
        resultado = np.zeros((nuevo_h, nuevo_w, c), dtype=np.uint8)
        for ch in range(c):
            resultado[:, :, ch] = _interpolacion_bilineal(imagen[:, :, ch], nuevo_h, nuevo_w)
        return resultado


def _interpolacion_bilineal(canal, nuevo_h, nuevo_w):
    """Interpolacion bilineal para un canal 2D."""
    h, w = canal.shape
    img = canal.astype(np.float64)

    ratio_filas = h / nuevo_h
    ratio_cols = w / nuevo_w

    coord_filas = np.arange(nuevo_h) * ratio_filas
    coord_cols = np.arange(nuevo_w) * ratio_cols

    f0 = np.floor(coord_filas).astype(int)
    c0 = np.floor(coord_cols).astype(int)
    f1 = np.minimum(f0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)

    df = coord_filas - f0
    dc = coord_cols - c0

    f0_grid, c0_grid = np.meshgrid(f0, c0, indexing='ij')
    f1_grid, c1_grid = np.meshgrid(f1, c1, indexing='ij')
    df_grid, dc_grid = np.meshgrid(df, dc, indexing='ij')

    sup_izq = img[f0_grid, c0_grid]
    sup_der = img[f0_grid, c1_grid]
    inf_izq = img[f1_grid, c0_grid]
    inf_der = img[f1_grid, c1_grid]

    superior = sup_izq * (1 - dc_grid) + sup_der * dc_grid
    inferior = inf_izq * (1 - dc_grid) + inf_der * dc_grid
    resultado = superior * (1 - df_grid) + inferior * df_grid

    return np.clip(resultado, 0, 255).astype(np.uint8)
