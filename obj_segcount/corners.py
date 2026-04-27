# Deteccion de esquinas - Harris

import numpy as np
from obj_segcount.filtering import convolucion2d, kernel_gaussiano
from obj_segcount.edges import sobel_x, sobel_y


def detector_harris(imagen, k=0.04, umbral=0.01,
                    tamano_ventana=5, sigma=1.0):
    """Detector de esquinas de Harris.
    1. Gradientes Ix, Iy con Sobel
    2. Productos Ix^2, Iy^2, Ix*Iy
    3. Suavizar con ventana gaussiana
    4. R = det(M) - k * trace(M)^2
    5. Umbral + supresion de no-maximos
    Retorna mascara booleana con las esquinas."""
    img = imagen.astype(np.float64)

    Ix = sobel_x(img)
    Iy = sobel_y(img)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Suavizar con gaussiana
    gauss = kernel_gaussiano(tamano_ventana, sigma)
    Sxx = convolucion2d(Ixx, gauss)
    Syy = convolucion2d(Iyy, gauss)
    Sxy = convolucion2d(Ixy, gauss)

    # Respuesta de Harris
    det_M = Sxx * Syy - Sxy ** 2
    traza_M = Sxx + Syy
    R = det_M - k * (traza_M ** 2)

    R_max = R.max()
    if R_max <= 0:
        return np.zeros(img.shape, dtype=bool)

    esquinas = R > (umbral * R_max)
    esquinas = _supresion_no_maximos(R, esquinas)
    return esquinas


def _supresion_no_maximos(R, esquinas, tamano=3):
    """Solo mantiene el maximo local en ventana tamano x tamano."""
    h, w = R.shape
    pad = tamano // 2
    resultado = np.zeros_like(esquinas, dtype=bool)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            if esquinas[i, j]:
                region = R[i - pad:i + pad + 1, j - pad:j + pad + 1]
                if R[i, j] == region.max():
                    resultado[i, j] = True

    return resultado


def marcar_esquinas(imagen, esquinas, color=(255, 0, 0), radio=3):
    """Dibuja circulos sobre las esquinas detectadas."""
    if imagen.ndim == 2:
        resultado = np.stack([imagen, imagen, imagen], axis=2).astype(np.uint8)
    else:
        resultado = imagen.copy().astype(np.uint8)

    h, w = esquinas.shape
    posiciones = np.argwhere(esquinas)

    for cy, cx in posiciones:
        y_min = max(0, cy - radio)
        y_max = min(h, cy + radio + 1)
        x_min = max(0, cx - radio)
        x_max = min(w, cx + radio + 1)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radio ** 2:
                    resultado[y, x] = color

    return resultado
