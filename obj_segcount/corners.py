import numpy as np
from obj_segcount.filtering import convolucion2d, kernel_gaussiano
from obj_segcount.edges import sobel_x, sobel_y


def detector_harris(imagen, k=0.04, umbral=0.01, tamano_ventana=5, sigma=1.0):
    # Calcula la respuesta R de Harris usando gradientes Sobel y una ventana gaussiana
    img = imagen.astype(np.float64)

    Ix = sobel_x(img)
    Iy = sobel_y(img)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    gauss = kernel_gaussiano(tamano_ventana, sigma)
    Sxx = convolucion2d(Ixx, gauss)
    Syy = convolucion2d(Iyy, gauss)
    Sxy = convolucion2d(Ixy, gauss)

    det_M = Sxx * Syy - Sxy ** 2
    traza_M = Sxx + Syy
    R = det_M - k * (traza_M ** 2)

    R_max = R.max()
    if R_max <= 0:
        return np.zeros(img.shape, dtype=bool)

    esquinas = R > (umbral * R_max)
    return esquinas


def marcar_esquinas(imagen, esquinas, color=(255, 0, 0), radio=3):
    # Pinta un circulo del color indicado en cada esquina detectada
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
