# Deteccion de bordes
# Sobel, Prewitt y Laplaciano

import numpy as np
from obj_segcount.filtering import convolucion2d

# Kernels de Sobel
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

# Kernels de Prewitt (pesos uniformes, a diferencia de Sobel)
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)

PREWITT_Y = np.array([[ 1,  1,  1],
                      [ 0,  0,  0],
                      [-1, -1, -1]], dtype=np.float64)

# Kernels del Laplaciano
LAPLACIANO_4 = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float64)

LAPLACIANO_8 = np.array([[1,  1, 1],
                         [1, -8, 1],
                         [1,  1, 1]], dtype=np.float64)


def sobel_x(imagen):
    """Derivada en X con Sobel."""
    return convolucion2d(imagen.astype(np.float64), SOBEL_X)


def sobel_y(imagen):
    """Derivada en Y con Sobel."""
    return convolucion2d(imagen.astype(np.float64), SOBEL_Y)


def magnitud_gradiente(imagen):
    """Magnitud del gradiente: sqrt(Gx^2 + Gy^2), normalizada a [0,255]."""
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    if magnitud.max() > 0:
        magnitud = (magnitud / magnitud.max()) * 255
    return magnitud.astype(np.uint8)


def direccion_gradiente(imagen):
    """Direccion del gradiente en radianes."""
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    return np.arctan2(gy, gx)


def deteccion_bordes(imagen, umbral=0.1):
    """Deteccion de bordes con Sobel + umbral relativo (0 a 1)."""
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    mag_max = magnitud.max()
    if mag_max > 0:
        magnitud = magnitud / mag_max

    bordes = np.zeros_like(magnitud, dtype=np.uint8)
    bordes[magnitud >= umbral] = 255
    return bordes


# --- Prewitt ---

def prewitt_x(imagen):
    """Derivada en X con Prewitt."""
    return convolucion2d(imagen.astype(np.float64), PREWITT_X)


def prewitt_y(imagen):
    """Derivada en Y con Prewitt."""
    return convolucion2d(imagen.astype(np.float64), PREWITT_Y)


def deteccion_prewitt(imagen, umbral=0.1):
    """Deteccion de bordes con Prewitt + umbral."""
    gx = prewitt_x(imagen)
    gy = prewitt_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    mag_max = magnitud.max()
    if mag_max > 0:
        magnitud = magnitud / mag_max

    bordes = np.zeros_like(magnitud, dtype=np.uint8)
    bordes[magnitud >= umbral] = 255
    return bordes


# --- Laplaciano ---

def laplaciano(imagen, usar_diagonales=False):
    """Aplica el operador Laplaciano.
    usar_diagonales=False usa 4 vecinos, True usa 8 vecinos."""
    kernel = LAPLACIANO_8 if usar_diagonales else LAPLACIANO_4
    resultado = convolucion2d(imagen.astype(np.float64), kernel)
    return resultado
