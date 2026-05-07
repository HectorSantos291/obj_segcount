import numpy as np
from obj_segcount.filtering import convolucion2d

# Kernels de Sobel
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

# Kernels de Prewitt (pesos uniformes, sin enfasis en el centro)
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
    return convolucion2d(imagen.astype(np.float64), SOBEL_X)


def sobel_y(imagen):
    return convolucion2d(imagen.astype(np.float64), SOBEL_Y)


def magnitud_gradiente(imagen):
    # sqrt(Gx^2 + Gy^2), normalizado a [0, 255]
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    if magnitud.max() > 0:
        magnitud = (magnitud / magnitud.max()) * 255
    return magnitud.astype(np.uint8)


def direccion_gradiente(imagen):
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    return np.arctan2(gy, gx)


def deteccion_bordes(imagen, umbral=0.1):
    # Aplica Sobel y umbraliza la magnitud del gradiente
    gx = sobel_x(imagen)
    gy = sobel_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    mag_max = magnitud.max()
    if mag_max > 0:
        magnitud = magnitud / mag_max

    bordes = np.zeros_like(magnitud, dtype=np.uint8)
    bordes[magnitud >= umbral] = 255
    return bordes


def prewitt_x(imagen):
    return convolucion2d(imagen.astype(np.float64), PREWITT_X)


def prewitt_y(imagen):
    return convolucion2d(imagen.astype(np.float64), PREWITT_Y)


def deteccion_prewitt(imagen, umbral=0.1):
    # Igual que Sobel pero usando los kernels de Prewitt
    gx = prewitt_x(imagen)
    gy = prewitt_y(imagen)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)

    mag_max = magnitud.max()
    if mag_max > 0:
        magnitud = magnitud / mag_max

    bordes = np.zeros_like(magnitud, dtype=np.uint8)
    bordes[magnitud >= umbral] = 255
    return bordes


def laplaciano(imagen, usar_diagonales=False):
    # usar_diagonales=False usa 4 vecinos, True usa 8 vecinos
    kernel = LAPLACIANO_8 if usar_diagonales else LAPLACIANO_4
    return convolucion2d(imagen.astype(np.float64), kernel)
