# Segmentacion por regiones

import numpy as np
from collections import deque


def componentes_conectados(imagen_binaria, conectividad=8):
    """Etiquetado de componentes conectados usando BFS.
    Retorna (imagen_etiquetas, num_componentes)."""
    h, w = imagen_binaria.shape
    etiquetas = np.zeros((h, w), dtype=np.int32)
    etiqueta_actual = 0

    if conectividad == 4:
        vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        vecinos = [(-1, -1), (-1, 0), (-1, 1),
                   (0,  -1),          (0,  1),
                   (1,  -1), (1,  0), (1,  1)]

    for i in range(h):
        for j in range(w):
            if imagen_binaria[i, j] == 255 and etiquetas[i, j] == 0:
                etiqueta_actual += 1
                cola = deque()
                cola.append((i, j))
                etiquetas[i, j] = etiqueta_actual

                while cola:
                    cy, cx = cola.popleft()
                    for dy, dx in vecinos:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and
                                imagen_binaria[ny, nx] == 255 and
                                etiquetas[ny, nx] == 0):
                            etiquetas[ny, nx] = etiqueta_actual
                            cola.append((ny, nx))

    return etiquetas, etiqueta_actual


def crecimiento_regiones(imagen, semillas, tolerancia=10):
    """Segmentacion por crecimiento de regiones desde puntos semilla."""
    img = imagen.astype(np.float64)
    h, w = img.shape
    segmentada = np.zeros((h, w), dtype=np.uint8)
    visitado = np.zeros((h, w), dtype=bool)

    vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for sem_y, sem_x in semillas:
        if visitado[sem_y, sem_x]:
            continue

        cola = deque()
        cola.append((sem_y, sem_x))
        visitado[sem_y, sem_x] = True
        segmentada[sem_y, sem_x] = 255

        suma_region = img[sem_y, sem_x]
        conteo_region = 1

        while cola:
            cy, cx = cola.popleft()
            media_region = suma_region / conteo_region

            for dy, dx in vecinos:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < h and 0 <= nx < w and not visitado[ny, nx]):
                    visitado[ny, nx] = True
                    if abs(img[ny, nx] - media_region) <= tolerancia:
                        segmentada[ny, nx] = 255
                        cola.append((ny, nx))
                        suma_region += img[ny, nx]
                        conteo_region += 1

    return segmentada


def quitar_regiones_pequenas(imagen_etiquetas, min_tamano):
    """Quita regiones con area menor a min_tamano pixeles."""
    max_etiqueta = imagen_etiquetas.max()
    nuevas_etiquetas = np.zeros_like(imagen_etiquetas)
    nuevo_conteo = 0

    for id_etiqueta in range(1, max_etiqueta + 1):
        mascara = (imagen_etiquetas == id_etiqueta)
        if np.sum(mascara) >= min_tamano:
            nuevo_conteo += 1
            nuevas_etiquetas[mascara] = nuevo_conteo

    return nuevas_etiquetas, nuevo_conteo


def quitar_regiones_grandes(imagen_etiquetas, max_tamano):
    """Quita regiones con area mayor a max_tamano pixeles."""
    max_etiqueta = imagen_etiquetas.max()
    nuevas_etiquetas = np.zeros_like(imagen_etiquetas)
    nuevo_conteo = 0

    for id_etiqueta in range(1, max_etiqueta + 1):
        mascara = (imagen_etiquetas == id_etiqueta)
        if np.sum(mascara) <= max_tamano:
            nuevo_conteo += 1
            nuevas_etiquetas[mascara] = nuevo_conteo

    return nuevas_etiquetas, nuevo_conteo


def quitar_regiones_borde(imagen_etiquetas):
    """Elimina objetos que tocan el borde de la imagen."""
    h, w = imagen_etiquetas.shape

    etiquetas_borde = set()
    etiquetas_borde.update(imagen_etiquetas[0, :].tolist())
    etiquetas_borde.update(imagen_etiquetas[h - 1, :].tolist())
    etiquetas_borde.update(imagen_etiquetas[:, 0].tolist())
    etiquetas_borde.update(imagen_etiquetas[:, w - 1].tolist())
    etiquetas_borde.discard(0)

    filtrada = imagen_etiquetas.copy()
    for id_etiqueta in etiquetas_borde:
        filtrada[filtrada == id_etiqueta] = 0

    # Re-etiquetar
    unicas = sorted(set(filtrada.ravel()) - {0})
    nuevas = np.zeros_like(filtrada)
    for nuevo_id, viejo_id in enumerate(unicas, start=1):
        nuevas[filtrada == viejo_id] = nuevo_id

    return nuevas, len(unicas)
