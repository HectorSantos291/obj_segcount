# Propiedades geometricas de objetos segmentados

import numpy as np


def medir_area(imagen_etiquetas, id_etiqueta):
    """Cuenta los pixeles del objeto."""
    return int(np.sum(imagen_etiquetas == id_etiqueta))


def medir_perimetro(imagen_etiquetas, id_etiqueta):
    """Cuenta pixeles de borde (que tienen al menos un vecino que no es del objeto)."""
    mascara = (imagen_etiquetas == id_etiqueta)
    h, w = mascara.shape
    perimetro = 0

    relleno = np.pad(mascara, 1, mode='constant', constant_values=False)

    for i in range(h):
        for j in range(w):
            if mascara[i, j]:
                pi, pj = i + 1, j + 1
                if (not relleno[pi - 1, pj] or not relleno[pi + 1, pj] or
                        not relleno[pi, pj - 1] or not relleno[pi, pj + 1]):
                    perimetro += 1

    return perimetro


def medir_centroide(imagen_etiquetas, id_etiqueta):
    """Calcula el centroide del objeto. Retorna (cy, cx)."""
    posiciones = np.argwhere(imagen_etiquetas == id_etiqueta)
    if len(posiciones) == 0:
        return (0.0, 0.0)

    cy = float(np.mean(posiciones[:, 0]))
    cx = float(np.mean(posiciones[:, 1]))
    return (cy, cx)


def medir_bbox(imagen_etiquetas, id_etiqueta):
    """Rectangulo envolvente. Retorna (y_min, x_min, y_max, x_max)."""
    posiciones = np.argwhere(imagen_etiquetas == id_etiqueta)
    if len(posiciones) == 0:
        return (0, 0, 0, 0)

    y_min = int(posiciones[:, 0].min())
    y_max = int(posiciones[:, 0].max())
    x_min = int(posiciones[:, 1].min())
    x_max = int(posiciones[:, 1].max())
    return (y_min, x_min, y_max, x_max)


def medir_circularidad(area, perimetro):
    """Circularidad = 4*pi*area / perimetro^2. Un circulo perfecto da ~1.0."""
    if perimetro == 0:
        return 0.0
    return (4 * np.pi * area) / (perimetro ** 2)


def medir_todo(imagen_etiquetas):
    """Mide las propiedades de todos los objetos.
    Retorna lista de dicts con id, area, perimetro, centroide, bbox y circularidad."""
    max_etiqueta = imagen_etiquetas.max()
    resultados = []

    for id_etiqueta in range(1, max_etiqueta + 1):
        area = medir_area(imagen_etiquetas, id_etiqueta)
        if area == 0:
            continue

        perimetro = medir_perimetro(imagen_etiquetas, id_etiqueta)
        centroide = medir_centroide(imagen_etiquetas, id_etiqueta)
        bbox = medir_bbox(imagen_etiquetas, id_etiqueta)
        circularidad = medir_circularidad(area, perimetro)

        resultados.append({
            'id': id_etiqueta,
            'area': area,
            'perimetro': perimetro,
            'centroide': centroide,
            'bbox': bbox,
            'circularidad': round(circularidad, 4),
        })

    return resultados
