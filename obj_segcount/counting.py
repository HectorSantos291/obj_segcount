import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from obj_segcount.preprocessing import rgb_a_gris, normalizar
from obj_segcount.intensity import ecualizacion_histograma, estiramiento_contraste
from obj_segcount.filtering import filtro_gaussiano, filtro_mediana
from obj_segcount.edges import magnitud_gradiente, deteccion_bordes
from obj_segcount.thresholding import umbral_otsu, umbral_adaptativo, umbral_global
from obj_segcount.segmentation import componentes_conectados, quitar_regiones_pequenas, quitar_regiones_grandes, quitar_regiones_borde
from obj_segcount.corners import detector_harris, marcar_esquinas
from obj_segcount.properties import medir_todo


def contar_objetos(imagen, tipo_filtro='gaussiano', tamano_filtro=5,
                   sigma_filtro=1.0, metodo_umbral='otsu',
                   valor_umbral=128, tamano_bloque=15, C_adaptativo=5,
                   min_tamano_objeto=50, max_tamano_objeto=None,
                   quitar_bordes=True, ecualizar=True,
                   invertir=False, conectividad=8, mostrar_pasos=False):
    resultado = {}

    # Convertir a escala de grises
    if imagen.ndim == 3:
        gris = rgb_a_gris(imagen)
    else:
        gris = imagen.copy()
    resultado['gris'] = gris

    # Ecualizar histograma para mejorar contraste
    if ecualizar:
        mejorada = ecualizacion_histograma(gris)
    else:
        mejorada = gris.copy()
    resultado['mejorada'] = mejorada

    # Filtrado espacial para reducir ruido
    if tipo_filtro == 'gaussiano':
        filtrada = filtro_gaussiano(mejorada, tamano=tamano_filtro, sigma=sigma_filtro)
    elif tipo_filtro == 'mediana':
        filtrada = filtro_mediana(mejorada, tamano=tamano_filtro)
    else:
        filtrada = mejorada.copy()
    resultado['filtrada'] = filtrada

    # Umbralizacion para obtener imagen binaria
    if metodo_umbral == 'otsu':
        binaria, val_otsu = umbral_otsu(filtrada)
        resultado['valor_umbral'] = val_otsu
    elif metodo_umbral == 'adaptativo':
        binaria = umbral_adaptativo(filtrada, tamano_bloque=tamano_bloque, C=C_adaptativo)
        resultado['valor_umbral'] = -1
    else:
        binaria = umbral_global(filtrada, T=valor_umbral)
        resultado['valor_umbral'] = valor_umbral

    if invertir:
        binaria = 255 - binaria
    resultado['binaria'] = binaria

    # Deteccion de bordes con Sobel
    bordes = deteccion_bordes(gris, umbral=0.15)
    resultado['bordes'] = bordes

    # Segmentacion por componentes conectados
    etiquetas, num_objetos = componentes_conectados(binaria, conectividad=conectividad)
    etiquetas, num_objetos = quitar_regiones_pequenas(etiquetas, min_tamano=min_tamano_objeto)

    if max_tamano_objeto is not None:
        etiquetas, num_objetos = quitar_regiones_grandes(etiquetas, max_tamano=max_tamano_objeto)

    if quitar_bordes:
        etiquetas, num_objetos = quitar_regiones_borde(etiquetas)

    resultado['etiquetas'] = etiquetas
    resultado['conteo'] = num_objetos

    # Medir propiedades de cada objeto
    props = medir_todo(etiquetas)
    resultado['propiedades'] = props

    if mostrar_pasos:
        visualizar_resultados(imagen, resultado)

    return resultado


def visualizar_resultados(imagen, resultado):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Segmentacion - {resultado['conteo']} objetos detectados", fontsize=14)

    ax = axes[0, 0]
    if imagen.ndim == 3:
        ax.imshow(imagen)
    else:
        ax.imshow(imagen, cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(resultado['mejorada'], cmap='gray')
    ax.set_title("Ecualizacion de histograma")
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(resultado['filtrada'], cmap='gray')
    ax.set_title("Filtrado espacial")
    ax.axis('off')

    ax = axes[1, 0]
    ax.imshow(resultado['binaria'], cmap='gray')
    info_umbral = ""
    if 'valor_umbral' in resultado and resultado['valor_umbral'] >= 0:
        info_umbral = f" (T={resultado['valor_umbral']})"
    ax.set_title(f"Umbralizacion{info_umbral}")
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(resultado['bordes'], cmap='gray')
    ax.set_title("Bordes (Sobel)")
    ax.axis('off')

    ax = axes[1, 2]
    etiquetas = resultado['etiquetas']
    coloreada = _colorear_etiquetas(etiquetas)
    ax.imshow(coloreada)

    for prop in resultado['propiedades']:
        y_min, x_min, y_max, x_max = prop['bbox']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        cy, cx = prop['centroide']
        ax.text(cx, cy, str(prop['id']), color='white', fontsize=10,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

    ax.set_title(f"Segmentacion - {resultado['conteo']} objetos")
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    _imprimir_tabla(resultado['propiedades'])


# Colores para pintar cada objeto detectado
COLORES_OBJETOS = [
    (255,  60,  60),
    ( 60, 180, 255),
    ( 60, 220,  60),
    (255, 200,   0),
    (255, 100, 255),
    (  0, 220, 200),
    (255, 140,   0),
    (160,  80, 255),
    (200, 255,  80),
    (255,  80, 140),
]


def _colorear_etiquetas(imagen_etiquetas):
    h, w = imagen_etiquetas.shape
    coloreada = np.zeros((h, w, 3), dtype=np.uint8)
    max_etiqueta = imagen_etiquetas.max()
    if max_etiqueta == 0:
        return coloreada

    for id_etiqueta in range(1, max_etiqueta + 1):
        color = COLORES_OBJETOS[(id_etiqueta - 1) % len(COLORES_OBJETOS)]
        mascara = imagen_etiquetas == id_etiqueta
        coloreada[mascara] = color

    return coloreada


def _imprimir_tabla(propiedades):
    if not propiedades:
        print("No se detectaron objetos.")
        return

    print(f"\n{'ID':>4} | {'Area':>8} | {'Perimetro':>10} | {'Centroide':>16} | {'Circularidad':>13}")
    print("-" * 65)

    for prop in propiedades:
        cy, cx = prop['centroide']
        print(f"{prop['id']:>4} | {prop['area']:>8} | {prop['perimetro']:>10} | "
              f"({cy:7.1f}, {cx:6.1f}) | {prop['circularidad']:>13.4f}")

    print(f"\nTotal: {len(propiedades)} objetos\n")



