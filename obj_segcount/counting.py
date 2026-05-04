# Pipeline de conteo
# Integra todos los modulos: preprocesamiento -> umbralizacion -> segmentacion -> conteo

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
    """Pipeline completo de segmentacion y conteo.

    Pasos: gris -> [ecualizacion] -> filtrado -> umbralizacion ->
           segmentacion -> filtrar regiones -> medir propiedades

    Retorna dict con: conteo, imagen_etiquetas, propiedades, imagen_binaria, etc.
    """
    resultado = {}

    # Paso 1: escala de grises
    if imagen.ndim == 3:
        gris = rgb_a_gris(imagen)
    else:
        gris = imagen.copy()
    resultado['gris'] = gris

    # Paso 2: ecualizacion de histograma (opcional)
    if ecualizar:
        mejorada = ecualizacion_histograma(gris)
    else:
        mejorada = gris.copy()
    resultado['mejorada'] = mejorada

    # Paso 3: filtrado
    if tipo_filtro == 'gaussiano':
        filtrada = filtro_gaussiano(mejorada, tamano=tamano_filtro, sigma=sigma_filtro)
    elif tipo_filtro == 'mediana':
        filtrada = filtro_mediana(mejorada, tamano=tamano_filtro)
    else:
        filtrada = mejorada.copy()
    resultado['filtrada'] = filtrada

    # Paso 4: umbralizacion
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

    # Paso 5: bordes
    bordes = deteccion_bordes(gris, umbral=0.15)
    resultado['bordes'] = bordes

    # Paso 6: segmentacion
    etiquetas, num_objetos = componentes_conectados(binaria, conectividad=conectividad)
    etiquetas, num_objetos = quitar_regiones_pequenas(etiquetas, min_tamano=min_tamano_objeto)

    if max_tamano_objeto is not None:
        etiquetas, num_objetos = quitar_regiones_grandes(etiquetas, max_tamano=max_tamano_objeto)

    if quitar_bordes:
        etiquetas, num_objetos = quitar_regiones_borde(etiquetas)

    resultado['etiquetas'] = etiquetas
    resultado['conteo'] = num_objetos

    # Paso 7: propiedades
    props = medir_todo(etiquetas)
    resultado['propiedades'] = props

    if mostrar_pasos:
        visualizar_resultados(imagen, resultado)

    return resultado


def visualizar_resultados(imagen, resultado):
    """Muestra los resultados del pipeline en 6 subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Segmentacion y Conteo - {resultado['conteo']} objetos detectados",
                 fontsize=14, fontweight='bold')

    # Original
    ax = axes[0, 0]
    if imagen.ndim == 3:
        ax.imshow(imagen)
    else:
        ax.imshow(imagen, cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    # Ecualizacion
    ax = axes[0, 1]
    ax.imshow(resultado['mejorada'], cmap='gray')
    ax.set_title("Ecualizacion de histograma")
    ax.axis('off')

    # Filtrado
    ax = axes[0, 2]
    ax.imshow(resultado['filtrada'], cmap='gray')
    ax.set_title("Filtrado espacial")
    ax.axis('off')

    # Binaria
    ax = axes[1, 0]
    ax.imshow(resultado['binaria'], cmap='gray')
    info_umbral = ""
    if 'valor_umbral' in resultado and resultado['valor_umbral'] >= 0:
        info_umbral = f" (T={resultado['valor_umbral']})"
    ax.set_title(f"Umbralizacion{info_umbral}")
    ax.axis('off')

    # Bordes
    ax = axes[1, 1]
    ax.imshow(resultado['bordes'], cmap='gray')
    ax.set_title("Bordes (Sobel)")
    ax.axis('off')

    # Segmentacion con bounding boxes
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Tabla de propiedades
    _imprimir_tabla(resultado['propiedades'])


def _colorear_etiquetas(imagen_etiquetas):
    """Asigna un color diferente a cada etiqueta."""
    h, w = imagen_etiquetas.shape
    coloreada = np.zeros((h, w, 3), dtype=np.uint8)
    max_etiqueta = imagen_etiquetas.max()
    if max_etiqueta == 0:
        return coloreada

    for id_etiqueta in range(1, max_etiqueta + 1):
        tono = (id_etiqueta - 1) / max_etiqueta
        color = _hsv_a_rgb(tono, 0.8, 0.9)
        mascara = imagen_etiquetas == id_etiqueta
        coloreada[mascara] = color

    return coloreada


def _hsv_a_rgb(h, s, v):
    """Conversion HSV a RGB."""
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def _imprimir_tabla(propiedades):
    """Imprime tabla con las propiedades de cada objeto."""
    if not propiedades:
        print("No se detectaron objetos.")
        return

    print("\n" + "=" * 75)
    print(f"{'ID':>4} | {'Area':>8} | {'Perimetro':>10} | {'Centroide':>16} | "
          f"{'Circularidad':>13}")
    print("-" * 75)

    for prop in propiedades:
        cy, cx = prop['centroide']
        print(f"{prop['id']:>4} | {prop['area']:>8} | {prop['perimetro']:>10} | "
              f"({cy:7.1f}, {cx:6.1f}) | {prop['circularidad']:>13.4f}")

    print("=" * 75)
    print(f"Total de objetos: {len(propiedades)}")
    print("=" * 75 + "\n")


# ============================================================================
# FUNCIONES CON CONFIGURACIONES PREDEFINIDAS
# ============================================================================

def contar_objetos_claros(imagen, mostrar_pasos=True):
    """Configuracion predefinida para objetos CLAROS sobre fondo OSCURO.
    
    Ideal para: monedas, piezas metalicas, objetos brillantes.
    
    Caracteristicas:
    - Umbral adaptativo (para iluminacion no uniforme)
    - Sin inversion de mascara
    - Filtro suave para evitar fusionar objetos cercanos
    - Area minima 200 pixeles
    
    Args:
        imagen: Imagen RGB o escala de grises
        mostrar_pasos: Si True, visualiza el pipeline completo
        
    Returns:
        dict con resultados (conteo, etiquetas, propiedades, etc.)
    """
    return contar_objetos(
        imagen,
        tipo_filtro='gaussiano',
        sigma_filtro=0.5,              # sigma bajo para no fusionar
        metodo_umbral='adaptativo',
        tamano_bloque=15,
        C_adaptativo=5,
        min_tamano_objeto=200,
        max_tamano_objeto=50000,
        quitar_bordes=False,
        ecualizar=False,               # no ecualizar con adaptativo
        invertir=False,                # objetos claros = blancos
        mostrar_pasos=mostrar_pasos
    )


def contar_objetos_oscuros(imagen, mostrar_pasos=True):
    """Configuracion predefinida para objetos OSCUROS sobre fondo CLARO.
    
    Ideal para: tornillos sobre mesa blanca, piezas oscuras, objetos opacos.
    
    Caracteristicas:
    - Umbral de Otsu (automatico)
    - Inversion de mascara activada
    - Ecualizacion de histograma para mejorar contraste
    - Area minima 100 pixeles
    
    Args:
        imagen: Imagen RGB o escala de grises
        mostrar_pasos: Si True, visualiza el pipeline completo
        
    Returns:
        dict con resultados (conteo, etiquetas, propiedades, etc.)
    """
    return contar_objetos(
        imagen,
        tipo_filtro='gaussiano',
        sigma_filtro=1.5,
        metodo_umbral='otsu',
        min_tamano_objeto=100,
        max_tamano_objeto=50000,
        quitar_bordes=True,
        ecualizar=True,                # mejorar contraste
        invertir=True,                 # objetos oscuros -> blancos
        mostrar_pasos=mostrar_pasos
    )


def contar_objetos_pequenos(imagen, mostrar_pasos=True):
    """Configuracion predefinida para objetos PEQUENOS y CERCANOS.
    
    Ideal para: granos (cafe, arroz, semillas), componentes electronicos pequenos.
    
    Caracteristicas:
    - Umbral de Otsu
    - Filtro muy suave (sigma bajo) para evitar fusion
    - Area minima MUY baja (50 pixeles)
    - Inversion de mascara activada
    
    Args:
        imagen: Imagen RGB o escala de grises
        mostrar_pasos: Si True, visualiza el pipeline completo
        
    Returns:
        dict con resultados (conteo, etiquetas, propiedades, etc.)
    """
    return contar_objetos(
        imagen,
        tipo_filtro='gaussiano',
        sigma_filtro=0.5,              # sigma muy bajo
        metodo_umbral='otsu',
        min_tamano_objeto=50,          # umbral bajo para pequenos
        max_tamano_objeto=5000,        # evitar regiones grandes
        quitar_bordes=True,
        ecualizar=True,
        invertir=True,
        mostrar_pasos=mostrar_pasos
    )
