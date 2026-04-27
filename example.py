# Ejemplo de uso de obj_segcount
# Script principal de demostracion. Ejecuta:
#   1. Pruebas con las imagenes de la carpeta images/ (coins, tornillos, etc.)
#   2. Imagen sintetica (fallback si no hay imagenes o no hay OpenCV)
#   3. Demo modular paso a paso
#   4. Comparativa de detectores de bordes y esquinas (Sobel, Prewitt,
#      Laplaciano, Harris)

import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import cv2
    TIENE_OPENCV = True
except ImportError:
    TIENE_OPENCV = False

import obj_segcount as osc


# =========================================================
# Configuraciones por imagen
# =========================================================
# Cada imagen puede requerir parametros distintos segun sus caracteristicas
# (iluminacion, tipo de objetos, contraste, etc.). Si una imagen no tiene
# entrada aqui, se usa CONFIG_DEFAULT (objetos oscuros sobre fondo claro).

CONFIG_DEFAULT = {
    'tipo_filtro': 'gaussiano',
    'tamano_filtro': 5,
    'sigma_filtro': 1.5,
    'metodo_umbral': 'otsu',
    'min_tamano_objeto': 100,
    'quitar_bordes': True,
    'ecualizar': True,
    'invertir': True,  # objetos oscuros sobre fondo claro
    'mostrar_pasos': True,
}

# Configuraciones especificas. La clave es una subcadena del nombre
# del archivo (sin extension, en minusculas).
CONFIGS_ESPECIFICAS = {
    # Monedas: iluminacion no uniforme -> umbral adaptativo.
    # sigma bajo para no fusionar monedas cercanas.
    # max_tamano filtra regiones muy grandes (fondo mal segmentado).
    'coins': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 0.5,
        'metodo_umbral': 'adaptativo',
        'tamano_bloque': 15,
        'C_adaptativo': 5,
        'min_tamano_objeto': 200,
        'max_tamano_objeto': 5000,
        'quitar_bordes': False,
        'ecualizar': False,
        'invertir': False,
        'mostrar_pasos': True,
    },
}


def obtener_config(nombre_archivo):
    """Devuelve la configuracion apropiada segun el nombre de la imagen."""
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0].lower()
    for clave, config in CONFIGS_ESPECIFICAS.items():
        if clave in nombre_sin_ext:
            return config
    return CONFIG_DEFAULT


# =========================================================
# Utilidades
# =========================================================

def crear_imagen_sintetica():
    """Crea una imagen de prueba con objetos sobre fondo claro."""
    h, w = 480, 640
    imagen = np.full((h, w, 3), 200, dtype=np.uint8)

    # Ruido en el fondo
    ruido = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    imagen = np.clip(imagen.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    # Objetos oscuros (simulando piezas)
    objetos = [
        ('circulo', 100, 100, 30),
        ('circulo', 250, 150, 25),
        ('circulo', 400, 100, 35),
        ('circulo', 530, 120, 28),
        ('rect', 80, 280, 60, 40),
        ('rect', 250, 300, 50, 70),
        ('rect', 450, 280, 45, 55),
        ('elipse', 150, 420, 40, 20),
        ('elipse', 350, 400, 35, 18),
        ('elipse', 520, 380, 30, 15),
    ]

    for obj in objetos:
        if obj[0] == 'circulo':
            _, cx, cy, r = obj
            yy, xx = np.ogrid[:h, :w]
            mascara = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            imagen[mascara] = np.random.randint(30, 80, 3)
        elif obj[0] == 'rect':
            _, x, y, rw, rh = obj
            imagen[y:y + rh, x:x + rw] = np.random.randint(30, 80, 3)
        elif obj[0] == 'elipse':
            _, cx, cy, a, b = obj
            yy, xx = np.ogrid[:h, :w]
            mascara = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
            imagen[mascara] = np.random.randint(30, 80, 3)

    return imagen


def cargar_imagen(ruta):
    """Carga una imagen con OpenCV y la convierte a RGB.
    Si la imagen es en escala de grises, la devuelve como array 2D."""
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Si viene como BGR, convertir a RGB. Si es gris, dejar como esta.
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imprimir_propiedades(resultados, nombre_objeto="Objeto"):
    """Imprime una tabla rapida con area/circularidad/centroide."""
    for p in resultados['propiedades']:
        cy, cx = p['centroide']
        print(f"  {nombre_objeto} {p['id']}: area={p['area']}, "
              f"circularidad={p['circularidad']:.3f}, "
              f"centroide=({cy:.0f}, {cx:.0f})")


# =========================================================
# Demos de pipeline
# =========================================================

def demo_imagen(ruta):
    """Ejecuta el pipeline sobre una imagen con la config adecuada segun su nombre."""
    nombre = os.path.basename(ruta)
    print("\n" + "=" * 60)
    print(f" DEMO: {nombre}")
    print("=" * 60)

    img = cargar_imagen(ruta)
    if img is None:
        print(f"  No se pudo cargar {ruta}")
        return None, None
    print(f"  Tamanio: {img.shape}")

    config = obtener_config(nombre)
    resultados = osc.contar_objetos(img, **config)

    print(f"\n  Objetos detectados: {resultados['conteo']}")
    if 'valor_umbral' in resultados and resultados['valor_umbral'] >= 0:
        print(f"  Umbral: {resultados['valor_umbral']}")
    imprimir_propiedades(resultados)

    return img, resultados


def demo_sintetica():
    """Fallback si no hay imagenes disponibles."""
    print("\n" + "=" * 60)
    print(" DEMO: Imagen sintetica")
    print("=" * 60)

    img = crear_imagen_sintetica()
    print(f"  Tamanio: {img.shape}")

    resultados = osc.contar_objetos(img, **CONFIG_DEFAULT)

    print(f"\n  Objetos detectados: {resultados['conteo']}")
    imprimir_propiedades(resultados)

    return img, resultados


# =========================================================
# Demo modular (paso a paso)
# =========================================================

def demo_modular(imagen):
    """Uso paso a paso de cada modulo."""
    print("\n" + "=" * 60)
    print(" DEMO MODULAR: Uso paso a paso")
    print("=" * 60)

    if imagen.ndim == 3:
        gris = osc.rgb_a_gris(imagen)
    else:
        gris = imagen.copy()
    print(f"  Gris: {gris.shape}")

    mejorada = osc.ecualizacion_histograma(gris)
    filtrada = osc.filtro_gaussiano(mejorada, tamano=5, sigma=1.5)

    binaria, valor_umbral = osc.umbral_otsu(filtrada)
    binaria = 255 - binaria  # invertir para objetos oscuros
    print(f"  Umbral Otsu: {valor_umbral}")

    etiquetas, conteo = osc.componentes_conectados(binaria)
    etiquetas, conteo = osc.quitar_regiones_pequenas(etiquetas, min_tamano=100)
    etiquetas, conteo = osc.quitar_regiones_borde(etiquetas)
    print(f"  Componentes: {conteo}")

    esquinas = osc.detector_harris(gris, k=0.04, umbral=0.01)
    print(f"  Esquinas Harris: {np.sum(esquinas)}")

    props = osc.medir_todo(etiquetas)
    for p in props:
        print(f"  Obj {p['id']}: area={p['area']}, circ={p['circularidad']:.3f}")

    # Imagen final con objetos marcados (centroide + ID)
    final = imagen.copy() if imagen.ndim == 3 else np.stack([gris]*3, axis=-1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Demo modular - pasos del pipeline", fontsize=14, fontweight='bold')

    axes[0, 0].imshow(imagen, cmap='gray' if imagen.ndim == 2 else None)
    axes[0, 0].set_title("1. Original")

    axes[0, 1].imshow(gris, cmap='gray')
    axes[0, 1].set_title("2. Escala de grises")

    axes[0, 2].imshow(mejorada, cmap='gray')
    axes[0, 2].set_title("3. Ecualizada")

    axes[0, 3].imshow(filtrada, cmap='gray')
    axes[0, 3].set_title("4. Filtro gaussiano")

    axes[1, 0].imshow(binaria, cmap='gray')
    axes[1, 0].set_title(f"5. Binaria (Otsu={valor_umbral})")

    axes[1, 1].imshow(etiquetas, cmap='nipy_spectral')
    axes[1, 1].set_title(f"6. Componentes ({conteo})")

    axes[1, 2].imshow(final)
    axes[1, 2].set_title(f"7. Resultado final ({len(props)} objetos)")
    for p in props:
        cy, cx = p['centroide']
        axes[1, 2].plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        axes[1, 2].annotate(str(p['id']), (cx, cy),
                            color='yellow', fontsize=10, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points')

    axes[1, 3].axis('off')  # celda vacía

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =========================================================
# Comparativa de detectores de bordes y esquinas
# =========================================================

def comparar_bordes(imagen, titulo="Imagen"):
    """Comparativa visual de Sobel, Prewitt, Laplaciano (4 y 8 vecinos) y Harris."""
    print("\n" + "=" * 60)
    print(f" COMPARATIVA DE BORDES Y ESQUINAS: {titulo}")
    print("=" * 60)

    # Asegurar escala de grises para los detectores
    if imagen.ndim == 3:
        gris = osc.rgb_a_gris(imagen)
    else:
        gris = imagen

    bordes_sobel = osc.deteccion_bordes(gris, umbral=0.15)
    bordes_prewitt = osc.deteccion_prewitt(gris, umbral=0.15)
    lap_4 = osc.laplaciano(gris, usar_diagonales=False)
    lap_8 = osc.laplaciano(gris, usar_diagonales=True)
    esquinas = osc.detector_harris(gris, k=0.04, umbral=0.01)

    imagen_para_marcar = imagen if imagen.ndim == 3 else gris
    marcada = osc.marcar_esquinas(imagen_para_marcar, esquinas,
                                   color=(255, 0, 0), radio=3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Comparativa de detectores - {titulo}",
                 fontsize=14, fontweight='bold')

    if imagen.ndim == 3:
        axes[0, 0].imshow(imagen)
    else:
        axes[0, 0].imshow(imagen, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(bordes_sobel, cmap='gray')
    axes[0, 1].set_title("Sobel")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(bordes_prewitt, cmap='gray')
    axes[0, 2].set_title("Prewitt")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(lap_4, cmap='gray')
    axes[1, 0].set_title("Laplaciano (4 vecinos)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(lap_8, cmap='gray')
    axes[1, 1].set_title("Laplaciano (8 vecinos)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(marcada)
    axes[1, 2].set_title(f"Harris ({np.sum(esquinas)} esquinas)")
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    print("obj_segcount - Demo")
    print(f"Version: {osc.__version__}")
    print(f"OpenCV disponible: {TIENE_OPENCV}")

    # ---- Pruebas con las imagenes de la carpeta images/ ----
    carpeta_imgs = os.path.join(os.path.dirname(__file__), "images")
    imagenes = []

    if os.path.exists(carpeta_imgs) and TIENE_OPENCV:
        for archivo in sorted(os.listdir(carpeta_imgs)):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                imagenes.append(os.path.join(carpeta_imgs, archivo))
    elif not TIENE_OPENCV:
        print("\n[!] OpenCV no esta instalado, no se pueden cargar imagenes.")
        print("    Para instalarlo: pip install opencv-python")

    if imagenes:
        # Para cada imagen: pipeline completo + comparativa de bordes y esquinas
        for ruta in imagenes:
            img, _ = demo_imagen(ruta)
            if img is not None:
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                comparar_bordes(img, titulo=nombre)
    else:
        if TIENE_OPENCV:
            print(f"\n[!] No se encontraron imagenes en {carpeta_imgs}")
        # Fallback a sintetica
        img_sint, _ = demo_sintetica()
        comparar_bordes(img_sint, titulo="Imagen sintetica")

    # ---- Demo modular (uso paso a paso) ----
    sintetica = crear_imagen_sintetica()
    demo_modular(sintetica)

    print("\n" + "=" * 60)
    print(" Demo completada.")
    print("=" * 60)