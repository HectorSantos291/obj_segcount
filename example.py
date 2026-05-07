import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

import obj_segcount as osc


# Configuracion por defecto para imagenes sin configuracion especifica
CONFIG_DEFAULT = {
    'tipo_filtro': 'gaussiano',
    'tamano_filtro': 5,
    'sigma_filtro': 1.5,
    'metodo_umbral': 'otsu',
    'min_tamano_objeto': 100,
    'max_tamano_objeto': 50000,
    'quitar_bordes': True,
    'ecualizar': False,
    'invertir': True,
    'mostrar_pasos': True,
}

# Configuraciones especificas por tipo de imagen
CONFIGS_ESPECIFICAS = {
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
    'cafe': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 50,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    # Frutas brillantes sobre fondo negro -> objetos mas claros que fondo, invertir=False
    'frutas': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 1500,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': False,
        'invertir': False,
        'mostrar_pasos': True,
    },
    'tornillos_largos': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 100,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    'tornillos_python': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 100,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    'botones': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 100,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    'almejas': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 100,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    # Engranes grises sobre fondo blanco, objetos pequenos incluidos
    'engranes': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 50,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    # Dulces coloridos sobre fondo blanco, sin ecualizar para no perder colores
    'dulces': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.0,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 80,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': False,
        'invertir': True,
        'mostrar_pasos': True,
    },
    'tornillos largos': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.5,
        'metodo_umbral': 'otsu',
        'min_tamano_objeto': 100,
        'max_tamano_objeto': 50000,
        'quitar_bordes': True,
        'ecualizar': True,
        'invertir': True,
        'mostrar_pasos': True,
    },
    'costuras': {
        'tipo_filtro': 'gaussiano',
        'tamano_filtro': 5,
        'sigma_filtro': 1.0,
        'metodo_umbral': 'adaptativo',
        'tamano_bloque': 25,
        'C_adaptativo': 8,
        'min_tamano_objeto': 200,
        'max_tamano_objeto': 50000,
        'quitar_bordes': False,
        'ecualizar': False,
        'invertir': True,
        'mostrar_pasos': True,
    },
}


def obtener_config(nombre_archivo):
    # Busca si el nombre del archivo coincide con alguna clave
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0].lower()
    for clave, config in CONFIGS_ESPECIFICAS.items():
        if clave in nombre_sin_ext:
            return config
    return CONFIG_DEFAULT


def crear_imagen_sintetica():
    # Imagen con fondo gris claro y objetos oscuros para probar el pipeline
    h, w = 480, 640
    imagen = np.full((h, w, 3), 200, dtype=np.uint8)

    cv2.circle(imagen, (100, 100), 30, (50, 60, 70), -1)
    cv2.circle(imagen, (250, 150), 25, (40, 55, 65), -1)
    cv2.circle(imagen, (400, 100), 35, (45, 50, 75), -1)
    cv2.circle(imagen, (530, 120), 28, (55, 60, 60), -1)

    cv2.rectangle(imagen, (80, 280), (140, 320), (50, 60, 70), -1)
    cv2.rectangle(imagen, (250, 300), (300, 370), (40, 55, 65), -1)
    cv2.rectangle(imagen, (450, 280), (495, 335), (45, 50, 75), -1)

    cv2.ellipse(imagen, (150, 420), (40, 20), 0, 0, 360, (50, 60, 70), -1)
    cv2.ellipse(imagen, (350, 400), (35, 18), 0, 0, 360, (40, 55, 65), -1)
    cv2.ellipse(imagen, (520, 380), (30, 15), 0, 0, 360, (45, 50, 75), -1)

    return imagen


def cargar_imagen(ruta):
    # cv2 carga en BGR, se convierte a RGB para mostrar con matplotlib
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imprimir_propiedades(resultados, nombre_objeto="Objeto"):
    for p in resultados['propiedades']:
        cy, cx = p['centroide']
        print(f"  {nombre_objeto} {p['id']}: area={p['area']}, "
              f"circularidad={p['circularidad']:.3f}, "
              f"centroide=({cy:.0f}, {cx:.0f})")


def demo_imagen(ruta):
    nombre = os.path.basename(ruta)
    print(f"\n--- {nombre} ---")

    img = cargar_imagen(ruta)
    if img is None:
        print(f"  No se pudo cargar {ruta}")
        return None, None
    print(f"  Tamanio: {img.shape}")

    config = obtener_config(nombre)
    print(f"  Umbral: {config['metodo_umbral']}")
    print(f"  Area minima: {config['min_tamano_objeto']}")

    resultados = osc.contar_objetos(img, **config)

    print(f"  Objetos detectados: {resultados['conteo']}")
    if 'valor_umbral' in resultados and resultados['valor_umbral'] >= 0:
        print(f"  Valor umbral: {resultados['valor_umbral']}")
    imprimir_propiedades(resultados)

    return img, resultados


def demo_sintetica():
    print("\n--- Imagen sintetica ---")

    img = crear_imagen_sintetica()
    print(f"  Tamanio: {img.shape}")

    resultados = osc.contar_objetos(img, **CONFIG_DEFAULT)

    print(f"  Objetos detectados: {resultados['conteo']}")
    imprimir_propiedades(resultados)

    return img, resultados


def demo_modular(imagen):
    # Muestra el pipeline paso a paso usando las funciones individuales
    print("\n--- Demo modular ---")

    if imagen.ndim == 3:
        gris = osc.rgb_a_gris(imagen)
    else:
        gris = imagen.copy()

    mejorada = osc.ecualizacion_histograma(gris)
    filtrada = osc.filtro_gaussiano(mejorada, tamano=5, sigma=1.5)

    binaria, valor_umbral = osc.umbral_otsu(filtrada)
    binaria = 255 - binaria
    print(f"  Umbral Otsu: {valor_umbral}")

    etiquetas, conteo = osc.componentes_conectados(binaria)
    etiquetas, conteo = osc.quitar_regiones_pequenas(etiquetas, min_tamano=100)
    etiquetas, conteo = osc.quitar_regiones_borde(etiquetas)
    print(f"  Componentes finales: {conteo}")

    esquinas = osc.detector_harris(gris, k=0.04, umbral=0.01)
    print(f"  Esquinas detectadas: {np.sum(esquinas)}")

    props = osc.medir_todo(etiquetas)
    for p in props:
        print(f"  Objeto {p['id']}: area={p['area']}, circularidad={p['circularidad']:.3f}")

    final = imagen.copy() if imagen.ndim == 3 else np.stack([gris]*3, axis=-1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Pipeline paso a paso", fontsize=14)

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
    axes[1, 2].set_title(f"7. Resultado ({len(props)} objetos)")
    for p in props:
        cy, cx = p['centroide']
        axes[1, 2].plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        axes[1, 2].annotate(str(p['id']), (cx, cy),
                            color='yellow', fontsize=10, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points')

    axes[1, 3].axis('off')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def comparar_bordes(imagen, titulo="Imagen"):
    # Aplica Sobel, Prewitt, Laplaciano y Harris para comparar resultados
    print(f"\n--- Comparativa de detectores: {titulo} ---")

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
    marcada = osc.marcar_esquinas(imagen_para_marcar, esquinas, color=(255, 0, 0), radio=3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Detectores de bordes - {titulo}", fontsize=14)

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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"obj_segcount v{osc.__version__}")

    carpeta_imgs = os.path.join(os.path.dirname(__file__), "images")
    imagenes = []

    if os.path.exists(carpeta_imgs):
        for archivo in sorted(os.listdir(carpeta_imgs)):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                imagenes.append(os.path.join(carpeta_imgs, archivo))

    if imagenes:
        print(f"\nImagenes encontradas: {len(imagenes)}")
        for ruta in imagenes:
            img, _ = demo_imagen(ruta)
            if img is not None:
                nombre = os.path.splitext(os.path.basename(ruta))[0]
                comparar_bordes(img, titulo=nombre)
    else:
        print(f"\nNo se encontraron imagenes en {carpeta_imgs}")
        img_sint, _ = demo_sintetica()
        comparar_bordes(img_sint, titulo="Imagen sintetica")

    sintetica = crear_imagen_sintetica()
    demo_modular(sintetica)
