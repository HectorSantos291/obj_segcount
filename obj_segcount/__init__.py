# obj_segcount - Segmentacion y conteo de objetos

__version__ = "1.0.0"

from obj_segcount.preprocessing import rgb_a_gris, normalizar, redimensionar

from obj_segcount.intensity import (
    negativo, transformacion_log, estiramiento_contraste,
    ajuste_lineal, ecualizacion_histograma,
)

from obj_segcount.filtering import (
    convolucion2d, kernel_gaussiano, filtro_promedio,
    filtro_gaussiano, filtro_mediana,
)

from obj_segcount.edges import (
    sobel_x, sobel_y, magnitud_gradiente,
    direccion_gradiente, deteccion_bordes,
    prewitt_x, prewitt_y, deteccion_prewitt,
    laplaciano,
)

from obj_segcount.thresholding import (
    umbral_global, umbral_otsu, umbral_adaptativo,
)

from obj_segcount.segmentation import (
    componentes_conectados, crecimiento_regiones,
    quitar_regiones_pequenas, quitar_regiones_grandes,
    quitar_regiones_borde,
)

from obj_segcount.corners import detector_harris, marcar_esquinas

from obj_segcount.properties import (
    medir_area, medir_perimetro, medir_centroide,
    medir_bbox, medir_circularidad, medir_todo,
)

from obj_segcount.counting import (
    contar_objetos, visualizar_resultados,
    contar_objetos_claros, contar_objetos_oscuros, contar_objetos_pequenos,
)
