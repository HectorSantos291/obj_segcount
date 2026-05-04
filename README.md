# obj_segcount

> Librería en Python para **segmentación y conteo de objetos** en superficies de trabajo.

`obj_segcount` permite procesar imágenes cenitales de objetos (tornillos, semillas, monedas, piezas industriales, etc.) sobre una mesa o banda transportadora para detectarlos, segmentarlos, contarlos y caracterizarlos geométricamente.

Todas las funciones centrales (convolución 2D, Otsu, componentes conectados, Harris, etc.) están **implementadas desde cero** usando únicamente NumPy.

---

## Características

- Pipeline completo de segmentación y conteo en una sola llamada
- Uso modular: cada etapa puede ejecutarse de forma independiente
- Múltiples métodos de umbralización (Otsu, adaptativo, fijo)
- Detectores de bordes propios: Sobel, Prewitt, Laplaciano
- Detector de esquinas Harris implementado desde cero
- Medición de propiedades geométricas (área, perímetro, centroide, circularidad, bbox)
- Visualización integrada de cada paso del pipeline

---

## Instalación

### Desde GitHub

```bash
pip install git+https://github.com/HectorSantos291/obj_segcount.git
```

### Para desarrollo local

```bash
git clone https://github.com/HectorSantos291/obj_segcount.git
cd obj_segcount
pip install -e .
```

---

## Uso rápido

El archivo `example.py` incluye una demostración completa con las imágenes de prueba (`coins.png`, `tornillos_python.jpg`, `cafe_python.jpg`):

```bash
python example.py
```

### Ejemplo mínimo

```python
import cv2
import obj_segcount as osc

# Cargar imagen
img = cv2.imread("images/tornillos_python.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Ejecutar pipeline completo con visualización
resultados = osc.contar_objetos(
    img,
    tipo_filtro='gaussiano',
    sigma_filtro=1.5,
    metodo_umbral='otsu',
    min_tamano_objeto=100,
    quitar_bordes=True,
    invertir=True,         # objetos oscuros sobre fondo claro
    mostrar_pasos=True,
)

print(f"Objetos detectados: {resultados['conteo']}")
for p in resultados['propiedades']:
    print(f"  Objeto {p['id']}: area={p['area']}, circularidad={p['circularidad']:.3f}")
```

### Ejemplo con iluminación no uniforme (umbral adaptativo)

```python
# Para imágenes con sombras o gradientes de iluminación (como coins.png)
resultados = osc.contar_objetos(
    img_gris,
    tipo_filtro='gaussiano',
    sigma_filtro=0.5,                  # sigma bajo para no fusionar objetos cercanos
    metodo_umbral='adaptativo',
    tamano_bloque=15,
    C_adaptativo=5,
    min_tamano_objeto=200,
    max_tamano_objeto=5000,
    ecualizar=False,
    mostrar_pasos=True,
)
```

---

## Configuraciones recomendadas según tipo de imagen

La librería incluye **funciones predefinidas** que automáticamente seleccionan los mejores parámetros según las características de tus objetos. Úsalas en lugar de configurar manualmente todos los parámetros.

### 1. Objetos claros sobre fondo oscuro

**Ejemplos:** Monedas, piezas metálicas, objetos brillantes, componentes con acabado reflectante.

**Función recomendada:**

```python
import cv2
import obj_segcount as osc

img = cv2.imread("monedas.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Configuracion automatica para objetos claros
resultados = osc.contar_objetos_claros(img, mostrar_pasos=True)
print(f"Objetos detectados: {resultados['conteo']}")
```

**Características de esta configuración:**
- Umbral adaptativo (maneja iluminación no uniforme)
- Sin inversión de máscara
- Filtro suave (σ=0.5) para evitar fusionar objetos cercanos
- Área mínima: 200 píxeles

### 2. Objetos oscuros sobre fondo claro

**Ejemplos:** Tornillos sobre mesa blanca, piezas de plástico oscuro, componentes opacos, objetos de metal sin brillo.

**Función recomendada:**

```python
import cv2
import obj_segcount as osc

img = cv2.imread("tornillos.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Configuracion automatica para objetos oscuros
resultados = osc.contar_objetos_oscuros(img, mostrar_pasos=True)
print(f"Objetos detectados: {resultados['conteo']}")
```

**Características de esta configuración:**
- Umbral de Otsu (automático)
- Inversión de máscara activada
- Ecualización de histograma para mejorar contraste
- Área mínima: 100 píxeles

### 3. Objetos pequeños y cercanos

**Ejemplos:** Granos de café, arroz, semillas, componentes electrónicos pequeños, partículas.

**Función recomendada:**

```python
import cv2
import obj_segcount as osc

img = cv2.imread("granos_cafe.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Configuracion automatica para objetos pequenos
resultados = osc.contar_objetos_pequenos(img, mostrar_pasos=True)
print(f"Objetos detectados: {resultados['conteo']}")
```

**Características de esta configuración:**
- Umbral de Otsu
- Filtro muy suave (σ=0.5) para evitar fusión entre objetos cercanos
- Área mínima MUY baja: 50 píxeles
- Área máxima: 5000 píxeles (evita regiones grandes)

### Tabla comparativa de configuraciones

| Tipo de objeto | Función | Umbral | Ecualizar | Invertir | σ filtro | Área mín |
|---|---|---|---|---|---|---|
| Claros / fondo oscuro | `contar_objetos_claros()` | Adaptativo | No | No | 0.5 | 200 |
| Oscuros / fondo claro | `contar_objetos_oscuros()` | Otsu | Sí | Sí | 1.5 | 100 |
| Pequeños y cercanos | `contar_objetos_pequenos()` | Otsu | Sí | Sí | 0.5 | 50 |

### ¿Cuándo usar configuración manual?

Si ninguna de las funciones predefinidas funciona bien para tu caso, usa `contar_objetos()` con parámetros personalizados:

```python
resultados = osc.contar_objetos(
    imagen,
    tipo_filtro='gaussiano',
    sigma_filtro=1.0,           # Ajusta según necesites
    metodo_umbral='otsu',       # 'otsu', 'adaptativo' o 'fijo'
    min_tamano_objeto=150,      # Filtra ruido
    max_tamano_objeto=10000,    # Filtra fondo
    ecualizar=True,             # Mejora contraste
    invertir=False,             # True si objetos oscuros
    mostrar_pasos=True
)
```

---

## Uso modular

Cada función puede usarse de forma independiente para construir pipelines personalizados:

```python
import obj_segcount as osc

# Preprocesamiento
gris = osc.rgb_a_gris(imagen)
mejorada = osc.ecualizacion_histograma(gris)
filtrada = osc.filtro_gaussiano(mejorada, tamano=5, sigma=1.0)

# Umbralización (Otsu, adaptativo o fijo)
binaria, valor_umbral = osc.umbral_otsu(filtrada)

# Segmentación
etiquetas, conteo = osc.componentes_conectados(binaria)
etiquetas, conteo = osc.quitar_regiones_pequenas(etiquetas, min_tamano=50)
etiquetas, conteo = osc.quitar_regiones_grandes(etiquetas, max_tamano=5000)

# Detección de bordes (3 algoritmos disponibles)
bordes_sobel = osc.deteccion_bordes(gris, umbral=0.15)
bordes_prewitt = osc.deteccion_prewitt(gris, umbral=0.15)
bordes_lap = osc.laplaciano(gris, usar_diagonales=True)

# Detección de esquinas (Harris)
esquinas = osc.detector_harris(gris, k=0.04, umbral=0.01)

# Propiedades geométricas
propiedades = osc.medir_todo(etiquetas)
for obj in propiedades:
    print(f"Objeto {obj['id']}: area={obj['area']}, "
          f"circularidad={obj['circularidad']:.3f}")
```

---

## Catálogo de funciones

### Preprocesamiento (`preprocessing.py`)
| Función | Descripción |
|---|---|
| `rgb_a_gris()` | Conversión RGB a escala de grises |
| `normalizar()` | Normalización min-max |
| `redimensionar()` | Redimensionado con interpolación bilineal |

### Transformaciones de intensidad (`intensity.py`)
| Función | Descripción |
|---|---|
| `negativo()` | Negativo de la imagen |
| `transformacion_log()` | Transformación logarítmica |
| `estiramiento_contraste()` | Estiramiento de contraste sigmoidal |
| `ajuste_lineal()` | Ajuste lineal de rango |
| `ecualizacion_histograma()` | Ecualización de histograma |

### Filtrado espacial (`filtering.py`)
| Función | Descripción |
|---|---|
| `convolucion2d()` | Convolución 2D propia |
| `filtro_gaussiano()` | Filtro gaussiano |
| `filtro_promedio()` | Filtro promedio |
| `filtro_mediana()` | Filtro de mediana |

### Detección de bordes (`edges.py`)
| Función | Descripción |
|---|---|
| `sobel_x()`, `sobel_y()` | Derivadas con Sobel |
| `prewitt_x()`, `prewitt_y()` | Derivadas con Prewitt |
| `deteccion_bordes()` | Bordes con Sobel + umbral |
| `deteccion_prewitt()` | Bordes con Prewitt + umbral |
| `laplaciano()` | Operador Laplaciano (4 u 8 vecinos) |
| `magnitud_gradiente()` | Magnitud del gradiente |
| `direccion_gradiente()` | Dirección del gradiente |

### Umbralización (`thresholding.py`)
| Función | Descripción |
|---|---|
| `umbral_otsu()` | Método de Otsu (implementación propia) |
| `umbral_global()` | Umbral fijo |
| `umbral_adaptativo()` | Umbral adaptativo local con imagen integral |

### Segmentación (`segmentation.py`)
| Función | Descripción |
|---|---|
| `componentes_conectados()` | Etiquetado por componentes conectados (BFS propio) |
| `crecimiento_regiones()` | Crecimiento de regiones |
| `quitar_regiones_pequenas()` | Filtrar regiones pequeñas |
| `quitar_regiones_grandes()` | Filtrar regiones grandes |
| `quitar_regiones_borde()` | Eliminar objetos en bordes |

### Detección de esquinas (`corners.py`)
| Función | Descripción |
|---|---|
| `detector_harris()` | Detector de Harris (implementación propia) |
| `marcar_esquinas()` | Marcar esquinas en la imagen |

### Propiedades geométricas (`properties.py`)
| Función | Descripción |
|---|---|
| `medir_area()` | Área en píxeles |
| `medir_perimetro()` | Perímetro |
| `medir_centroide()` | Centroide |
| `medir_bbox()` | Rectángulo envolvente |
| `medir_circularidad()` | Circularidad (4π·área / perímetro²) |
| `medir_todo()` | Todas las propiedades de todos los objetos |

### Pipeline completo (`counting.py`)
| Función | Descripción |
|---|---|
| `contar_objetos()` | Pipeline completo de segmentación y conteo con parámetros configurables |
| `contar_objetos_claros()` | Configuración predefinida para objetos claros sobre fondo oscuro |
| `contar_objetos_oscuros()` | Configuración predefinida para objetos oscuros sobre fondo claro |
| `contar_objetos_pequenos()` | Configuración predefinida para objetos pequeños y cercanos |
| `visualizar_resultados()` | Visualización de resultados en mosaico |

---

## Estructura del proyecto

```
Proyecto_P2/
├── obj_segcount/
│   ├── __init__.py
│   ├── preprocessing.py   # RGB a gris, normalización, redimensionado
│   ├── intensity.py       # Transformaciones de intensidad
│   ├── filtering.py       # Convolución 2D, filtros espaciales
│   ├── edges.py           # Sobel, Prewitt, Laplaciano
│   ├── thresholding.py    # Otsu, umbral adaptativo, umbral fijo
│   ├── segmentation.py    # Componentes conectados, crecimiento de regiones
│   ├── corners.py         # Detector de Harris
│   ├── properties.py      # Propiedades geométricas
│   └── counting.py        # Pipeline completo + visualización
├── images/
│   ├── coins.png          # Monedas (iluminación no uniforme)
│   ├── tornillos_python.jpg
│   └── cafe_python.jpg    # Granos de café
├── example.py             # Script de demostración
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Imágenes de prueba incluidas

| Imagen | Tipo de objetos | Configuración recomendada |
|---|---|---|
| `coins.png` | Monedas con iluminación no uniforme | Umbral adaptativo, σ=0.5 |
| `tornillos_python.jpg` | Tornillos sobre fondo claro | Otsu + invertir |
| `cafe_python.jpg` | Granos de café | Otsu + invertir |

El script `example.py` selecciona automáticamente la configuración adecuada según el nombre de la imagen.

---

## Requisitos

- Python ≥ 3.9
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- OpenCV (`opencv-python`) — para cargar imágenes

---

## Autores

- **Héctor Alexis Santos**
- **Kane Aaron Soto Rodríguez**
- **Gerardo Alejandro Orozco Gutiérrez**

---

## Licencia

Proyecto académico desarrollado para la materia de Visión Robótica.
