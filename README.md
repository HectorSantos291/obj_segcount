# obj_segcount

Librería en Python para segmentación y conteo de objetos en superficies de trabajo.

`obj_segcount` procesa imágenes cenitales de objetos (tornillos, semillas, monedas, piezas industriales, etc.) para detectarlos, segmentarlos, contarlos y medir sus propiedades geométricas.

---

## Instalación

```bash
pip install git+https://github.com/HectorSantos291/obj_segcount.git
```

O en modo desarrollo:

```bash
git clone https://github.com/HectorSantos291/obj_segcount.git
cd obj_segcount
pip install -e .
```

---

## Uso rápido

```bash
python example.py
```

### Ejemplo básico

```python
import cv2
import obj_segcount as osc

img = cv2.imread("images/tornillos_python.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

resultados = osc.contar_objetos(
    img,
    tipo_filtro='gaussiano',
    sigma_filtro=1.5,
    metodo_umbral='otsu',
    min_tamano_objeto=100,
    quitar_bordes=True,
    invertir=True,
    mostrar_pasos=True,
)

print(f"Objetos detectados: {resultados['conteo']}")
for p in resultados['propiedades']:
    print(f"  Objeto {p['id']}: area={p['area']}, circularidad={p['circularidad']:.3f}")
```

### Umbral adaptativo (para iluminación no uniforme)

```python
resultados = osc.contar_objetos(
    img,
    tipo_filtro='gaussiano',
    sigma_filtro=0.5,
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

## Uso modular

Cada función puede usarse de forma independiente:

```python
import obj_segcount as osc

# Preprocesamiento
gris = osc.rgb_a_gris(imagen)
mejorada = osc.ecualizacion_histograma(gris)
filtrada = osc.filtro_gaussiano(mejorada, tamano=5, sigma=1.0)

# Umbralización
binaria, valor_umbral = osc.umbral_otsu(filtrada)

# Segmentación
etiquetas, conteo = osc.componentes_conectados(binaria)
etiquetas, conteo = osc.quitar_regiones_pequenas(etiquetas, min_tamano=50)

# Detección de bordes
bordes_sobel = osc.deteccion_bordes(gris, umbral=0.15)
bordes_prewitt = osc.deteccion_prewitt(gris, umbral=0.15)
bordes_lap = osc.laplaciano(gris, usar_diagonales=True)

# Detección de esquinas
esquinas = osc.detector_harris(gris, k=0.04, umbral=0.01)

# Propiedades
propiedades = osc.medir_todo(etiquetas)
for obj in propiedades:
    print(f"Objeto {obj['id']}: area={obj['area']}, circularidad={obj['circularidad']:.3f}")
```

---

## Funciones disponibles

### Preprocesamiento (`preprocessing.py`)
| Función | Descripción |
|---|---|
| `rgb_a_gris()` | Conversión RGB a escala de grises |
| `normalizar()` | Normalización min-max |

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
| `convolucion2d()` | Convolución 2D |
| `filtro_gaussiano()` | Filtro gaussiano |
| `filtro_promedio()` | Filtro promedio |
| `filtro_mediana()` | Filtro de mediana |

### Detección de bordes (`edges.py`)
| Función | Descripción |
|---|---|
| `deteccion_bordes()` | Bordes con Sobel + umbral |
| `deteccion_prewitt()` | Bordes con Prewitt + umbral |
| `laplaciano()` | Operador Laplaciano (4 u 8 vecinos) |
| `magnitud_gradiente()` | Magnitud del gradiente |

### Umbralización (`thresholding.py`)
| Función | Descripción |
|---|---|
| `umbral_otsu()` | Método de Otsu |
| `umbral_global()` | Umbral fijo |
| `umbral_adaptativo()` | Umbral adaptativo local |

### Segmentación (`segmentation.py`)
| Función | Descripción |
|---|---|
| `componentes_conectados()` | Etiquetado por componentes conectados |
| `quitar_regiones_pequenas()` | Filtrar regiones pequeñas |
| `quitar_regiones_grandes()` | Filtrar regiones grandes |
| `quitar_regiones_borde()` | Eliminar objetos en bordes |

### Detección de esquinas (`corners.py`)
| Función | Descripción |
|---|---|
| `detector_harris()` | Detector de Harris |
| `marcar_esquinas()` | Marcar esquinas en la imagen |

### Propiedades geométricas (`properties.py`)
| Función | Descripción |
|---|---|
| `medir_area()` | Área en píxeles |
| `medir_perimetro()` | Perímetro |
| `medir_centroide()` | Centroide |
| `medir_bbox()` | Rectángulo envolvente |
| `medir_circularidad()` | Circularidad |
| `medir_todo()` | Todas las propiedades de todos los objetos |

### Pipeline completo (`counting.py`)
| Función | Descripción |
|---|---|
| `contar_objetos()` | Pipeline completo con parámetros configurables |
| `visualizar_resultados()` | Visualización del resultado en subplots |

---

## Estructura del proyecto

```
Proyecto_P2/
├── obj_segcount/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── intensity.py
│   ├── filtering.py
│   ├── edges.py
│   ├── thresholding.py
│   ├── segmentation.py
│   ├── corners.py
│   ├── properties.py
│   └── counting.py
├── images/
│   ├── coins.png
│   ├── tornillos_python.jpg
│   └── cafe_python.jpg
├── example.py
├── prueba_manual.py
├── pyproject.toml
└── README.md
```

---

## Imágenes de prueba

| Imagen | Configuración sugerida |
|---|---|
| `coins.png` | Umbral adaptativo, σ=0.5, invertir=False |
| `tornillos_python.jpg` | Otsu, invertir=True |
| `cafe_python.jpg` | Otsu, invertir=True, ecualizar=True |

---

## Requisitos

- Python ≥ 3.9
- NumPy
- Matplotlib
- OpenCV (`opencv-python`)

---

## Autores

- Héctor Alexis Santos
- Kane Aaron Soto Rodríguez
- Gerardo Alejandro Orozco Gutiérrez

Proyecto académico — Visión Robótica
