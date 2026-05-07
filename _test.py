import cv2, obj_segcount as osc, sys
sys.path.insert(0, '.')
from example import obtener_config

for nombre in ['almejas.png', 'costuras.jpg']:
    img = cv2.imread(f'images/{nombre}', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c = dict(obtener_config(nombre))
    c['mostrar_pasos'] = False
    r = osc.contar_objetos(img, **c)
    print(f'{nombre}: {r["conteo"]} objetos  metodo={c["metodo_umbral"]}  umbral={r["valor_umbral"]}')
