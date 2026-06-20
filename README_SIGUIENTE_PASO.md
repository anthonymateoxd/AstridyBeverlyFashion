# Sistema de inspección textil — siguiente etapa

## Qué está listo

- Cámara integrada mediante `CAMERA_SOURCE=0`.
- Video MJPEG en Flask.
- Captura manual y automática.
- Región de inspección configurable mediante ROI.
- Registro de imágenes y resultados en MySQL.
- Carga automática de `models/best.pt` cuando el entrenamiento termina.
- Modo prototipo honesto mientras no exista un modelo entrenado: no inventa resultados aleatorios.

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python check_system.py
python app.py
```

En Linux/macOS, active el entorno con `source venv/bin/activate` y copie el archivo con `cp .env.example .env`.

## Clases del detector

El detector tendrá únicamente cuatro clases:

- 0: mancha
- 1: rotura
- 2: agujero
- 3: variacion_color

`Sin defecto` no se dibuja como caja. Una imagen limpia debe incluir un archivo `.txt` vacío con el mismo nombre.

Ejemplo:

```text
dataset/images/train/blusa_limpia_001.jpg
dataset/labels/train/blusa_limpia_001.txt   # vacío

dataset/images/train/blusa_mancha_001.jpg
dataset/labels/train/blusa_mancha_001.txt  # contiene la caja de la mancha
```

## Tipos de prenda

Blusa, top corto y camisa cropped no deben mezclarse como clases de defecto. En esta fase son metadatos de la inspección. Posteriormente se puede agregar:

1. selección manual del tipo y talla en `station.html`; o
2. un segundo modelo de clasificación para reconocer automáticamente la prenda.

Para la tesis, conviene terminar primero el detector de defectos, porque es la variable principal de validación.

## Distribución recomendada

Use 70 % para entrenamiento, 20 % para validación y 10 % para prueba. Las fotografías muy parecidas de una misma secuencia deben permanecer en el mismo grupo para evitar fuga de datos.


## Capturar y etiquetar con la cámara de la computadora

El archivo `capture_dataset.py` permite tomar la fotografía y dibujar la caja del defecto inmediatamente. Ejecute sesiones separadas para evitar mezclar fotografías casi idénticas entre entrenamiento y evaluación:

```bash
python capture_dataset.py --split train
python capture_dataset.py --split val
python capture_dataset.py --split test
```

Controles dentro de la ventana:

- `B`, `T`, `C`: tipo de prenda.
- `0`: prenda limpia; genera una etiqueta vacía.
- `1`: mancha.
- `2`: rotura.
- `3`: agujero.
- `4`: variación de color.
- `S`: guardar. Para defectos, después se dibuja la caja y se presiona Enter.
- `Q`: salir.

No tome ráfagas de la misma escena. Cambie posición, iluminación controlada, prenda, tamaño, fondo y ubicación del defecto.

## Entrenamiento

Cuando las imágenes y etiquetas estén colocadas:

```bash
python train_model.py
```

El script valida el dataset, entrena, evalúa y copia el mejor peso a:

```text
models/best.pt
```

Después basta reiniciar Flask para que el sistema cargue la IA real.
