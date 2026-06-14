# Sistema de Gestión de Control de Calidad - Prototipo Flask

Base funcional para la tesis de visión artificial aplicada al control de calidad textil.

## Incluye

- Login.
- Dashboard similar al diseño enviado.
- Inspección en vivo.
- Captura desde imagen cargada o cámara conectada.
- Procesamiento base con OpenCV.
- Alerta visual: aprobado, defecto o revisar.
- Registro en SQLite.
- Historial trazable con ID, fecha, tipo de prenda, talla, defecto, confianza e imagen.
- Reportes preliminares.

## Usuario demo

Usuario: `admin`
Contraseña: `admin123`

## Instalación Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abrir:

```text
http://127.0.0.1:5000
```

## Cámara

Si existe webcam o cámara USB, intenta usar `CAMERA_INDEX=0`.

Para cambiar cámara:

```bash
set CAMERA_INDEX=1
python app.py
```

## Nota técnica

El detector incluido es base para demostrar el flujo funcional:

captura -> análisis -> alerta -> registro -> trazabilidad.

Para la versión final se recomienda reemplazar `detect_defect()` por YOLOv8 entrenado con imágenes propias.
