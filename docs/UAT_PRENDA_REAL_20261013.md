# UAT con prenda real y panel iluminado — 13 de octubre de 2026

## Contexto

Prueba autorizada por el responsable de Astrid & Beverly Fashion. Se encendió el panel de iluminación y se colocó una blusa rosa estampada frente a la cámara Dahua. Se obtuvo una muestra RTSP temporal de **2688×1520 píxeles**, sin crear inspecciones, cambiar estados ni entrenar IA.

La muestra se mantuvo fuera del repositorio Git. Se guardó para trazabilidad en almacenamiento S3 privado y en una carpeta temporal del equipo de trabajo; no se publica la imagen ni credenciales RTSP.

## Comparación verificable

El mismo fotograma se procesó localmente con la versión base anterior a la corrección de segmentación y con la rama `fix/segmentacion-guiada-regresion` (commit `d49534a`). **La medición final reproduce explícitamente los valores ROI de AWS: `ROI_X1=0.16`, `ROI_Y1=0.05`, `ROI_X2=0.84`, `ROI_Y2=0.88`.** Una primera medición local con los valores predeterminados (`0.10, 0.10, 0.90, 0.90`) se descartó de las conclusiones y se conserva como dato auxiliar, no representativo de producción.

| Condición | Versión base | Candidata corregida |
| --- | --- | --- |
| Segmentador con selección normal | `semilla` | `semilla` |
| Cobertura normal del ROI productivo | **50.765 %** | **50.765 %** |
| Caja de la máscara | (543, 136, 2004, 1255) | igual |
| Segmentación adaptativa aislada, sin semilla | rechazada: 10.79 % antes de la guarda | rechazada: 11.68 % antes de la guarda |
| Razón adaptativa | `cobertura_adaptativa` | `cobertura_adaptativa` |
| Guarda mínima | 20 % | 20 %, sin modificaciones |

La máscara del método semilla se comparó visualmente contra la prenda en la imagen: cubre casi toda la superficie rosa, incluyendo ambas mangas y el cuerpo, sin incluir visiblemente grandes sectores del fondo. **No se midió precisión de defectos PatchCore ni se realizó un dictamen de calidad de la prenda.**

El método adaptativo aislado toma una isla irregular de aproximadamente 11.68 % del ROI productivo, situada principalmente en la parte central y derecha de la prenda. El diagnóstico del segmentador indica que otras regiones se fragmentan y algunas tocan los límites del panel; la franja de recuperación 5×5 no resuelve ese caso.

El ROI de AWS empieza en y=76, mientras que la máscara detectada comienza en y=136. Por tanto, el encuadre superior no presenta el recorte aparente de la primera prueba con ROI predeterminado.

## Repetición automatizable y de solo lectura

Se agregó el comando `scripts/uat_guided_real_frame.py` a la rama para analizar fotogramas existentes **sin abrir la cámara, sin operaciones SQL y sin crear capturas de producción**:

```powershell
python scripts/uat_guided_real_frame.py --image RUTA_PRIVADA_FOTO.jpg --roi 0.16 0.05 0.84 0.88
python scripts/uat_guided_real_frame.py --image RUTA_PRIVADA_FOTO.jpg --roi 0.16 0.05 0.84 0.88 --force-adaptive
```

Resultados reproducidos con los mismos parámetros AWS: `detected/semilla/0.50765` y `not_detected/ninguno/0.1168` (el último valor es la candidata descartada antes de aplicar el umbral del 20 %). El código no exporta imágenes; las muestras se mantienen privadas.

## Resultado y decisión de despliegue

1. **Prueba de cámara y segmentación normal para esta blusa rosa: favorable.** Imagen recibida correctamente, máscara normal 50.77 % y método semilla activo.
2. **Validación de segmentación adaptativa con prenda real: NO aprobada.** La corrección del borde supera las pruebas sintéticas (408 tests en total, 148 omitidos, 0 fallos), pero no detecta esta blusa cuando se obliga al método adaptativo.
3. **Despliegue: NO realizado** a causa del límite de validación real. No se tocó Docker en producción ni los modelos ni los registros.
4. Antes de habilitar el cambio, probar con varias prendas reales de colores y tejidos distintos, fondos vacíos e iluminación variable. Diseñar un método adaptativo que no dependa de unir indiscriminadamente regiones y volver a medir falsos positivos y cobertura.
5. La validación óptica de la silueta no sustituye la prueba de inferencia PatchCore, la revisión humana ni la generación de informes.

## Evidencia preservada

Los archivos capturados para UAT son temporales y permanecen fuera de Git; ruta S3 privada restringida bajo `astrid/operations/uat/20261013/`. No se copiaron imágenes a `static/captures` ni a `static/results`, y no se registró ninguna inspección nueva.
