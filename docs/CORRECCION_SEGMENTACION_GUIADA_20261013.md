# Corrección de segmentación guiada — 13 de octubre de 2026

## Alcance y seguridad

Rama: `fix/segmentacion-guiada-regresion`, basada en `fix/validar-meta-antes-de-crear-lote` (`ea73e06`).

Se trabajó únicamente sobre una copia Git aislada en la computadora Anthony. **No se modificó AWS, la cámara, MySQL, los lotes ni los modelos entrenados.** La corrección todavía **no se desplegó en producción**.

## Defecto reproducido

Las dos pruebas `GuidedSegmentationTests.test_synthetic_garment_is_detected` y `test_guided_segment_uses_a_single_mask` fallaban porque la segmentación adaptativa extraía correctamente una candidata de prenda, pero la banda de bordes producida por el gradiente LAB/Scharr y el cierre morfológico recortaba parte del contorno.

En el escenario sintético principal, la elipse conocida ocupa alrededor del 20.6 % del ROI. Antes de corregir, la máscara detectada tenía **19.755 %** de cobertura y era rechazada por `GUIDED_COVERAGE_MIN = 0.20`. El diagnóstico era `reason='cobertura_adaptativa'`, `method='ninguno'`.

## Corrección

Después de reconstruir el contorno interior de una candidata válida, se recupera la franja perdida mediante una dilatación elíptica pequeña, aplicada **antes de verificar la cobertura**. El núcleo está escalado al tamaño de trabajo: en la configuración de laboratorio (`GUIDED_SEGMENTATION_SCALE=0.5`) es de **5×5 píxeles**. No se modifica el umbral del 20 % ni se inventa una máscara cuando no existe contorno candidato.

Esta operación recupera aproximadamente 0.4 puntos porcentuales de cobertura para la elipse sintética, suficiente para superar el mínimo, y mantiene la clasificación de escenas vacías.

## Resultados cuantitativos de calibración

Imagen de referencia: `calibration/background_full.jpg` con elipse de geometría conocida superpuesta en diferentes colores. Coincidencia IoU dentro del ROI:

| Color simulado | Método | Cobertura | IoU respecto a la elipse |
| --- | --- | ---: | ---: |
| Verde | Adaptativo | 20.17 % | 98.09 % |
| Azul | Semilla | 20.69 % | 99.38 % |
| Rojo | Semilla | 20.69 % | 99.38 % |
| Blanco | Adaptativo | 20.13 % | 97.90 % |
| Negro | Adaptativo | 20.09 % | 97.75 % |

Seis imágenes de fondo vacío (`background_full.jpg`, `blusa_actual_full.jpg`, `blusa_ahora_full.jpg`, `blusa_confirmada_full.jpg`, `current_full.jpg`, `rtsp_confirmacion.jpg`) devolvieron **máscara vacía** y `method='ninguno'`. Una elipse deliberadamente menor que el umbral también fue rechazada.

Las pruebas nuevas comprueban recuperación de borde con umbral intacto, prendas de cinco colores, solapamiento IoU con geometría conocida y rechazo de un objeto pequeño. Las pruebas anteriores de ausencia de prenda y consistencia entre máscara, cobertura y caja delimitadora siguen pasando.

## Validación automatizada

Comando local (con `SECRET_KEY` temporal exclusiva del proceso, nunca la de AWS):

```
python -m unittest discover -s tests -p "test_*.py" -q
```

Resultado final: **408 tests; 260 ejecutados correctamente; 148 omitidos; 0 fallos; 0 errores**.

## UAT posterior con blusa real

El 13 de octubre se realizó una prueba autorizada con una blusa rosa iluminada y fotograma RTSP de 2688×1520. La segmentación normal, basada en semilla cromática, reconoció la prenda (50.77 % del ROI productivo configurado en AWS) tanto en el código previo como en esta rama. Sin embargo, la segmentación adaptativa aislada reconstruyó solo 11.68 % del ROI y **no superó el mínimo del 20 %**. La prueba real del fallback **no está aprobada**; ver `docs/UAT_PRENDA_REAL_20261013.md`. **No desplegar la corrección del segmentador únicamente por aprobar la suite sintética.**

## Limitaciones y siguiente validación

Las métricas IoU anteriores son de una elipse **sintética** dibujada sobre imágenes reales de fondo; **no equivalen a la precisión de inspección de defectos PatchCore ni a un ensayo con prendas reales**.

Antes del despliegue se requiere UAT con prendas reales en la cámara, contraste contra escenas vacías con cambios de iluminación y revisión del flujo de captura guiada. La rama contiene además la corrección previa de validación de la meta al crear lotes, todavía no desplegada. No mezclar despliegues sin verificar ambos cambios y conservar imagen anterior para reversión.

Pendientes independientes: trazabilidad de 25 inspecciones con lote y 318 históricas cuyas imágenes no se localizaron, más pruebas de inferencia con prendas reales, seguridad y CI/CD.
