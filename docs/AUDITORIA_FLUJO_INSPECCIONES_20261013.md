# Auditoría del flujo de inspecciones — 13 de octubre de 2026

## Alcance y garantías

Inspección de código real versionado desde la imagen activa de AWS, plantillas, esquemas MySQL y **consultas SELECT únicamente**. No se generaron, eliminaron ni actualizaron inspecciones, lotes, checkpoints o modelos de IA. Los cambios correctivos se desarrollaron y probaron fuera de producción.

Flujo revisado: captura RTSP → selección de fotograma → inferencia PatchCore asociada al lote → persistencia de inspección y evidencia → revisión humana de alertas → cierre del lote → historial e informes.

## Datos verificados en AWS (fotografía del diagnóstico)

- **539 inspecciones**: 221 vinculadas a lotes (flujo actual), 318 históricas sin lote; por diseño, los informes productivos que exigen `batch_id IS NOT NULL` no incluyen las históricas.
- De las 221 productivas: 132 `NORMAL`/`NO_REQUIERE_REVISION` y 89 `ANOMALIA`.
- Las 89 alertas: 52 pendientes, 20 defectos confirmados y 17 alertas descartadas.
- Lotes: 19 `COMPLETADO`, 45 `REVISION_PENDIENTE`, 9 `PREPARACION`. Ningún lote completado contenía alertas pendientes conforme a la consulta de consistencia.
- Las inspecciones normales y las alertas no presentan valores de `review_status` fuera de los previstos en sus respectivas categorías.

## Hallazgo 1 — historial de prendas aceptadas (corrección preparada)

**Causa:** `records()` calculaba `decision_class='pending'` para una prenda `NORMAL` sin revisión humana, mientras la plantilla mostraba su decisión como «Aceptada». Resultado: «Aceptada» y responsable «Pendiente de revisión» en la misma ficha.

**Alcance medido:** 132 registros del flujo de lotes actual.

**Corrección preparada:** `classify_record_final_decision()` unifica la decisión final; la plantilla muestra «No requiere revisión humana» para `NORMAL` y utiliza la clasificación consistente para representar la decisión.

**Pruebas:** casos normal, histórico, alerta pendiente, defecto confirmado, alerta descartada y prioridad de resolución explícita sobre estados contradictorios. No modifica filas ni estados persistidos.

## Hallazgo 2 — inicio de lote sin IA utilizable (corrección preparada)

**Causa:** `start_batch()` verificaba aprobación del modelo de prenda y asociación con la versión IA, pero no exigía versión `ACTIVO`, `active=1`, ruta de checkpoint ni umbral definitivo congelado. Estos requisitos sí se verificaban después al intentar inferir; por tanto, un lote podía pasar a `EN_INSPECCION` y fallar en la primera captura.

**Caso verificable:** IA del modelo de prueba `TEST-AI-B-14040_5948`, versión `id=607`, figura activa pero sin checkpoint ni umbral final; lote `id=45` en `PREPARACION`. Existen otros lotes en preparación vinculados a versiones de IA retiradas.

**Corrección preparada:** `ai_model_ready_for_batch()` y comprobación del archivo en el mismo directorio de artefactos utilizado por la inferencia. Rechaza el inicio con mensaje claro sin cambiar el estado del lote. No elimina modelos ni modifica datos.

## Hallazgo 3 — estado de la jornada al 100 % (comportamiento previsto)

`get_goal_history_data()` conserva el estado «En curso» mientras la jornada actual no alcance el final del turno. Una meta lograda antes del cierre sigue en curso hasta la hora configurada y luego pasa a «Cumplida». Es semánticamente distinto a «meta alcanzada».

En la fecha revisada existía una meta de 100 prendas, 10 lotes y turno 08:00–17:00 (Guatemala). **No modificar esta regla sin acuerdo funcional.** Como mejora opcional, presentar por separado «Meta alcanzada» y «Jornada en curso».

## Hallazgo 4 — endpoint de salud y modelo IA (pendiente)

`/health` determina `model_ready` comprobando un archivo global `MODEL_PATH` de la integración antigua YOLO y produce `detection_mode='prototype'` si falta. **Eso no comprueba PatchCore**, cuyo runtime productivo carga por `batches.ai_model_id` y valida checkpoint, estado y umbral. `model_ready=false` no prueba que PatchCore esté fallando.

Pendiente: diseñar una respuesta de salud diferenciada para web, BD, cámara y versión PatchCore seleccionada, sin cambiar por accidente el contrato usado por monitorización.

## Seguridad, despliegue y UAT pendientes

- Los arreglos todavía deben incorporarse a la imagen activa con despliegue aislado, respaldo previo, validación y reversión.
- Probar sesión autenticada con una prenda NORMAL, una alerta pendiente y una revisada; comprobar resumen, detalle, historial e informes.
- Intentar iniciar **solo en entorno de pruebas** un lote cuyo modelo no tiene checkpoint y verificar que permanece en preparación.
- Verificar existencia de archivos de imágenes históricos mediante comprobación de solo lectura, sin exponerlos.
- Conservar la separación de los 318 registros históricos respecto de los informes productivos salvo decisión explícita.
- La carpeta fuente de EC2 no es idéntica a la imagen en ejecución: no reconstruir con `docker compose build` ni hacer `git reset/pull` hasta reconciliar el repositorio.
- Los respaldos automáticos permanecen configurados; revisar primera copia programada, restauración periódica y crecimiento de EBS/S3.
- Rotar las credenciales de cámara y Wi-Fi expuestas previamente de forma coordinada.

## Cambios versionados

- `fix/historial-decisiones-finales`: clasificador y pruebas.
- `fix/lotes-validar-modelo-ia`: guardia de inicio y pruebas.
- `fix/auditoria-flujo-inspecciones`: integración conjunta para futura validación y despliegue.

**Estado:** auditoría y correcciones locales completadas; despliegue y UAT de navegador pendientes hasta completar la verificación previa de producción.
