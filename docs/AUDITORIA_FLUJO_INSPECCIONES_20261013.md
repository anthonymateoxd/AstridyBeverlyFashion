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

## Hallazgo 4 — salud informada por YOLO antiguo (corrección preparada)

El endpoint público `/health` confundía un archivo antiguo `models/best.pt` (YOLO) con la disponibilidad del motor productivo actual, PatchCore. Por ello mostraba `detection_mode=prototype` y `model_ready=false` aunque existían modelos PatchCore activos.

La nueva función `summarize_patchcore_readiness()` calcula, sin cargar modelos ni procesar imágenes, cuántas versiones activas tienen checkpoint accesible y umbral definitivo congelado. Si existe un lote en inspección, el indicador principal corresponde a **su versión específica** y comprueba también la asociación del modelo de prenda. Se conservan `status=ok`, `database`, `inspections`, `camera_source`, `model_ready` y `detection_mode`; estos dos últimos pasan a describir PatchCore en lugar de YOLO heredado.

La respuesta distingue `active_batch_model_ready=null` si no hay lote activo. `readiness_check=metadata_and_checkpoint_file_only` aclara que esto **no demuestra inferencia exitosa, precisión ni calidad óptica**.

Se agregan seis pruebas unitarias para casos sin lote, lote válido, lote con modelo ausente, modelo de otra prenda y configuración incompleta, además del contrato de salud. Pruebas integradas de regresión: **25 correctas**.

## Seguridad, despliegue y UAT pendientes

- Los arreglos se incorporaron a la imagen activa mediante despliegue aislado con respaldo previo, validación y reversión disponible. Falta UAT funcional en navegador.
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
- `fix/auditoria-flujo-inspecciones`: integración conjunta publicada y desplegada en AWS.

## Despliegue controlado confirmado

- Imagen activada: `astrid-quality-control-app:hotfix-auditoria-d84ec7f`, SHA-256 de imagen `5a27fe46cbeeaeb0220bc0a8afe8d3573fa2a39c3879a2e3a39adf2712c59c46`.
- Fuente versionada: `fix/auditoria-flujo-inspecciones` (commits `31b2ad3`, `de7f6e2`, `d84ec7f`).
- Respaldo SQL previo: `astrid/production/snapshots/2026/10/13/20261013T064611Z-ip-172-31-2-214/` en S3 privado.
- Imagen anterior etiquetada: `astrid-quality-control-app:rollback-before-audit-20261013`.
- Se aplicaron `app.py` y `templates/records.html` sobre la imagen que estaba activa, sin reconstrucción de dependencias ni migración de BD.
- Confirmados los hashes del código y la plantilla nueva dentro del contenedor. La comprobación `/health` devolvió `status=ok` y 539 inspecciones. Acceso no autenticado al historial redirige al inicio de sesión (HTTP 302).
- MySQL conservó estado saludable; el trabajador IA no se reinició. Monitoreo RTSP y ambos timers de respaldo permanecen activos. Disco al 84 %, aproximadamente 8 GB disponibles.
- No se detectaron nuevas excepciones de aplicación en la ventana de logs revisada.

**Estado:** correcciones desplegadas y verificadas técnicamente. UAT en navegador pendiente para confirmar visualmente el historial y el rechazo del inicio de un lote con IA incompleta, sin manipular datos valiosos. Los indicadores de salud de YOLO/PatchCore siguen siendo un hallazgo independiente pendiente de corregir.

**Nota de reversión:** si aparece una regresión confirmada, etiquetar `astrid-quality-control-app:rollback-before-audit-20261013` como `astrid-quality-control-app:latest`, recrear exclusivamente el servicio `app` mediante Docker Compose y repetir `/health`. No restaurar la base de datos por un error exclusivamente de presentación.
