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

## Hallazgo 4 — salud informada por YOLO antiguo (corregido y desplegado)

El endpoint público `/health` confundía un archivo antiguo `models/best.pt` (YOLO) con la disponibilidad del motor productivo actual, PatchCore. Por ello mostraba `detection_mode=prototype` y `model_ready=false` aunque existían modelos PatchCore activos.

La nueva función `summarize_patchcore_readiness()` calcula, sin cargar modelos ni procesar imágenes, cuántas versiones activas tienen checkpoint accesible y umbral definitivo congelado. Si existe un lote en inspección, el indicador principal corresponde a **su versión específica** y comprueba también la asociación del modelo de prenda. Se conservan `status=ok`, `database`, `inspections`, `camera_source`, `model_ready` y `detection_mode`; estos dos últimos pasan a describir PatchCore en lugar de YOLO heredado.

La respuesta distingue `active_batch_model_ready=null` si no hay lote activo. `readiness_check=metadata_and_checkpoint_file_only` aclara que esto **no demuestra inferencia exitosa, precisión ni calidad óptica**.

Se agregaron seis pruebas unitarias para casos sin lote, lote válido, lote con modelo ausente, modelo de otra prenda y configuración incompleta, además del contrato de salud. Pruebas integradas de regresión: **25 correctas**.

**Despliegue confirmado:** nueva imagen Docker `astrid-quality-control-app:hotfix-patchcore-health-c9906fd`, ID `sha256:bfc2541af97cd1d7eeae13187ecd642456fe052b4dc91b4e53169c0facd00fed`. El endpoint devolvió `status=ok`, `detection_mode=patchcore`, `active_patchcore_models=4`, `ready_patchcore_models=3`, `active_batch_present=false`, `model_ready=true`. Respaldo MySQL previo en S3: `astrid/production/snapshots/2026/10/13/20261013T073342Z-ip-172-31-2-214/`; rollback de imagen: `astrid-quality-control-app:rollback-before-patchcore-health-20261013`. MySQL y worker IA siguieron operativos, temporizadores activos.

## Hallazgo 5 — referencias de imágenes históricas sin archivo localizable

Se realizó una inspección de solo lectura de `inspections.image_original` y `inspections.image_result` en el contenedor productivo, resolviendo las rutas desde `/app/static`. **Los registros de BD se conservaron; no se modificaron imágenes ni inspecciones.**

| Grupo | Inspecciones | Original presente | Original ausente | Resultado presente | Resultado ausente | Resultado sin ruta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Con lote | 221 | 196 | 25 | 196 | 25 | 0 |
| Histórico sin lote | 318 | 0 | 318 | 0 | 317 | 1 |
| Total | 539 | 196 | 343 | 196 | 342 | 1 |

Los **25 registros con lote afectados** se distribuyen así: 1 del 8 de septiembre de 2026, 23 del 10 de septiembre y 1 del 11 de septiembre. Sus identificadores de inspección son: 327, 354, 358–378, 379 y 380 (el conjunto exacto está registrado en las comprobaciones operativas). Debe reconstruirse una lista de rutas originales y cotejarla con archivos de respaldo y directorios históricos antes de cualquier recuperación. No sustituir evidencias por imágenes nuevas ni alterar resultados para ocultar ausencias.

El servidor conserva una copia de seguridad de directorios de capturas y resultados en S3 (`astrid/production/emergency/20261013T023413Z/`), pero **su existencia no confirma que incluya los archivos ausentes**. La recuperación debe ser selectiva, verificando nombres y hashes y sin sobreescribir archivos existentes. Hasta entonces, el hallazgo queda abierto.

## Hallazgo 6 — listado de versiones IA disponibles para lotes (preparado, sin desplegar)

La suite local detectó que las consultas `GET /lotes` y `POST /lotes` permitían seleccionar versiones de IA activas sin checkpoint ni umbral congelado. Se añadió a las consultas SQL el requisito de checkpoint no vacío, `threshold_final` y `threshold_frozen_at`; el subquery que escoge la versión también aplica los criterios. Permanece además el control del archivo real cuando se inicia el lote, ya desplegado en la corrección anterior.

La corrección está en la rama de auditoría, **pendiente de despliegue**, para evitar más reinicios del lector de cámara durante la prueba nocturna. La prueba del flujo de consulta existente y 26 pruebas focalizadas pasaron después del cambio.

## Resultado de la suite amplia

Se ejecutaron 400 pruebas locales (148 omitidas) con una `SECRET_KEY` temporal exclusiva del proceso de prueba; no se utilizaron secretos ni base de datos productivos. El resultado original fue **5 fallos**:

- 2 fallos en segmentación de prendas sintéticas (requieren reproducir el entorno de prueba y comparar con requisitos de captura reales).
- 1 fallo en la consulta de versiones IA del formulario de lotes (corregido en rama, prueba individual posterior superada).
- 1 fallo de la prueba de cantidad del lote (el simulador de `fetch_one` y sus expectativas deben reconciliarse con las validaciones de línea/modelo actuales; no afirmar todavía un fallo en producción).
- 1 fallo de prueba de eliminación de un modelo incompleto (espera parámetros antiguos; el código añade correctamente el timestamp de eliminación; requiere revisar contrato de test).

La primera corrida sin `SECRET_KEY` había registrado errores de importación de Flask, atribuibles al entorno de pruebas. **No declarar suite completa en verde** hasta resolver los fallos restantes y repetirla en un entorno controlado.

## Seguridad, despliegue y UAT pendientes

- Los arreglos se incorporaron a la imagen activa mediante despliegue aislado con respaldo previo, validación y reversión disponible. Falta UAT funcional en navegador.
- Probar sesión autenticada con una prenda NORMAL, una alerta pendiente y una revisada; comprobar resumen, detalle, historial e informes.
- Intentar iniciar **solo en entorno de pruebas** un lote cuyo modelo no tiene checkpoint y verificar que permanece en preparación.
- Buscar en fuentes anteriores las imágenes de los 318 registros históricos y las 25 inspecciones productivas afectadas, preservando evidencia y trazabilidad.
- Conservar la separación de los 318 registros históricos respecto de los informes productivos salvo decisión explícita.
- La carpeta fuente de EC2 no es idéntica a la imagen en ejecución: no reconstruir con `docker compose build` ni hacer `git reset/pull` hasta reconciliar el repositorio.
- Los respaldos automáticos permanecen configurados; revisar primera copia programada, restauración periódica y crecimiento de EBS/S3.
- Rotar las credenciales de cámara y Wi-Fi expuestas previamente de forma coordinada.

## Cambios versionados

- `fix/historial-decisiones-finales`: clasificador y pruebas.
- `fix/lotes-validar-modelo-ia`: guardia de inicio y pruebas.
- `fix/auditoria-flujo-inspecciones`: integración conjunta publicada y desplegada en AWS.

## Despliegue anterior del historial y lotes

- Imagen activada: `astrid-quality-control-app:hotfix-auditoria-d84ec7f`, SHA-256 de imagen `5a27fe46cbeeaeb0220bc0a8afe8d3573fa2a39c3879a2e3a39adf2712c59c46`.
- Fuente versionada: `fix/auditoria-flujo-inspecciones` (commits `31b2ad3`, `de7f6e2`, `d84ec7f`).
- Respaldo SQL previo: `astrid/production/snapshots/2026/10/13/20261013T064611Z-ip-172-31-2-214/` en S3 privado.
- Imagen anterior etiquetada: `astrid-quality-control-app:rollback-before-audit-20261013`.
- Se aplicaron `app.py` y `templates/records.html` sobre la imagen que estaba activa, sin reconstrucción de dependencias ni migración de BD.
- Confirmados los hashes del código y la plantilla nueva dentro del contenedor. La comprobación `/health` devolvió `status=ok` y 539 inspecciones. Acceso no autenticado al historial redirige al inicio de sesión (HTTP 302).
- MySQL conservó estado saludable; el trabajador IA no se reinició. Monitoreo RTSP y ambos timers de respaldo permanecen activos. Disco al 84 %, aproximadamente 8 GB disponibles.
- No se detectaron nuevas excepciones de aplicación en la ventana de logs revisada.

**Estado actualizado:** las correcciones del historial y de inicio de lotes están desplegadas, y el diagnóstico de salud PatchCore quedó corregido posteriormente mediante `c9906fd`. Continúan pendientes la validación visual autenticada, las pruebas de inferencia sobre prendas reales y el hallazgo de imágenes históricas ausentes.

**Nota de reversión:** la etiqueta `astrid-quality-control-app:rollback-before-patchcore-health-20261013` conserva la aplicación con las correcciones de historial y lotes, pero sin el diagnóstico nuevo. La etiqueta `astrid-quality-control-app:rollback-before-audit-20261013` es anterior a ambas correcciones. En una regresión confirmada, elegir la versión de destino adecuada, recrear solamente el servicio web y comprobar `/health`. No restaurar MySQL por un fallo de presentación.
