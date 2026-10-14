# Corrección controlada: meta de producción antes de guardar lotes

Fecha: 13 de octubre de 2026.
Rama: `fix/validar-meta-antes-de-crear-lote`, basada en `fix/auditoria-flujo-inspecciones`.

## Hallazgo confirmado

En la ruta `POST /lotes`, la sección `SMART_BATCH_QUANTITY_GUARD_V1` aparecía **después** de `conn.commit()` y de `return redirect(...)`. Por tanto, la validación del objetivo diario nunca se ejecutaba en solicitudes POST. Podía guardarse un lote con cantidad diferente de la sugerida sin requerir la anulación explícita.

## Corrección preparada

- Mover la guarda dentro del bloque POST, tras comprobar que `planned_quantity` está entre 1 y 5000 y **antes** de consultar la línea o abrir una transacción.
- Reutilizar `production_plan`, ya calculado en el inicio de la vista, sin realizar una consulta repetida.
- Si la meta está completamente planificada y no existe `quantity_override=1`, rechazar la creación.
- Si la cantidad no coincide con la sugerencia y no hay anulación explícita, rechazarla.
- Si no se configuró meta o la anulación está explícita, continuar a las validaciones normales de línea y modelo.
- Eliminar el bloque original muerto después del `return`.

Se actualizaron las pruebas del módulo de lotes para afirmar que la guarda evita incluso abrir la conexión a BD y que la anulación explícita sí permite avanzar hasta la validación de la línea. Se corrigió la prueba de borrado lógico para contemplar `updated_at` en UTC sin alterar el código productivo de eliminación.

## Resultados reproducibles

- **33 pruebas focalizadas:** 33 correctas, sin errores.
- **Suite de regresión completa:** 404 pruebas, 148 omitidas, 2 fallos, 0 errores de ejecución.
- **Dos fallos restantes:** `GuidedSegmentationTests.test_synthetic_garment_is_detected` y `test_guided_segment_uses_a_single_mask`.
- Diagnóstico reproducido: la máscara verde sintética alcanzó cobertura adaptativa de **0.1976** y fue rechazada por el umbral mínimo `GUIDED_COVERAGE_MIN = 0.20`. Se devolvió `method='ninguno'` con causa `cobertura_adaptativa`. No se ha modificado ese umbral, porque afecta la detección real de prendas y requiere contrastarlo con escenas vacías y fotogramas válidos.

Pruebas ejecutadas en la computadora Anthony con una `SECRET_KEY` de prueba solo en el proceso, sin secretos reales ni acceso de escritura a producción.

## Pendientes y límites

1. Integrar y desplegar esta corrección de lotes únicamente tras verificación y copia de reversión; **no está desplegada en AWS**.
2. Investigar las dos pruebas del segmentador con datos de calibración representativos, evitando relajar el umbral por una única muestra.
3. En concurrencia, dos creaciones simultáneas podrían consultar la misma planificación antes de escribir. La corrección recupera la guarda existente, pero **no aporta todavía un bloqueo transaccional de la meta**. Validar este riesgo antes de una carga productiva con varios operadores.
4. Validar con administrador desde navegador los mensajes de cantidad, anulación y la creación normal de un lote de prueba, sin alterar inspecciones históricas.

No hubo migraciones, cambios de cámara ni despliegue de esta rama.
