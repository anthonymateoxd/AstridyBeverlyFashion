# Registro de estado, respaldo y recuperación — Astrid y Beverly Fashion

**Fecha de corte:** 18 de noviembre de 2026 (Guatemala, UTC−6)  
**Proyecto:** Automatización del control de calidad textil mediante visión artificial  
**Repositorio:** https://github.com/anthonymateoxd/AstridyBeverlyFashion  
**Propósito:** conservar evidencia de qué código está publicado en GitHub, qué sigue solo en la computadora o en AWS y cómo recuperar el trabajo sin sobrescribir datos productivos.

> **ADVERTENCIA IMPORTANTE:** Este documento es un inventario de estado y una guía de recuperación, **no demuestra que exista un respaldo completo del servidor, base de datos, fotografías y modelos entrenados**. No fusionar ni desplegar una rama por el solo hecho de estar guardada en GitHub.

## 1. GitHub: qué quedó guardado

| Referencia | Estado observado |
|---|---|
| Rama principal `main` | Commit `c963cb07e4aa93690afa75240808f3cd19d7fea2` |
| Rama publicada de preservación | `sync/astrid-local-20261118` |
| Commit de preservación del trabajo local | `233a48f18593c14dc8f6dc681969feff3207ca94` |
| Diferencia de esta rama frente a `main` (antes de agregar esta nota) | 17 commits por delante, 0 detrás, 93 archivos modificados en conjunto |
| Último commit de preservación | 42 archivos incorporados/actualizados: 4,942 líneas agregadas y 654 eliminadas |
| Revisión para futura incorporación a `main` | PR #1, **borrador, no fusionada**: https://github.com/anthonymateoxd/AstridyBeverlyFashion/pull/1 |

**Significado:** los avances de código incluidos en esa rama ya están en GitHub, pero **`main` no incluye todos esos cambios**. El servidor AWS no recibió ningún despliegue con esta operación.

El commit `233a48f` provino de una **copia temporal aislada del repositorio local**. Esa copia temporal fue eliminada tras confirmarse el push. No se hicieron commit, reset, limpieza, checkout ni cambios a la copia original del usuario.

## 2. Computadora original: estado que se preservó

- Carpeta usada en la comprobación: `C:\Users\antho\Desktop\astrid_quality_control`.
- Rama activa: `feat/eliminacion-logica-lotes`.
- Último commit local original: `acfea8fe44dc45b90087834f34ff4f39e518e579`.
- Inicialmente había **24 archivos versionados modificados** más archivos sin seguimiento, incluyendo nuevos módulos, JavaScript y pruebas.
- Se trasladaron a la rama de preservación los cambios de esos archivos versionados y **una selección explícita de archivos nuevos de código y pruebas**.
- **No se subieron** los ZIP de explicación/video, el directorio local `ops/` ni los demás elementos no incluidos expresamente; tampoco fotografías, respaldos binarios o datos operativos.
- **Consecuencia:** la rama de GitHub NO es un espejo bit a bit de toda la carpeta local. Antes de borrar, formatear o sustituir la computadora se requiere una copia integral verificada de lo que falte.

## 3. AWS: diferencias de producción que NO están respaldadas en esta rama

**Región de consulta:** `us-east-2`; verificación por AWS Systems Manager (SSM).  
**Directorio de código en el host:** `/opt/astrid-quality-control`.

- El repositorio de ese directorio estaba en `main`, commit `c963cb0`, pero con **17 archivos versionados modificados**, incluidos `app.py`, `ai_training.py`, `docker-compose.yaml` y plantillas.
- Existen además **archivos no versionados de producción**: código auxiliar, pruebas, referencias/calibraciones, carpetas de respaldo y archivos auxiliares.
- Contenedor de aplicación: `astrid_quality_app`, imagen `astrid-candidate:d733420`, **en ejecución**.
- La aplicación usa **montajes bind** desde rutas del host: por ejemplo, `/opt/astrid-release-d733420/ui-daily-batch-numbering-20261028/app.py` se monta como `/app/app.py` y otras rutas del release montan plantillas y `ai_training.py`. Por tanto, ni `git log` del host ni la etiqueta de imagen describen por sí solos la versión efectiva.
- La huella SHA-256 de `/app/app.py` en ejecución se observó como `e674d8efffa96651a2a255c8ff10c990e3d5b2626b1460158ff5a745370d1012`. **No coincidía** con `app.py` de la computadora, `main` o el commit original de la rama de trabajo.
- También diferían las huellas de `/app/ai_training.py`, `/app/templates/batches.html` y `/app/templates/dashboard.html`.
- Contenedor `astrid_ai_worker`: **pausado** al momento de consultar; investigar si fue intencional antes de modificarlo.
- Contenedor MySQL `textile_mysql`: **saludable** en esa consulta.
- La web pública `https://www.astridbeverlyfashion.com/` respondió **HTTP 200** en la comprobación, lo que **no** certifica todos los módulos ni la IA.

**Consecuencia:** los cambios exclusivos de AWS y los montajes del contenedor **todavía no fueron extraídos, reconciliados ni guardados en GitHub como un respaldo completo**. No sustituir ese código desde `main` ni desde la rama de preservación sin reconciliación.

## 4. Verificaciones de la rama de preservación

- Compilación de sintaxis Python: **completada**.
- `git diff --cached --check` sobre los 42 archivos del último commit: **sin incidencias**.
- 13 pruebas puntuales pasaron: 2 de evidencias de historial, 4 de filtro de modelos y 7 de preparación de IA.
- La suite total **no** quedó aprobada: varios grupos no arrancaron por dependencias faltantes en el intérprete de pruebas (`mysql.connector`, `torch`, `pytest`) y una imagen diagnóstica ausente en el entorno aislado.
- Una revisión por patrones de secretos sobre **los 42 archivos de la última incorporación** halló una coincidencia en una clave de sesión de prueba. **No equivale** a una auditoría completa de toda la historia Git ni de AWS.

## 5. Qué aún requiere un respaldo independiente

1. Código y configuración efectivamente activos en AWS (imagen Docker, digest, `docker inspect`, montajes bind, archivos fuente del release y código auxiliar), sin capturar/publicar secretos.
2. Archivos modificados/no versionados del host AWS, discriminando código frente a datos temporales o personales.
3. Base de datos MySQL: dump consistente y restauración probada. **No se creó ni verificó uno nuevo en esta operación**.
4. Modelos IA entrenados, checkpoints, referencias, calibraciones y artefactos, conservando estructura, permisos y hashes.
5. Fotografías de inspección, evidencias e imágenes de entrenamiento, tanto locales como en S3, verificando políticas y accesibilidad.
6. Inventario de variables de entorno, nombres y ubicación de secretos; **nunca** subir valores de contraseñas, claves o tokens a un repositorio público.
7. ZIP, `ops/` y otros archivos locales excluidos, si se necesitan para recuperación integral.
8. Validación de pruebas completas y reconciliación antes de fusionar PR #1.

## 6. Instrucciones de recuperación del código guardado en GitHub

En una máquina nueva o una carpeta vacía y segura:

```bash
git clone https://github.com/anthonymateoxd/AstridyBeverlyFashion.git
cd AstridyBeverlyFashion
git fetch origin
git switch -c recuperacion-astrid origin/sync/astrid-local-20261118
git rev-parse HEAD
```

- El último comando mostrará el commit actual de esa rama (puede ser posterior a `233a48f` por la incorporación de esta nota).
- Para recuperar **exactamente el código publicado en el momento de preservación**, consultar o crear una rama nueva desde `233a48f18593c14dc8f6dc681969feff3207ca94` sin modificar `main`.
- Esa clonación recupera **solo los archivos incluidos en Git**, no la base de datos ni artefactos operativos de AWS, ni los archivos locales excluidos.
- **Nunca ejecutar un despliegue directo al servidor con este checkout**. Primero preservar el sistema actual y comparar versiones.

## 7. Protocolo mínimo antes de cualquier actualización futura

1. Confirmar que `main`, la rama de preservación y PR #1 siguen accesibles en GitHub.
2. Tomar y **verificar** respaldos completos independientes de AWS/MySQL/S3/modelos/fotos.
3. Inventariar y comparar los bind mounts y el código realmente usado por el contenedor.
4. Integrar correcciones de AWS con la rama preservada **en otra rama de trabajo**, no con un force-push a `main`.
5. Ejecutar pruebas funcionales, IA, historial, lotes, permisos y respaldos; documentar las que no se puedan ejecutar.
6. Revisar cambios por seguridad, entonces fusionar con aprobación y desplegar de manera controlada con opción de reversión.

### Regla de seguridad

**No borrar ni reemplazar carpetas de AWS o de la computadora, contenedores, volúmenes, MySQL, fotos ni checkpoints basándose solamente en esta nota.** Este registro debe acompañar a un respaldo de datos real, y no sustituirlo.
