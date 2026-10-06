# Astrid Quality Control

Sistema de control de calidad textil desarrollado para la tesis de inspección de prendas de Astrid y Beverly Fashion. La aplicación integra gestión operativa, captura de cámara, inspección en tiempo real y un flujo de visión artificial basado en PatchCore.

## Arquitectura

- Backend y aplicación web: Flask / Python 3.11.
- Base de datos: MySQL 8.
- Visión artificial: OpenCV, anomalib y PatchCore.
- Captura: cámara local o RTSP.
- Ejecución reproducible: Docker y Docker Compose.
- Worker independiente para entrenamiento y preparación de modelos IA.
- Persistencia de modelos, datasets y evidencias en `AI_ARTIFACTS_ROOT`.

## Funcionalidades principales

- Autenticación y control de acceso por roles.
- Dashboard operativo.
- Gestión de modelos de prenda.
- Gestión de lotes e inspecciones.
- Estación de inspección manual y automática.
- Captura guiada para datasets normales.
- Entrenamiento y versionado de modelos PatchCore.
- Banco de validación, congelamiento de umbral y prueba final.
- Activación de una versión IA por modelo de prenda.
- Historial, revisión de resultados e informes.
- Trazabilidad de eventos del ciclo de IA.

## Requisitos

Para ejecución con Docker:

- Docker Engine o Docker Desktop.
- Docker Compose v2.
- Acceso de red a la cámara RTSP cuando corresponda.

Para ejecución directa:

- Python 3.11.
- MySQL 8.
- Dependencias de `requirements.txt`.

## Configuración

Cree el archivo local de entorno a partir de la plantilla:

```bash
cp .env.example .env
```

En Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Complete como mínimo las credenciales de MySQL, `SECRET_KEY`, las credenciales de cámara y, en una instalación nueva, `ADMIN_PASSWORD`.

`ADMIN_PASSWORD` no se versiona y debe tener al menos 12 caracteres cuando se crea el administrador inicial.

Nunca suba el archivo `.env` al repositorio.

## Ejecución con Docker

```bash
docker compose up --build -d
```

La aplicación queda disponible en:

```text
http://localhost:5000
```

Para revisar el estado:

```bash
docker compose ps
docker compose logs -f app
docker compose logs -f ai-worker
```

## Ejecución local

Cree y active un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Pruebas

La suite usa `unittest`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Algunas pruebas de integración utilizan MySQL cuando está disponible.

## Datos y artefactos IA

Los siguientes directorios contienen datos generados en ejecución y deliberadamente no forman parte del repositorio Git:

- `ai_artifacts/`
- `patchcore_results/`
- `static/captures/`
- `static/results/`
- `diagnostics/`
- `instance/`

Esto evita publicar datasets, checkpoints, evidencias, bases locales y archivos de gran tamaño en un repositorio público.

Para desplegar una instancia existente no basta con clonar Git. Deben migrarse por separado:

1. La base de datos MySQL.
2. Los artefactos IA requeridos por los modelos activos.
3. El archivo `.env` de producción, creado directamente en el servidor.
4. Los datos persistentes que deban conservarse.

## Modelo IA en producción

La estación productiva resuelve la versión IA activa asociada al lote y al modelo de prenda. El checkpoint y el umbral se obtienen desde la metadata persistida en MySQL y desde `AI_ARTIFACTS_ROOT`.

Los modelos de entrenamiento, validación y producción no deben mezclarse manualmente ni copiarse sobre una versión histórica.

## Seguridad

- `.env` está excluido de Git y del contexto de Docker.
- No se publican contraseñas, claves, tokens ni credenciales de cámara.
- La contraseña del administrador inicial debe definirse mediante variable de entorno.
- Los checkpoints y datasets se almacenan fuera del repositorio.
- En producción, MySQL no debe exponerse públicamente a Internet.
- El acceso web de producción debe publicarse detrás de HTTPS.

## Despliegue

El repositorio contiene el código fuente y configuración reproducible. El despliegue en servidor se realiza en cuatro partes:

```text
GitHub
   |
   +--> código de aplicación
   |
Servidor
   +--> Docker / Flask / MySQL
   +--> .env privado
   +--> restauración de base de datos
   +--> restauración de AI_ARTIFACTS_ROOT
   +--> cámara accesible por red
```

Para AWS, el destino previsto es una instancia EC2 con almacenamiento persistente. Los artefactos IA y los respaldos deben migrarse de forma independiente al código.

## Estado de desarrollo

La rama principal debe representar una versión validada. El trabajo experimental de IA se conserva en ramas de funcionalidad antes de integrarse a `main`.
