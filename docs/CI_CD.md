# CI/CD de Astrid & Beverly Fashion

El repositorio utiliza GitHub Actions para validar cambios y desplegar producción de forma controlada.

## Flujo

- Todo `push` y todo pull request hacia `main` ejecuta CI.
- CI usa Python 3.11 y MySQL 8, valida sintaxis, Docker Compose y ejecuta `unittest`.
- Un `push` a `main` puede desplegar al EC2 únicamente cuando la variable de repositorio `CD_ENABLED` vale `true`.
- El CD autentica GitHub Actions contra AWS mediante OIDC. No requiere guardar `AWS_ACCESS_KEY_ID` ni `AWS_SECRET_ACCESS_KEY` en GitHub.
- El runner localiza la instancia con la etiqueta `Name=astrid-aws` y ejecuta el despliegue mediante AWS Systems Manager (SSM).
- Antes de cambiar código, el servidor genera un respaldo comprimido de MySQL.
- El despliegue no ejecuta `git clean`, por lo que `.env`, checkpoints, capturas y artefactos persistentes no se eliminan.
- Si el health local no vuelve a `status=ok`, el script restaura el commit anterior y reconstruye los contenedores.

## Variables de GitHub

En **Settings → Secrets and variables → Actions → Variables**:

| Variable | Valor |
| --- | --- |
| `CD_ENABLED` | `false` durante la preparación; cambiar a `true` al habilitar CD |
| `AWS_DEPLOY_ROLE_ARN` | ARN del rol IAM asumible por GitHub Actions |
| `AWS_REGION` | `us-east-2` |
| `EC2_NAME` | `astrid-aws` |
| `APP_DIR` | `/opt/astrid-quality-control` |
| `DEPLOY_USER` | `ubuntu` |

No se necesitan credenciales AWS permanentes como secrets.
## AWS OIDC

Debe existir un proveedor OIDC de GitHub en IAM para:

- URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

El rol indicado en `AWS_DEPLOY_ROLE_ARN` debe confiar únicamente en este repositorio y, de preferencia, en el environment `production`.

Como GitHub cambió el formato de algunos claims OIDC para repositorios creados en 2026, la política de confianza debe configurarse con los claims reales mostrados por GitHub/AWS para este repositorio en lugar de copiar una cadena `sub` antigua.

Permisos mínimos que necesita el rol de despliegue:

- `ec2:DescribeInstances`
- `ssm:GetConnectionStatus`
- `ssm:SendCommand`
- `ssm:GetCommandInvocation`

Restrinja `ssm:SendCommand` a la instancia de producción y al documento `AWS-RunShellScript` cuando se conozca el Instance ID definitivo.

## Requisitos del EC2

La instancia debe:

1. Estar registrada y conectada a AWS Systems Manager.
2. Tener Docker Engine y Docker Compose v2.
3. Mantener el repositorio en `/opt/astrid-quality-control`.
4. Mantener el `.env` de producción fuera de Git.
5. Mantener los artefactos IA y directorios persistentes fuera del control de versiones.
6. Permitir a `ubuntu` ejecutar `git fetch` del repositorio.
7. Poder resolver `http://127.0.0.1:5000/health` después de levantar la aplicación.
## Activación segura

No habilite `CD_ENABLED=true` hasta que `main` contenga la misma versión funcional que producción.

La razón es que CD ejecuta:

```text
origin/main → EC2
```

y, por diseño, el EC2 termina exactamente en el SHA que disparó el workflow.

Secuencia recomendada:

1. Probar este workflow en su rama.
2. Integrar los cambios de preproducción pendientes a `main`.
3. Configurar OIDC y el rol IAM.
4. Crear/proteger el environment `production` en GitHub.
5. Mantener `CD_ENABLED=false` durante una primera ejecución de CI en `main`.
6. Cambiar `CD_ENABLED=true`.
7. Hacer un cambio controlado o ejecutar el siguiente push a `main`.
8. Confirmar `https://www.astridbeverlyfashion.com/health`.

## Rollback y respaldo

Los respaldos se almacenan en:

```text
/opt/astrid-backups/mysql/
```

Por defecto se conservan los 10 más recientes.

El rollback automático restaura código y contenedores al SHA anterior si falla el health. No restaura automáticamente la base de datos: un restore automático sería más riesgoso que conservar el respaldo para una recuperación supervisada.
