#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

AWS_REGION_VALUE="${AWS_REGION:-us-east-2}"
EC2_NAME="${EC2_NAME:-astrid-aws}"
BUCKET="${BACKUP_S3_BUCKET:-}"
INSTANCE_ROLE_NAME="${INSTANCE_ROLE_NAME:-}"
SKIP_IAM=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bucket)
      BUCKET="${2:-}"
      shift 2
      ;;
    --region)
      AWS_REGION_VALUE="${2:-}"
      shift 2
      ;;
    --ec2-name)
      EC2_NAME="${2:-}"
      shift 2
      ;;
    --instance-role)
      INSTANCE_ROLE_NAME="${2:-}"
      shift 2
      ;;
    --skip-iam)
      SKIP_IAM=1
      shift
      ;;
    *)
      echo "Opción desconocida: $1" >&2
      exit 2
      ;;
  esac
done

command -v aws >/dev/null 2>&1 || {
  echo "AWS CLI no está instalado." >&2
  exit 1
}

account_id="$(
  aws sts get-caller-identity     --query Account     --output text
)"

if [ -z "${BUCKET}" ]; then
  BUCKET="astrid-beverly-backups-${account_id}-${AWS_REGION_VALUE}"
fi

echo "[s3] Cuenta AWS: ${account_id}"
echo "[s3] Región: ${AWS_REGION_VALUE}"
echo "[s3] Bucket: ${BUCKET}"
if aws s3api head-bucket     --bucket "${BUCKET}"     --region "${AWS_REGION_VALUE}"     >/dev/null 2>&1; then
  echo "[s3] El bucket ya existe y es accesible."
else
  echo "[s3] Creando bucket privado."
  if [ "${AWS_REGION_VALUE}" = "us-east-1" ]; then
    aws s3api create-bucket       --bucket "${BUCKET}"       --region "${AWS_REGION_VALUE}"       >/dev/null
  else
    aws s3api create-bucket       --bucket "${BUCKET}"       --region "${AWS_REGION_VALUE}"       --create-bucket-configuration         "LocationConstraint=${AWS_REGION_VALUE}"       >/dev/null
  fi
fi

aws s3api put-public-access-block   --bucket "${BUCKET}"   --region "${AWS_REGION_VALUE}"   --public-access-block-configuration     BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-encryption   --bucket "${BUCKET}"   --region "${AWS_REGION_VALUE}"   --server-side-encryption-configuration     '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-bucket-versioning   --bucket "${BUCKET}"   --region "${AWS_REGION_VALUE}"   --versioning-configuration Status=Enabled
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

cat >"${tmp_dir}/lifecycle.json" <<'JSON'
{
  "Rules": [
    {
      "ID": "astrid-daily",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "astrid/production/snapshots/",
          "Tags": [{"Key": "Retention", "Value": "daily"}]
        }
      },
      "Expiration": {"Days": 8},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "astrid-weekly",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "astrid/production/snapshots/",
          "Tags": [{"Key": "Retention", "Value": "weekly"}]
        }
      },
      "Expiration": {"Days": 35},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "astrid-monthly",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "astrid/production/snapshots/",
          "Tags": [{"Key": "Retention", "Value": "monthly"}]
        }
      },
      "Expiration": {"Days": 190},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "astrid-predeploy",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "astrid/production/snapshots/",
          "Tags": [{"Key": "Retention", "Value": "predeploy"}]
        }
      },
      "Expiration": {"Days": 35},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "astrid-manual",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "astrid/production/snapshots/",
          "Tags": [{"Key": "Retention", "Value": "manual"}]
        }
      },
      "Expiration": {"Days": 35},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "astrid-latest-old-versions",
      "Status": "Enabled",
      "Filter": {"Prefix": "astrid/production/latest/"},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 30}
    },
    {
      "ID": "astrid-abort-multipart",
      "Status": "Enabled",
      "Filter": {"Prefix": "astrid/"},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    }
  ]
}
JSON

aws s3api put-bucket-lifecycle-configuration   --bucket "${BUCKET}"   --region "${AWS_REGION_VALUE}"   --lifecycle-configuration "file://${tmp_dir}/lifecycle.json"
if aws s3api get-bucket-policy     --bucket "${BUCKET}"     --region "${AWS_REGION_VALUE}"     >/dev/null 2>&1; then
  echo "[s3] El bucket ya tiene policy; no se sobrescribe."
else
  cat >"${tmp_dir}/tls-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
JSON

  aws s3api put-bucket-policy     --bucket "${BUCKET}"     --region "${AWS_REGION_VALUE}"     --policy "file://${tmp_dir}/tls-policy.json"
fi
if [ "${SKIP_IAM}" != "1" ] && [ -z "${INSTANCE_ROLE_NAME}" ]; then
  instance_id="$(
    aws ec2 describe-instances       --region "${AWS_REGION_VALUE}"       --filters         "Name=tag:Name,Values=${EC2_NAME}"         "Name=instance-state-name,Values=running"       --query 'Reservations[].Instances[].InstanceId'       --output text
  )"

  if [ -n "${instance_id}" ] && [ "${instance_id}" != "None" ]       && [ "$(wc -w <<<"${instance_id}")" -eq 1 ]; then
    profile_arn="$(
      aws ec2 describe-instances         --region "${AWS_REGION_VALUE}"         --instance-ids "${instance_id}"         --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn'         --output text
    )"

    if [ -n "${profile_arn}" ] && [ "${profile_arn}" != "None" ]; then
      profile_name="${profile_arn##*/}"
      INSTANCE_ROLE_NAME="$(
        aws iam get-instance-profile           --instance-profile-name "${profile_name}"           --query 'InstanceProfile.Roles[0].RoleName'           --output text
      )"
    fi
  fi
fi
cat >"${tmp_dir}/instance-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AstridBackupBucketMetadata",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketVersions",
        "s3:ListBucketMultipartUploads"
      ],
      "Resource": "arn:aws:s3:::${BUCKET}"
    },
    {
      "Sid": "AstridBackupObjects",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObjectTagging",
        "s3:GetObjectTagging",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "arn:aws:s3:::${BUCKET}/astrid/*"
    },
    {
      "Sid": "AstridSecureConfigParameter",
      "Effect": "Allow",
      "Action": [
        "ssm:PutParameter",
        "ssm:GetParameter",
        "ssm:GetParameterHistory"
      ],
      "Resource": "arn:aws:ssm:${AWS_REGION_VALUE}:${account_id}:parameter/astrid/production/env-file"
    }
  ]
}
JSON

if [ "${SKIP_IAM}" = "1" ]; then
  echo "[iam] Configuración IAM omitida por --skip-iam."
elif [ -n "${INSTANCE_ROLE_NAME}" ] && [ "${INSTANCE_ROLE_NAME}" != "None" ]; then
  echo "[iam] Aplicando permisos mínimos al rol ${INSTANCE_ROLE_NAME}."
  aws iam put-role-policy     --role-name "${INSTANCE_ROLE_NAME}"     --policy-name AstridProductionBackupsS3     --policy-document "file://${tmp_dir}/instance-policy.json"
else
  echo "[iam] No se pudo resolver el IAM Role del EC2." >&2
  echo "[iam] Policy requerida:" >&2
  cat "${tmp_dir}/instance-policy.json" >&2
  exit 1
fi
echo
echo "Configuración S3 completada."
echo "BACKUP_S3_BUCKET=${BUCKET}"
echo "AWS_REGION=${AWS_REGION_VALUE}"
echo
echo "Siguiente paso en el EC2:"
echo "sudo scripts/install-backup-timer.sh --bucket ${BUCKET} --region ${AWS_REGION_VALUE} --run-now"
