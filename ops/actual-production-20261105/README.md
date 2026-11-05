# Running production operations snapshot

Read-only backup of operational scripts and active Docker Compose files on 2026-11-05. AWS environment values, MySQL data, S3 images, and trained model weights are NOT included. The live application was not restarted, modified, or taken offline to produce this branch.

S3 inspection object lifecycle: 60 days under astrid/inspection-images/v1/. Local cleanup: older than 7 days and verified in S3; trained AI images excluded.

Restore requires securely provisioned secrets, volumes, datasets, and credentials outside GitHub.
