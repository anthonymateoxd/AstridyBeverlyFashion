# Production code backup - November 5, 2026

Snapshot source: **running** `astrid_quality_app` Docker container and the production operational scripts, not the modified but uncommitted server working directory. `main` was intentionally not changed.

Includes deployed Python, HTML templates, CSS, JS, and Compose overrides and S3 retention operational scripts. Does not include `.env`, credentials, databases, private inspection photos, training images, PatchCore model checkpoints, weights, or private backups.

Changes to roles and persistent notification handling which have not yet been deployed are stored as separate patches in `ops/not-yet-deployed-role-flash-20261105/`.

Not a bootable data-complete server image; full disaster recovery also requires AWS snapshots, MySQL backups, S3 media and model artifacts.
