# PENDING changes (NOT yet running in production)

These are unified patches to the active production files. Role access for MODEL_MANAGER (pause, resume, finish batches) and persistent flash auto-dismiss are prepared but require a controlled deployment when inspection is idle. DO NOT assume these patches are deployed.

The flash JS hotfix exists in the running container and is backed up at `static/js/global-flash.js`; this file may not survive container recreation until included in the next Docker image or mounted.
