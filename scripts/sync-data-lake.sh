#!/bin/bash
# ==============================================================================
# ColabBio Data Lake Sync Utility
# Syncs persistent assets (models, MLflow artifacts, OMERO data, DB dumps)
# between local POSIX storage (/data) and Google Cloud Storage.
# ==============================================================================
set -euo pipefail

BUCKET_NAME="${COLABBIO_DATA_LAKE_BUCKET:-colabbio-data-lake-prevvy1}"
LOCAL_ROOT="${COLABBIO_DATA_ROOT:-/data}"
DIRECTION="${1:-push}" # "push" (local -> GCS) or "pull" (GCS -> local)

echo "===================================================================="
echo " ColabBio Data Lake Sync"
echo " Local Storage Root: ${LOCAL_ROOT}"
echo " Cloud Data Lake:    gs://${BUCKET_NAME}"
echo " Mode:               ${DIRECTION}"
echo "===================================================================="

FOLDERS=("models" "mlflow-artifacts" "omero-data" "features" "db-backups")

for folder in "${FOLDERS[@]}"; do
    mkdir -p "${LOCAL_ROOT}/${folder}"
    
    if [ "$DIRECTION" = "push" ]; then
        echo "--> [PUSH] Syncing ${LOCAL_ROOT}/${folder} -> gs://${BUCKET_NAME}/${folder}/"
        gcloud storage rsync "${LOCAL_ROOT}/${folder}" "gs://${BUCKET_NAME}/${folder}" --recursive || true
    elif [ "$DIRECTION" = "pull" ]; then
        echo "--> [PULL] Syncing gs://${BUCKET_NAME}/${folder}/ -> ${LOCAL_ROOT}/${folder}"
        gcloud storage rsync "gs://${BUCKET_NAME}/${folder}" "${LOCAL_ROOT}/${folder}" --recursive || true
    else
        echo "Error: Unknown sync direction '${DIRECTION}'. Use 'push' or 'pull'."
        exit 1
    fi
done

echo "===================================================================="
echo " Data Lake Sync Completed Successfully!"
echo "===================================================================="
