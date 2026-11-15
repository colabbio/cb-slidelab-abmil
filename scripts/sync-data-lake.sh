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

FOLDERS=("models" "triton_models" "mlflow-artifacts" "omero-data" "features" "db-backups")

if [ "$DIRECTION" = "publish-release" ]; then
    VERSION_TAG="${2:-$(date -u +%Y%m%d_%H%M%SZ)}"
    echo "===================================================================="
    echo " Publishing Model Release: ${VERSION_TAG}"
    echo " Destination: gs://${BUCKET_NAME}/models/releases/${VERSION_TAG}/"
    echo "===================================================================="
    
    # 1. Models & weights
    if [ -d "${LOCAL_ROOT}/models" ] && [ -n "$(ls -A "${LOCAL_ROOT}/models" 2>/dev/null)" ]; then
        echo "--> [RELEASE] Syncing /models -> gs://${BUCKET_NAME}/models/releases/${VERSION_TAG}/"
        gcloud storage rsync "${LOCAL_ROOT}/models" "gs://${BUCKET_NAME}/models/releases/${VERSION_TAG}/" --recursive || true
        
        echo "--> [RELEASE] Updating latest pointer -> gs://${BUCKET_NAME}/models/latest/"
        gcloud storage rsync "${LOCAL_ROOT}/models" "gs://${BUCKET_NAME}/models/latest/" --recursive || true
    else
        echo "⚠️  No local models found in ${LOCAL_ROOT}/models to publish."
    fi

    # 2. Triton Models
    if [ -d "${LOCAL_ROOT}/triton_models" ] && [ -n "$(ls -A "${LOCAL_ROOT}/triton_models" 2>/dev/null)" ]; then
        echo "--> [RELEASE] Syncing /triton_models -> gs://${BUCKET_NAME}/triton_models/releases/${VERSION_TAG}/"
        gcloud storage rsync "${LOCAL_ROOT}/triton_models" "gs://${BUCKET_NAME}/triton_models/releases/${VERSION_TAG}/" --recursive || true
        
        echo "--> [RELEASE] Updating latest triton pointer -> gs://${BUCKET_NAME}/triton_models/latest/"
        gcloud storage rsync "${LOCAL_ROOT}/triton_models" "gs://${BUCKET_NAME}/triton_models/latest/" --recursive || true
    fi

    # 3. Metrics & Manifest snapshot
    if [ -d "${LOCAL_ROOT}/omero-data" ] && [ -n "$(ls -A "${LOCAL_ROOT}/omero-data" 2>/dev/null)" ]; then
        echo "--> [RELEASE] Syncing metrics -> gs://${BUCKET_NAME}/models/releases/${VERSION_TAG}/metrics/"
        gcloud storage rsync "${LOCAL_ROOT}/omero-data" "gs://${BUCKET_NAME}/models/releases/${VERSION_TAG}/metrics/" --recursive || true
    fi

    echo "✅ Published release ${VERSION_TAG} to GCS Data Lake."
    exit 0
fi

for folder in "${FOLDERS[@]}"; do
    mkdir -p "${LOCAL_ROOT}/${folder}"
    
    if [ "$DIRECTION" = "push" ]; then
        echo "--> [PUSH] Syncing ${LOCAL_ROOT}/${folder} -> gs://${BUCKET_NAME}/${folder}/"
        gcloud storage rsync "${LOCAL_ROOT}/${folder}" "gs://${BUCKET_NAME}/${folder}" --recursive || true
    elif [ "$DIRECTION" = "pull" ]; then
        echo "--> [PULL] Syncing gs://${BUCKET_NAME}/${folder}/ -> ${LOCAL_ROOT}/${folder}"
        gcloud storage rsync "gs://${BUCKET_NAME}/${folder}" "${LOCAL_ROOT}/${folder}" --recursive || true
    else
        echo "Error: Unknown sync direction '${DIRECTION}'. Use 'push', 'pull', or 'publish-release <TAG>'."
        exit 1
    fi
done

echo "===================================================================="
echo " Data Lake Sync Completed Successfully!"
echo "===================================================================="
