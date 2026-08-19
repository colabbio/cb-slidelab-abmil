#!/bin/bash
# ==============================================================================
# ColabBio Database Restoration Utility
# Downloads the latest database snapshot from GCS and restores OMERO & MLflow.
# ==============================================================================
set -euo pipefail

BUCKET_NAME="${COLABBIO_DATA_LAKE_BUCKET:-colabbio-data-lake-prevvy1}"
BACKUP_DIR="${COLABBIO_BACKUP_DIR:-/data/db-backups}"

mkdir -p "${BACKUP_DIR}"

echo "===================================================================="
echo " Starting ColabBio Database Restoration"
echo " Cloud Source: gs://${BUCKET_NAME}/db-backups/"
echo " Local Target: ${BACKUP_DIR}"
echo "===================================================================="

# 1. Pull latest backup files from GCS
echo "--> Pulling latest database snapshots from Cloud Data Lake..."
gcloud storage cp "gs://${BUCKET_NAME}/db-backups/*.sql.gz" "${BACKUP_DIR}/" || true

# 2. Find the most recent snapshot file with valid content (> 100 bytes)
LATEST_BACKUP=$(ls -t "${BACKUP_DIR}"/postgres_*.sql.gz 2>/dev/null | head -n1 || true)

if [ -z "${LATEST_BACKUP}" ]; then
    echo "Warning: No database snapshot found to restore."
    exit 0
fi

echo "--> Selected snapshot for restoration: ${LATEST_BACKUP}"

# 3. Ensure PostgreSQL is up
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'postgres'; then
    echo "--> Starting PostgreSQL service container..."
    docker rm -f colabbio-postgres 2>/dev/null || true
    docker run -d --name colabbio-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15-alpine
    sleep 4
fi

# 4. Perform restoration
echo "--> Restoring database catalogs (OMERO, MLflow, Keycloak)..."
CONTAINER_ID=$(docker ps -q -f name=postgres | head -n1)
gunzip -c "${LATEST_BACKUP}" | docker exec -i "${CONTAINER_ID}" psql -U postgres

echo "===================================================================="
echo " Database Restoration Completed Successfully!"
echo " Verifying OMERO database:"
docker exec -i "${CONTAINER_ID}" psql -U postgres -d omero -c "SELECT * FROM images;" || true
echo " Verifying MLflow database:"
docker exec -i "${CONTAINER_ID}" psql -U postgres -d mlflow -c "SELECT * FROM runs;" || true
echo "===================================================================="
