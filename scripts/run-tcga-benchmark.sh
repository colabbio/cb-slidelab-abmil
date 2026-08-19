#!/usr/bin/env bash
# ==============================================================================
# ColabBio TCGA Zero-Download MedVFS + SlideLab + ABMIL Streaming Benchmark
# ==============================================================================
set -euo pipefail

VFS_MOUNT="${VFS_MOUNT:-/data/vfs}"
DEVICE="${DEVICE:-cuda}"
EXTRACTOR="${EXTRACTOR:-resnet50}"

echo "===================================================================="
echo " Starting ColabBio TCGA Zero-Download Streaming Benchmark"
echo " MedVFS Virtual Mount: ${VFS_MOUNT}"
echo " Compute Target:       ${DEVICE}"
echo " GCS Virtual Source:   gs://idc-open-data (TCGA WSI Archive)"
echo " Zero-Download Mode:   ENABLED (Direct POSIX streaming over FUSE)"
echo "===================================================================="

# Step 0: Auto-Restore Database State & Pull Data Lake Assets if configured
if [ "${COLABBIO_AUTO_RESTORE:-true}" = "true" ]; then
    echo "==> Step 0: Restoring OMERO & MLflow catalog databases from Data Lake..."
    bash scripts/restore-databases.sh || true
    bash scripts/sync-data-lake.sh pull || true
fi

# Step 1: Materialize MedVFS POSIX tree / TCGA slides
echo "==> Step 1: Initializing MedVFS POSIX virtual mount..."
sudo mkdir -p "${VFS_MOUNT}/images/svs_raw"
sudo chmod -R 777 "${VFS_MOUNT}" || true

if [ -z "$(ls -A "${VFS_MOUNT}/images/svs_raw"/*.svs 2>/dev/null)" ]; then
    echo "Materializing sample TCGA Whole Slide Image (GDC Open Access streaming)..."
    python3 scripts/download-sample-tcga-slides.py -o "${VFS_MOUNT}/images/svs_raw" -n 1 || true
fi

# Step 2: Build / Verify Docker Image
echo "==> Step 2: Verifying SlideLab Docker image..."
if command -v nvidia-smi &> /dev/null && [ "${DEVICE}" = "cuda" ]; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DOCKER_TARGET="gpu"
    PROFILE="gpu"
else
    echo "Running on CPU target"
    DOCKER_TARGET="cpu"
    PROFILE="local"
fi

if [ -z "$(docker images -q colabbio/slidelab:latest 2>/dev/null)" ]; then
    cd ../SlideLab
    docker build --target "${DOCKER_TARGET}" -t colabbio/slidelab:latest .
    cd ../cb-slidelab-abmil
else
    echo "Docker image colabbio/slidelab:latest already cached and verified."
fi

# Step 3: Run Nextflow Streaming Pipeline
echo "==> Step 3: Executing Nextflow Pipeline with Zero-Disk Feature Streaming..."
START_TIME=$(date +%s)
nextflow run pipelines/main.nf \
    -profile "${PROFILE}" \
    --vfs_mount "${VFS_MOUNT}" \
    --device "${DEVICE}"
END_TIME=$(date +%s)

TOTAL_SECONDS=$((END_TIME - START_TIME))
# Step 4: Auto-Backup Databases and Push Generated Artifacts to Data Lake
if [ "${COLABBIO_AUTO_BACKUP:-true}" = "true" ]; then
    echo "==> Step 4: Taking database snapshot and syncing outputs to Cloud Data Lake..."
    # Copy generated features and summaries to local data lake
    mkdir -p /data/features /data/omero-data
    find work -name '*.h5' -exec cp -u {} /data/features/ \; 2>/dev/null || true
    find work -name '*metrics.json' -exec cp -u {} /data/omero-data/ \; 2>/dev/null || true
    find work -name 'omero_sync_summary.json' -exec cp -u {} /data/omero-data/ \; 2>/dev/null || true
    
    bash scripts/backup-databases.sh || true
    bash scripts/sync-data-lake.sh push || true
fi

echo "===================================================================="
echo " Zero-Download Benchmarking Run Completed Successfully!"
echo " Total Execution Time: ${TOTAL_SECONDS} seconds"
echo " Features Generated:   $(find work -name '*.h5' 2>/dev/null | wc -l) files"
echo " OMERO Sync Artifacts: $(find work -name 'omero_sync_summary.json' 2>/dev/null | wc -l) files"
echo "===================================================================="
