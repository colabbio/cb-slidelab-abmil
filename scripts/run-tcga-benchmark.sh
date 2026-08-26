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
sudo mkdir -p "${VFS_MOUNT}/images/svs_raw" "${VFS_MOUNT}/features" "${VFS_MOUNT}/clinical" 2>/dev/null || true
sudo chmod -R 777 "${VFS_MOUNT}" 2>/dev/null || true

# Copy existing cohort manifest and features if available locally or in data lake
for src_csv in \
    "/home/jmolmo/git/colabbio/cb-nextflow-modules/scripts/cohort_real_brca_gcp.csv" \
    "/home/jmolmo/colabbio/cohort_real_brca_gcp.csv" \
    "$HOME/colabbio/cohort_real_brca_gcp.csv" \
    "cohort_real_brca_gcp.csv"; do
    if [ -f "$src_csv" ]; then
        cp "$src_csv" "${VFS_MOUNT}/clinical/cohort_real_brca.csv" 2>/dev/null || true
        cp "$src_csv" "/data/cohort_real_brca.csv" 2>/dev/null || true
        break
    fi
done

# Step 2: Auto-Detect Hardware & Profile
echo "==> Step 2: Checking Compute Hardware & Docker Profile..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
    PROFILE="gpu"
else
    echo "💻 Running on CPU target (No physical GPU detected)"
    DEVICE="cpu"
    PROFILE="local"
fi

# Step 3: Run Nextflow Streaming Pipeline
echo "==> Step 3: Executing Nextflow Pipeline with Zero-Disk Feature Streaming (Profile: ${PROFILE}, Device: ${DEVICE})..."
START_TIME=$(date +%s)
nextflow run pipelines/main.nf \
    -profile "${PROFILE}" \
    --vfs_mount "${VFS_MOUNT}" \
    --cohort_csv "/data/cohort_real_brca.csv" \
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
