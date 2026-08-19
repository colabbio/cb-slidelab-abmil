#!/usr/bin/env bash
# ==============================================================================
# ColabBio TCGA SlideLab & ABMIL End-to-End Benchmark Runner
# ==============================================================================
set -euo pipefail

VFS_MOUNT="${VFS_MOUNT:-/tmp/tcga_vfs}"
DEVICE="${DEVICE:-cuda}"
EXTRACTOR="${EXTRACTOR:-resnet50}"
NUM_SLIDES="${NUM_SLIDES:-1}"

echo "===================================================================="
echo " Starting ColabBio TCGA Real-World Benchmarking Run"
echo " VFS Mount:        ${VFS_MOUNT}"
echo " Device:           ${DEVICE}"
echo " Extractor:        ${EXTRACTOR}"
echo " Target Slides:    ${NUM_SLIDES}"
echo "===================================================================="

mkdir -p "${VFS_MOUNT}/images/svs_raw"

echo "==> Step 1: Preparing sample TCGA diagnostic slides..."
python3 scripts/download-sample-tcga-slides.py -o "${VFS_MOUNT}/images/svs_raw" -n "${NUM_SLIDES}"

echo "==> Step 2: Building/Verifying SlideLab Docker image..."
if command -v nvidia-smi &> /dev/null && [ "${DEVICE}" = "cuda" ]; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DOCKER_TARGET="gpu"
    PROFILE="local"
else
    echo "Running on CPU target"
    DOCKER_TARGET="cpu"
    PROFILE="local"
fi

cd ../SlideLab
docker build --target "${DOCKER_TARGET}" -t colabbio/slidelab:latest .
cd ../cb-slidelab-abmil

echo "==> Step 3: Executing Nextflow On-The-Fly Extraction and Training Pipeline..."
START_TIME=$(date +%s)
nextflow run pipelines/main.nf \
    -profile "${PROFILE}" \
    --vfs_mount "${VFS_MOUNT}" \
    --device "${DEVICE}"
END_TIME=$(date +%s)

TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "===================================================================="
echo " Benchmarking Run Completed Successfully!"
echo " Total Wallclock Time: ${TOTAL_SECONDS} seconds"
echo " Features Generated:   $(find work -name '*.h5' 2>/dev/null | wc -l) files"
echo " Metrics Generated:    $(find work -name '*_metrics.json' 2>/dev/null | wc -l) files"
echo " OMERO Sync Artifacts: $(find work -name 'omero_sync_summary.json' 2>/dev/null | wc -l) files"
echo "===================================================================="
