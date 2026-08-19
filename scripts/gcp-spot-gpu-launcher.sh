#!/usr/bin/env bash
# ==============================================================================
# ColabBio GCP Spot GPU Launcher for SlideLab & WSI_MIL TCGA Benchmarking
# ==============================================================================
set -euo pipefail

INSTANCE_NAME="${INSTANCE_NAME:-colabbio-slidelab-benchmark-spot}"
ZONE="${ZONE:-europe-west4-a}"  # Or us-central1-a, us-east4-c
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"  # 4 vCPUs, 16 GB RAM, 1x NVIDIA L4 (24GB VRAM)
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}" # Or nvidia-tesla-t4
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu121-debian-11-py310}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

echo "===================================================================="
echo " Launching ColabBio Spot GPU Benchmark VM on Google Cloud Platform"
echo " Instance:       ${INSTANCE_NAME}"
echo " Zone:           ${ZONE}"
echo " Machine Type:   ${MACHINE_TYPE}"
echo " Accelerator:    ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo " Provisioning:   SPOT (Up to 70-90% discount, ~0.20-0.30 EUR/hr)"
echo "===================================================================="

# Check gcloud authentication
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "${CURRENT_PROJECT}" ]; then
    echo "ERROR: No active GCP project found. Run 'gcloud auth login' and 'gcloud config set project <PROJECT_ID>'."
    exit 1
fi
echo "Active GCP Project: ${CURRENT_PROJECT}"

# Launch Spot Instance with GPU drivers pre-installed
gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${CURRENT_PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --maintenance-policy="TERMINATE" \
    --provisioning-model="SPOT" \
    --instance-termination-action="DELETE" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type="pd-balanced" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --metadata="install-nvidia-driver=True" \
    --tags="colabbio-benchmark,http-server,https-server"

echo ""
echo " Spot GPU instance '${INSTANCE_NAME}' created successfully in ${ZONE}!"
echo "To connect to your VM:"
echo "    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo ""
echo "To copy repository and run the benchmark:"
echo "    gcloud compute scp --recurse /home/jmolmo/git/colabbio ${INSTANCE_NAME}:~/ --zone=${ZONE}"
echo "    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='cd ~/colabbio/cb-slidelab-abmil && bash scripts/run-tcga-benchmark.sh'"
echo "===================================================================="
