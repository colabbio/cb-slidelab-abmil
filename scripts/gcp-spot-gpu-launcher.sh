#!/usr/bin/env bash
# ==============================================================================
# ColabBio GCP Spot GPU Launcher for SlideLab & WSI_MIL TCGA Benchmarking
# ==============================================================================
set -euo pipefail

INSTANCE_NAME="${INSTANCE_NAME:-colabbio-slidelab-benchmark-spot}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-150GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"

echo "===================================================================="
echo " Launching ColabBio Spot GPU VM (Docker + K3s) on GCP us-central1"
echo " Instance:       ${INSTANCE_NAME}"
echo " Zone:           ${ZONE}"
echo " Machine Type:   ${MACHINE_TYPE}"
echo " Accelerator:    ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo " Environment:    Docker + K3s Dual-Mode GPU Enabled"
echo " Provisioning:   SPOT (Up to 70-90% discount, ~0.25 EUR/hr)"
echo "===================================================================="

CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "${CURRENT_PROJECT}" ]; then
    echo "ERROR: No active GCP project found. Run 'gcloud auth login' and 'gcloud config set project <PROJECT_ID>'."
    exit 1
fi
echo "Active GCP Project: ${CURRENT_PROJECT}"

# Startup script to configure NVIDIA drivers, Docker, NVIDIA Container Toolkit and K3s
STARTUP_SCRIPT=$(cat << 'EOF_STARTUP'
#!/bin/bash
set -ex
export DEBIAN_FRONTEND=noninteractive

# Update system
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release build-essential git openjdk-17-jre-headless libopenslide-dev

# 1. Install Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2. Install NVIDIA Drivers & Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" nvidia-driver-535 nvidia-container-toolkit || true
nvidia-ctk runtime configure --runtime=docker || true
systemctl restart docker || true

# 3. Install Nextflow
curl -s https://get.nextflow.io | bash
mv nextflow /usr/local/bin/
chmod +x /usr/local/bin/nextflow

# 4. Install K3s with Docker support
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker" sh - || true
chmod 644 /etc/rancher/k3s/k3s.yaml || true
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
echo "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml" >> /root/.bashrc
echo "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml" >> /home/*/.bashrc || true

echo "ColabBio GPU VM Provisioning Complete!"
EOF_STARTUP
)

# Launch Spot Instance in us-central1
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
    --metadata="startup-script=${STARTUP_SCRIPT}" \
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
