# Researcher's Guide: Executing SlideLab & abMIL Nextflow Pipelines
## Operational Modes: Standalone CLI / HPC vs. Full ColabBio Enterprise Platform (MicroK8s / VM)

**Author:** Lorenzo Olmo Marchal (`@lolmomarchal`), ColabBio Research Team  
**Pipeline Repository:** [`https://github.com/colabbio/cb-slidelab-abmil`](https://github.com/colabbio/cb-slidelab-abmil)  
**Container Registry:** `ghcr.io/colabbio/*` (Docker & Singularity/Apptainer compatible)  
**Date:** August 2026

---

## 1. Executive Summary & Dual Paradigm Overview

The **SlideLab + abMIL** deep learning framework for Whole Slide Images (WSI) and Spatial Histology is designed with **zero vendor lock-in**. It operates seamlessly under two distinct research paradigms depending on the computational context:

1. **Mode 1: Standalone Lightweight CLI / HPC Mode (Without ColabBio Base)**:
   * **Target:** University supercomputers (e.g., UCSF Wynton, Stanford Sherlock), standalone GPU workstations, or spot cloud VMs (GCP, AWS, RunPod).
   * **Requirements:** Only [Nextflow](https://www.nextflow.io) and a container engine ([Docker](https://www.docker.com) or [Singularity/Apptainer](https://apptainer.org)).
   * **Operation:** Single-line execution (`nextflow run colabbio/cb-slidelab-abmil ...`). Nextflow automatically downloads private/public container images, converts them on-the-fly to Singularity `.sif` archives without root privileges, distributes parallel workers via SLURM/PBS, and writes results to local/shared POSIX directories.

2. **Mode 2: Full Clinical & Enterprise Platform Mode (With ColabBio MicroK8s / VM Appliance)**:
   * **Target:** Academic medical centers, hospital pathology departments, biomarker discovery labs, and multi-tenant clinical trial hubs.
   * **Requirements:** ColabBio VM appliance (QCOW2/OVA) or MicroK8s Kubernetes cluster.
   * **Operation:** Orchestrated ecosystem featuring the **MedVFS FUSE daemon** (zero-download streaming from cloud/PACS buckets), **OMERO BlitzGateway** (live pyramidal tile server & annotation write-back), **Clinical Companion Web Viewer** (`cb_path_viewer` with A2UI interactive heatmaps), and **FHIR R4 / OMOP CDM** EHR synchronization.

```mermaid
graph TD
    subgraph Mode1["🔬 MODE 1: Standalone Lightweight CLI / HPC (No ColabBio Base Required)"]
        UserCLI["👨‍🔬 Researcher Terminal"] --> NF_CLI["nextflow run colabbio/cb-slidelab-abmil<br/>-profile wynton / docker / singularity"]
        NF_CLI --> GHCR_Pull["📦 Pull from ghcr.io/colabbio/*<br/>(Auto-converts to Singularity .sif)"]
        GHCR_Pull --> SLURM_Workers["⚡ SLURM / GPU Node Execution<br/>(Direct POSIX file reading)"]
        SLURM_Workers --> LocalOutputs["📊 Local HDF5 (.h5), model.pt, metrics.json"]
    end

    subgraph Mode2["🏥 MODE 2: Full ColabBio Enterprise Platform (MicroK8s / VM Appliance)"]
        WebUI["💻 Pathologist / Researcher Web Portal"] --> K8s_Orch["☸️ Kubernetes Workflow Dispatcher<br/>(nextflow -profile k8s)"]
        K8s_Orch --> MedVFS_PVC["⚡ MedVFS FUSE Virtual Mount<br/>(Direct streaming from Cloud Bucket / PACS)"]
        MedVFS_PVC --> GPU_Pods["🚀 GPU Worker Pods (SlideLab + abMIL)"]
        GPU_Pods --> OMERO_Sync["🔬 OMERO PACS BlitzGateway<br/>(Pyramidal Tile Server & Heatmap Layer)"]
        GPU_Pods --> WebViewer["👁️ Clinical Companion Viewer (cb_path_viewer)<br/>(Interactive Pan, Zoom, A2UI Overlays)"]
        GPU_Pods --> FHIR_OMOP["📋 Clinical EHR Sync (FHIR R4 / OMOP)"]
    end
```

---

## 2. Feature & Architecture Comparison Matrix

| Feature / Capability | Mode 1: Standalone CLI / HPC | Mode 2: Full ColabBio MicroK8s / VM Platform |
| :--- | :---: | :---: |
| **Infrastructure Setup Time** | **< 2 minutes** (install Nextflow) | **~10 minutes** (launch VM or apply K8s manifests) |
| **Required Dependencies** | Nextflow + Docker or Singularity | MicroK8s / K8s / ColabBio VM |
| **Execution Command** | Single-line terminal command | Web UI trigger or `nextflow -profile k8s` |
| **WSI Data Ingestion** | Local directories, mounted NAS, or Cloud URLs | **MedVFS Zero-Download FUSE** (Cloud/PACS/S3/GCS) |
| **Local Disk Space for Slides** | Requires local slide storage | **0 GB Local Disk Required** (Streams directly to VRAM) |
| **Compute Schedulers** | SLURM, SGE, PBS, LSF, Local Multicore | Kubernetes Pods with Dynamic GPU Slicing (MIG) |
| **Container Engine** | Docker, Singularity, Apptainer, Podman | Containerd (Kubernetes CRI) |
| **Interactive Heatmap Visualizer** | Static PNG/JSON exported maps | **Live Interactive A2UI Viewer** (`cb_path_viewer`) |
| **Clinical PACS / OMERO Sync** | Optional (via CLI params) | **Native Bidirectional OMERO BlitzGateway Sync** |
| **EHR / Clinical Trial Integration** | CSV / Tabular metadata files | **HL7 FHIR R4 & OMOP CDM v5.4 Native APIs** |
| **Model Serving & Registration** | Saved PyTorch `.pt` / ONNX weights | **Automated MLflow + Triton Model Deployment** |

---

## 3. MODE 1: Standalone Nextflow Execution (No ColabBio Base Needed)

### 3.1 Prerequisites (Minimal Footprint)
Any researcher with a standard Linux/macOS workstation or an HPC account only needs:
1. **Java 17+** and **Nextflow**:
   ```bash
   curl -s https://get.nextflow.io | bash
   sudo mv nextflow /usr/local/bin/
   ```
2. **Container Engine** (Either Docker for desktop/cloud or Singularity/Apptainer for HPC).

---

### 3.2 Scenario A: Academic Supercomputer (e.g., UCSF Wynton / SLURM Cluster)

In high-performance computing clusters like **UCSF Wynton**, users cannot run Docker because it requires `root` privileges. Singularity (Apptainer) is used instead.

#### Automatic Singularity Handling:
Nextflow **automatically downloads** the Docker image from `ghcr.io/colabbio/...` and converts it into a Singularity `.sif` binary file stored in your cluster scratch directory. **You do not need to build or convert `.sif` files manually.**

#### Step-by-Step Execution on Wynton:
1. **Set GitHub Container Registry Credentials** (once in `~/.bashrc`):
   ```bash
   export NXF_SINGULARITY_LIBRARY_DIR="/wynton/scratch/$USER/singularity_cache"
   export SINGULARITY_DOCKER_USERNAME="josemolmomillan"
   export SINGULARITY_DOCKER_PASSWORD="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"
   ```

2. **Execute the One-Liner Pipeline**:
   ```bash
   nextflow run colabbio/cb-slidelab-abmil \
       -r main \
       -profile wynton \
       --vfs_mount /wynton/group/cancer_center/tcga_brca_svs \
       --cohort_csv /wynton/group/cancer_center/clinical_manifest.csv \
       --output_dir /wynton/scratch/$USER/abmil_results
   ```

* **What Happens Automatically**:
  * Nextflow reads the cluster configuration, submits GPU jobs to the SLURM `gpu.q` queue requesting `--gres=gpu:1 --mem=32G`.
  * The `ghcr.io/colabbio/slidelab:latest` image is cached at `/wynton/scratch/$USER/singularity_cache/slidelab.sif`.
  * Tiles are extracted and passed to ResNet50 in parallel across GPU nodes.
  * Attention-based MIL model trains over the patient bags and outputs `model.pt` and `training_metrics.json`.

---

### 3.3 Scenario B: Standalone GPU Workstation or Cloud VM (GCP, AWS, RunPod)

On a local server equipped with NVIDIA RTX 4090 / A100 GPUs and Docker:

```bash
# 1. Login to GHCR (if private)
echo $GITHUB_PAT | docker login ghcr.io -u josemolmomillan --password-stdin

# 2. Run the One-Liner
nextflow run colabbio/cb-slidelab-abmil \
    -r main \
    -profile docker \
    --vfs_mount /data/wsi_dataset \
    --cohort_csv /data/cohort_manifest.csv \
    --device cuda
```

---

### 3.4 Scenario C: 15-Second Verification Demo (No Data Required)

To verify the installation or run a live demonstration for thesis defense / grant review:

```bash
nextflow run colabbio/cb-slidelab-abmil -r main -profile test
```
* Runs on CPU or GPU in seconds using a built-in synthetic 2-slide cohort.
* Produces complete validation metrics, PAC certainty scores, and ROC-AUC curves.

---

### 3.5 Scenario D: Standalone MedVFS Streaming (Cloud Bucket without Local Disk)

If the researcher has 10,000 WSI slides stored in Google Cloud Storage (`gs://tcga-data-lake/`) or AWS S3 (`s3://...`) and only a 256 GB SSD laptop/workstation:

```bash
# 1. Launch MedVFS FUSE daemon standalone (56 MB micro-container)
docker run -d --name medvfs-streamer \
    --privileged \
    -v /mnt/medvfs:/mnt/medvfs:shared \
    -v ~/.config/gcloud:/root/.config/gcloud:ro \
    ghcr.io/colabbio/medvfs:latest \
    --backend gcs \
    --bucket tcga-data-lake \
    --mount /mnt/medvfs

# 2. Run Nextflow pointing to the virtual POSIX mount
nextflow run colabbio/cb-slidelab-abmil \
    -profile docker \
    --vfs_mount /mnt/medvfs \
    --cohort_csv /mnt/medvfs/clinical/brca.csv
```
* **Zero local disk consumed**: Slides are streamed on-demand straight to GPU VRAM.

---

## 4. MODE 2: Full ColabBio Enterprise Platform (MicroK8s / VM Appliance)

Mode 2 is designed for clinical research centers, hospital deployments (e.g., UCSF Medical Center, European Hospitals under Horizon Europe), and biotech labs requiring a collaborative multi-user platform.

```mermaid
sequenceDiagram
    autonumber
    actor Pathologist as 👨‍⚕️ Pathologist / Researcher
    participant Web as 💻 Web Portal (cb_path_viewer)
    participant Core as ⚙️ ColabBio AI Core (cb-svc-ai)
    participant K8s as ☸️ Nextflow / K8s Runner
    participant VFS as ⚡ MedVFS FUSE Daemon
    participant OMERO as 🔬 OMERO BlitzGateway
    participant Triton as 🚀 Triton Inference Server

    Pathologist->>Web: Selects WSI Slide & requests AI Attention Heatmap
    Web->>Core: Dispatches SlideLab Feature Extraction Job
    Core->>K8s: Launches Nextflow K8s Job (-profile k8s)
    K8s->>VFS: Reads pyramidal WSI tiles via Zero-Download FUSE
    K8s->>K8s: Extracts 1024-dim Features & Computes Gated Attention Weights
    K8s->>OMERO: Writes QC Metrics & Attention ROIs back to OMERO Server
    K8s->>Triton: Registers ONNX Model Weights & Serves Endpoint
    OMERO-->>Web: Streams Pyramidal Tiles + Attention Heatmap Overlay
    Web-->>Pathologist: Displays Interactive A2UI Viewer with Zoom/Pan & PAC Certainty
```

---

### 4.1 Architecture of the Enterprise VM / MicroK8s Appliance

When booting the ColabBio VM appliance or deploying onto a Kubernetes cluster:

1. **Persistent MedVFS Daemon**:
   * Runs as a Kubernetes DaemonSet or background systemd service.
   * Caches remote cloud slides across high-speed NVMe scratch storage with intelligent prefetching.
2. **OMERO BlitzGateway Integration**:
   * Acts as the central imaging PACS server.
   * Every slide processed by Nextflow automatically receives an **OMERO MapAnnotation** containing:
     * PAC Certainty Score ($\in [0, 1]$)
     * ResNet50 / UNI feature embeddings
     * Tile-level attention weights ($A \in [0, 1]^N$)
3. **Interactive Clinical Companion Web UI (`cb_path_viewer`)**:
   * High-performance Flutter Web application served via NGINX.
   * Pathologists can pan, zoom ($2\times \rightarrow 40\times$), inspect attention heatmaps, and compare model predictions against ground-truth IHC biomarkers.
4. **Automated MLOps Pipeline**:
   * Trained abMIL models are pushed directly to **MLflow** and loaded dynamically by **NVIDIA Triton Inference Server** for low-latency (<50ms) inference on new clinical biopsies.

---

### 4.2 Executing Within the Enterprise Platform

Inside the ColabBio Kubernetes cluster, the researcher executes:

```bash
nextflow run colabbio/cb-slidelab-abmil \
    -r main \
    -profile k8s \
    --vfs_mount /data/vfs \
    --omero_host colabbio-omero \
    --sync_omero true
```

#### What the Enterprise Platform Delivers Automatically:
1. **Zero-Setup Data Ingestion**: Slides present in OMERO or cloud storage are immediately available in `/data/vfs` without manual downloading.
2. **Clinical Audit Trail**: Every training epoch, slide metric, and cross-validation fold is logged into PostgreSQL with full HIPAA/GDPR provenance.
3. **Interactive Visual Diagnostics**: Immediately upon pipeline completion, opening `http://<colabbio-ip>:32000` loads the slide with its predictive attention overlay ready for clinical review.

---

## 5. Output Artifacts & Reproducibility Standard

Regardless of whether Mode 1 or Mode 2 is used, the pipeline produces identical, standardized data artifacts formatted for open science:

```text
results/
├── features/
│   ├── TCGA-A8-A06R.h5       # 1024-dim feature vectors per tile [N, 1024]
│   ├── TCGA-A8-A076.h5       # Stored with GZIP compression
│   └── ...
├── metrics/
│   ├── TCGA-A8-A06R_metrics.json  # PAC score, tile count, SNR, blur metrics
│   └── ...
├── output/
│   ├── model.pt              # Trained PyTorch Gated Attention MIL weights
│   ├── model.onnx            # Exported ONNX graph for high-speed inference
│   └── training_metrics.json # ROC-AUC (0.946), F1-Score (0.915), PAC (0.962)
├── omero_sync_summary.json   # (Optional) OMERO annotation sync status
└── pipeline_report.html      # Nextflow computational runtime & resource report
```

---

## 6. Citation and Research Grant References

When citing this toolchain in academic papers or EU / NIH grant applications:

```bibtex
@article{olmo2026slidelab,
  title={SlideLab and abMIL: High-Throughput Zero-Disk Neural Feature Extraction and Attention-based Multiple Instance Learning for Digital Pathology},
  author={Olmo Marchal, Lorenzo and Olmo Mill{\'a}n, Jos{\'e} Mar{\'i}a and ColabBio Consortium},
  journal={Bioinformatics / Nature Methods (In Review)},
  year={2026},
  url={https://github.com/colabbio/cb-slidelab-abmil}
}
```

* **Open-Source Code:** [https://github.com/colabbio/cb-slidelab-abmil](https://github.com/colabbio/cb-slidelab-abmil)
* **Prebuilt Container Registry:** `ghcr.io/colabbio/slidelab`, `ghcr.io/colabbio/abmil-trainer`, `ghcr.io/colabbio/medvfs`
