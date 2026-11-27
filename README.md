# ColabBio SlideLab + ABMIL Pipeline

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

End-to-end **Nextflow DSL2** digital pathology pipeline combining **SlideLab** zero-disk Whole Slide Image (WSI) streaming over **MedVFS** (`cb-vfs`) with **Attention-Based Multiple Instance Learning (ABMIL)** and **Polarized Attentional Certainty (PAC)** scoring.

## Pipeline Stages (`main.nf`)

1. **`SLIDELAB_EXTRACT`**: Streams WSIs lazily from MedVFS POSIX mounts (`/data/vfs`), applies tissue segmentation, Laplacian blur & pen-marker QC, Macenko stain normalization, and encodes features into `features.h5`.
2. **`WRITE_OMERO_ANNOTATIONS`**: Persists QC metrics, classification probabilities, and PAC scores back to OMERO as `MapAnnotation` records.
3. **`ABMIL_TRAIN`**: Trains a PyTorch Attention-Based MIL classifier on extracted bag features.
4. **`MLFLOW_REGISTER`** *(ColabBio Platform)*: Registers model weights and metrics via [`cb-nextflow-modules`](https://github.com/colabbio/cb-nextflow-modules).
5. **`TRITON_DEPLOY`** *(ColabBio Platform)*: Deploys the trained model for inference on Triton Inference Server.

## Documentation & Thesis

- **[SlideLab Preprint (`slidelab_article.md`)](https://github.com/colabbio/SlideLab/blob/v2.0/docs/slidelab_article.md)** — 1,000-WSI TCGA-COAD benchmarks, ABMIL MSI vs. MSS results, and zero-disk MedVFS streaming comparison.
- **[Polarized Attentional Certainty (PAC) Preprint (`pac_article.md`)](https://github.com/colabbio/SlideLab/blob/v2.0/docs/pac_article.md)** — Interpretability metric across colorectal cancer WSIs and B-ALL flow cytometry.
- **[UC San Diego M.S. Thesis Defense Slides (PDF)](https://github.com/colabbio/SlideLab/blob/v2.0/docs/thesis_presentation.pdf)** · **[Open in Google Slides](https://docs.google.com/presentation/d/1AZUdJUrf6N98_nFGE76qFn8rHJB9XNI0SPVWnQrF70k/present)** — Lorenzo Olmo Marchal (2025).
- **[Researcher Guide](docs/RESEARCHER_GUIDE.md)** — Standalone HPC vs. ColabBio Kubernetes execution.

## Quick Run

```bash
# ColabBio Kubernetes Cluster
kubectl apply -f pipelines/nextflow-runner-job.yaml

# Local Multi-GPU Workstation (e.g. 4 NVIDIA GPUs)
nextflow run main.nf -profile local_multigpu --num_gpus 4
```

## License

Licensed under the [Apache License 2.0](LICENSE) — Copyright 2026 ColabBio.
