# Changelog

## [0.4.1] - 2026-11-15

### Added / Changed
- Consolidate platform WIP, hygiene updates (`.gitignore`), and multi-repo GitHub workflow alignment.

## [0.3.0] - 2026-08-19

### Added
- **Transatlantic Federated Learning Hub & Client**:
  - Implemented `colabbio_federated_hub.py` (port `:8085`) providing global model registry, round progression, and `FedAvg` weighted tensor aggregation.
  - Implemented `colabbio_hospital_trainer.py` supporting ABMIL feature extraction, local batch training directly over MedVFS FUSE points, and secure tensor gradient shipping ($\Delta W$).
- **Live Multi-VM Test Validation**:
  - Validated 2-round federated training across 3 isolated Multipass VMs (`slidelab-abmil-vm`, `hosp-edge-1`, `hosp-edge-2`) achieving 57.1% loss reduction ($0.7160 \to 0.3069$) and 100% accuracy with Zero-Data Egress.

## [0.2.0] - 2026-08-17

- **Study Orchestration**: Configured `vfs.yaml` and pipeline orchestrators for end-to-end SlideLab ABMIL pathology execution.
- **Nextflow Module Integration**: Linked shared DSL2 ingestion and MLOps deployment pipelines.

## [0.1.0] - 2026-07-05

- Initial release.
