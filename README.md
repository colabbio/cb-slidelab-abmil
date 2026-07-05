# ColabBio Tenant Pilot: SlideLab + ABMIL

Este repositorio contiene la implementación piloto de un Tenant (Data Scientist) utilizando la arquitectura Data Mesh de ColabBio basado en **Nextflow** sobre **Kubernetes**.

## Pipeline Principal (`main.nf`)
El pipeline consta de 2 fases Core (propiedad del tenant) y 2 fases MLOps (abstraídas por la plataforma):
1. `SLIDELAB_EXTRACT`: Consume imágenes WSI desde MedVFS y extrae características visuales (Features).
2. `ABMIL_TRAIN`: Entrena un clasificador Multiple Instance Learning usando PyTorch sobre las características previas.
3. `MLFLOW_REGISTER` (Plataforma): Registra automáticamente los pesos y métricas en el Model Registry central.
4. `TRITON_DEPLOY` (Plataforma): Despliega el modelo de inferencia en tiempo real en Triton Inference Server.

## Integración con ColabBio MLOps
La gran ventaja de este diseño es que el código de entrenamiento del Tenant permanece completamente aislado de la complejidad de la infraestructura. El MLOps se incluye automáticamente desde el repositorio central:

```groovy
include { MLFLOW_REGISTER; TRITON_DEPLOY } from 'github.com/colabbio/cb-nextflow-modules/mlops.nf'
```

Para lanzar la ejecución en el cluster de ColabBio:
```bash
kubectl apply -f pipelines/nextflow-runner-job.yaml
```
