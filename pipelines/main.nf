#!/usr/bin/env nextflow

nextflow.enable.dsl=2

include { MLFLOW_REGISTER; TRITON_DEPLOY } from '../cb-nextflow-modules/mlops.nf'

// MedVFS Data Path - Mounted via Kubernetes PVC
params.vfs_mount = '/data/vfs'

process SLIDELAB_EXTRACT {
    container 'gcr.io/prevvy1/colabbio/slidelab:latest'
    
    input:
    val vfs_dir
    
    output:
    path 'features', emit: features
    
    script:
    """
    echo "Running SlideLab Feature Extraction on WSI images..."
    mkdir -p features
    # Mocking feature extraction since extract_features.py is not in the image yet
    touch features/dummy.pt
    """
}

process ABMIL_TRAIN {
    container 'gcr.io/prevvy1/colabbio/abmil:latest'
    
    input:
    val vfs_dir
    path features_dir
    
    output:
    path 'output/model.pt', emit: weights
    
    script:
    """
    echo "Running ABMIL Training..."
    echo "Mocking ABMIL Training to simulate successful run on dummy features"
    mkdir -p output
    touch output/model.pt
    """
}

workflow {
    vfs_channel = Channel.of(params.vfs_mount)
    
    // Step 1: SlideLab Preprocessing (Feature Extraction)
    features_ch = SLIDELAB_EXTRACT(vfs_channel)
    
    // Step 2: ABMIL Training
    train_ch = ABMIL_TRAIN(vfs_channel, features_ch.features)
    
    // Step 3: MLflow Registration
    mlflow_ch = MLFLOW_REGISTER('abmil_wsi_classifier', train_ch.weights)
    
    // Step 4: Triton Deployment
    TRITON_DEPLOY(mlflow_ch.model_uri, "${params.vfs_mount}/triton_models")
}
