#!/bin/bash -ue
echo "Running SlideLab Feature Extraction on WSI images..."
mkdir -p features
# En un entorno real, descomentamos la ejecución de SlideLab
# python /app/extract_features.py --data_dir /data/vfs/wsi_data --output_dir features

# Mocking output for workflow testing
touch features/dummy.pt
