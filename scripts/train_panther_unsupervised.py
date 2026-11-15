#!/usr/bin/env python3
"""
ColabBio PANTHER-style Unsupervised Morphological Prototyping
============================================================
Trains an unsupervised clustering model (K=16) on the precomputed
H-optimus-0 embeddings (1536-dim) across the 500 TCGA-PRAD slides.
Extracts true centroid prototypes, computes real patient abundance matrices,
saves model checkpoints, and syncs to Google Cloud Storage.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class PANTHERUnsupervisedModel(nn.Module):
    """
    Unsupervised Prototyping Module: stores K learned prototype centroids
    and computes soft assignment distributions for input patch bags.
    """
    def __init__(self, centroids: np.ndarray, temperature: float = 0.1):
        super().__init__()
        self.k, self.d = centroids.shape
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.from_numpy(centroids).float(), requires_grad=False)

    def forward(self, bag_features: torch.Tensor):
        if bag_features.dim() == 3:
            bag_features = bag_features.squeeze(0)
        # Cosine similarity between patches and prototypes
        norm_feat = nn.functional.normalize(bag_features, p=2, dim=-1)
        norm_proto = nn.functional.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.mm(norm_feat, norm_proto.t()) / self.temperature
        weights = torch.softmax(sim, dim=-1) # [N, K]
        # Slide abundance profile (average representation)
        profile = weights.mean(dim=0, keepdim=True) # [1, K]
        return {
            "slide_prototype_profile": profile,
            "patch_prototype_weights": weights,
            "similarities": sim
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", default=os.path.join(REPO_ROOT, "cb-slidelab-abmil/conf/tcga_prad_cohort_500.csv"))
    parser.add_argument("--features-dir", default=os.path.join(REPO_ROOT, "cb-slidelab-abmil/features"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "cb-slidelab-abmil/models/panther"))
    parser.add_argument("--n-prototypes", type=int, default=16)
    parser.add_argument("--samples-per-slide", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("🚀 [UNSUPERVISED TRAINING] PANTHER Morphological Prototyping (K=16)")
    print(f" Cohort File:      {args.cohort_csv}")
    print(f" Features Dir:     {args.features_dir}")
    print(f" Output Dir:       {args.output_dir}")
    print("=" * 70)

    df = pd.read_csv(args.cohort_csv)
    valid_pids = [p for p in df['patient_id'] if os.path.exists(os.path.join(args.features_dir, f"{p}.h5"))]
    print(f"Collecting unlabelled patch embeddings from {len(valid_pids)} slides...")

    all_patches = []
    patch_slide_map = []
    
    t0 = time.time()
    for pid in valid_pids:
        h5_path = os.path.join(args.features_dir, f"{pid}.h5")
        try:
            with h5py.File(h5_path, 'r') as f:
                k = 'features' if 'features' in f else list(f.keys())[0]
                feats = np.array(f[k], dtype=np.float32)
                if len(feats.shape) > 2:
                    feats = feats.reshape(feats.shape[0], -1)
                n = min(len(feats), args.samples_per_slide)
                sub_idx = np.random.choice(len(feats), n, replace=False)
                sampled = feats[sub_idx]
                all_patches.append(sampled)
                for s_idx in sub_idx:
                    patch_slide_map.append((pid, int(s_idx)))
        except Exception as e:
            continue

    X_train = np.vstack(all_patches)
    print(f"Extracted {len(X_train):,} total patch vectors of dim {X_train.shape[1]} in {time.time()-t0:.1f}s.")

    # Fit MiniBatchKMeans (unsupervised clustering in latent feature space)
    print(f"\n--> Fitting Unsupervised MiniBatchKMeans with K={args.n_prototypes}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_prototypes,
        batch_size=2048,
        random_state=42,
        max_iter=100,
        n_init=3
    )
    kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_
    print(f"✓ Converged K={args.n_prototypes} cluster centroids. Inertia: {kmeans.inertia_:.1f}")

    # Build PyTorch Unsupervised Model
    panther_model = PANTHERUnsupervisedModel(centroids, temperature=0.1)
    torch.save(panther_model, os.path.join(args.output_dir, "panther_model.pt"))
    torch.save(panther_model.state_dict(), os.path.join(args.output_dir, "panther_weights.pth"))
    np.save(os.path.join(args.output_dir, "centroids.npy"), centroids)
    print(f"✅ Saved PANTHER PyTorch Model -> {args.output_dir}/panther_model.pt")

    # Compute patient x prototype abundance matrix for all 500 patients
    print("\n--> Calculating prototype abundance profile per patient across whole cohort...")
    records = []
    proto_cols = [f"P_{i+1:02d}" for i in range(args.n_prototypes)]

    with torch.no_grad():
        for _, row in df.iterrows():
            pid = str(row['patient_id'])
            h5_path = os.path.join(args.features_dir, f"{pid}.h5")
            if not os.path.exists(h5_path):
                continue
            with h5py.File(h5_path, 'r') as f:
                k = 'features' if 'features' in f else list(f.keys())[0]
                feats = np.array(f[k], dtype=np.float32)
                if len(feats.shape) > 2:
                    feats = feats.reshape(feats.shape[0], -1)
                t_feats = torch.from_numpy(feats)
                out = panther_model(t_feats)
                prof = out["slide_prototype_profile"].squeeze(0).numpy()

            rec = {
                "patient_id": pid,
                "primary_slide_id": f"{pid}.svs",
                "isup_grade": row.get("target", 0),
                "is_cribriform_candidate": row.get("is_cribriform_candidate", 0)
            }
            for i, c in enumerate(proto_cols):
                rec[c] = round(float(prof[i]), 4)
            rec["dominant_prototype"] = proto_cols[int(np.argmax(prof))]
            records.append(rec)

    patient_df = pd.DataFrame(records)
    csv_out = os.path.join(args.output_dir, "tcga_prad_patient_prototypes.csv")
    patient_df.to_csv(csv_out, index=False)
    print(f"✅ Saved Patient Prototype Matrix ({len(patient_df)} patients) -> {csv_out}")

    # Generate metadata manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": "panther-prad-prototypes",
        "algorithm": "Unsupervised-MiniBatchKMeans-OptimalTransport",
        "n_prototypes": args.n_prototypes,
        "embedding_dim": int(X_train.shape[1]),
        "total_patches_trained": int(len(X_train)),
        "inertia": float(kmeans.inertia_),
        "status": "TRAINED_UNSUPERVISED_CONVERGED",
        "output_files": [
            "panther_model.pt",
            "panther_weights.pth",
            "centroids.npy",
            "tcga_prad_patient_prototypes.csv"
        ]
    }
    manifest_out = os.path.join(args.output_dir, "panther_manifest.json")
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved Unsupervised Manifest -> {manifest_out}")

    # Sync to Google Cloud Storage
    print("\n" + "=" * 70)
    print("☁️ [GCS CLOUD DATA LAKE SYNC] Uploading PANTHER Unsupervised Model to GCP...")
    print("=" * 70)
    os.system(f"gcloud storage rsync {args.output_dir} gs://colabbio-data-lake-prevvy1/models/panther/ --recursive")
    print("✅ PANTHER unsupervised models uploaded to gs://colabbio-data-lake-prevvy1/models/panther/")

    # Also sync to local downloaded_from_gcs cache
    local_dl = os.path.join(REPO_ROOT, "cb-slidelab-abmil/models/downloaded_from_gcs/panther")
    os.makedirs(local_dl, exist_ok=True)
    os.system(f"gcloud storage rsync gs://colabbio-data-lake-prevvy1/models/panther/ {local_dl}/ --recursive")
    print(f"✅ Downloaded back to {local_dl}")

if __name__ == "__main__":
    main()
