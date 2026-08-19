#!/usr/bin/env python3
"""
download-sample-tcga-slides.py

Utility for streaming/downloading a small set of open-access Whole Slide Images (WSI)
from the NCI Genomic Data Commons (GDC) public API or Google Cloud Storage.
"""

import os
import sys
import argparse
import urllib.request
import json

SAMPLE_TCGA_SLIDES = [
    {
        "patient_id": "TCGA-AA-3660",
        "file_id": "97e682d3-2e0f-4fa1-840a-5b12da61cf2c",
        "filename": "TCGA-AA-3660-01A-01-TS1.08ddba85-f0e7-4950-ba5d-e26ee8bc78b7.svs",
        "cancer_type": "COAD",
        "label": "Colorectal Adenocarcinoma"
    },
    {
        "patient_id": "TCGA-3L-AA1B",
        "file_id": "3b4f6b4a-3fb9-4f7f-8d26-7bc28c894ce5",
        "filename": "TCGA-3L-AA1B-01A-01-TS1.10986799-8d48-43d9-95e3-4d436cfdc5f2.svs",
        "cancer_type": "COAD",
        "label": "Colorectal Adenocarcinoma"
    },
    {
        "patient_id": "TCGA-A7-A13E",
        "file_id": "76df626c-9c76-4d1e-8e43-cbfe99ec9871",
        "filename": "TCGA-A7-A13E-01A-01-TS1.78e6da58-d98c-4a39-9556-9d33b3780308.svs",
        "cancer_type": "BRCA",
        "label": "Breast Invasive Carcinoma"
    }
]

def download_file(url: str, target_path: str):
    print(f"--> Streaming from {url} to {target_path}...")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progress: {percent}% ({mb:.1f} MB / {total_mb:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print("\n    [DONE]")

def main():
    parser = argparse.ArgumentParser(description="Download sample TCGA WSI slides for benchmarking")
    parser.add_argument("--output_dir", "-o", type=str, default="/data/vfs/images/svs_raw", help="Target folder for SVS slides")
    parser.add_argument("--count", "-n", type=int, default=1, help="Number of sample slides to download (1-3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    slides_to_fetch = SAMPLE_TCGA_SLIDES[:max(1, min(len(SAMPLE_TCGA_SLIDES), args.count))]

    manifest = []
    for s in slides_to_fetch:
        target = os.path.join(args.output_dir, s["filename"])
        if os.path.exists(target) and os.path.getsize(target) > 1000000:
            print(f"Slide {s['filename']} already exists ({os.path.getsize(target)/(1024*1024):.1f} MB), skipping download.")
        else:
            url = f"https://api.gdc.cancer.gov/data/{s['file_id']}"
            try:
                download_file(url, target)
            except Exception as e:
                print(f"Warning: Could not stream from GDC API: {e}")
                print("Tip: If running on GCP with gsutil/MedVFS, point directly to gs://gdc-tcga-phs000178-open/")
        manifest.append(s)

    with open(os.path.join(args.output_dir, "tcga_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Successfully processed {len(manifest)} TCGA slide definitions in {args.output_dir}.")

if __name__ == "__main__":
    main()
