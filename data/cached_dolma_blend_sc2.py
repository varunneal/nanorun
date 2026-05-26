"""
Download the StarCoder2-tokenized Dolma blend from HuggingFace.
4 domains: code, math, literature, news. Vocab size: 49,152.

Usage:
  python data/cached_dolma_blend_sc2.py
"""

import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "varunneal/dolma-blend-starcoder2"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "dolma_blend_sc2")


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    files = list_repo_files(REPO_ID, repo_type="dataset")
    bin_files = [f for f in files if f.endswith(".bin")]
    print(f"Found {len(bin_files)} files to download")
    for i, fname in enumerate(sorted(bin_files)):
        dest = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(dest):
            print(f"  [{i+1}/{len(bin_files)}] Already exists: {fname}")
            continue
        print(f"  [{i+1}/{len(bin_files)}] Downloading: {fname}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
        )
    print(f"Done! Files in {LOCAL_DIR}")


if __name__ == "__main__":
    main()
