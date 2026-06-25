"""
Download the SmolLM2-tokenized Dolma blend from HuggingFace.
3 domains: code, math, literature. Vocab size: 49,152. BOS token: 0.

Usage:
  python data/cached_dolma_blend_smol.py
"""

import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download

REPO_ID = "varunneal/dolma-blend-smollm2"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "dolma_blend_smol")


def main():
    print("Downloading val shards first...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=["dolma_val_*.bin"],
        max_workers=16,
    )
    print("Val shards done. Downloading train shards...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=["dolma_train_*.bin"],
        max_workers=16,
    )
    print(f"Done! Files in {LOCAL_DIR}")


if __name__ == "__main__":
    main()
