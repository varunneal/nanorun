"""
Download the GPT-2 tokenized Dolma blend from HuggingFace.
4 domains: code, math, literature, news.

Usage:
  python data/cached_dolma_blend.py
"""

import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download

REPO_ID = "varunneal/dolma-blend-gpt2"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "dolma_blend")


def main():
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=["*.bin"],
        max_workers=16,
    )
    print(f"Done! Files in {LOCAL_DIR}")


if __name__ == "__main__":
    main()
