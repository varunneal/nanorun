"""Download GPT-2 tokens of FineWeb-10B from HuggingFace.

Uses snapshot_download with high worker count for maximum throughput.
"""
import os
import sys

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download

LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'fineweb10B')
REPO_ID = "kjj0/fineweb10B-gpt2"

num_chunks = 103
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])

patterns = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, num_chunks + 1)]

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=patterns,
    max_workers=16,
)
