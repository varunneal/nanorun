"""
Download, tokenize, and shard a blended dataset from Dolma subsets.

Produces .bin files in the same format as fineweb10B:
  - 256 int32 header: magic=20240520, version=1, num_tokens=N, rest zeros
  - N uint16 token values (GPT-2 tokenizer)

Domains: code (starcoder), math (algebraic-stack), literature (books/gutenberg), news (cc_news)

Usage:
  uv run python data/prepare_dolma_blend.py [--tokens-per-domain 250_000_000] [--upload]
"""

import argparse
import gzip
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import tiktoken
import requests

DOMAINS = {
    "code": "https://olmo-data.org/dolma-v1_7/starcoder/starcoder-0000.json.gz",
    "math": "https://olmo-data.org/dolma-v1_7/proof_pile_2-algebraic_stack/algebraic-stack-train-0000.json.gz",
    "literature": "https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz",
    "news": "https://olmo-data.org/dolma-v1_7/cc_news_head/cc_news-0000.json.gz",
}

SHARD_SIZE = 100_000_000  # 100M tokens per shard (same as fineweb)
BOS_TOKEN = 50256
MAGIC = 20240520
VERSION = 1


def download_gz(url: str, dest: Path):
    """Download a .json.gz file with progress."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)", end="", flush=True)
    print()


def tokenize_jsonl_gz(path: Path, max_tokens: int, enc) -> np.ndarray:
    """Tokenize a gzipped JSONL file, prepending BOS to each document."""
    tokens = []
    total = 0
    docs = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if total >= max_tokens:
                break
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            doc_tokens = enc.encode_ordinary(text)
            tokens.append(BOS_TOKEN)
            tokens.extend(doc_tokens)
            total += len(doc_tokens) + 1
            docs += 1
            if docs % 10000 == 0:
                print(f"\r  Tokenized {docs} docs, {total / 1e6:.1f}M tokens", end="", flush=True)
    print(f"\r  Tokenized {docs} docs, {total / 1e6:.1f}M tokens (done)")
    return np.array(tokens[:max_tokens], dtype=np.uint16)


def write_shard(tokens: np.ndarray, path: Path):
    """Write a .bin shard with the standard header."""
    num_tokens = len(tokens)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = num_tokens
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    print(f"  Wrote {path.name}: {num_tokens / 1e6:.1f}M tokens")


def shard_tokens(tokens: np.ndarray, domain: str, split: str, out_dir: Path):
    """Split a token array into shards."""
    paths = []
    for i in range(0, len(tokens), SHARD_SIZE):
        shard = tokens[i : i + SHARD_SIZE]
        fname = f"dolma_{split}_{domain}_{i // SHARD_SIZE:04d}.bin"
        path = out_dir / fname
        write_shard(shard, path)
        paths.append(path)
    return paths


def compute_stats(tokens: np.ndarray, domain: str, enc):
    """Compute sequence length statistics."""
    bos_positions = np.where(tokens == BOS_TOKEN)[0]
    if len(bos_positions) < 2:
        return
    lengths = np.diff(bos_positions)
    print(f"  {domain}: {len(lengths)} sequences")
    print(f"    median: {int(np.median(lengths))}, mean: {int(np.mean(lengths))}")
    print(f"    p25: {int(np.percentile(lengths, 25))}, p75: {int(np.percentile(lengths, 75))}")
    print(f"    p95: {int(np.percentile(lengths, 95))}, max: {int(np.max(lengths))}")
    return {
        "domain": domain,
        "num_sequences": int(len(lengths)),
        "total_tokens": int(len(tokens)),
        "median_len": int(np.median(lengths)),
        "mean_len": int(np.mean(lengths)),
        "p25_len": int(np.percentile(lengths, 25)),
        "p75_len": int(np.percentile(lengths, 75)),
        "p95_len": int(np.percentile(lengths, 95)),
        "max_len": int(np.max(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-domain", type=int, default=250_000_000)
    parser.add_argument("--val-tokens-per-domain", type=int, default=5_000_000)
    parser.add_argument("--out-dir", type=str, default="data/dolma_blend")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo-id", type=str, default="varunneal/dolma-blend-gpt2")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats, don't shard")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(tempfile.gettempdir()) / "dolma_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    total_needed = args.tokens_per_domain + args.val_tokens_per_domain

    all_stats = []
    all_paths = []

    for domain, url in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        # Download
        gz_path = cache_dir / f"{domain}.json.gz"
        download_gz(url, gz_path)

        # Tokenize
        print(f"  Tokenizing (target: {total_needed / 1e6:.0f}M tokens)...")
        tokens = tokenize_jsonl_gz(gz_path, total_needed, enc)

        # Stats
        stats = compute_stats(tokens, domain, enc)
        if stats:
            all_stats.append(stats)

        if args.stats_only:
            continue

        # Split into val + train
        val_tokens = tokens[: args.val_tokens_per_domain]
        train_tokens = tokens[args.val_tokens_per_domain : args.val_tokens_per_domain + args.tokens_per_domain]

        print(f"  Val: {len(val_tokens) / 1e6:.1f}M tokens, Train: {len(train_tokens) / 1e6:.1f}M tokens")

        # Shard
        val_paths = shard_tokens(val_tokens, domain, "val", out_dir)
        train_paths = shard_tokens(train_tokens, domain, "train", out_dir)
        all_paths.extend(val_paths + train_paths)

    # Print summary
    if all_stats:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for s in all_stats:
            print(f"  {s['domain']:12s}: {s['num_sequences']:>8} seqs, "
                  f"median={s['median_len']:>5}, mean={s['mean_len']:>5}, "
                  f"p95={s['p95_len']:>6}, max={s['max_len']:>7}")

    # Upload
    if args.upload and all_paths:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        print(f"\nUploading {len(all_paths)} files to {args.repo_id}...")
        for path in all_paths:
            print(f"  Uploading {path.name}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        print("Done!")


if __name__ == "__main__":
    main()
