"""
Download, tokenize, and shard a blended dataset from Dolma subsets.
Uses SmolLM2 tokenizer (49,152 vocab, good code+prose compression).

Produces .bin files in the same format as fineweb10B:
  - 256 int32 header: magic=20240520, version=1, num_tokens=N, rest zeros
  - N uint16 token values (SmolLM2 tokenizer, vocab_size=49152)

Domains: code (starcoder), math (algebraic-stack), literature (books/gutenberg)

Usage:
  uv run --with transformers python data/prepare_dolma_blend_smol.py [--tokens-per-domain 2_500_000_000] [--upload]
"""

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

DOMAINS = {
    "code": {
        "base_url": "https://olmo-data.org/dolma-v1_7/starcoder/starcoder-{shard:04d}.json.gz",
        "max_shards": 100,
    },
    "math": {
        "base_url": "https://olmo-data.org/dolma-v1_7/proof_pile_2-algebraic_stack/algebraic-stack-train-{shard:04d}.json.gz",
        "max_shards": 100,
    },
    "literature": {
        "base_url": "https://olmo-data.org/dolma-v1_7/books/books-{shard:04d}.json.gz",
        "max_shards": 3,
    },
}

SHARD_SIZE = 100_000_000  # 100M tokens per shard
MAGIC = 20240520
VERSION = 1


def download_gz(url: str, dest: Path) -> bool:
    """Download a .json.gz file with progress. Returns False if 404."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return True
    print(f"  Downloading {url}...")
    resp = requests.get(url, stream=True)
    if resp.status_code == 404:
        print(f"  Not found (404), no more shards available")
        return False
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
    return True


def tokenize_jsonl_gz(path: Path, max_tokens: int, tokenizer, bos_token: int, existing_count: int = 0):
    """Tokenize a gzipped JSONL file, prepending BOS to each document."""
    tokens = []
    total = existing_count
    docs = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if total >= max_tokens:
                break
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            doc_tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(bos_token)
            tokens.extend(doc_tokens)
            total += len(doc_tokens) + 1
            docs += 1
            if docs % 10000 == 0:
                print(f"\r  Tokenized {docs} docs, {total / 1e6:.1f}M tokens", end="", flush=True)
    print(f"\r  Tokenized {docs} docs from this file, {total / 1e6:.1f}M tokens cumulative")
    remaining = max_tokens - existing_count
    return np.array(tokens[:remaining], dtype=np.uint16)


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


def compute_stats(tokens: np.ndarray, domain: str, bos_token: int):
    """Compute sequence length statistics."""
    bos_positions = np.where(tokens == bos_token)[0]
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
    parser.add_argument("--tokens-per-domain", type=int, default=2_500_000_000)
    parser.add_argument("--val-tokens-per-domain", type=int, default=10_000_000)
    parser.add_argument("--out-dir", type=str, default="data/dolma_blend_smol")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo-id", type=str, default="varunneal/dolma-blend-smollm2")
    parser.add_argument("--stats-only", action="store_true", help="Just print stats, don't shard")
    parser.add_argument("--domains", type=str, default=None, help="Comma-separated list of domains to process (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(tempfile.gettempdir()) / "dolma_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SmolLM2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    bos_token = tokenizer.bos_token_id
    print(f"  Vocab size: {tokenizer.vocab_size}, BOS token: {bos_token} ({repr(tokenizer.bos_token)})")

    total_needed = args.tokens_per_domain + args.val_tokens_per_domain

    all_stats = []
    all_paths = []
    selected_domains = args.domains.split(",") if args.domains else list(DOMAINS.keys())

    for domain, domain_cfg in DOMAINS.items():
        if domain not in selected_domains:
            continue
        print(f"\n{'='*60}")
        print(f"Domain: {domain} (target: {total_needed / 1e9:.2f}B tokens)")
        print(f"{'='*60}")

        token_chunks = []
        collected = 0

        for shard_idx in range(domain_cfg["max_shards"]):
            if collected >= total_needed:
                break

            url = domain_cfg["base_url"].format(shard=shard_idx)
            gz_path = cache_dir / f"{domain}_{shard_idx:04d}.json.gz"

            if not download_gz(url, gz_path):
                break

            print(f"  Tokenizing shard {shard_idx} (have {collected / 1e9:.2f}B / {total_needed / 1e9:.2f}B)...")
            chunk = tokenize_jsonl_gz(gz_path, total_needed, tokenizer, bos_token, existing_count=collected)
            if len(chunk) == 0:
                print(f"  Shard {shard_idx} yielded 0 tokens, stopping")
                break
            token_chunks.append(chunk)
            collected += len(chunk)
            print(f"  Cumulative: {collected / 1e9:.3f}B tokens")

        if collected < total_needed:
            print(f"  WARNING: only got {collected / 1e9:.2f}B tokens (wanted {total_needed / 1e9:.2f}B)")

        tokens = np.concatenate(token_chunks)
        del token_chunks

        stats = compute_stats(tokens, domain, bos_token)
        if stats:
            all_stats.append(stats)

        if args.stats_only:
            continue

        val_tokens = tokens[: args.val_tokens_per_domain]
        train_tokens = tokens[args.val_tokens_per_domain : args.val_tokens_per_domain + args.tokens_per_domain]
        del tokens

        print(f"  Val: {len(val_tokens) / 1e6:.1f}M tokens, Train: {len(train_tokens) / 1e6:.1f}M tokens")

        val_paths = shard_tokens(val_tokens, domain, "val", out_dir)
        del val_tokens
        train_paths = shard_tokens(train_tokens, domain, "train", out_dir)
        del train_tokens
        all_paths.extend(val_paths + train_paths)

    if all_stats:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_tokens = 0
        for s in all_stats:
            print(f"  {s['domain']:12s}: {s['total_tokens'] / 1e9:.2f}B tokens, {s['num_sequences']:>8} seqs, "
                  f"median={s['median_len']:>5}, mean={s['mean_len']:>5}, "
                  f"p95={s['p95_len']:>6}, max={s['max_len']:>7}")
            total_tokens += s["total_tokens"]
        print(f"  {'TOTAL':12s}: {total_tokens / 1e9:.2f}B tokens")

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
