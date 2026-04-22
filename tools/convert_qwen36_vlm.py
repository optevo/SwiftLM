#!/usr/bin/env python3
"""
Convert Qwen3.6-35B-A3B VLM from HuggingFace safetensors to MLX format.

Applies the same key remapping and weight transformations as Qwen35MoE.sanitize()
in SwiftLM, then quantises eligible weights with mx.quantize. Sets format=mlx in
safetensors metadata so the Swift loader skips sanitize() on load.

Usage:
    python3 convert_qwen36_vlm.py [options]

Options:
    --src PATH        Source directory (default: ~/models/Qwen3.6-35B-A3B/pending-mlx-conversion)
    --dst PATH        Output directory (default: ~/models/Qwen3.6-35B-A3B-vlm-staging)
    --bits INT        Quantisation bits (default: 4)
    --group-size INT  Quantisation group size (default: 64)
    --no-quantise     Skip quantisation, output full-precision BF16
    --shard-mb INT    Target output shard size in MB (default: 2000)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

# --- Key remapping ---

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

# Files to copy verbatim from source to destination.
COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "configuration.json",
    "README.md",
]


def remap_key(key: str) -> str:
    """Apply prefix remapping mirroring Qwen35.sanitize()."""
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return "vision_tower." + key[len("model.visual."):]
    if key.startswith("lm_head."):
        return "language_model.lm_head." + key[len("lm_head."):]
    return key


def should_skip(key: str) -> bool:
    """Skip MTP (multi-token prediction) auxiliary weights."""
    return key.startswith("mtp.")


def needs_norm_correction(key: str) -> bool:
    """True for 1-D norm weights that store residuals from 1.0 in HF format."""
    return any(key.endswith(s) for s in NORM_SUFFIXES)


def should_quantise(key: str, arr: mx.array, group_size: int) -> bool:
    """True if this weight should be quantised."""
    if not key.endswith(".weight"):
        return False
    if arr.ndim < 2:
        return False
    if arr.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if arr.shape[-1] < group_size or arr.shape[-1] % group_size != 0:
        return False
    # Skip conv weights (last dim = 1 means it's a kernel dim, not an in-feature dim)
    if arr.shape[-1] == 1:
        return False
    return True


# --- Weight processing ---

def process_weights(
    raw: dict[str, mx.array],
    bits: int,
    group_size: int,
    quantise: bool,
) -> dict[str, mx.array]:
    """
    Apply all transformations to a shard of weights and return the result dict.

    Handles:
      - MTP skip
      - Prefix remapping
      - Norm weight +1 correction
      - MoE gate_up_proj split into switch_mlp.{gate,up}_proj.weight
      - MoE experts.down_proj rename to switch_mlp.down_proj.weight
      - Quantisation of eligible weights
    """
    result: dict[str, mx.array] = {}

    for key, value in raw.items():
        if should_skip(key):
            continue

        # --- MoE expert splitting (must happen before prefix remap for simpler matching) ---
        # gate_up_proj: (num_experts, gate+up, hidden) → split along dim -2
        if key.endswith(".mlp.experts.gate_up_proj"):
            base = key[: -len(".mlp.experts.gate_up_proj")]
            mid = value.shape[-2] // 2
            gate_key = remap_key(base) + ".mlp.switch_mlp.gate_proj.weight"
            up_key = remap_key(base) + ".mlp.switch_mlp.up_proj.weight"
            gate_arr = value[..., :mid, :]
            up_arr = value[..., mid:, :]
            _emit(result, gate_key, gate_arr, bits, group_size, quantise)
            _emit(result, up_key, up_arr, bits, group_size, quantise)
            continue

        # down_proj: (num_experts, hidden, expert_dim) → just rename
        if key.endswith(".mlp.experts.down_proj"):
            base = key[: -len(".mlp.experts.down_proj")]
            new_key = remap_key(base) + ".mlp.switch_mlp.down_proj.weight"
            _emit(result, new_key, value, bits, group_size, quantise)
            continue

        # --- Standard key remap ---
        new_key = remap_key(key)

        # --- Norm correction ---
        if needs_norm_correction(new_key) and value.ndim == 1:
            value = value + mx.array(1.0, dtype=value.dtype)

        # --- conv1d weight axis correction ---
        # HF stores as (out, in_channels, kernel_size); Conv1d expects (out, kernel_size, in_channels).
        # Mirror Swift's: movedAxis(source: 2, destination: 1) when dim(-1) != 1.
        if "conv1d.weight" in new_key and value.ndim == 3 and value.shape[-1] != 1:
            value = mx.transpose(value, [0, 2, 1])

        # --- patch_embed.proj.weight axis correction ---
        # HF stores as (out, in_channels, D, H, W); Conv3d expects (out, D, H, W, in_channels).
        # Mirror Swift's VisionModel.sanitize: transpose(0, 2, 3, 4, 1) unless already in MLX format
        # (which is when last dim == in_channels=3).
        if "patch_embed.proj.weight" in new_key and value.ndim == 5 and value.shape[-1] != value.shape[1]:
            value = mx.transpose(value, [0, 2, 3, 4, 1])

        # --- Emit (possibly quantised) ---
        _emit(result, new_key, value, bits, group_size, quantise)

    return result


def _emit(
    result: dict[str, mx.array],
    key: str,
    arr: mx.array,
    bits: int,
    group_size: int,
    quantise: bool,
) -> None:
    """Add weight to result dict, quantising if eligible."""
    if quantise and should_quantise(key, arr, group_size):
        qw, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        result[key] = qw
        result[key.removesuffix(".weight") + ".scales"] = scales
        result[key.removesuffix(".weight") + ".biases"] = biases
    else:
        result[key] = arr


# --- Sharding and saving ---

def _weight_bytes(arr: mx.array) -> int:
    dtype_bytes = {
        mx.float32: 4, mx.float16: 2, mx.bfloat16: 2,
        mx.uint32: 4, mx.uint8: 1, mx.int32: 4,
    }
    return arr.size * dtype_bytes.get(arr.dtype, 2)


def save_shards(
    weights: dict[str, mx.array],
    dst: Path,
    shard_mb: int,
    metadata: dict[str, str],
) -> None:
    """Save weights as numbered safetensors shards and write the index file."""
    shard_bytes = shard_mb * 1024 * 1024
    shards: list[dict[str, mx.array]] = []
    current: dict[str, mx.array] = {}
    current_size = 0

    for key in sorted(weights.keys()):
        arr = weights[key]
        size = _weight_bytes(arr)
        if current and current_size + size > shard_bytes:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = arr
        current_size += size

    if current:
        shards.append(current)

    n = len(shards)
    weight_map: dict[str, str] = {}

    for i, shard in enumerate(shards, 1):
        filename = f"model-{i:05d}-of-{n:05d}.safetensors"
        path = dst / filename
        print(f"  writing {filename} ({len(shard)} tensors)...")
        mx.save_safetensors(str(path), shard, metadata=metadata)
        for key in shard:
            weight_map[key] = filename

    index = {
        "metadata": {"total_size": sum(_weight_bytes(w) for w in weights.values())},
        "weight_map": weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )


# --- Config ---

def write_config(src: Path, dst: Path, bits: int, group_size: int, quantise: bool) -> None:
    """Copy config.json, adding quantisation metadata when quantising."""
    cfg = json.loads((src / "config.json").read_text())
    if quantise:
        cfg["quantization"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def copy_support_files(src: Path, dst: Path) -> None:
    """Copy tokeniser and other non-weight files."""
    for name in COPY_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


# --- Main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home()
    p.add_argument("--src", default=str(home / "models/Qwen3.6-35B-A3B/pending-mlx-conversion"))
    p.add_argument("--dst", default=str(home / "models/Qwen3.6-35B-A3B-vlm-staging"))
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--no-quantise", action="store_true")
    p.add_argument("--shard-mb", type=int, default=2000)
    p.add_argument("--force", action="store_true", help="Overwrite destination without prompting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    quantise = not args.no_quantise
    bits = args.bits
    group_size = args.group_size

    if not src.exists():
        sys.exit(f"Source not found: {src}")
    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            print(f"Destination exists: {dst}")
            resp = input("Overwrite? [y/N] ").strip().lower()
            if resp != "y":
                sys.exit("Aborted.")
            shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"Source : {src}")
    print(f"Dest   : {dst}")
    print(f"Quant  : {'%d-bit group_size=%d' % (bits, group_size) if quantise else 'none (BF16)'}")
    print()

    # Load index to discover shards
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit("model.safetensors.index.json not found in source")
    index = json.loads(index_path.read_text())
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Input shards: {len(shard_files)}")

    # Process each input shard
    all_weights: dict[str, mx.array] = {}
    for i, fname in enumerate(shard_files, 1):
        print(f"[{i:2d}/{len(shard_files)}] loading {fname}...")
        raw = mx.load(str(src / fname))
        processed = process_weights(raw, bits, group_size, quantise)
        all_weights.update(processed)
        mx.eval(list(processed.values()))  # materialise before moving on

    print(f"\nTotal output tensors: {len(all_weights)}")
    print("Saving shards...")
    metadata = {"format": "mlx"}
    save_shards(all_weights, dst, args.shard_mb, metadata)

    print("Writing config and support files...")
    write_config(src, dst, bits, group_size, quantise)
    copy_support_files(src, dst)

    print(f"\nDone. Output at: {dst}")
    total_mb = sum(_weight_bytes(w) for w in all_weights.values()) / 1024 / 1024
    print(f"Total weight data: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
