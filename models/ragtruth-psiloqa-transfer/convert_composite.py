"""Restore base composite tensors around a full-fine-tuned text checkpoint."""

import argparse
import glob
import os
import shutil

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def load_all(files):
    tensors = {}
    for filename in files:
        with safe_open(filename, framework="pt") as source:
            for key in source.keys():
                tensors[key] = source.get_tensor(key)
    return tensors


def resolve_base(value):
    if os.path.isdir(value):
        return os.path.abspath(value)
    return snapshot_download(
        value,
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors", "merges.txt"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="local snapshot directory or HF repo")
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_dir = resolve_base(args.base)
    base_files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    tuned_files = sorted(glob.glob(os.path.join(args.tuned, "*.safetensors")))
    if not base_files:
        raise SystemExit(f"no safetensors found under base {base_dir}")
    if not tuned_files:
        raise SystemExit(f"no safetensors found under tuned checkpoint {args.tuned}")

    base = load_all(base_files)
    tuned = load_all(tuned_files)
    missing = [key for key in tuned if key not in base]
    if missing:
        raise SystemExit(f"tuned keys absent from base: {missing[:5]}")
    for key, value in tuned.items():
        base[key] = value.to(base[key].dtype)
    print(f"overrode {len(tuned)}/{len(base)} tensors with tuned weights", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for filename in glob.glob(os.path.join(base_dir, "*")):
        basename = os.path.basename(filename)
        if basename.endswith(".safetensors") or basename == "model.safetensors.index.json":
            continue
        if os.path.isfile(filename):
            shutil.copy2(filename, os.path.join(args.out, basename))
    save_file(
        base,
        os.path.join(args.out, "model.safetensors"),
        metadata={"format": "pt"},
    )
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
