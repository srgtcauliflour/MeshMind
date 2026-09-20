"""Acquire and hash TinyStories without committing corpus bytes to Git."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

FILES = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as output:
        digest = hashlib.sha256()
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=FILES, default="valid")
    parser.add_argument("--output-dir", default="data/raw/tinystories")
    args = parser.parse_args()
    target = Path(args.output_dir) / f"{args.split}.txt"
    sha = download(FILES[args.split], target)
    print(f"{target} sha256={sha}")


if __name__ == "__main__":
    main()
