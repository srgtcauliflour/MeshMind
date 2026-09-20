"""Verify that local MM-E001 corpus bytes exactly match the pinned manifest."""

import argparse

from meshmind.provenance import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/tinystories-v2.json")
    parser.add_argument("--root", default="data/raw/tinystories")
    args = parser.parse_args()
    verified = verify_artifacts(args.manifest, args.root)
    for name, digest in verified.items():
        print(f"{name}: {digest}")


if __name__ == "__main__":
    main()
