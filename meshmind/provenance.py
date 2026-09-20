"""Strict artifact provenance for MeshMind experiments."""

import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 2:
        raise ValueError("artifact manifest version 2 required")
    return payload


def verify_artifacts(manifest_path, root):
    manifest = load_artifact_manifest(manifest_path)
    root = Path(root)
    results = {}
    for artifact in manifest["artifacts"]:
        expected = artifact["sha256"]
        if not expected or expected.startswith("UNPINNED"):
            raise ValueError(f"{artifact['name']} is not cryptographically pinned")
        path = root / artifact["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {artifact['name']}: {actual} != {expected}")
        results[artifact["name"]] = actual
    return results
