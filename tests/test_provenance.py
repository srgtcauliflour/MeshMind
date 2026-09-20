import json

import pytest

from meshmind.provenance import sha256_file, verify_artifacts


def test_manifest_verifies_exact_bytes(tmp_path):
    corpus = tmp_path / "train.txt"
    corpus.write_text("mesh mind\n", encoding="utf-8")
    digest = sha256_file(corpus)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 2,
        "artifacts": [{"name": "train", "filename": "train.txt", "sha256": digest}],
    }))
    assert verify_artifacts(manifest, tmp_path)["train"] == digest


def test_unpinned_and_changed_data_are_rejected(tmp_path):
    corpus = tmp_path / "train.txt"
    corpus.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 2,
        "artifacts": [{"name": "train", "filename": "train.txt", "sha256": "UNPINNED"}],
    }))
    with pytest.raises(ValueError, match="not cryptographically pinned"):
        verify_artifacts(manifest, tmp_path)

    digest = sha256_file(corpus)
    manifest.write_text(json.dumps({
        "version": 2,
        "artifacts": [{"name": "train", "filename": "train.txt", "sha256": digest}],
    }))
    corpus.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifacts(manifest, tmp_path)
