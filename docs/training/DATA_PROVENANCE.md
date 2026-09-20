# MM-E001 Data Provenance

MeshMind treats the exact bytes used for training as part of the experiment.

The version-2 corpus manifest identifies each split separately by canonical source URI and SHA-256.
A dataset name alone is not sufficient provenance. `scripts/verify_corpus.py` refuses unpinned
artifacts and rejects local bytes that differ from the recorded digest.

## Promotion rule

No MM-E001 result may be promoted, compared, or used as a baseline until both train and validation
artifacts are pinned and verification succeeds. The tokenizer must be trained from those verified
bytes, and its serialized artifact must be archived with the run.

The repository intentionally does not contain the downloaded corpus.
