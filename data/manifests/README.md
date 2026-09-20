# Dataset manifests

Every corpus admitted to MeshMind training must have a manifest entry before use.

Required fields:

- source name
- canonical source URI
- declared license or usage basis
- SHA-256 of the exact acquired artifact
- notes describing transformations or restrictions

Raw and processed corpora are intentionally excluded from Git. Manifests, filtering code,
statistics, and provenance records belong in the repository.

No training dataset should be silently introduced.
