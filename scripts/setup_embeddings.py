#!/usr/bin/env python3
"""Download a pinned CPU sentence encoder inside DreamProver."""
import argparse
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/models")
    args = parser.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    cache = args.output_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "all-mpnet-base-v2"
    metadata_path = cache / "embedding-model.json"
    if metadata_path.exists() and path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["repository"] != MODEL or metadata["revision"] != REVISION:
            raise ValueError("Existing embedding model differs; preserve it and choose another output directory")
    else:
        # The exported model is self-contained. Keep the download cache
        # temporary instead of retaining a second copy of the weights.
        with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as download:
            model = SentenceTransformer(MODEL, revision=REVISION, cache_folder=download, device="cpu")
            model.save(str(path))
        metadata = {"repository": MODEL, "revision": REVISION, "path": str(path),
                    "device": "cpu", "dimension": model.get_sentence_embedding_dimension()}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
