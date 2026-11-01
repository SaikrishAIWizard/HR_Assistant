"""Command-line entry point for validating and inspecting an ingestion run."""

from __future__ import annotations

import argparse
import json
import os

from veltrix_assistant.ingestion.pipeline import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Veltrix HR document corpus.")
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", "./data/corpus"),
        help="Directory containing the source documents (or CORPUS_DIR).",
    )
    parser.add_argument(
        "--manifest-path",
        default=os.environ.get("MANIFEST_PATH", "./data/manifest.json"),
        help="Manifest JSON path (or MANIFEST_PATH).",
    )
    parser.add_argument("--chunk-tokens", type=int, default=500)
    args = parser.parse_args()

    corpus = ingest_corpus(
        args.corpus_dir,
        args.manifest_path,
        chunk_tokens=args.chunk_tokens,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "documents": len(corpus.documents),
                "chunks": len(corpus.chunks),
            }
        )
    )
