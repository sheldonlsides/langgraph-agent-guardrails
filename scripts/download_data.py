#!/usr/bin/env python
"""Download the OpenNutrition dataset into ``src/data/``.

Run from the project root:

    uv run python scripts/download_data.py [--force]

Idempotent: skips the download when the TSV is already present (pass ``--force``
to re-fetch). Thin wrapper around ``common.dataset.ensure_opennutrition_tsv`` so
the CLI and the notebook share one implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DEST = _REPO_ROOT / "src" / "data" / "opennutrition_foods.tsv"


def _load_dataset_module():
    """Load `src/common/dataset.py` directly (stdlib-only, no LLM deps needed)."""
    spec = importlib.util.spec_from_file_location(
        "opennutrition_dataset", _REPO_ROOT / "src" / "common" / "dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dataset = _load_dataset_module()
DATASET_URL = _dataset.DATASET_URL
ensure_opennutrition_tsv = _dataset.ensure_opennutrition_tsv


def main() -> None:
    """Parse CLI args and ensure the OpenNutrition TSV is present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(_DEFAULT_DEST),
                        help="Target path for the TSV (default: src/data/...).")
    parser.add_argument("--url", default=DATASET_URL, help="Dataset zip URL.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the TSV already exists.")
    args = parser.parse_args()

    ensure_opennutrition_tsv(args.dest, url=args.url, force=args.force)


if __name__ == "__main__":
    main()
