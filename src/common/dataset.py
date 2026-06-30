"""Bootstrap the OpenNutrition nutrition dataset used by the vector store.

The full dataset (~327K foods, ~269 MB TSV) is too large for git, so the repo
ships without it. ``ensure_opennutrition_tsv`` fetches the official OpenNutrition
release zip on demand and unzips the TSV into place -- but only when the file is
missing, so repeated notebook/script runs never re-download.

The data is licensed under the Open Database License (ODbL), with contents under
the DbCL. Credit "OpenNutrition" (https://www.opennutrition.app) wherever the
data is displayed.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

# Official OpenNutrition full-dataset download. Version-pinned; bump on new releases.
DATASET_URL = "https://downloads.opennutrition.app/opennutrition-dataset-2025.1.zip"

# Name of the TSV inside the release archive (located wherever it sits in the tree).
TSV_NAME = "opennutrition_foods.tsv"

# Read/copy buffer for streaming the download and extraction.
_CHUNK = 256 * 1024


def ensure_opennutrition_tsv(
    dest: str | Path = "src/data/opennutrition_foods.tsv",
    *,
    url: str = DATASET_URL,
    force: bool = False,
) -> Path:
    """Make sure the OpenNutrition TSV exists at ``dest``, downloading it if needed.

    Idempotent: if ``dest`` already exists and is non-empty, returns it WITHOUT
    downloading. Otherwise it streams the release zip from ``url``, extracts the
    ``opennutrition_foods.tsv`` member (plus any LICENSE/README files) next to
    ``dest``, and moves the TSV into place atomically (an interrupted download
    never leaves a partial file at ``dest``).

    Args:
        dest: Target path for the TSV.
        url: OpenNutrition dataset zip URL.
        force: Re-download even if ``dest`` already exists.

    Returns:
        The resolved path to the TSV.

    Raises:
        FileNotFoundError: if the archive contains no ``opennutrition_foods.tsv``.
    """
    dest = Path(dest)

    # Idempotency guard: never re-download a dataset that is already in place.
    if not force and dest.exists() and dest.stat().st_size > 0:
        print(f"✅ OpenNutrition dataset already present at {dest} "
              f"({dest.stat().st_size / 1_048_576:.0f} MB) — skipping download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading OpenNutrition dataset (~60 MB) from {url} …")
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "opennutrition.zip"
        _download(url, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            member = _find_member(zf, TSV_NAME)
            print(f"📦 Extracting {member} …")
            zf.extract(member, tmp_dir)

            # Atomic publish: land the TSV on a sibling temp path, then rename.
            staged = dest.with_name(dest.name + ".part")
            shutil.move(str(tmp_dir / member), str(staged))
            staged.replace(dest)

            # Keep the license/readme files alongside the data (ODbL attribution).
            _extract_license_files(zf, dest.parent)

    print(f"✅ OpenNutrition dataset ready at {dest} "
          f"({dest.stat().st_size / 1_048_576:.0f} MB).")
    return dest


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target`` in chunks, printing percentage progress."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "langgraph-agent-guardrails/1.0"}
    )
    with urllib.request.urlopen(req) as resp, open(target, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r   {read / total * 100:5.1f}%  "
                      f"({read // 1_048_576} / {total // 1_048_576} MB)",
                      end="", flush=True)
    print()  # newline after the in-place progress line


def _find_member(zf: zipfile.ZipFile, name: str) -> str:
    """Return the archive member whose basename is ``name`` (root or subfolder)."""
    for info in zf.infolist():
        if not info.is_dir() and Path(info.filename).name == name:
            return info.filename

    raise FileNotFoundError(
        f"{name!r} not found in the OpenNutrition archive at this URL; "
        "the dataset layout may have changed — check DATASET_URL."
    )


def _extract_license_files(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Copy any LICENSE/README members from the archive into ``dest_dir``."""
    for info in zf.infolist():
        base = Path(info.filename).name
        if not info.is_dir() and base.upper().startswith(("LICENSE", "README")):
            with zf.open(info) as src, open(dest_dir / base, "wb") as out:
                shutil.copyfileobj(src, out)
