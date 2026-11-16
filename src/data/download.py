"""Download the Davidson et al. (2017) dataset from the authors' GitHub repository.

    python -m src.data.download            # skips if the file already exists
    python -m src.data.download --force

Source : https://github.com/t-davidson/hate-speech-and-offensive-language
Licence: MIT (repository licence). See docs/DATASET_CARD.md.

The file is validated (columns, row count) before it replaces anything, and a
SHA-256 checksum is written next to it so you can tell if the upstream file changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.data.load import DataValidationError, load_raw
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


class DatasetDownloadError(Exception):
    """Raised when the dataset cannot be downloaded or fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(cfg: dict, force: bool = False) -> Path:
    target = resolve_path(cfg["paths"]["raw_data"])
    url = cfg["data"]["url"]
    if target.is_file() and not force:
        log.info("Dataset already present at %s (use --force to re-download).", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "labeled_data.csv"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp_file, "wb") as out:
                shutil.copyfileobj(response, out)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DatasetDownloadError(
                f"Download failed: {exc}\n"
                f"Manual fix: open {url} in a browser, save the file as\n  {target}"
            ) from exc

        try:
            df = load_raw(tmp_file)
        except DataValidationError as exc:
            raise DatasetDownloadError(f"Downloaded file is not the expected dataset: {exc}") from exc

        expected = cfg["data"].get("expected_rows")
        if expected and len(df) != expected:
            log.warning("Expected %s rows but found %s. Upstream file may have changed.", expected, len(df))

        shutil.move(str(tmp_file), str(target))

    info = {
        "source_url": url,
        "repository": "https://github.com/t-davidson/hate-speech-and-offensive-language",
        "licence": "MIT (repository licence)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "sha256": _sha256(target),
    }
    with open(target.parent / "dataset_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    log.info("Saved %s rows to %s (sha256 %s...)", info["rows"], target, info["sha256"][:12])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        download_dataset(load_config(args.config), force=args.force)
    except DatasetDownloadError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
