"""Download the M5 dataset from Kaggle into data/raw/."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "m5-forecasting-accuracy"
RAW_DIR = Path("data/raw")
EXPECTED_SHA256 = {
    "calendar.csv": "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
    "sales_train_evaluation.csv": "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
    "sell_prices.csv": "9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef",
}


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def invalid_files(raw_dir: Path) -> list[str]:
    """Return the expected files that are missing or have the wrong hash."""
    return [
        name
        for name, expected in EXPECTED_SHA256.items()
        if not (raw_dir / name).exists() or sha256(raw_dir / name) != expected
    ]


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Download and verify the M5 files, skipping if already present."""
    if not force and not invalid_files(raw_dir):
        print(f"M5 data already present and verified in {raw_dir}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    # Always re-download: invalid_files() already decided it is needed, and
    # Kaggle's own freshness check would reuse a partial zip from a failed run.
    api.competition_download_files(
        COMPETITION, path=str(raw_dir), force=True, quiet=False
    )

    archive = raw_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        for name in EXPECTED_SHA256:
            zf.extract(name, raw_dir)
    archive.unlink()

    bad = invalid_files(raw_dir)
    if bad:
        raise RuntimeError(f"Checksum mismatch after download: {', '.join(bad)}")
    print(f"M5 data downloaded and verified in {raw_dir}")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify the M5 dataset from Kaggle."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"directory for the raw CSV files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the files are present and valid",
    )
    args = parser.parse_args(argv)
    download(args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
