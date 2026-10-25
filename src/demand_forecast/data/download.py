"""Download the M5 dataset from Kaggle into data/raw/."""

import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "m5-forecasting-accuracy"
RAW_DIR = Path("data/raw")
FILES = ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download the competition archive and extract the files we use."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION, path=str(raw_dir), quiet=False)

    archive = raw_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        for name in FILES:
            zf.extract(name, raw_dir)
    archive.unlink()


if __name__ == "__main__":
    download()
