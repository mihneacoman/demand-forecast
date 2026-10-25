import hashlib
from pathlib import Path

import pytest

from demand_forecast.data import download as dl

FILES = {"a.csv": b"alpha\n", "b.csv": b"beta\n"}


@pytest.fixture
def expected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real M5 hashes with hashes of small fake files."""
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}
    monkeypatch.setattr(dl, "EXPECTED_SHA256", hashes)


def write_files(raw_dir: Path) -> None:
    for name, data in FILES.items():
        (raw_dir / name).write_bytes(data)


def test_sha256_reads_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    data = b"0123456789" * 10
    path.write_bytes(data)

    assert dl.sha256(path, chunk_size=4) == hashlib.sha256(data).hexdigest()


@pytest.mark.usefixtures("expected")
def test_invalid_files_empty_when_all_valid(tmp_path: Path) -> None:
    write_files(tmp_path)

    assert dl.invalid_files(tmp_path) == []


@pytest.mark.usefixtures("expected")
def test_invalid_files_reports_missing_and_corrupted(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_bytes(b"corrupted\n")

    assert dl.invalid_files(tmp_path) == ["a.csv", "b.csv"]


@pytest.mark.usefixtures("expected")
def test_download_skips_kaggle_when_files_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingKaggleApi:
        def __init__(self) -> None:
            raise AssertionError("Kaggle should not be called")

    monkeypatch.setattr(dl, "KaggleApi", FailingKaggleApi)
    write_files(tmp_path)

    dl.download(tmp_path)
