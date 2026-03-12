"""Download MIMII fan and slider (0 dB) subsets and summarize them.

This script:
1. Loads project configuration from ``config.yaml``.
2. Downloads the 0 dB fan and slider archives from Zenodo.
3. Verifies archive checksums where an expected hash is configured.
4. Extracts archives into the configured raw data directory.
5. Scans extracted WAV files to report basic dataset statistics.

Run from the project root with:

    .venv\\Scripts\\python.exe scripts\\download_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import requests
import soundfile as sf
import yaml


LOGGER = logging.getLogger("vehicleanomalynet.download_data")


ZENODO_BASE_URL = "https://zenodo.org/record/3384388/files"


@dataclass(frozen=True)
class ArchiveSpec:
    """Specification for a single dataset archive."""

    name: str
    url: str
    filename: str
    expected_md5: Optional[str]


ARCHIVES: Tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        name="fan_0dB",
        url=f"{ZENODO_BASE_URL}/0_dB_fan.zip?download=1",
        filename="0_dB_fan.zip",
        # MD5 from MIMII dataset documentation for 0_dB_fan.zip
        expected_md5="6354d1cc2165c52168f9ef1bcd9c7c52",
    ),
    ArchiveSpec(
        name="slider_0dB",
        url=f"{ZENODO_BASE_URL}/0_dB_slider.zip?download=1",
        filename="0_dB_slider.zip",
        expected_md5="4d674c21474f0646ecd75546db6c0c4e",
    ),
)


def load_config(config_path: Path) -> Dict[str, object]:
    """Load the YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return config


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def md5_for_file(path: Path, chunk_size: int = 8192) -> str:
    """Compute the MD5 checksum for a file."""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Download a file from ``url`` to ``destination``."""
    if destination.exists() and not overwrite:
        LOGGER.info("File already exists, skipping download: %s", destination)
        return

    LOGGER.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(destination)


def verify_checksum(path: Path, expected_md5: Optional[str]) -> None:
    """Verify the MD5 checksum of ``path`` if an expected value is provided."""
    actual_md5 = md5_for_file(path)
    if expected_md5 is None:
        LOGGER.warning(
            "No expected MD5 configured for %s. Computed MD5: %s",
            path.name,
            actual_md5,
        )
        return

    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {expected_md5}, got {actual_md5}"
        )

    LOGGER.info("Checksum OK for %s (%s)", path.name, actual_md5)


def find_existing_archive(raw_dir: Path, archive: ArchiveSpec) -> Optional[Path]:
    """Return path to an existing zip in raw_dir (accepts 0_dB_* or 0_db_* naming)."""
    candidates = [
        raw_dir / archive.filename,
        raw_dir / archive.filename.replace("0_dB_", "0_db_"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract a ZIP archive into ``target_dir``."""
    import zipfile

    LOGGER.info("Extracting %s into %s", archive_path, target_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def iter_wav_files(root: Path, machine_type: str) -> Iterable[Path]:
    """Yield all WAV files under ``root / machine_type``."""
    type_root = root / machine_type
    if not type_root.exists():
        LOGGER.warning("Machine type directory not found: %s", type_root)
        return []
    return type_root.rglob("*.wav")


def infer_label_from_path(path: Path) -> Optional[str]:
    """Infer 'normal' or 'abnormal' label from a file path."""
    parts = [p.lower() for p in path.parts]
    if "normal" in parts:
        return "normal"
    if "abnormal" in parts:
        return "abnormal"
    return None


def summarize_dataset(raw_dir: Path, sample_rate: int, machine_types: Iterable[str]) -> None:
    """Scan WAV files for each machine type and print dataset statistics."""
    from collections import defaultdict

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    durations: Dict[Tuple[str, str], float] = defaultdict(float)

    for machine in machine_types:
        for wav_path in iter_wav_files(raw_dir, machine):
            label = infer_label_from_path(wav_path)
            if label is None:
                continue

            try:
                info = sf.info(str(wav_path))
            except RuntimeError:
                LOGGER.warning("Failed to read audio info for %s", wav_path)
                continue

            if info.samplerate != sample_rate:
                LOGGER.warning(
                    "Sample rate mismatch for %s: file=%d, config=%d",
                    wav_path,
                    info.samplerate,
                    sample_rate,
                )

            key = (machine, label)
            counts[key] += 1
            durations[key] += info.frames / float(sample_rate)

    if not counts:
        LOGGER.warning("No labeled WAV files found under %s", raw_dir)
        return

    print("\n=== MIMII Dataset Summary (0 dB, fan + slider) ===")
    print(f"Raw directory: {raw_dir}")
    print(f"Configured sample rate for duration calculations: {sample_rate} Hz\n")
    header = f"{'Machine':10s} {'Class':10s} {'N clips':>10s} {'Total duration [h]':>20s}"
    print(header)
    print("-" * len(header))

    for machine in machine_types:
        for label in ("normal", "abnormal"):
            key = (machine, label)
            n = counts.get(key, 0)
            total_sec = durations.get(key, 0.0)
            total_hours = total_sec / 3600.0
            print(f"{machine:10s} {label:10s} {n:10d} {total_hours:20.3f}")

    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download MIMII fan + slider (0 dB) archives and summarize them."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if they already exist.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entry point for downloading and summarizing the MIMII dataset."""
    configure_logging()
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Config field 'data' must be a mapping.")

    raw_dir_value = data_cfg.get("raw_dir")
    if not isinstance(raw_dir_value, str):
        raise ValueError("Config field 'data.raw_dir' must be a string.")

    raw_dir = (project_root / raw_dir_value).resolve()
    ensure_directory(raw_dir)

    sample_rate = data_cfg.get("sample_rate")
    if not isinstance(sample_rate, int):
        raise ValueError("Config field 'data.sample_rate' must be an integer.")

    machine_types = data_cfg.get("machine_types")
    if not isinstance(machine_types, list):
        raise ValueError("Config field 'data.machine_types' must be a list of strings.")

    for archive in ARCHIVES:
        existing = find_existing_archive(raw_dir, archive)
        if existing is not None and not args.force:
            destination = existing
            LOGGER.info("Using existing archive: %s", destination)
        else:
            destination = raw_dir / archive.filename
            download_file(archive.url, destination, overwrite=args.force)
        verify_checksum(destination, archive.expected_md5)
        extract_archive(destination, raw_dir)

    summarize_dataset(raw_dir, sample_rate, machine_types)


if __name__ == "__main__":
    main()

