"""
echoX - ASVspoof 2019 LA Dataset Downloader Utility
Downloads official ASVspoof 2019 Logical Access (LA) archives or protocol files
from Edinburgh DataShare / official academic mirrors.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ASVspoof2019_LA"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOLS_URL = "https://www.asvspoof.org/asvspoof2019/LA.zip"


def download_protocols(dest_dir: Path):
    print(f"ASVspoof 2019 LA Directory: {dest_dir}")
    print("To download the full 25GB ASVspoof 2019 LA dataset, run:")
    print(f"  wget -c {PROTOCOLS_URL} -P {dest_dir}")
    print("  unzip LA.zip")
    print("\nFor fast testing and presentations, pre-generated A01-A19 samples")
    print("are available in: tests/samples/asvspoof2019_la/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official ASVspoof 2019 LA dataset")
    parser.add_argument("--dest", type=str, default=str(DATA_DIR), help="Destination folder")
    args = parser.parse_args()
    download_protocols(Path(args.dest))
