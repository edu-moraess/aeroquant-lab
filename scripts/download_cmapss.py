#!/usr/bin/env python3
"""Baixa NASA C-MAPSS para data/external/ (~12 MB)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT / "CMAPSSData.zip"
    if not (OUT / "train_FD001.txt").exists():
        print(f"Downloading {URL} ...")
        req = Request(URL, headers={"User-Agent": "AeroQuantLab/1.0"})
        with urlopen(req, timeout=180) as resp:
            zip_path.write_bytes(resp.read())
        print(f"Saved {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(OUT)
        print("Extracted to", OUT)
    else:
        print("train_FD001.txt already present — skip download")
    for name in (
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
        "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
        "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    ):
        p = OUT / name
        print(f"  {name}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
