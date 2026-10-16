#!/usr/bin/env python3
"""Download CARLA AdditionalMaps (Town06, Town07) for the bargaining evaluation."""

import os, sys, requests, zipfile, shutil

CARLA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation", "carla")
URL = "https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/AdditionalMaps_0.9.10.zip"
ZIP_PATH = os.path.join(CARLA_DIR, "AdditionalMaps_0.9.10.zip")


def download():
    if not os.path.isdir(CARLA_DIR):
        print(f"[ERROR] CARLA directory not found at {CARLA_DIR}")
        print("Please extract CARLA_0.9.10.1 to this directory first.")
        sys.exit(1)

    if os.path.isfile(ZIP_PATH):
        print(f"[INFO] AdditionalMaps zip already exists ({os.path.getsize(ZIP_PATH) // 1024**2} MB)")
    else:
        print(f"[INFO] Downloading AdditionalMaps_0.9.10.1.zip...")
        response = requests.get(URL, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"  Downloaded: {downloaded // 1024**2} MB / {total // 1024**2} MB ({pct:.1f}%)", end="\r")
        print(f"\n[INFO] Download complete.")

    print(f"[INFO] Extracting to {CARLA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(CARLA_DIR)
    print(f"[INFO] AdditionalMaps extracted. Town06 and Town07 should now be available.")

    # Clean up
    os.remove(ZIP_PATH)
    print(f"[INFO] Cleaned up zip file.")


if __name__ == "__main__":
    download()
