"""
Download SAM 2.1 hiera-large checkpoint if not present.

Standalone: just `python3 download_sam2_ckpt.py`. ~900 MB.
"""
import os
import sys
import urllib.request

URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
OUT = "sam2.1_hiera_large.pt"


def progress(blocks, block_size, total):
    done = blocks * block_size
    pct = 100.0 * done / total if total > 0 else 0
    mb_done = done / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    sys.stdout.write(f"\r  downloading {OUT}: {mb_done:7.1f} / {mb_total:7.1f} MB  ({pct:5.1f}%)")
    sys.stdout.flush()


def main():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 800 * 1024 * 1024:
        print(f"[OK] {OUT} already present ({os.path.getsize(OUT) / 1024**2:.1f} MB) — skipping")
        return
    print(f"fetching {URL}", flush=True)
    urllib.request.urlretrieve(URL, OUT, reporthook=progress)
    print(f"\n[OK] saved -> {OUT}")


if __name__ == "__main__":
    main()
