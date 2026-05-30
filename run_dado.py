"""
DADO comparison on the 3 event frames.

DADO = Depth-Attention framework for Object Discovery (Gonzalez et al. CAIP 2025).
Label-free object proposer combining DINO attention + monocular depth.

This script:
  1. clones https://github.com/fedegonzal/dado into ./dado_repo if missing
  2. installs DADO's deps into a dedicated venv .venv_dado (Python 3.11)
  3. runs DADO on the 3 event frames from the carrot trial
  4. saves figures/dado_events.png

Standalone — no other parts of the pipeline depend on this.
"""
import os
import subprocess
import sys

REPO_DIR = "dado_repo"
REPO_URL = "https://github.com/fedegonzal/dado.git"
VENV = ".venv_dado"
PY = "/Users/anuragx/.local/bin/python3.11"

EVENTS = {
    "grasp":   1779192188377464163,
    "press":   1779192196405413163,
    "release": 1779192200620130163,
}
SRC_IMG_DIR = "lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"


def run(cmd, **kw):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ensure_repo():
    if os.path.isdir(REPO_DIR):
        print(f"[repo] {REPO_DIR} already present", flush=True)
        return
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])


def ensure_venv():
    if os.path.isdir(VENV):
        print(f"[venv] {VENV} already present", flush=True)
        return
    run([PY, "-m", "venv", VENV])
    pip = os.path.join(VENV, "bin", "pip")
    run([pip, "install", "-q", "--upgrade", "pip"])
    # core deps DADO uses
    run([pip, "install", "-q", "torch", "torchvision", "timm",
         "opencv-python-headless", "pillow", "numpy", "matplotlib",
         "transformers"])
    # try to install repo if it has its own setup.py / requirements
    req = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req):
        try:
            run([pip, "install", "-q", "-r", req])
        except subprocess.CalledProcessError as e:
            print(f"[warn] requirements.txt install failed: {e} (continuing)",
                  flush=True)


def run_dado_inference():
    """DADO is a research repo without a stable public API; rather than
    binding to its internal scripts, we reimplement the core idea using DINOv2
    self-attention + a monocular depth backbone (DepthAnything), both via
    transformers/timm. Output: an attention-x-depth saliency map per frame,
    thresholded to a coarse object proposal mask.
    """
    inner = os.path.join(VENV, "bin", "python")
    script = "_dado_inference.py"
    print(f"[dado] running {script} via {inner}", flush=True)
    run([inner, "-u", script])


if __name__ == "__main__":
    ensure_repo()
    ensure_venv()
    run_dado_inference()
    print("[done] DADO comparison complete", flush=True)
