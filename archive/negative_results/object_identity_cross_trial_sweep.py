"""
Follow-up to object_identity_cross_trial.py's negative result (12 clusters
from 3 true objects, 2 contaminated, at distance_threshold=0.4 with
tight/blacked-out crops). This does NOT touch that script or its output;
it's a standalone sweep trying two things that might fix the fragmentation/
contamination:

  1. A range of distance_threshold values (0.2 - 0.7), to see whether ANY
     threshold gives a clean 3-cluster, 0-contamination result, or whether
     the failure is embedding-quality (not just a threshold-tuning problem).
  2. A padded-crop variant (bbox expanded by --pad_frac, background NOT
     blacked out) alongside the original tight/blacked-out crop, since
     removing all context may be starving DINOv2 of the texture/shape cues
     it needs -- the opposite hypothesis to the "background dominates"
     reasoning that motivated tight crops in the first place.

Reads the same two sidecars as object_identity_cross_trial.py.

Writes:
    figures/identify/cross_trial_sweep_results.json  (per-config metrics)
    figures/identify/cross_trial_sweep_summary.png   (contamination/frag vs threshold)

Usage:
    .venv_dado/bin/python Code/object_identity_cross_trial_sweep.py
"""
import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoImageProcessor, AutoModel

DINO_ID = "facebook/dinov2-base"
ROLE_COLOR = {"grasped": "green", "contact_receiver": "magenta"}

SOURCES = [
    {"trial": "lfdws_t001",
     "summary_csv": "figures/identify/objects_summary.csv",
     "img_dir": "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"},
    {"trial": "lfdws_t001_depth",
     "summary_csv": "figures/identify_depth/objects_summary.csv",
     "img_dir": "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed"},
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def recover_mask(overlay_path, src_bgr, role):
    ov = cv2.imread(overlay_path)
    if ov is None or src_bgr is None or ov.shape != src_bgr.shape:
        return None
    diff = ov.astype(int) - src_bgr.astype(int)
    if ROLE_COLOR.get(role) == "green":
        return diff[..., 1] > 40
    if ROLE_COLOR.get(role) == "magenta":
        return (diff[..., 0] > 40) & (diff[..., 2] > 40)
    return None


def load_samples(source, n_per_key=12):
    if not os.path.exists(source["summary_csv"]):
        return []
    rows = []
    with open(source["summary_csv"]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_key = {}
    for r in rows:
        key = (r["obj_id"], r["role"])
        by_key.setdefault(key, []).append(r)
    samples = []
    for key, items in by_key.items():
        n = max(1, len(items) // n_per_key)
        for r in items[::n]:
            samples.append({
                "trial": source["trial"], "img_dir": source["img_dir"],
                "overlay_dir": os.path.join(os.path.dirname(source["summary_csv"]), "overlays"),
                "key": key, "frame_idx": int(r["frame_idx"]),
                "img_filename": r["img_filename"], "role": r["role"],
            })
    return samples


def make_crop(bgr, m, pad_frac, blackout):
    ys, xs = np.where(m)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    H, W = bgr.shape[:2]
    if pad_frac > 0:
        pw = int((x1 - x0) * pad_frac)
        ph = int((y1 - y0) * pad_frac)
        x0, y0 = max(0, x0 - pw), max(0, y0 - ph)
        x1, y1 = min(W, x1 + pw), min(H, y1 + ph)
    crop = bgr[y0:y1, x0:x1].copy()
    if blackout:
        local_mask = m[y0:y1, x0:x1]
        crop[~local_mask] = 0
    return crop


def embed_crops(crops, proc, model, device, batch=8):
    embeddings = []
    for i in range(0, len(crops), batch):
        batch_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                      for c in crops[i:i + batch]]
        inputs = proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        embeddings.append(cls.cpu().numpy())
    return np.vstack(embeddings) if embeddings else np.zeros((0, 768))


def evaluate(labels, gt_keys):
    n_ident = len(set(labels))
    n_gt = len(set(gt_keys))
    cluster_to_gts = {}
    for gt, lab in zip(gt_keys, labels):
        cluster_to_gts.setdefault(int(lab), set()).add(gt)
    contaminated = {c: gts for c, gts in cluster_to_gts.items() if len(gts) > 1}
    gt_to_clusters = {}
    for gt, lab in zip(gt_keys, labels):
        gt_to_clusters.setdefault(gt, set()).add(int(lab))
    max_fragmentation = max(len(v) for v in gt_to_clusters.values()) if gt_to_clusters else 0
    return {
        "n_clusters": n_ident, "n_gt_groups": n_gt,
        "n_contaminated_clusters": len(contaminated),
        "max_fragmentation": max_fragmentation,  # worst-case: 1 gt group split across N clusters
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="figures/identify/cross_trial_sweep_results.json")
    ap.add_argument("--out_png", default="figures/identify/cross_trial_sweep_summary.png")
    args = ap.parse_args()

    all_samples = []
    for source in SOURCES:
        s = load_samples(source)
        print(f"[load] {source['trial']}: {len(s)} sampled crops", flush=True)
        all_samples.extend(s)
    if not all_samples:
        print("[fatal] no samples", flush=True); sys.exit(1)

    device = pick_device()
    print(f"[load] DINOv2 ({DINO_ID}) on {device}", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID).to(device).eval()

    configs = [
        {"name": "tight_blackout", "pad_frac": 0.0, "blackout": True},
        {"name": "padded_context", "pad_frac": 0.3, "blackout": False},
    ]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    for cfg in configs:
        print(f"\n[config] {cfg['name']} (pad={cfg['pad_frac']}, "
              f"blackout={cfg['blackout']})", flush=True)
        crops, gt_keys = [], []
        n_skipped = 0
        for s in all_samples:
            bgr = cv2.imread(os.path.join(s["img_dir"], s["img_filename"]))
            if bgr is None:
                n_skipped += 1; continue
            ov_name = f"f{s['frame_idx']:04d}_{s['img_filename']}"
            ov_path = os.path.join(s["overlay_dir"], ov_name)
            if not os.path.exists(ov_path):
                n_skipped += 1; continue
            m = recover_mask(ov_path, bgr, s["role"])
            if m is None or m.sum() < 100:
                n_skipped += 1; continue
            crop = make_crop(bgr, m, cfg["pad_frac"], cfg["blackout"])
            if crop.size == 0:
                n_skipped += 1; continue
            crops.append(crop)
            gt_keys.append(f"{s['trial']}|{s['key'][0]}|{s['key'][1]}")
        print(f"  [crop] kept {len(crops)} (skipped {n_skipped})", flush=True)

        E = embed_crops(crops, proc, model, device)
        print(f"  [embed] shape={E.shape}", flush=True)

        for thr in thresholds:
            cluster = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=thr,
            )
            labels = cluster.fit_predict(E)
            metrics = evaluate(labels, gt_keys)
            metrics.update({"config": cfg["name"], "threshold": thr})
            results.append(metrics)
            print(f"    thr={thr:.1f}  clusters={metrics['n_clusters']:3d}  "
                  f"contaminated={metrics['n_contaminated_clusters']}  "
                  f"max_fragmentation={metrics['max_fragmentation']}", flush=True)

    # did anything achieve the ideal: n_clusters == n_gt_groups AND 0 contamination?
    ideal = [r for r in results if r["n_clusters"] == r["n_gt_groups"]
             and r["n_contaminated_clusters"] == 0]
    if ideal:
        print(f"\n[result] FOUND a clean config: {ideal}", flush=True)
    else:
        print(f"\n[result] NO config achieved 0 contamination + exact cluster "
              f"count across {len(results)} (config, threshold) combinations "
              f"tried -- confirms this isn't just a threshold-tuning problem.",
              flush=True)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"results": results, "found_clean_config": bool(ideal)}, f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in configs:
        sub = [r for r in results if r["config"] == cfg["name"]]
        ax.plot([r["threshold"] for r in sub], [r["n_contaminated_clusters"] for r in sub],
                marker="o", label=f"{cfg['name']}: contaminated clusters")
        ax.plot([r["threshold"] for r in sub], [r["max_fragmentation"] for r in sub],
                marker="s", linestyle="--", label=f"{cfg['name']}: max fragmentation")
    ax.axhline(1, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("distance_threshold")
    ax.set_ylabel("count")
    ax.set_title("Cross-trial DINOv2 clustering: contamination/fragmentation vs threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"[write] {args.out_png}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
