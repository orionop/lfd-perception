"""
Cross-trial object identity: does DINOv2 correctly separate objects that
come from DIFFERENT trials with DIFFERENT scenes?

Code/object_identity.py clusters (obj_id, role) crops from a single trial's
objects_summary.csv. Until today only one image-bearing trial existed, so
that was never actually exercised across trials. This is a separate,
standalone script (Code/object_identity.py is intentionally left
untouched) that pools crops from MULTIPLE trials into one embedding space
and clusters them together -- the real test of whether the embeddings
generalize, not just separate objects within one demo.

Sources here: lfdws_t001 (carrot=grasped, cup=contact_receiver) and
lfdws_t001_depth (plate=contact_receiver only, no grasped role -- see
Docs/FAILURE_MODES.md B3/B4; uses the corrected plate-seeded propagation,
NOT the bad table-texture one).

Reads:
    figures/identify/objects_summary.csv        (lfdws_t001)
    figures/identify_depth/objects_summary.csv  (lfdws_t001_depth, corrected)

Writes (new files only, does not touch either trial's existing outputs):
    figures/identify/cross_trial_identity.json
    figures/identify/cross_trial_identity.png

Usage:
    .venv_dado/bin/python Code/object_identity_cross_trial.py
"""
import argparse
import csv
import json
import os
import sys

import cv2
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
    {"trial": "lfdws_t002_new",
     "summary_csv": "figures/t002new/identify/objects_summary.csv",
     "img_dir": "Data/lfdws_t002_new/zed_zed_node_rgb_color_rect_image_compressed"},
    {"trial": "lfdws_t002_labexport",
     "summary_csv": "figures/t002labexport/identify/objects_summary.csv",
     "img_dir": "Data/lfdws_t002_labexport/lfdws_t002/zed_zed_node_rgb_color_rect_image_compressed"},
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
    """Return list of {trial, key, frame_idx, img_filename, role, overlay_path}."""
    if not os.path.exists(source["summary_csv"]):
        print(f"[skip] {source['trial']}: {source['summary_csv']} missing", flush=True)
        return []
    rows = []
    with open(source["summary_csv"]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"[load] {source['trial']}: {len(rows)} (frame, obj_id) entries", flush=True)

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
    print(f"[load] {source['trial']}: {len(samples)} sampled (obj_id, role) crops "
          f"across {len(by_key)} distinct objects", flush=True)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="figures/identify/cross_trial_identity.json")
    ap.add_argument("--out_png", default="figures/identify/cross_trial_identity.png")
    ap.add_argument("--distance_threshold", type=float, default=0.4)
    args = ap.parse_args()

    all_samples = []
    for source in SOURCES:
        all_samples.extend(load_samples(source))
    if not all_samples:
        print("[fatal] no samples from any source", flush=True); sys.exit(1)
    print(f"[load] {len(all_samples)} total samples pooled across "
          f"{len(SOURCES)} trials", flush=True)

    device = pick_device()
    print(f"[load] DINOv2 ({DINO_ID}) on {device}", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID).to(device).eval()

    crops, sample_meta = [], []
    n_skipped = 0
    for i, s in enumerate(all_samples):
        bgr = cv2.imread(os.path.join(s["img_dir"], s["img_filename"]))
        if bgr is None:
            n_skipped += 1
            continue
        ov_name = f"f{s['frame_idx']:04d}_{s['img_filename']}"
        ov_path = os.path.join(s["overlay_dir"], ov_name)
        if not os.path.exists(ov_path):
            n_skipped += 1
            continue
        m = recover_mask(ov_path, bgr, s["role"])
        if m is None or m.sum() < 100:
            n_skipped += 1
            continue
        ys, xs = np.where(m)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = bgr[y0:y1, x0:x1].copy()
        crop[~m[y0:y1, x0:x1]] = 0
        if crop.size == 0:
            n_skipped += 1
            continue
        crops.append(crop)
        sample_meta.append(s)
        if (i + 1) % 20 == 0:
            print(f"  [crop] {i+1}/{len(all_samples)}", flush=True)
    print(f"[crop] kept {len(crops)} tight crops (skipped {n_skipped})", flush=True)

    embeddings = []
    BATCH = 8
    for i in range(0, len(crops), BATCH):
        batch_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                      for c in crops[i:i + BATCH]]
        inputs = proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        embeddings.append(cls.cpu().numpy())
        print(f"  [embed] {min(i+BATCH, len(crops))}/{len(crops)}", flush=True)
    if not embeddings:
        print("[fatal] no embeddings", flush=True); sys.exit(1)
    E = np.vstack(embeddings)
    print(f"[embed] shape={E.shape}", flush=True)

    print(f"[cluster] agglomerative, distance_threshold={args.distance_threshold}",
          flush=True)
    cluster = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=args.distance_threshold,
    )
    labels = cluster.fit_predict(E)
    n_ident = len(set(labels))
    print(f"[cluster] {n_ident} stable identity cluster(s) across "
          f"{len(set(s['trial'] for s in sample_meta))} trials", flush=True)

    # ground-truth key per sample: (trial, obj_id, role) -- what we WANT
    # separated, to check if DINOv2 actually keeps different objects apart
    gt_keys = [f"{s['trial']}|{s['key'][0]}|{s['key'][1]}" for s in sample_meta]
    unique_gt = sorted(set(gt_keys))
    print(f"[check] {len(unique_gt)} ground-truth (trial, obj_id, role) groups: "
          f"{unique_gt}", flush=True)

    # for each ground-truth group, which cluster(s) did it get split across?
    gt_to_clusters = {}
    for gt, lab in zip(gt_keys, labels):
        gt_to_clusters.setdefault(gt, set()).add(int(lab))
    print("[check] cluster assignment per ground-truth group:", flush=True)
    for gt, cls_set in gt_to_clusters.items():
        print(f"  {gt:45s} -> cluster(s) {sorted(cls_set)}", flush=True)

    # cross-contamination check: did any two DIFFERENT ground-truth groups
    # share a cluster? (that would mean DINOv2 confused two different objects)
    cluster_to_gts = {}
    for gt, lab in zip(gt_keys, labels):
        cluster_to_gts.setdefault(int(lab), set()).add(gt)
    contaminated = {c: gts for c, gts in cluster_to_gts.items() if len(gts) > 1}
    if contaminated:
        print(f"[check] CONTAMINATED clusters (mixed different objects): "
              f"{contaminated}", flush=True)
    else:
        print("[check] no cross-object contamination -- every cluster maps "
              "to exactly one (trial, obj_id, role) ground-truth group",
              flush=True)

    out_doc = {
        "sources": [s["summary_csv"] for s in SOURCES],
        "n_samples": len(sample_meta),
        "n_identities": n_ident,
        "ground_truth_groups": unique_gt,
        "gt_to_clusters": {gt: sorted(list(v)) for gt, v in gt_to_clusters.items()},
        "contaminated_clusters": {str(c): sorted(list(v)) for c, v in contaminated.items()},
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    # montage: one row per discovered cluster, labelled with its dominant
    # ground-truth group so you can eyeball correctness
    by_cluster = {}
    for crop, lab, gt in zip(crops, labels, gt_keys):
        by_cluster.setdefault(int(lab), {"crops": [], "gts": []})
        by_cluster[int(lab)]["crops"].append(crop)
        by_cluster[int(lab)]["gts"].append(gt)
    panels = []
    target_h = 96
    for lab in sorted(by_cluster.keys()):
        cs = by_cluster[lab]["crops"][:10]
        resized = [cv2.resize(c, (int(c.shape[1] * target_h / max(c.shape[0], 1)),
                                  target_h)) for c in cs if c.size > 0]
        if not resized:
            continue
        row = np.hstack(resized)
        gts = by_cluster[lab]["gts"]
        dom = max(set(gts), key=gts.count)
        banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"cluster {lab}: {dom}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(np.vstack([banner, row]))
    if panels:
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)
        cv2.imwrite(args.out_png, np.vstack(padded))
        print(f"[write] {args.out_png}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
