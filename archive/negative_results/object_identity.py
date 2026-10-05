"""
Object identity across phases / episodes.

Within one demo, the SAM 2 propagation already keeps a stable obj_id while
it's tracking. But:
  - if the carrot leaves the frame and re-enters, SAM 2 may renumber it
  - across DIFFERENT demos, obj_id=1 in trial 001 has no relation to obj_id=1
    in trial 005

This script embeds each (frame, obj_id) mask crop with a frozen DINOv2
backbone, then clusters the embeddings to assign stable object identities.

Crop tightening:
  - the bbox in objects_summary.csv was derived from the propagation
    overlay PNG and includes background pixels. We recover the actual
    mask (by colour-distance vs source) and crop to the mask's tight
    bbox, then black out non-mask pixels inside the crop. This makes
    DINOv2 embed the object itself, not the surrounding table.

Reads:
    figures/identify/objects_summary.csv

Writes:
    figures/identify/object_identity.json  (per object_id -> stable identity)
    figures/identify/object_identity.png   (one row per identity, sample crops)

Standalone — uses the same .venv_dado that already has transformers + torch.

Usage:
    .venv_dado/bin/python Code/object_identity.py
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

SUMMARY_CSV = "figures/identify/objects_summary.csv"
SRC_IMG_DIR = "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
OUT_JSON = "figures/identify/object_identity.json"
OUT_PNG = "figures/identify/object_identity.png"
DINO_ID = "facebook/dinov2-base"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default=SUMMARY_CSV)
    ap.add_argument("--img_dir", default=SRC_IMG_DIR)
    ap.add_argument("--out_json", default=OUT_JSON)
    ap.add_argument("--out_png", default=OUT_PNG)
    ap.add_argument("--distance_threshold", type=float, default=0.4,
                    help="agglomerative clustering distance threshold on cosine distance")
    args = ap.parse_args()

    if not os.path.exists(args.summary_csv):
        print(f"[fatal] {args.summary_csv} missing -- run build_sidecar.py first",
              flush=True); sys.exit(1)

    print(f"[load] {args.summary_csv}", flush=True)
    rows = []
    with open(args.summary_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"[load] {len(rows)} (frame, obj_id) entries", flush=True)

    # sample a manageable number of crops per (obj_id, role) — 1 in every K frames
    samples = []  # (key, frame_idx, img_filename, bbox, role, overlay_path)
    by_key = {}
    for r in rows:
        key = (r["obj_id"], r["role"])
        by_key.setdefault(key, []).append(r)
    for key, items in by_key.items():
        # take 12 evenly-spaced samples
        n = max(1, len(items) // 12)
        for r in items[::n]:
            try:
                bb = [int(r["bbox_x0"]), int(r["bbox_y0"]),
                      int(r["bbox_x1"]), int(r["bbox_y1"])]
            except Exception:
                continue
            if bb[0] < 0 or bb[1] < 0 or bb[2] <= bb[0] or bb[3] <= bb[1]:
                continue
            samples.append({
                "key": key, "frame_idx": int(r["frame_idx"]),
                "img_filename": r["img_filename"], "bbox": bb,
                "role": r["role"],
            })
    print(f"[load] {len(samples)} sample crops across {len(by_key)} (obj_id, role)",
          flush=True)

    device = pick_device()
    print(f"[load] DINOv2 ({DINO_ID}) on {device}", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID).to(device).eval()

    # mask colours per role (from build_sidecar.py overlay convention)
    ROLE_COLOR = {"grasped": "green", "contact_receiver": "magenta"}

    def recover_mask(overlay_path, src_bgr, role):
        """Return a tight boolean mask of the role-coloured pixels in overlay."""
        ov = cv2.imread(overlay_path)
        if ov is None or src_bgr is None or ov.shape != src_bgr.shape:
            return None
        diff = ov.astype(int) - src_bgr.astype(int)
        if ROLE_COLOR.get(role) == "green":
            return diff[..., 1] > 40
        if ROLE_COLOR.get(role) == "magenta":
            return (diff[..., 0] > 40) & (diff[..., 2] > 40)
        return None

    crops = []
    sample_meta = []
    n_skipped_no_mask = 0
    for i, s in enumerate(samples):
        path = os.path.join(args.img_dir, s["img_filename"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        # Find the per-frame overlay PNG: build_sidecar wrote them with the
        # naming f<idx:04d>_<img_filename>. Find the one matching this frame.
        ov_dir = os.path.join(os.path.dirname(args.summary_csv), "overlays")
        ov_name = None
        if os.path.isdir(ov_dir):
            cand = f"f{s['frame_idx']:04d}_{s['img_filename']}"
            if os.path.exists(os.path.join(ov_dir, cand)):
                ov_name = cand
        if ov_name is None:
            # fall back to the loose bbox crop (no tightening possible)
            x0, y0, x1, y1 = s["bbox"]
            crop = bgr[y0:y1, x0:x1]
        else:
            m = recover_mask(os.path.join(ov_dir, ov_name), bgr, s["role"])
            if m is None or m.sum() < 100:
                n_skipped_no_mask += 1
                continue
            ys, xs = np.where(m)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            crop = bgr[y0:y1, x0:x1].copy()
            # black out background inside the tight crop
            local_mask = m[y0:y1, x0:x1]
            crop[~local_mask] = 0
        if crop.size == 0:
            continue
        crops.append(crop)
        sample_meta.append(s)
        if (i + 1) % 20 == 0:
            print(f"  [crop] {i+1}/{len(samples)}", flush=True)
    print(f"[crop] kept {len(crops)} tight crops "
          f"(skipped {n_skipped_no_mask} with no recoverable mask)", flush=True)

    # embed in small batches
    embeddings = []
    BATCH = 8
    for i in range(0, len(crops), BATCH):
        batch_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                      for c in crops[i:i + BATCH]]
        inputs = proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # CLS token = pooled embedding
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        embeddings.append(cls.cpu().numpy())
        print(f"  [embed] {min(i+BATCH, len(crops))}/{len(crops)}", flush=True)
    if not embeddings:
        print("[fatal] no embeddings", flush=True); sys.exit(1)
    E = np.vstack(embeddings)
    print(f"[embed] shape={E.shape}", flush=True)

    # cluster: cosine distance via 1 - dot since vectors are L2-normalised
    print(f"[cluster] agglomerative, distance_threshold={args.distance_threshold}",
          flush=True)
    cluster = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=args.distance_threshold,
    )
    labels = cluster.fit_predict(E)
    n_ident = len(set(labels))
    print(f"[cluster] {n_ident} stable identity cluster(s) discovered", flush=True)

    # map each (obj_id, role) -> majority cluster
    obj_to_ident = {}
    cluster_to_role = {}
    for s, lab in zip(sample_meta, labels):
        key = f"{s['key'][0]}|{s['key'][1]}"
        obj_to_ident.setdefault(key, []).append(int(lab))
    obj_to_ident_final = {}
    for k, v in obj_to_ident.items():
        # majority vote
        vals, cnts = np.unique(v, return_counts=True)
        winner = int(vals[np.argmax(cnts)])
        obj_to_ident_final[k] = winner

    # human-readable names per identity (carrot/cup heuristic from role majority)
    cluster_role_counts = {}
    for s, lab in zip(sample_meta, labels):
        cluster_role_counts.setdefault(int(lab), {}).setdefault(s["role"], 0)
        cluster_role_counts[int(lab)][s["role"]] += 1
    cluster_name = {}
    for lab, roles in cluster_role_counts.items():
        winner_role = max(roles.items(), key=lambda kv: kv[1])[0]
        if winner_role == "grasped":
            cluster_name[lab] = f"identity_{lab}_grasped"
        elif winner_role == "contact_receiver":
            cluster_name[lab] = f"identity_{lab}_contact_receiver"
        else:
            cluster_name[lab] = f"identity_{lab}"

    out_doc = {
        "summary_csv": args.summary_csv,
        "n_samples": len(sample_meta),
        "n_identities": n_ident,
        "obj_to_identity": {k: cluster_name[v] for k, v in obj_to_ident_final.items()},
        "identity_role_histogram": {cluster_name[k]: v
                                    for k, v in cluster_role_counts.items()},
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    # ----- montage of sample crops grouped by identity -----
    by_identity = {}
    for crop, lab in zip(crops, labels):
        by_identity.setdefault(int(lab), []).append(crop)
    panels = []
    target_h = 96
    for lab in sorted(by_identity.keys()):
        cs = by_identity[lab][:10]
        resized = [cv2.resize(c, (int(c.shape[1] * target_h / max(c.shape[0], 1)),
                                  target_h)) for c in cs if c.size > 0]
        if not resized:
            continue
        row = np.hstack(resized)
        banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, cluster_name[lab], (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.vstack([banner, row]))
    if panels:
        # pad all to same width
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
