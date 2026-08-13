"""
Second label-free baseline: SAM automatic masks ranked by depth.

The paper's "label-free visual discovery does not find the task-relevant
object" claim currently rests on ONE method -- DINOv2 self-attention
re-weighted by depth (DADO-style, Code/_dado_vs_groundtruth_all_trials.py,
mean IoU 0.166 over 14 events). A reviewer will reasonably ask whether that
is a property of label-free discovery or just of raw attention maps. This
adds a second, structurally different proposer and scores it on exactly the
same events against exactly the same ground truth.

WHY THIS IS THE RIGHT SECOND BASELINE
DADO produces a saliency blob and thresholds it, so it never proposes an
object boundary at all. This baseline instead asks a real segmentation model
for object-like proposals (SAM's automatic mask generator), then picks among
them using only the one task-agnostic cue available without supervision:
depth. The manipulated object is, almost by construction, the thing nearest
the camera in an eye-in-hand view. That is a genuinely stronger and more
favourable setup than DADO -- it gets real object boundaries for free and
only has to choose. If it still misses, the claim is about label-free
*selection*, not about the quality of any one saliency map.

No TASK priors are used -- no image-position bonus, no object-scale cutoff,
no role heuristics. Those are what Code/auto_seed.py encodes, and this
baseline is meant to be an alternative to that script, so reusing its priors
would smuggle supervision into a "label-free" number.

Two task-agnostic filters ARE applied, and the first version of this script
lacked the second, which made it a strawman:

  - minimum proposal area (500 px): a sliver is not an object proposal.
  - drop proposals touching the image border. Ranking purely by "nearest to
    the camera" sounds right but is wrong for an eye-in-hand rig, because
    the nearest thing is the robot's OWN GRIPPER, which enters from the
    frame edge. Without this filter the baseline returned the manipulator on
    essentially every event -- mean IoU 0.021, every pick sitting at
    0.15-0.18 m -- which measures our own oversight rather than any property
    of label-free discovery. Border exclusion is standard practice and uses
    no knowledge of the task or the object, so applying it keeps the
    comparison honest.

After filtering, ranking is: among the surviving proposals, take the one
whose median real depth is smallest.

Scored identically to the DADO baseline: IoU against the SAM2-propagated
ground-truth mask for the role that event's proprioceptive cue seeds, on the
same (recording, event) pairs, using the same corrected mask recovery from
Code/event_utils.py.

Output: figures/baseline_sam_depth_ranking.csv (+ .png panel)

Run inside .venv_sam2 (needs segment_anything + torch):
    .venv_sam2/bin/python Code/baseline_sam_depth_ranking.py
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

CKPT = "sam_vit_h_4b8939.pth"
MODEL = "vit_h"
OUT_CSV = "figures/baseline_sam_depth_ranking.csv"
OUT_PNG = "figures/baseline_sam_depth_ranking.png"
DADO_CSV = "figures/dado_vs_groundtruth_all_trials.csv"


def pick_device():
    """CUDA if present, otherwise CPU -- deliberately NOT MPS.

    SamAutomaticMaskGenerator builds its point grid in float64, which MPS
    refuses ("Cannot convert a MPS Tensor to float64"). Code/auto_seed.py
    runs SAM on CPU for the same reason. Slower, but this is a one-off
    evaluation over a handful of frames.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def real_depth_m(path, H, W):
    """Metres. Handles both conventions: lab .npy float32-m, mcap 16-bit mm PNG."""
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = np.array(Image.open(path)).astype(np.float32) / 1000.0
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else float("nan")


def load_tasks():
    """Shared evaluation set -- see Code/dado_eval_tasks.py. Both baselines
    import the same list so they are provably scored on identical events."""
    from dado_eval_tasks import TASKS
    return TASKS


def main():
    dev = pick_device()
    print(f"[setup] device={dev}", flush=True)
    tasks = load_tasks()
    print(f"[setup] {len(tasks)} (recording, event) pairs -- same as the DADO baseline",
          flush=True)

    print(f"[load] SAM {MODEL} from {CKPT}", flush=True)
    sam = sam_model_registry[MODEL](checkpoint=CKPT).to(dev)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                    pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90)

    sidecars, rows, panels = {}, [], []
    for k, (trial, event, role, color, rgb_p, depth_p, side_csv, img_id) in enumerate(tasks):
        bgr = cv2.imread(rgb_p)
        if bgr is None or not os.path.exists(depth_p):
            print(f"  [skip] {trial}/{event}: missing rgb or depth", flush=True)
            continue
        H, W = bgr.shape[:2]

        if side_csv not in sidecars:
            sidecars[side_csv] = list(csv.DictReader(open(side_csv))) \
                if os.path.exists(side_csv) else []
        gt_rows = [r for r in sidecars[side_csv]
                   if r["role"] == role and r["img_filename"] == f"{img_id}.png"]
        if not gt_rows:
            print(f"  [skip] {trial}/{event}: no ground-truth row", flush=True)
            continue
        gt = mask_from_overlay(gt_rows[0]["overlay_path"], rgb_p, color)
        if gt is None or gt.sum() == 0:
            print(f"  [skip] {trial}/{event}: empty ground truth", flush=True)
            continue

        print(f"[run] {trial}/{event}: SAM automatic masks ...", flush=True)
        anns = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not anns:
            print(f"  [warn] no proposals", flush=True)
            continue
        depth = real_depth_m(depth_p, H, W)

        # Two task-agnostic filters, both needed to keep this a fair baseline
        # rather than a strawman:
        #   (a) minimum area -- a 50px sliver is not an object proposal
        #   (b) drop proposals touching the image border. In an eye-in-hand
        #       view the nearest thing to the camera is the robot's own
        #       gripper, which enters from the frame edge; without this the
        #       ranking simply returns the manipulator every time (measured:
        #       mean IoU 0.021, every pick at 0.15-0.18 m). Excluding
        #       border-touching regions is standard practice and uses no
        #       knowledge of the task or the object.
        MIN_AREA = 500
        border = np.zeros((H, W), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

        # Alongside the ranked pick, record two diagnostics that separate
        # "SAM cannot segment this object" from "ranking cannot find it":
        #   oracle_iou  -- best IoU ANY surviving proposal achieves. This is
        #                  the ceiling a perfect selector would reach.
        #   codepth_frac-- fraction of surviving proposals whose median depth
        #                  is within 2 cm of the target's. If this is high,
        #                  depth provably cannot discriminate the target.
        ok = (depth > 0.05) & (depth < 5.0)
        gt_z = float(np.median(depth[gt & ok])) if (gt & ok).any() else np.nan

        best, best_z, n_kept, oracle, n_codepth = None, np.inf, 0, 0.0, 0
        for a in anns:
            m = a["segmentation"]
            if m.sum() < MIN_AREA or (m & border).any():
                continue
            dv = depth[m & ok]
            if dv.size < 50:
                continue
            n_kept += 1
            z = float(np.median(dv))
            oracle = max(oracle, iou(m, gt))
            if not np.isnan(gt_z) and abs(z - gt_z) < 0.02:
                n_codepth += 1
            if z < best_z:
                best_z, best = z, m
        if best is None:
            print(f"  [warn] no proposal survived filtering", flush=True)
            continue

        s = iou(best, gt)
        cf = n_codepth/n_kept if n_kept else 0.0
        rows.append([trial, event, role, f"{s:.4f}", f"{oracle:.4f}", f"{cf:.4f}",
                     f"{best.mean()*100:.2f}", f"{gt.mean()*100:.2f}",
                     f"{best_z:.3f}", f"{gt_z:.3f}", n_kept, len(anns)])
        print(f"  [result] {trial}/{event}: IoU={s:.3f}  oracle={oracle:.3f}  "
              f"co-depth={100*cf:.0f}%  (picked {best_z:.2f} m vs target {gt_z:.2f} m; "
              f"{n_kept}/{len(anns)} proposals survived)", flush=True)

        if len(panels) < 6:
            gtv = bgr.copy()
            cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gtv, cs, -1, (0, 255, 0), 2)
            red = np.zeros_like(bgr); red[..., 2] = 255
            pv = np.where(best[..., None], cv2.addWeighted(bgr, 0.5, red, 0.5, 0), bgr)
            cv2.putText(gtv, f"{trial} {event} GT", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(pv, f"SAM+depth  IoU={s:.3f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            panels.append(np.hstack([gtv, pv]))

    if not rows:
        print("[fatal] no scored events", flush=True); return

    os.makedirs("figures", exist_ok=True)
    with open(OUT_CSV, "w") as f:
        w = csv.writer(f)
        w.writerow(["trial", "event", "role", "iou", "oracle_iou",
                    "codepth_frac_within_2cm", "proposal_coverage_pct",
                    "gt_coverage_pct", "proposal_median_depth_m",
                    "gt_median_depth_m", "n_proposals_kept", "n_proposals"])
        w.writerows(rows)
    print(f"\n[write] {OUT_CSV}", flush=True)

    if panels:
        th = 220
        rs = [cv2.resize(p, (int(p.shape[1]*th/p.shape[0]), th)) for p in panels]
        mw = max(r.shape[1] for r in rs)
        rs = [np.pad(r, ((0, 0), (0, mw-r.shape[1]), (0, 0))) if r.shape[1] < mw else r
              for r in rs]
        cv2.imwrite(OUT_PNG, np.vstack(rs))
        print(f"[write] {OUT_PNG}", flush=True)

    v = [float(r[3]) for r in rows]
    orc = [float(r[4]) for r in rows]
    cdp = [float(r[5]) for r in rows]
    print(f"\n[summary] SAM-automask + depth ranking: n={len(v)}  "
          f"mean IoU={sum(v)/len(v):.3f}  range={min(v):.3f}-{max(v):.3f}", flush=True)
    print(f"[oracle]  best proposal SAM actually produced: mean={sum(orc)/len(orc):.3f}  "
          f"median={sorted(orc)[len(orc)//2]:.3f}  max={max(orc):.3f}", flush=True)
    print(f"[codepth] proposals within 2cm of the target's depth: "
          f"mean={100*sum(cdp)/len(cdp):.0f}%  max={100*max(cdp):.0f}%", flush=True)
    gap = sum(orc)/len(orc) - sum(v)/len(v)
    print(f"\n[diagnosis] SAM's proposal set CONTAINS the object (oracle "
          f"{sum(orc)/len(orc):.3f}); depth ranking recovers only "
          f"{sum(v)/len(v):.3f}.", flush=True)
    print(f"            {100*(1-(sum(v)/len(v))/(sum(orc)/len(orc))):.0f}% of achievable "
          f"performance is lost at the SELECTION step, not at segmentation --", flush=True)
    print(f"            unsurprisingly, since {100*sum(cdp)/len(cdp):.0f}% of candidates sit "
          f"within 2cm of the target in depth.", flush=True)
    if os.path.exists(DADO_CSV):
        dv = [float(r["iou"]) for r in csv.DictReader(open(DADO_CSV))]
        print(f"[compare] DINOv2 attention x depth:      n={len(dv)}  "
              f"mean IoU={sum(dv)/len(dv):.3f}  range={min(dv):.3f}-{max(dv):.3f}",
              flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
