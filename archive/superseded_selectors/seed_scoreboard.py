"""
One scored table: every seeding method x every trial x every object role.

WHY
---
Three seeding methods are now in the repo and only one of them has ever been
quantified:

  auto_seed.py        vision-only SAM-automatic-mask heuristic. The original.
                      NEVER SCORED. Its one generalisation attempt is visible
                      in figures/identify/auto_seeds_VERIFY_generalized.csv,
                      which took contact_receiver from 64,297 px to 287,734 px
                      -- it grabbed background -- and was reverted.
  constant pixel      grasped_seed_pixel.py (works, 95.5%/96.4% in-sample) and
                      contact_seed_pixel.py (fails, held-out 3/7).
  projected           the wrench line of action through calibration.yaml's
                      T_bota_camera. 6/7 on the shared contact events.

"Does auto seeding work?" currently has no answer. This produces one.

THE CRITERION
-------------
A seed is scored on the only thing that matters downstream: does the seed
pixel fall inside the ground-truth mask of the role it is meant to seed, on
the event frame it is placed at. Distance to the mask centroid is reported
alongside, because a near miss and a wild miss are different failures.

Ground truth is the propagated track in each trial's objects_summary.csv,
recovered through event_utils.mask_from_overlay so the caption-colour fix
applies.

ADMISSIBILITY IS EXPLICIT, NOT AUTOMATIC
----------------------------------------
lfdws_t004 and lfdws_t005 have a "grasped" track but no carried object -- the
task there is a bolt latch on a hinged panel. Their tracks sit on scene
hardware (verified by eye, see Code/grasped_seed_pixel.py's docstring). Those
cells are reported as NO GROUND TRUTH, not as scores. Scoring a seeder against
a track that is wrong would reward landing on the wrong thing.

lfdws_t002_labexport is the same physical bag as lfdws_t002_new, so the two
are marked as one recording and must not be counted as independent evidence.

TWO PASSES
----------
    .venv_sam2/bin/python     Code/seed_scoreboard.py --run_auto_seed
    .venv_analysis/bin/python Code/seed_scoreboard.py

The first runs auto_seed.py on every admissible trial (SAM 1 ViT-H, needs the
sam2 venv and the checkpoint) and caches its seeds. The second scores whatever
is cached and prints the table. Missing cells are printed as missing rather
than silently dropped.

Writes only into figures/seed_scoreboard/. Touches no pipeline artifact.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "figures/seed_scoreboard"
OUT_CSV = os.path.join(OUT_DIR, "scoreboard.csv")
RGB = "zed_zed_node_rgb_color_rect_image_compressed"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

COLOR = {
    "grasped": (0, 255, 0),
    "contact_receiver": (255, 0, 255),
    "tool_contact": (0, 165, 255),
    "charger_contact": (0, 215, 255),
}

# label, trial dir, sidecar, {role: (admit, why)}, bag-identity group
TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001",
     "figures/identify/objects_summary.csv",
     {"grasped": (True, "carrot in gripper"),
      "contact_receiver": (True, "cup")}, "t001"),
    ("lfdws_t002_new", "Data/lfdws_t002_new",
     "figures/t002new/identify/objects_summary.csv",
     {"grasped": (True, "cube in gripper")}, "t002"),
    ("lfdws_t002_labexport", "Data/lfdws_t002_labexport/lfdws_t002",
     "figures/t002labexport/identify/objects_summary.csv",
     {"grasped": (True, "cube in gripper; SAME BAG as lfdws_t002_new")},
     "t002"),
    ("lfdws_t001_depth", "Data/lfdws_t001_depth",
     "figures/identify_depth_multi/objects_summary.csv",
     {"contact_receiver": (True, "plate")}, "t001_depth"),
    ("lfdws_t001_labexport", "Data/lfdws_t001_labexport/lfdws_t001",
     "figures/t001labexport/identify/objects_summary.csv",
     {"contact_receiver": (True, "latch")}, "t001_lab"),
    ("lfdws_t004", "Data/lfdws_t004",
     "figures/t004/identify/objects_summary.csv",
     {"grasped": (False, "no carried object; track drifts over hardware")},
     "t004"),
    ("lfdws_t005", "Data/lfdws_t005",
     "figures/t005/identify/objects_summary.csv",
     {"grasped": (False, "no carried object; track on a rig bracket")},
     "t005"),
]


def merged_csv(tdir):
    for pat in ("*_0.csv", "*.csv"):
        hits = [p for p in sorted(glob.glob(os.path.join(tdir, pat)))
                if not os.path.basename(p).startswith("config")]
        if hits:
            return hits[0]
    return None


def auto_seed_path(label):
    return os.path.join(OUT_DIR, f"auto_seeds_{label}.csv")


def run_auto_seed():
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, tdir, _, _, _ in TRIALS:
        out = auto_seed_path(label)
        if os.path.exists(out):
            print(f"[skip] {label}: cached at {out}", flush=True)
            continue
        cmd = [sys.executable, "Code/auto_seed.py", "--trial", tdir,
               "--out_csv", out,
               "--out_overlay", os.path.join(OUT_DIR, f"auto_seeds_{label}.png")]
        print(f"\n[run ] {label}\n       {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        print(f"[{'ok  ' if r.returncode == 0 else 'FAIL'}] {label} "
              f"rc={r.returncode}", flush=True)


def gt_mask(tdir, sidecar, role, img_id):
    """Ground-truth mask for one role on one frame, or None."""
    from event_utils import mask_from_overlay
    for r in csv.DictReader(open(sidecar)):
        if r["role"] != role:
            continue
        if os.path.splitext(r["img_filename"])[0] != str(img_id):
            continue
        if float(r["mask_px"]) <= 0:
            return None
        rgb = os.path.join(tdir, RGB, f"{img_id}.png")
        if not os.path.exists(rgb):
            return None
        return mask_from_overlay(r["overlay_path"], rgb, COLOR[role])
    return None


def judge(mask, x, y):
    H, W = mask.shape
    if not (0 <= x < W and 0 <= y < H):
        return "OUT-OF-FRAME", float("nan")
    ys, xs = np.nonzero(mask)
    d = float(np.hypot(x - xs.mean(), y - ys.mean())) if len(xs) else float("nan")
    return ("HIT" if mask[int(y), int(x)] else "miss"), d


def const_pixels():
    out = {}
    for path, role in (("figures/grasped_seed_pixel.json", "grasped"),
                       ("figures/contact_seed_pixel.json", "contact_receiver")):
        if os.path.exists(path):
            out[role] = tuple(json.load(open(path))["seed_pixel"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_auto_seed", action="store_true")
    args = ap.parse_args()
    if args.run_auto_seed:
        run_auto_seed()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    const = const_pixels()
    print(f"[const] constant pixels: {const}\n", flush=True)

    rows = []
    print(f"{'trial':22s} {'role':17s} {'seeder':12s} {'seed':>12s} "
          f"{'verdict':13s} {'dist px':>8s}", flush=True)
    print("-" * 92, flush=True)
    for label, tdir, sidecar, roles, group in TRIALS:
        if not os.path.exists(sidecar):
            print(f"{label:22s} {'-':17s} {'-':12s} sidecar missing", flush=True)
            continue
        aseeds = {}
        ap_ = auto_seed_path(label)
        if os.path.exists(ap_):
            for r in csv.DictReader(open(ap_)):
                aseeds[r["role"]] = (float(r["seed_x"]), float(r["seed_y"]),
                                     str(r["img_id"]))
        for role, (admit, why) in roles.items():
            if not admit:
                print(f"{label:22s} {role:17s} {'-':12s} "
                      f"{'':>12s} NO GROUND TRUTH  ({why})", flush=True)
                rows.append([label, group, role, "-", "", "",
                             "NO_GROUND_TRUTH", "", why])
                continue
            # the frame to judge on: auto_seed's event frame if we have it
            img_id = aseeds.get(role, (None, None, None))[2]
            if img_id is None:
                print(f"{label:22s} {role:17s} {'auto_seed':12s} "
                      f"{'MISSING':>12s} (run --run_auto_seed)", flush=True)
                rows.append([label, group, role, "auto_seed", "", "",
                             "MISSING", "", "no cached seed"])
                continue
            mask = gt_mask(tdir, sidecar, role, img_id)
            if mask is None:
                print(f"{label:22s} {role:17s} {'-':12s} "
                      f"{'':>12s} no GT mask on frame {img_id}", flush=True)
                rows.append([label, group, role, "-", "", img_id,
                             "NO_MASK_ON_FRAME", "", ""])
                continue
            cands = [("auto_seed", aseeds[role][0], aseeds[role][1])]
            if role in const:
                cands.append(("const_pixel", const[role][0], const[role][1]))
            for sname, x, y in cands:
                v, d = judge(mask, x, y)
                print(f"{label:22s} {role:17s} {sname:12s} "
                      f"{f'({x:.0f},{y:.0f})':>12s} {v:13s} "
                      f"{d:8.1f}", flush=True)
                rows.append([label, group, role, sname, f"({x:.0f},{y:.0f})",
                             img_id, v, f"{d:.1f}", ""])

    print("\n[tally] HIT rate per seeder, over cells with ground truth",
          flush=True)
    for sname in ("auto_seed", "const_pixel"):
        sub = [r for r in rows if r[3] == sname and r[6] in ("HIT", "miss",
                                                             "OUT-OF-FRAME")]
        if not sub:
            print(f"  {sname:12s} no scored cells", flush=True)
            continue
        h = sum(1 for r in sub if r[6] == "HIT")
        groups = sorted({r[1] for r in sub})
        print(f"  {sname:12s} {h}/{len(sub)} cells   "
              f"({len(groups)} independent recordings: {', '.join(groups)})",
              flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "bag_group", "role", "seeder", "seed_px",
                    "img_id", "verdict", "dist_to_centroid_px", "note"])
        w.writerows(rows)
    print(f"\n[write] {OUT_CSV}", flush=True)
    print("[note] cells within one bag_group are not independent evidence",
          flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
