"""
Constant-pixel seed for the grasped role, fitted and cross-checked on the
verified-good tracks.

WHY A CONSTANT PIXEL
--------------------
On this eye-in-hand rig the arm pose cancels out of the projection:

    p_cam = inv(T_bota_camera) @ inv(T_base_bota(t)) @ T_base_bota(t) @ p_bota
          = inv(T_bota_camera) @ p_bota                       -- no t

Verified: a fixed bota-frame offset projects to the SAME pixel on every
sample, std 0.0 px. So "seed the grasped role from a fixed 3-D offset" and
"seed it at a fixed pixel" are the same statement, and the pixel needs neither
depth nor T_bota_camera. That is why lfdws_t001, which has no depth, can be
used here although it was excluded from the earlier offset search.

The objective -- "how many held-object masks contain this pixel" -- is just
the sum of the masks, so the whole image is scored at once. No grid, no step
size, no search.

WHICH TRACKS ARE ADMISSIBLE, AND WHY THIS IS HARD-CODED
-------------------------------------------------------
Two automatic gates were tried and BOTH were wrong, so the admission list is
hard-coded to what was verified by eye, and the evidence is recorded here.

  * "the track is stationary in the image" -- a held object is stationary on
    an eye-in-hand rig, but so is rig hardware bolted in view. lfdws_t005's
    track is the MOST stationary of all (in-hold centroid std 3.0, 6.6 px)
    and is pure hardware: a black bracket at a fixed image position while the
    scene behind it changes completely.
  * "the track moves once the object is released" -- lfdws_t005 passes this
    too (out-of-hold std 131.8, 129.1 px, ratio 19.9) because the track
    DRIFTS off the bracket after the hold. And it wrongly rejects lfdws_t001,
    whose carrot is on the table before pickup and legitimately visible.

Verified by inspecting the in-hold frames directly:
    lfdws_t001      GOOD    mask on the carrot in the gripper
    lfdws_t002_new  GOOD    mask on the cube in the gripper
    lfdws_t004      BAD     drifts across black scene hardware
                            (in-hold centroid std 64.7, 86.1 px)
    lfdws_t005      BAD     locked onto a black rig bracket, not a held object

lfdws_t004 and lfdws_t005 contain no carried object at all -- the task there
is a bolt latch on a hinged panel, moved in place -- so there is nothing to
admit, not merely a bad track.

WHY THE ESTIMATOR IS MINIMAX AND NOT THE ARGMAX
-----------------------------------------------
Each trial on its own has a pixel inside its object on 100% of hold frames,
but those two pixels are 250 px apart and neither transfers:

    fit on t001 -> pixel (762, 11) -> t002:  0/281
    fit on t002 -> pixel (515,117) -> t001: 15/177

That is not evidence against a constant pixel. The argmax sits at the deepest
point of whichever object was fitted, which is object-specific by
construction. The quantity actually wanted is the pixel that works for EVERY
recording, i.e. the one maximising the WORST per-recording rate.

VALIDATION STATUS -- READ BEFORE SHIPPING
-----------------------------------------
The minimax pixel is fitted on both recordings, so it is IN-SAMPLE. With only
two independent recordings a joint fit cannot also be held out. What is
genuine evidence is the shape of the solution set: the pixels scoring >=90% on
both form a single connected region of a few hundred pixels, not the isolated
knife-edge points that (correctly) sank Code/grasp_offset_search.py. A third
carried-object recording is what would turn this into a held-out result.

Read only apart from its own outputs. Writes the JSON, a per-trial heatmap,
and a seed CSV in auto_seed.py's column format so the propagation scripts
consume it unmodified.

Usage: .venv_analysis/bin/python Code/grasped_seed_pixel.py
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (gripper_closed_window, mask_from_overlay,
                         parse_gripper_width)

GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
RGB = "zed_zed_node_rgb_color_rect_image_compressed"
GRASPED_BGR = (0, 255, 0)
OUT_JSON = "figures/grasped_seed_pixel.json"
OUT_DIR = "figures/grasped_seed_pixel"
OUT_SEED = "figures/grasped_seed_pixel/grasped_seed.csv"

# admit: verified by eye on the in-hold frames (see docstring)
TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001",
     "figures/identify/objects_summary.csv", True, "carrot in gripper"),
    ("lfdws_t002_new", "Data/lfdws_t002_new",
     "figures/t002new/identify/objects_summary.csv", True, "cube in gripper"),
    ("lfdws_t004", "Data/lfdws_t004",
     "figures/t004/identify/objects_summary.csv", False,
     "no carried object; track drifts over scene hardware"),
    ("lfdws_t005", "Data/lfdws_t005",
     "figures/t005/identify/objects_summary.csv", False,
     "no carried object; track locked on a rig bracket"),
]

MIN_RATE = 0.90     # the level at which the solution region is reported


def accumulate(label, tdir, sidecar):
    cpath = glob.glob(os.path.join(tdir, "*_0.csv"))
    if not (cpath and os.path.exists(sidecar)):
        print(f"[skip] {label}: missing inputs", flush=True)
        return None
    rows = list(csv.DictReader(open(cpath[0])))
    if GRIP not in rows[0]:
        print(f"[skip] {label}: no gripper topic", flush=True)
        return None
    w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
    closed = gripper_closed_window(w)
    if not closed.any():
        print(f"[skip] {label}: gripper span guard tripped", flush=True)
        return None
    held = {}
    for i, r in enumerate(rows):
        iid = str(r[IMG])
        held[iid] = held.get(iid, False) or bool(closed[i])

    acc, n, cents, first = None, 0, [], None
    for r in csv.DictReader(open(sidecar)):
        if r["role"] != "grasped" or float(r["mask_px"]) <= 0:
            continue
        iid = os.path.splitext(r["img_filename"])[0]
        if not held.get(iid, False):
            continue
        rgb = os.path.join(tdir, RGB, f"{iid}.png")
        if not os.path.exists(rgb):
            continue
        m = mask_from_overlay(r["overlay_path"], rgb, GRASPED_BGR)
        if m is None or not m.any():
            continue
        if acc is None:
            acc = np.zeros(m.shape, np.int32)
        acc += m.astype(np.int32)
        ys, xs = np.nonzero(m)
        cents.append((xs.mean(), ys.mean()))
        if first is None:
            first = iid
        n += 1
        if n % 50 == 0:
            print(f"    [{label}] {n} in-hold frames", flush=True)
    if not n:
        print(f"[skip] {label}: no usable in-hold frames", flush=True)
        return None
    c = np.array(cents)
    return {"label": label, "acc": acc, "n": n, "first_img": first,
            "std": (float(c[:, 0].std()), float(c[:, 1].std()))}


def main():
    print("[note] the grasped seed is a CONSTANT PIXEL on this eye-in-hand "
          "rig; the arm pose cancels out of the projection\n", flush=True)
    ok, rej = [], []
    for label, tdir, sidecar, admit, why in TRIALS:
        if not admit:
            rej.append((label, why))
            print(f"[reject] {label}: {why}", flush=True)
            continue
        print(f"[load] {label}", flush=True)
        d = accumulate(label, tdir, sidecar)
        if d is None:
            continue
        d["why"] = why
        by, bx = np.unravel_index(int(d["acc"].argmax()), d["acc"].shape)
        d["own_best"] = (int(bx), int(by))
        print(f"  {label:16s} {d['n']:4d} in-hold frames  centroid std "
              f"({d['std'][0]:5.1f},{d['std'][1]:5.1f}) px  own best pixel "
              f"({bx},{by}) = {d['acc'][by,bx]}/{d['n']}\n", flush=True)
        ok.append(d)

    if len(ok) < 2:
        print("[fatal] need at least two admitted recordings", flush=True)
        return
    shape = ok[0]["acc"].shape
    if any(d["acc"].shape != shape for d in ok):
        print("[fatal] frame sizes differ across recordings", flush=True)
        return

    print("[naive] fit the argmax on one recording, test on the other",
          flush=True)
    for a in ok:
        for b in ok:
            if a is b:
                continue
            x, y = a["own_best"]
            h = int(b["acc"][y, x])
            print(f"  fit {a['label']:16s} -> ({x},{y}) -> test "
                  f"{b['label']:16s} {h}/{b['n']} ({h/b['n']:.1%})", flush=True)
    print("  the argmax is object-specific, so this failing is expected and "
          "is not evidence against a constant pixel\n", flush=True)

    rates = [d["acc"].astype(np.float64) / d["n"] for d in ok]
    worst = np.minimum.reduce(rates)
    by, bx = np.unravel_index(int(worst.argmax()), worst.shape)
    print(f"[minimax] pixel that maximises the WORST per-recording rate: "
          f"({bx},{by})", flush=True)
    for d, r in zip(ok, rates):
        print(f"    {d['label']:16s} {r[by,bx]:.1%}  "
              f"({int(d['acc'][by,bx])}/{d['n']} in-hold frames)", flush=True)

    sel = worst >= MIN_RATE
    n_sel = int(sel.sum())
    print(f"\n[region] pixels scoring >= {MIN_RATE:.0%} on EVERY recording: "
          f"{n_sel}", flush=True)
    if n_sel:
        ys, xs = np.nonzero(sel)
        nlab, _, stats, _ = cv2.connectedComponentsWithStats(
            sel.astype(np.uint8), 8)
        big = max(range(1, nlab), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        print(f"    spans x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}], "
              f"{nlab-1} component(s), largest {stats[big, cv2.CC_STAT_AREA]} px",
              flush=True)
        print("    a connected region of this size is what a real rig "
              "constant looks like; isolated points are what overfitting "
              "looks like (see Code/grasp_offset_search.py)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    vis = cv2.applyColorMap((255 * worst).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.drawMarker(vis, (bx, by), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.imwrite(os.path.join(OUT_DIR, "minimax_heatmap.png"), vis)

    with open(OUT_SEED, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["role", "event", "img_id", "seed_x", "seed_y",
                      "frac_x", "frac_y", "mask_px"])
        for d in ok:
            wtr.writerow(["grasped", "grasp", d["first_img"], bx, by,
                          round(bx / shape[1], 4), round(by / shape[0], 4),
                          int(d["acc"][by, bx])])
    json.dump({"seed_pixel": [int(bx), int(by)],
               "per_recording": {d["label"]: {
                   "rate": float(r[by, bx]), "n_hold_frames": d["n"],
                   "own_best_pixel": list(d["own_best"])}
                   for d, r in zip(ok, rates)},
               "region_min_rate": MIN_RATE,
               "region_px": n_sel,
               "rejected": [{"trial": t, "why": w} for t, w in rej],
               "held_out_validated": False,
               "note": ("fitted jointly on all admitted recordings, so "
                        "in-sample; a third carried-object recording is "
                        "needed for held-out validation")},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n[write] {OUT_JSON}\n[write] {OUT_SEED}\n"
          f"[write] {OUT_DIR}/minimax_heatmap.png", flush=True)
    print("\n[status] IN-SAMPLE. Two recordings cannot both fit and hold out. "
          "Usable as a seed; not yet a validated rig constant.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
