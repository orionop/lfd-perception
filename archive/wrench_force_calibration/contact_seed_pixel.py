"""
Constant-pixel seed for the CONTACT role, and the test of whether the camera
extrinsic can leave the pipeline entirely.

THE ARGUMENT
------------
On this eye-in-hand rig the arm pose cancels out of the projection:

    p_cam = inv(T_base_bota(t) @ T_bota_camera) @ (T_base_bota(t) @ p_bota)
          = inv(T_bota_camera) @ p_bota                        -- no t

Verified numerically (four random poses give byte-identical pixels). So any
point rigidly fixed to the hand projects to the SAME pixel in every frame.
Code/grasped_seed_pixel.py already exploits this: the grasped object is held
by the fingers, so its seed is one constant pixel needing neither depth nor
T_bota_camera, and it lands inside the held-object mask on 95.5% / 96.4% of
hold frames.

The contact object is NOT hand-fixed in general -- it sits in the world. But
AT THE MOMENT OF CONTACT it is touching the fingertips, which are hand-fixed.
So the same argument should make the contact seed a constant pixel too, and
only at the contact events, which is exactly where the seed is needed.

If that holds, nothing in the pipeline needs T_bota_camera, and the unresolved
bota->camera extrinsic stops being a blocker.

WHY THIS IS A FAIR TEST AND NOT A FIT
-------------------------------------
Scored on Code/contact_eval_set.py's shared 7 events -- the SAME events, with
the same ground-truth masks, that the wrench-ray projection is scored on. So
the comparison against the extrinsic-based seed is like for like, and the
wrench-ray number is recomputed live here rather than quoted.

The pixel test is strictly HARDER than the ray test: a ray hits if it crosses
the mask anywhere along its length, a pixel hits only if it is inside.

Three things are reported separately and must not be conflated:
  * in-sample     the argmax over all 7 events. Optimistic by construction.
  * minimax       the pixel maximising the WORST per-recording rate. This is
                  the estimator, for the reason given in
                  Code/grasped_seed_pixel.py: a per-recording argmax sits at
                  the deepest point of whichever object was fitted and does
                  not transfer.
  * held out      leave one RECORDING out, fit the minimax pixel on the other
                  two, score on the held-out one. Three recordings, so unlike
                  the grasped case this is actually possible.

The size and connectedness of the solution region is reported too. An isolated
knife-edge optimum is what sank Code/grasp_offset_search.py; a broad connected
region is evidence, a single pixel is not.

HONEST LIMITS
-------------
Seven events across three recordings, and the five events inside
lfdws_t001_depth are not independent of each other. This can show the idea
works or fails; it cannot certify a production seed on its own.

Read only apart from its own outputs. Does not touch calibration.yaml or any
existing pipeline artifact.

Usage: .venv_analysis/bin/python Code/contact_seed_pixel.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from verify_mark_transform import load_K, score as ray_score

OUT_JSON = "figures/contact_seed_pixel.json"
OUT_PNG = "figures/contact_seed_pixel.png"
CALIB = "calibration.yaml"
MIN_RATE = 0.60          # level at which the solution region is measured


def load_T():
    import yaml
    with open(CALIB) as f:
        c = yaml.safe_load(f)
    return np.array(c["bota_to_camera"]["T"], dtype=float)


def per_recording_rates(events, shape):
    """{recording: float array of per-pixel hit rate over its events}."""
    out = {}
    for r in sorted({e["trial"] for e in events}):
        evs = [e for e in events if e["trial"] == r]
        acc = np.zeros(shape, np.float64)
        for e in evs:
            acc += e["mask"].astype(np.float64)
        out[r] = acc / float(len(evs))
    return out


def minimax_pixel(rates):
    """Pixel maximising the worst per-recording rate; ties broken by total
    rate, then by being most interior to the optimal region."""
    stack = np.stack(list(rates.values()))
    worst = stack.min(axis=0)
    total = stack.sum(axis=0)
    best = worst.max()
    cand = (worst >= best - 1e-12)
    tot_best = total[cand].max()
    cand &= (total >= tot_best - 1e-12)
    ys, xs = np.nonzero(cand)
    cx, cy = xs.mean(), ys.mean()
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return (int(xs[i]), int(ys[i])), float(best), int(cand.sum())


def hits_at(events, px):
    x, y = px
    return [bool(e["mask"][y, x]) for e in events]


def main():
    print("[note] testing whether the CONTACT seed is a constant pixel, "
          "which would remove T_bota_camera from the pipeline\n", flush=True)

    events = build_events(verbose=True)
    recs = sorted({e["trial"] for e in events})
    shapes = {(e["H"], e["W"]) for e in events}
    print(f"\n[events] {len(events)} events, {len(recs)} recordings, "
          f"frame shapes {shapes}", flush=True)
    if len(shapes) != 1:
        print("[fatal] frame sizes differ across recordings", flush=True)
        return
    shape = shapes.pop()

    print("\n[baseline] wrench ray through calibration.yaml's extrinsic, "
          "same events", flush=True)
    K, T = load_K(), load_T()
    r_rate, r_dist, r_px, r_hits = ray_score(T, events, K)
    print(f"  hit {r_rate:.3f} ({int(round(r_rate*len(events)))}/{len(events)})"
          f"  mean centroid dist {r_dist:.1f} px", flush=True)

    rates = per_recording_rates(events, shape)
    for r in recs:
        n = sum(1 for e in events if e["trial"] == r)
        by, bx = np.unravel_index(int(rates[r].argmax()), shape)
        print(f"\n[recording] {r:24s} {n} event(s)  own best pixel "
              f"({bx},{by}) = {rates[r][by,bx]:.3f}", flush=True)

    print("\n[naive] fit each recording's argmax, test on the others",
          flush=True)
    for a in recs:
        by, bx = np.unravel_index(int(rates[a].argmax()), shape)
        other = [f"{b} {rates[b][by,bx]:.2f}" for b in recs if b != a]
        print(f"  from {a:24s} ({bx},{by}) -> " + ", ".join(other), flush=True)

    px, worst, region = minimax_pixel(rates)
    hits = hits_at(events, px)
    print(f"\n[minimax] pixel {px}  worst per-recording rate {worst:.3f}",
          flush=True)
    print(f"[minimax] in-sample {sum(hits)}/{len(hits)} events  "
          f"{''.join('1' if h else '0' for h in hits)}", flush=True)
    print(f"[minimax] optimal region {region} px", flush=True)
    stack = np.stack(list(rates.values()))
    broad = int((stack.min(axis=0) >= MIN_RATE).sum())
    print(f"[region]  pixels with every recording >= {MIN_RATE}: {broad}",
          flush=True)

    print("\n[held-out] leave one RECORDING out", flush=True)
    ho = []
    for r in recs:
        sub = {k: v for k, v in rates.items() if k != r}
        p, _, _ = minimax_pixel(sub)
        te = [e for e in events if e["trial"] == r]
        h = hits_at(te, p)
        ho.append((r, p, sum(h), len(h)))
        print(f"  hold out {r:24s} n={len(h)}  fitted pixel {p}  "
              f"held-out {sum(h)}/{len(h)}", flush=True)
    tot_h = sum(a for _, _, a, _ in ho)
    tot_n = sum(b for _, _, _, b in ho)
    print(f"  held-out total {tot_h}/{tot_n} = {tot_h/tot_n:.3f}", flush=True)

    print("\n[verdict]", flush=True)
    if tot_h / tot_n >= r_rate:
        print("  A constant pixel matches or beats the extrinsic-based ray on")
        print("  held-out recordings. T_bota_camera is not needed for seeding.")
    elif tot_h / tot_n >= 0.5:
        print("  A constant pixel works but is behind the ray on held-out")
        print("  events. Promising, not yet a replacement.")
    else:
        print("  The constant-pixel hypothesis FAILS for the contact role.")
        print("  The contact object is not sufficiently hand-fixed at contact;")
        print("  keep the projected seed and the extrinsic with it.")

    os.makedirs("figures", exist_ok=True)
    for p in (OUT_JSON, OUT_PNG):
        if os.path.exists(p):
            import shutil
            shutil.copy2(p, p + ".bak")
    with open(OUT_JSON, "w") as f:
        json.dump({
            "seed_pixel": list(px),
            "worst_per_recording_rate": worst,
            "in_sample_hits": f"{sum(hits)}/{len(hits)}",
            "per_event_hits": "".join("1" if h else "0" for h in hits),
            "optimal_region_px": region,
            f"region_px_all_recordings_ge_{MIN_RATE}": broad,
            "held_out": [{"recording": r, "pixel": list(p), "hits": a,
                          "n": b} for r, p, a, b in ho],
            "held_out_total": f"{tot_h}/{tot_n}",
            "wrench_ray_baseline": {
                "hit_rate": r_rate,
                "mean_centroid_dist_px": r_dist,
                "per_event_hits": "".join("1" if h else "0" for h in r_hits),
                "note": "same events, uses calibration.yaml T_bota_camera",
            },
            "n_events": len(events),
            "n_recordings": len(recs),
            "caveat": ("7 events; the 5 events inside lfdws_t001_depth are "
                       "not independent of each other"),
        }, f, indent=2)
    print(f"\n[write] {OUT_JSON}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(recs)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.0))
    for ax, r in zip(axes, recs):
        ax.imshow(rates[r], cmap="magma", vmin=0, vmax=1)
        ax.plot(*px, "c+", ms=12, mew=2)
        ax.set_title(r, fontsize=8)
        ax.axis("off")
    axes[-1].imshow(np.stack(list(rates.values())).min(axis=0), cmap="magma",
                    vmin=0, vmax=1)
    axes[-1].plot(*px, "c+", ms=12, mew=2)
    axes[-1].set_title(f"worst-case, seed {px}", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Contact-role constant-pixel seed: per-recording hit rate",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[write] {OUT_PNG}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
