# Negative results — measured, documented, do not re-run expecting a different answer

Nothing here is broken code. Each script ran, produced a result, and the result
was that the approach does not work. They are kept because the measurement is
the value: without them, each idea looks untried and gets attempted again.

## Image stationarity (`motion_selection_*.py`)

Hypothesis: on an eye-in-hand rig the held object is the one thing that does
*not* move in the image while the background sweeps past, so it can be selected
by minimum flow.

- `motion_selection_probe.py` — bbox-centroid proxy, found the effect on 3/5
  trials (12-28x drop in image motion per metre travelled).
- `motion_selection_flow.py` — real dense flow as a selection rule: **0.037
  mean IoU**, worse than both existing baselines.
- `motion_selection_held.py` — the proper retest: in-hold frames instead of
  event instants, contiguous-run cycle splitting, adaptive texture gate,
  camera-motion requirement. Result in `figures/motion_selection_held.csv`:
  stationarity IoU **0.000 on 14 of 16 rows**, mean ~0.010, against a
  random-pick null of ~0.033. **It loses to random.**
- `motion_selection_diagnose.py` — found that lfdws_t004's gripper-closed mask
  is several cycles on differently-sized objects, not one grasp.

Why it fails: the premise does not hold on these events. The true object's flow
is *higher* than the background's on most frames (e.g. t004 18.2 px vs 7.5 px
background; t005 22.5 vs 12.9).

## Object identity across trials (`object_identity*.py`)

DINOv2-embedding clustering for stable object identity. Within one trial SAM 2's
own `obj_id` already works, which is all the deliverable needs. Across trials it
fragments and cross-contaminates; `object_identity_cross_trial_sweep.py` swept
6 distance thresholds x 2 crop styles and found **zero clean configurations**.

## DADO-style label-free proposer (`run_dado.py`, `_dado_*.py`)

DINOv2 attention x Depth-Anything depth as a label-free object proposer, scored
against the propagated ground-truth mask. **Mean IoU ~0.16 across 16 events**
(`_dado_vs_groundtruth_all_trials.py`). Not competitive.

## Depth ranking baseline (`baseline_sam_depth_ranking.py`)

Established that discovery is not limited by segmentation: SAM's proposal set
contains the right object (oracle IoU 0.612 mean) while depth ranking recovers
only 0.104. **83% of achievable performance is lost at selection** — the finding
that framed everything after it.

## Note on re-running

These import `event_utils` and some import `select_objects`, which now live one
directory up in `Code/`. They will need a `sys.path` entry pointing at `Code/`
to run from here.
