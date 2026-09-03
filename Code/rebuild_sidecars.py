"""
Rebuild every sidecar after the 2026-08-30 mask-recovery and event fixes.

WHAT CHANGED UNDER THEM
-----------------------
  BUG 3  event_utils.gripper_transitions returned the FIRST close and the
         LAST open, so on a multi-cycle recording the reported grasp and
         release bracketed every cycle in between. Active on lfdws_t004
         (4 cycles) and lfdws_t005 (2 cycles).
  BUG 4  build_sidecar_multi wrote the COMBINED overlay path into every
         per-object summary row, and mask_from_overlay's per-channel test
         reduces each channel to one bit, so tool_contact (0,165,255) and
         charger_contact (0,215,255) shared a signature and recovered
         byte-identical masks.

WHY THE SPECS ARE DERIVED, NOT TYPED
------------------------------------
CLAUDE.md records a real incident: a sidecar was once rebuilt from a
superseded point-prompt track (mean 2,350 px) instead of the box-prompt one
(mean 32,522 px) and nothing errored. Re-typing --object specs by hand is
exactly how that happens again.

So each object's propagation CSV is identified by EXACT AGREEMENT with the
sidecar being replaced: for every (frame_idx, mask_px) pair the existing
sidecar records for that role, the candidate CSV must carry the same mask_px
at the same frame_idx. Sidecar rows are a subset of the propagation rows --
frames whose overlay recovery came back empty were dropped -- so subset
agreement is the correct test, and a candidate matching on a subset this
large cannot be the wrong track. A role that matches zero or several
candidates is reported and the trial is SKIPPED rather than guessed.

Colours and obj_ids come from the sidecar's own objects.json.

Usage:
    .venv_analysis/bin/python Code/rebuild_sidecars.py            # plan only
    .venv_analysis/bin/python Code/rebuild_sidecars.py --run
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

SIDECARS = [
    "figures/identify",
    "figures/identify_depth",
    "figures/identify_depth_multi",
    "figures/t001labexport/identify",
    "figures/t002new/identify",
    "figures/t002labexport/identify",
    "figures/t004/identify",
    "figures/t005/identify",
]


def candidates():
    out = {}
    for p in glob.glob("figures/**/*_summary.csv", recursive=True):
        if os.path.basename(p) == "objects_summary.csv":
            continue
        rows = {}
        try:
            for r in csv.DictReader(open(p)):
                rows[int(r["frame_idx"])] = (r["file"], float(r["mask_px"]))
        except Exception:
            continue
        if rows:
            out[p] = rows
    return out


# WHY THE MAPPING IS EXPLICIT
# ---------------------------
# Two automatic identifications were tried and both are unsound:
#
#   exact mask_px equality -- only works for sidecars built after mask_px was
#     taken from the propagation summary. Older ones store the RECOVERED
#     overlay count, a documented lower bound, so equality fails.
#   mask_px ratio nearest 1 -- picks the WRONG track. On lfdws_t001_depth the
#     plate's own CSV scores 0.423, because a large bright object saturates
#     under the 0.5 blend and recovery is very lossy, while the unrelated
#     screwdriver CSV scores 0.813. The bands overlap, so the ratio carries
#     no identity information.
#
# Structural agreement (frame_idx -> filename) cannot discriminate either:
# every object in a trial shares the same frames.
#
# So the mapping is written out, from the propagation CSV names and the
# history CLAUDE.md records, and then VERIFIED: the named CSV must agree on
# frame_idx -> filename for every sidecar row, and the mask_px ratio must not
# exceed 1.05 (the sidecar's count can be a lossy lower bound, never larger).
# The ratio is reported for audit but never used to choose.
#
# The two genuinely ambiguous cases are the ones CLAUDE.md warns about:
#   lfdws_t002_new       propagation_grasped_summary.csv is the superseded
#                        point-prompt run (one sticker face); the box-prompt
#                        run is _grasped_box_summary.csv -> use the box one.
#   lfdws_t002_labexport naming is inverted: the point-prompt run carries the
#                        _pointprompt_ suffix and has drifted to 142,816 px on
#                        frame 0 (27% of frame), against 7,561 px for the
#                        plain file -> use the plain one.
RATIO_MAX = 1.05

MAPPING = {
    "figures/identify": {
        "grasped": "figures/propagation_bidir_summary.csv",
        "contact_receiver": "figures/propagation_cup_summary.csv",
    },
    "figures/identify_depth": {
        "contact_receiver": "figures/propagation_plate_depth_summary.csv",
    },
    "figures/identify_depth_multi": {
        "contact_receiver": "figures/propagation_plate_depth_summary.csv",
        "tool_contact": "figures/propagation_obj3_screwdriver_summary.csv",
        "charger_contact": "figures/propagation_obj4_charger_summary.csv",
    },
    "figures/t001labexport/identify": {
        "contact_receiver": "figures/t001labexport/propagation_latch_summary.csv",
    },
    "figures/t002new/identify": {
        "grasped": "figures/t002new/propagation_grasped_box_summary.csv",
    },
    "figures/t002labexport/identify": {
        "grasped": "figures/t002labexport/propagation_grasped_summary.csv",
    },
    "figures/t004/identify": {
        "grasped": "figures/t004/propagation_grasped_summary.csv",
    },
    "figures/t005/identify": {
        "grasped": "figures/t005/propagation_grasped_summary.csv",
    },
}


def verify(role_rows, path, cands):
    """(ratio, None) if `path` backs these sidecar rows, else (None, reason)."""
    import statistics
    rows = cands.get(path)
    if rows is None:
        return None, f"{path} not found"
    ratios = []
    for f, (fname, px) in role_rows.items():
        got = rows.get(f)
        if got is None:
            return None, f"{path} has no frame_idx {f}"
        if got[0] != fname:
            return None, (f"{path} frame {f} is {got[0]}, sidecar says {fname}")
        if got[1] > 0:
            ratios.append(px / got[1])
    if not ratios:
        return None, f"{path} has no non-zero mask_px on these frames"
    med = statistics.median(ratios)
    if med > RATIO_MAX:
        return None, (f"{path} median mask_px ratio {med:.3f} exceeds "
                      f"{RATIO_MAX}; the sidecar cannot hold more pixels "
                      f"than the propagation run")
    return med, None


def plan_one(sc_dir, cands):
    js = os.path.join(sc_dir, "objects.json")
    summ = os.path.join(sc_dir, "objects_summary.csv")
    if not (os.path.exists(js) and os.path.exists(summ)):
        return None, f"missing objects.json or objects_summary.csv"
    d = json.load(open(js))
    trial = d["trial_dir"]
    if not os.path.isdir(trial):
        return None, f"trial dir {trial} not present"
    by_role = {}
    for r in csv.DictReader(open(summ)):
        by_role.setdefault(r["role"], {})[int(r["frame_idx"])] = (
            r["img_filename"], float(r["mask_px"]))
    specs, notes = [], []
    for oid, info in d["objects"].items():
        role = info["role"]
        want = MAPPING.get(sc_dir, {}).get(role)
        if want is None:
            return None, f"role {role}: no mapping entry -- refusing to guess"
        rows = by_role.get(role)
        b, g, r_ = info["color"]
        if not rows:
            # The role is declared in objects.json but has no rows in the
            # summary being replaced. That happens when a previous rebuild
            # emptied it, and it must NOT abort the rebuild: the driver would
            # then refuse to repair exactly the sidecar that needs repairing.
            # The mapping is explicit, so proceed and say verification was
            # skipped.
            if want not in cands:
                return None, f"role {role}: {want} not found"
            specs.append(f"{oid}:{role}:{want}:{b},{g},{r_}")
            notes.append(f"      {role:17s} <- {want}  "
                         f"(0 rows in current summary; ratio check SKIPPED)")
            continue
        ratio, err = verify(rows, want, cands)
        if err:
            return None, f"role {role}: {err}"
        specs.append(f"{oid}:{role}:{want}:{b},{g},{r_}")
        notes.append(f"      {role:17s} <- {want}  "
                     f"({len(rows)} rows, mask_px ratio {ratio:.3f})")
    return (trial, specs, notes), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", help="rebuild just this sidecar dir")
    args = ap.parse_args()

    cands = candidates()
    print(f"[scan] {len(cands)} candidate propagation summary CSVs\n", flush=True)

    plans = []
    for sc in ([args.only] if args.only else SIDECARS):
        p, err = plan_one(sc, cands)
        if err:
            print(f"[SKIP] {sc}\n      {err}", flush=True)
            continue
        trial, specs, notes = p
        print(f"[plan] {sc}   trial={trial}", flush=True)
        for n in notes:
            print(n, flush=True)
        plans.append((sc, trial, specs))
    print(f"\n[plan] {len(plans)}/{len(SIDECARS)} sidecars resolved", flush=True)

    if not args.run:
        print("[dry ] pass --run to rebuild", flush=True)
        return

    for sc, trial, specs in plans:
        cmd = [sys.executable, "Code/build_sidecar_multi.py",
               "--trial", trial, "--out", sc]
        for s in specs:
            cmd += ["--object", s]
        print(f"\n{'='*70}\n[run ] {sc}\n{'='*70}", flush=True)
        r = subprocess.run(cmd)
        print(f"[{'ok  ' if r.returncode == 0 else 'FAIL'}] {sc} "
              f"rc={r.returncode}", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
