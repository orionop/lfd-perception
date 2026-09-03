"""One guarded command for Deliverable A.

Exit codes: 0 accepted output, 2 automatic selection requires review,
1 invalid input or execution failure. A review-required run never publishes
``objects.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from deliverable_events import find_demo_csv

RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
COLORS = [(0, 255, 0), (255, 0, 255), (0, 165, 255), (255, 165, 0)]


def run(command):
    print("[exec] " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def write_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam1-python", default=".venv_sam2/bin/python")
    ap.add_argument("--sam2-python", default=".venv_sam2/bin/python")
    ap.add_argument("--analysis-python", default=".venv_analysis/bin/python")
    ap.add_argument("--sam1-ckpt", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--sam2-ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--rig-profile", default="config/deliverable_rig.yaml")
    ap.add_argument("--offload-video-to-cpu", action="store_true")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    work = os.path.join(args.out, "work")
    selection_dir = os.path.join(work, "selection")
    report_path = os.path.join(args.out, "run_report.json")
    report = {"schema_version": "1.0", "trial": args.trial,
              "started_utc": started, "status": "running", "stages": []}

    try:
        demo_csv = find_demo_csv(args.trial)
        image_dir = os.path.join(args.trial, RGB_DIR)
        # Selection-only evaluation has no dependency on SAM 2 or the
        # analysis environment. Keeping those out of its preflight makes the
        # calibration-free selector independently runnable and testable.
        required = [demo_csv, image_dir, args.sam1_python,
                    args.sam1_ckpt, args.rig_profile]
        if not args.select_only:
            required.extend([args.sam2_python, args.analysis_python,
                             args.sam2_ckpt])
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
        os.makedirs(selection_dir, exist_ok=True)

        select_cmd = [args.sam1_python, "Code/select_objects.py",
                      "--trial", args.trial, "--out", selection_dir,
                      "--ckpt", args.sam1_ckpt,
                      "--rig-profile", args.rig_profile]
        for override in args.override:
            select_cmd.extend(["--override", override])
        rc = run(select_cmd)
        report["stages"].append({"name": "selection", "exit_code": rc})
        selection_path = os.path.join(selection_dir, "selection_report.json")
        if not os.path.exists(selection_path):
            raise RuntimeError("selector did not produce selection_report.json")
        with open(selection_path) as f:
            selection = json.load(f)
        report["selection_status"] = selection.get("status")
        if rc == 2 or selection.get("status") == "review_required":
            report["status"] = "review_required"
            write_report(report_path, report)
            print("[stop] selection requires review; no objects.json published", flush=True)
            return 2
        if rc != 0:
            raise RuntimeError(f"selector failed with exit code {rc}")
        if args.select_only:
            report["status"] = selection.get("status", "accepted")
            write_report(report_path, report)
            return 0

        jpg_dir = os.path.join(work, "frames_jpg")
        rc = run([args.sam2_python, "Code/prepare_sam2_frames.py",
                  "--src", image_dir, "--dst", jpg_dir])
        report["stages"].append({"name": "prepare_frames", "exit_code": rc})
        if rc:
            raise RuntimeError("frame preparation failed")

        object_specs = []
        accepted = [s for s in selection["selections"] if s["status"] == "accepted"]
        for obj_id, selected in enumerate(accepted, 1):
            role = selected["role"]
            color = COLORS[(obj_id - 1) % len(COLORS)]
            prop_out = os.path.join(work, f"propagation_obj{obj_id}_{role}")
            box = ",".join(str(float(v)) for v in selected["seed_box_xyxy"])
            command = [args.sam2_python, "Code/propagate_object_n.py",
                       "--trial", args.trial, "--jpg_dir", jpg_dir,
                       "--ckpt", args.sam2_ckpt, "--out", prop_out,
                       "--obj_id", str(obj_id), "--role", role,
                       "--seed_img_id", str(selected["img_id"]),
                       "--seed_box", box,
                       "--color", ",".join(str(v) for v in color)]
            if args.offload_video_to_cpu:
                command.append("--offload_video_to_cpu")
            rc = run(command)
            report["stages"].append({"name": f"propagate_{obj_id}",
                                     "role": role, "exit_code": rc})
            if rc:
                raise RuntimeError(f"propagation failed for {role}")
            summary = f"{prop_out}_summary.csv"
            object_specs.append(
                f"{obj_id}:{role}:{summary}:{','.join(str(v) for v in color)}")

        final_dir = os.path.join(args.out, "sidecar")
        command = [args.analysis_python, "Code/build_sidecar_multi.py",
                   "--trial", args.trial, "--out", final_dir,
                   "--selection-report", selection_path, "--portable-paths"]
        for spec in object_specs:
            command.extend(["--object", spec])
        rc = run(command)
        report["stages"].append({"name": "build_sidecar", "exit_code": rc})
        if rc:
            raise RuntimeError("sidecar build failed")

        sidecar = os.path.join(final_dir, "objects.json")
        validation_report = os.path.join(args.out, "validation_report.json")
        command = [args.analysis_python, "Code/validate_sidecar.py", sidecar,
                   "--report", validation_report]
        for role in sorted({s["role"] for s in accepted}):
            command.extend(["--required-role", role])
        rc = run(command)
        report["stages"].append({"name": "validate_sidecar", "exit_code": rc})
        if rc:
            raise RuntimeError("sidecar quality gate failed")

        report.update(status=selection.get("status", "accepted"),
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      sidecar=os.path.relpath(sidecar, args.out),
                      validation_report=os.path.relpath(validation_report, args.out))
        write_report(report_path, report)
        print(f"[done] accepted sidecar: {sidecar}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(report_path, report)
        print(f"[fatal] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
