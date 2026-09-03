import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Code"))
from deliverable_events import FX, FY, FZ, GRIP, IMG, POSE_TS
from select_objects import proposal_provenance


class RunnerContractTests(unittest.TestCase):
    def test_abstention_never_publishes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = os.path.join(directory, "trial")
            image_dir = os.path.join(
                trial, "zed_zed_node_rgb_color_rect_image_compressed")
            out = os.path.join(directory, "out")
            cache = os.path.join(out, "work", "selection", "proposal_cache")
            os.makedirs(image_dir)
            os.makedirs(cache)

            fields = [POSE_TS, IMG, GRIP, FX, FY, FZ]
            rows = []
            for index in range(60):
                image_id = str(index // 3)
                width = 0.02 if 10 <= index < 50 else 0.08
                force = 10.0 if index == 30 else 0.0
                rows.append({POSE_TS: str(1_700_000_000_000_000_000 +
                                          index * 10_000_000),
                             IMG: image_id,
                             GRIP: f"[{width / 2}, {width / 2}]",
                             FX: str(force), FY: "0", FZ: "0"})
            with open(os.path.join(trial, "trial_0.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            image = np.zeros((540, 960, 3), dtype=np.uint8)
            for image_id in {row[IMG] for row in rows}:
                cv2.imwrite(os.path.join(image_dir, image_id + ".png"), image)

            # Two genuinely different candidates of identical size, both wholly
            # inside both role regions, so every ranking feature ties and the
            # winner margin is exactly zero. They must be disjoint: identical
            # masks are two views of one object and are suppressed before the
            # margin is measured, which is the opposite of an ambiguous scene.
            # The selector must abstain, and the runner must stop before any
            # propagation or objects.json publication.
            first = np.zeros((540, 960), dtype=np.uint8)
            first[150:220, 420:520] = 1
            second = np.zeros((540, 960), dtype=np.uint8)
            second[150:220, 560:660] = 1
            meta = json.dumps([
                {"bbox": [420, 150, 519, 219], "area": int(first.sum()),
                 "predicted_iou": 0.95, "stability": 0.95},
                {"bbox": [560, 150, 659, 219], "area": int(second.sum()),
                 "predicted_iou": 0.95, "stability": 0.95},
            ])
            checkpoint = os.path.join(directory, "sam.pth")
            open(checkpoint, "wb").close()
            provenance = proposal_provenance(SimpleNamespace(
                ckpt=checkpoint, model="vit_h", points_per_side=24))
            for image_id in ("6", "7", "8", "10"):
                np.savez_compressed(os.path.join(cache, image_id + ".npz"),
                                    masks=np.stack([first, second]), meta=meta,
                                    provenance=json.dumps(provenance,
                                                          sort_keys=True))

            result = subprocess.run([
                sys.executable, "Code/run_deliverable.py",
                "--trial", trial, "--out", out, "--select-only",
                "--sam1-python", sys.executable,
                "--sam1-ckpt", checkpoint,
                "--rig-profile", "config/deliverable_rig.yaml",
            ], cwd=ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(os.path.exists(os.path.join(out, "sidecar", "objects.json")))
            with open(os.path.join(out, "run_report.json")) as report_file:
                report = json.load(report_file)
            self.assertEqual(report["status"], "review_required")


if __name__ == "__main__":
    unittest.main()
