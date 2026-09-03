import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from deliverable_events import FX, FY, FZ, GRIP, IMG, POSE_TS, detect_events


def dataframe(widths=None, peaks=()):
    n = 120
    rows = [{POSE_TS: str(1_700_000_000_000_000_000 + i * 10_000_000),
             IMG: str(i // 3)} for i in range(n)]
    if widths is not None:
        for row, width in zip(rows, widths):
            row[GRIP] = f"[{width / 2}, {width / 2}]"
    if peaks is not None:
        force = np.zeros(n)
        for p in peaks:
            force[p] = 10.0
        for i, row in enumerate(rows):
            row.update({FX: str(force[i]), FY: "0", FZ: "0"})
    return rows


class EventDetectionTests(unittest.TestCase):
    def test_inactive_gripper_does_not_invent_transitions(self):
        widths = np.full(120, 0.08) + np.linspace(0, 1e-7, 120)
        events, cycles, summary = detect_events(dataframe(widths, (60,)))
        self.assertEqual(summary["n_grasps"], 0)
        self.assertEqual(summary["n_releases"], 0)
        self.assertEqual(summary["n_presses"], 1)
        self.assertEqual(events[0]["event"], "press")

    def test_multiple_closed_runs_become_multiple_cycles(self):
        widths = np.full(120, 0.08)
        widths[20:45] = 0.02
        widths[70:100] = 0.03
        events, cycles, summary = detect_events(dataframe(widths, (30, 80)),
                                                 peak_distance_s=0.1)
        self.assertEqual(summary["n_cycles"], 2)
        self.assertEqual([len(c["presses"]) for c in cycles], [1, 1])
        self.assertEqual((cycles[0]["start_idx"], cycles[0]["end_idx"]),
                         (20, 45))

    def test_gripper_only_has_no_press(self):
        widths = np.full(120, 0.08)
        widths[20:45] = 0.02
        _, cycles, summary = detect_events(dataframe(widths, None))
        self.assertFalse(summary["has_force"])
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0]["presses"], [])


if __name__ == "__main__":
    unittest.main()
