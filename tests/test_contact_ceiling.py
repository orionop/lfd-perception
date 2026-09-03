import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from contact_ceiling_study import (Candidate, Case, contact_flow_contrast,
                                   evaluate, score_case)


def candidate(x0, value, overlap):
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, x0:x0 + 6] = True
    return Candidate(mask, (x0, 2, x0 + 5, 7),
                     np.array([value] + [0.0] * 7), overlap)


class ContactCeilingTests(unittest.TestCase):
    def test_contact_flow_contrast_is_local_relative_motion(self):
        mask = np.zeros((80, 80), bool)
        mask[25:55, 25:55] = True
        flow = np.ones((80, 80), dtype=float)
        flow[mask] = 3.0
        self.assertAlmostEqual(contact_flow_contrast(mask, flow), 0.5)

    def test_margin_and_correctness_are_separate(self):
        case = Case("r", "g", 1,
                    [candidate(1, 0.8, 0.9), candidate(12, 0.1, 0.0)])
        result = score_case(case, np.array([1.0] + [0.0] * 7),
                            margin_min=0.1, suppression_iou=0.5)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["margin"], 0.7)

    def test_wrong_high_margin_counts_against_precision(self):
        case = Case("r", "g", 1,
                    [candidate(1, 0.8, 0.0), candidate(12, 0.1, 0.9)])
        result = evaluate([case], np.array([1.0] + [0.0] * 7),
                          margin_min=0.1, suppression_iou=0.5)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
