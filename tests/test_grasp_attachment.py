import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from grasp_attachment_study import attachment_features
from select_objects import Proposal


def proposal(mask):
    ys, xs = np.nonzero(mask)
    return Proposal(mask, (int(xs.min()), int(ys.min()), int(xs.max()),
                           int(ys.max())), int(mask.sum()), 0.9, 0.9)


class GraspAttachmentTests(unittest.TestCase):
    def test_stable_post_close_track_scores_high(self):
        mask = np.zeros((100, 100), bool)
        mask[30:60, 40:70] = True
        shifted = np.zeros_like(mask)
        shifted[31:61, 41:71] = True
        change = np.zeros(mask.shape, np.float32)
        change[mask] = 0.8
        values = attachment_features(proposal(mask),
                                     [[proposal(shifted)], [proposal(mask)]],
                                     change)
        self.assertGreater(values["post_mask_iou"], 0.85)
        self.assertEqual(values["post_support"], 1.0)
        self.assertGreater(values["preclose_change"], 0.7)

    def test_missing_post_close_match_has_no_support(self):
        mask = np.zeros((100, 100), bool)
        mask[10:30, 10:30] = True
        other = np.zeros_like(mask)
        other[70:90, 70:90] = True
        values = attachment_features(proposal(mask), [[proposal(other)]],
                                     np.zeros(mask.shape, np.float32))
        self.assertEqual(values["post_mask_iou"], 0.0)
        self.assertEqual(values["post_support"], 0.0)


if __name__ == "__main__":
    unittest.main()
