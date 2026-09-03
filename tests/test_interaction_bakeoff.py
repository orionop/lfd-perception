import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from score_interaction_bakeoff import (best_proposal, box_mask, hoi_evidence,
                                       load_proposals)
from select_objects import Proposal


class ProposalEvidenceTests(unittest.TestCase):
    def test_box_maps_to_best_mask_by_iou(self):
        a = np.zeros((20, 20), dtype=bool)
        b = np.zeros((20, 20), dtype=bool)
        a[1:5, 1:5] = True
        b[10:18, 10:18] = True
        proposals = [Proposal(a, (1, 1, 4, 4), 16, 0.9, 0.9),
                     Proposal(b, (10, 10, 17, 17), 64, 0.9, 0.9)]
        score, selected = best_proposal(
            proposals, box_mask((20, 20), [9, 9, 18, 18]))
        self.assertEqual(selected.bbox, (10, 10, 17, 17))
        self.assertGreater(score, 0.5)

    def test_cached_proposal_loader_rejects_pickle_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            mask = np.zeros((1, 8, 8), dtype=np.uint8)
            mask[0, 2:6, 2:6] = 1
            meta = [{"bbox": [2, 2, 5, 5], "area": 16,
                     "predicted_iou": 0.9, "stability": 0.95}]
            np.savez_compressed(path, masks=mask, meta=json.dumps(meta))
            proposals = load_proposals(path, "contact_receiver")
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].bbox, (2, 2, 5, 5))


class HoiRelationTests(unittest.TestCase):
    RECORD = {
        "detections": [
            {"class_id": 0, "score": 0.9, "box": [0, 0, 4, 4]},
            {"class_id": 1, "score": 0.8, "box": [5, 5, 10, 10]},
            {"class_id": 2, "score": 0.7, "box": [11, 11, 18, 18]},
        ],
        "hf": [{"a": 0, "b": 1, "prob": 0.85}],
        "fs": [{"a": 1, "b": 2, "prob": 0.75}],
    }

    def test_grasp_uses_first_object_from_hf_relation(self):
        evidence = hoi_evidence(self.RECORD, "grasped")
        self.assertEqual(evidence[2], [5, 5, 10, 10])

    def test_contact_requires_full_hf_to_fs_chain(self):
        evidence = hoi_evidence(self.RECORD, "contact_receiver")
        self.assertEqual(evidence[2], [11, 11, 18, 18])
        broken = dict(self.RECORD, hf=[])
        self.assertIsNone(hoi_evidence(broken, "contact_receiver"))

    def test_unlinked_second_object_is_ignored(self):
        record = dict(self.RECORD,
                      fs=[{"a": 0, "b": 2, "prob": 0.99}])
        self.assertIsNone(hoi_evidence(record, "contact_receiver"))


if __name__ == "__main__":
    unittest.main()
