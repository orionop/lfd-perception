import os
import sys
import tempfile
import unittest
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from select_objects import (Proposal, _load_proposals, _save_proposals, choose,
                            hold_probe_indices, interior_point,
                            merged_grasp_proposals, polygon_mask, rank_proposals,
                            write_json_safely)


class SelectionLogicTests(unittest.TestCase):
    def test_safe_json_write_preserves_previous_as_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "result.json")
            with open(path, "w") as f:
                json.dump({"version": 1}, f)
            write_json_safely(path, {"version": np.int64(2)})
            with open(path) as f:
                self.assertEqual(json.load(f), {"version": 2})
            with open(path + ".bak") as f:
                self.assertEqual(json.load(f), {"version": 1})

    def test_empty_proposal_cache_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.npz")
            _save_proposals(path, [])
            self.assertEqual(_load_proposals(path), [])

    def test_cache_provenance_mismatch_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.npz")
            _save_proposals(path, [], {"points_per_side": 8})
            self.assertEqual(
                _load_proposals(path, {"points_per_side": 8}), [])
            self.assertIsNone(
                _load_proposals(path, {"points_per_side": 24}))

    def setUp(self):
        self.config = {
            "proposal_filter": {"min_area_fraction": 0.001,
                                "max_area_fraction": 0.65,
                                "max_border_sides": 2}}

    def test_interior_point_avoids_mask_hole(self):
        mask = np.zeros((100, 100), bool)
        mask[10:90, 10:90] = True
        mask[40:60, 40:60] = False
        x, y = interior_point(mask)
        self.assertTrue(mask[int(y), int(x)])

    def test_region_proposal_ranks_over_distant_proposal(self):
        region = polygon_mask((100, 100), [[0.4, 0.4], [0.6, 0.4],
                                           [0.6, 0.6], [0.4, 0.6]])
        near = np.zeros((100, 100), bool); near[35:65, 35:65] = True
        far = np.zeros((100, 100), bool); far[5:25, 5:25] = True
        proposals = [Proposal(far, (5, 5, 24, 24), 400, 0.9, 0.9),
                     Proposal(near, (35, 35, 64, 64), 900, 0.9, 0.9)]
        ranked = rank_proposals(proposals, region, "contact_receiver", self.config)
        self.assertEqual(ranked[0][0].bbox, proposals[1].bbox)

    @staticmethod
    def _disjoint_pair():
        a = np.zeros((20, 20), bool); a[0:8, 0:8] = True
        b = np.zeros((20, 20), bool); b[12:20, 12:20] = True
        return (Proposal(a, (0, 0, 7, 7), int(a.sum()), 0.9, 0.9),
                Proposal(b, (12, 12, 19, 19), int(b.sum()), 0.9, 0.9))

    def test_low_margin_abstains(self):
        first, second = self._disjoint_pair()
        ranked = [(first, {"score": 0.5}), (second, {"score": 0.49})]
        result = choose(ranked, {"min_score": 0.4,
                                 "confidence_margin_min": 0.1})
        self.assertEqual(result["status"], "review_required")

    def test_nested_duplicate_does_not_consume_the_margin(self):
        """SAM's own coarse/fine views of one object must not read as ambiguity."""
        whole = np.zeros((20, 20), bool); whole[0:12, 0:12] = True
        part = np.zeros((20, 20), bool); part[0:10, 0:10] = True
        ranked = [(Proposal(whole, (0, 0, 11, 11), int(whole.sum()), .9, .9),
                   {"score": 0.5}),
                  (Proposal(part, (0, 0, 9, 9), int(part.sum()), .9, .9),
                   {"score": 0.49})]
        result = choose(ranked, {"min_score": 0.4,
                                 "confidence_margin_min": 0.1})
        self.assertEqual(result["status"], "accepted")
        self.assertAlmostEqual(result["runner_up_score"], 0.49)
        self.assertAlmostEqual(result["distinct_runner_up_score"], 0.0)

    def test_distinct_runner_up_is_found_past_duplicates(self):
        whole = np.zeros((20, 20), bool); whole[0:12, 0:12] = True
        part = np.zeros((20, 20), bool); part[0:10, 0:10] = True
        other = np.zeros((20, 20), bool); other[14:20, 14:20] = True
        ranked = [(Proposal(whole, (0, 0, 11, 11), int(whole.sum()), .9, .9),
                   {"score": 0.5}),
                  (Proposal(part, (0, 0, 9, 9), int(part.sum()), .9, .9),
                   {"score": 0.49}),
                  (Proposal(other, (14, 14, 19, 19), int(other.sum()), .9, .9),
                   {"score": 0.45})]
        result = choose(ranked, {"min_score": 0.4,
                                 "confidence_margin_min": 0.1})
        # The genuinely different object still sets the margin, so 0.5 - 0.45
        # remains below the bar and the selector still abstains.
        self.assertAlmostEqual(result["distinct_runner_up_score"], 0.45)
        self.assertEqual(result["status"], "review_required")

    def test_sole_candidate_has_full_margin(self):
        first, _ = self._disjoint_pair()
        result = choose([(first, {"score": 0.5})],
                        {"min_score": 0.4, "confidence_margin_min": 0.1})
        self.assertEqual(result["status"], "accepted")
        self.assertAlmostEqual(result["distinct_runner_up_score"], 0.0)

    def test_grasp_uses_multiple_hold_probe_frames(self):
        cycle = {"start_idx": 100, "end_idx": 200}
        self.assertEqual(hold_probe_indices(cycle, "grasped", 150),
                         [120, 127, 135])
        self.assertEqual(hold_probe_indices(cycle, "contact_receiver", 177),
                         [177])

    def test_fragmented_grasp_proposals_can_form_local_union(self):
        a = np.zeros((100, 100), bool); a[20:50, 20:45] = True
        b = np.zeros((100, 100), bool); b[20:50, 44:70] = True
        proposals = [Proposal(a, (20, 20, 44, 49), int(a.sum()), .9, .9),
                     Proposal(b, (44, 20, 69, 49), int(b.sum()), .9, .9)]
        merged = merged_grasp_proposals(proposals, (100, 100), .4)
        self.assertTrue(any(p.mask[30, 60] and p.mask[30, 30]
                            for p in merged))


if __name__ == "__main__":
    unittest.main()
