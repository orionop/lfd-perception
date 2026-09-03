import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from evaluate_selection import declared_reference_role, role_category


class EvaluationTests(unittest.TestCase):
    def test_contact_roles_cannot_match_grasped_reference(self):
        self.assertEqual(role_category("grasped"), "grasped")
        self.assertEqual(role_category("contact_receiver"), "contact")
        self.assertEqual(role_category("tool_contact"), "contact")


class DeclaredGroundTruthTests(unittest.TestCase):
    """A frame holding several contact objects must not be scored by category.

    This is the version 1 defect: role_category collapsed contact_receiver,
    tool_contact and charger_contact into one bucket, so the evaluator scored
    the selector against whichever overlay recovered first.
    """

    RECORDING = {
        "id": "multi_contact",
        "ground_truth": [
            {"selector_role": "contact_receiver", "cycle_idx": 1,
             "reference_role": "contact_receiver"},
            {"selector_role": "contact_receiver", "cycle_idx": 2,
             "reference_role": "tool_contact"},
            {"selector_role": "grasped", "cycle_idx": 1,
             "reference_role": "grasped"},
        ],
    }

    def test_same_selector_role_resolves_per_cycle(self):
        first = declared_reference_role(
            self.RECORDING, {"role": "contact_receiver", "cycle_idx": 1})
        second = declared_reference_role(
            self.RECORDING, {"role": "contact_receiver", "cycle_idx": 2})
        self.assertEqual(first, "contact_receiver")
        self.assertEqual(second, "tool_contact")
        self.assertNotEqual(first, second)

    def test_cycle_idx_may_be_a_string(self):
        self.assertEqual(
            declared_reference_role(
                self.RECORDING, {"role": "contact_receiver", "cycle_idx": "2"}),
            "tool_contact")

    def test_undeclared_cycle_is_not_guessed(self):
        self.assertIsNone(declared_reference_role(
            self.RECORDING, {"role": "contact_receiver", "cycle_idx": 9}))

    def test_undeclared_role_is_not_guessed(self):
        self.assertIsNone(declared_reference_role(
            self.RECORDING, {"role": "charger_contact", "cycle_idx": 1}))

    def test_recording_without_ground_truth_scores_nothing(self):
        self.assertIsNone(declared_reference_role(
            {"id": "excluded", "ground_truth": []},
            {"role": "grasped", "cycle_idx": 1}))
        self.assertIsNone(declared_reference_role(
            {"id": "legacy"}, {"role": "grasped", "cycle_idx": 1}))


class ManifestIntegrityTests(unittest.TestCase):
    """The shipped manifest must stay consistent with what it declares."""

    @classmethod
    def setUpClass(cls):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config",
                            "evaluation_manifest.yaml")
        with open(path) as f:
            cls.manifest = yaml.safe_load(f)

    def test_declared_reference_roles_are_eligible(self):
        for recording in self.manifest["recordings"]:
            eligible = set(recording.get("eligible_roles", []))
            for entry in recording.get("ground_truth", []):
                self.assertIn(entry["reference_role"], eligible,
                              f"{recording['id']} declares an ineligible role")

    def test_no_duplicate_cycle_declarations(self):
        for recording in self.manifest["recordings"]:
            keys = [(e["selector_role"], int(e["cycle_idx"]))
                    for e in recording.get("ground_truth", [])]
            self.assertEqual(len(keys), len(set(keys)),
                             f"{recording['id']} declares a cycle twice")

    def test_excluded_recordings_declare_nothing(self):
        for recording in self.manifest["recordings"]:
            if not recording.get("eligible_roles"):
                self.assertFalse(recording.get("ground_truth"),
                                 f"{recording['id']} is excluded but declares "
                                 "ground truth")

    def test_grasped_meets_minimum_independent_group_count(self):
        groups = {r["independent_group"] for r in self.manifest["recordings"]
                  for e in r.get("ground_truth", [])
                  if e["selector_role"] == "grasped"}
        self.assertGreaterEqual(
            len(groups), 3,
            "grasped cannot meet the evidence bar with fewer than 3 groups")


if __name__ == "__main__":
    unittest.main()
