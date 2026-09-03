import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))
from validate_sidecar import validate


def valid_sidecar():
    return {
        "schema_version": "1.0", "trial_dir": "trial", "events": {},
        "objects": {"1": {"role": "grasped", "color": [0, 255, 0]}},
        "frames": [{"frame_idx": 0, "img_filename": "1.png",
                    "objects": [{"obj_id": 1, "role": "grasped",
                                 "mask_px": 100, "bbox_xyxy": [10, 50, 30, 80]}]}],
    }


class SidecarValidationTests(unittest.TestCase):
    def test_valid_minimal_sidecar(self):
        errors, _, summary = validate(valid_sidecar(), "objects.json",
                                      require_paths=False)
        self.assertEqual(errors, [])
        self.assertEqual(summary["object_records"], 1)

    def test_caption_artifact_is_rejected(self):
        data = valid_sidecar()
        data["frames"][0]["objects"][0]["bbox_xyxy"] = [10, 10, 150, 35]
        errors, _, _ = validate(data, "objects.json", require_paths=False)
        self.assertTrue(any("caption artifact" in error for error in errors))

    def test_review_role_contract(self):
        errors, _, _ = validate(valid_sidecar(), "objects.json",
                                required_roles=["contact_receiver"],
                                require_paths=False)
        self.assertTrue(any("required role missing" in error for error in errors))

    def test_portable_missing_overlay_is_rejected(self):
        data = valid_sidecar()
        data["path_base"] = "sidecar_directory"
        data["frames"][0]["overlay_path"] = "overlays/missing.png"
        with tempfile.TemporaryDirectory() as directory:
            sidecar = os.path.join(directory, "objects.json")
            errors, _, _ = validate(data, sidecar, require_paths=True)
        self.assertTrue(any("overlay_path does not exist" in error
                            for error in errors))


if __name__ == "__main__":
    unittest.main()
