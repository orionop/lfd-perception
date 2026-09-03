"""Export the frozen Deliverable-A cases for external GPU inference.

The bundle deliberately contains no reference masks.  External models see only
the event frames; proposal matching and scoring happen after their predictions
return to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from select_objects import RGB_DIR, polygon_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_partner(primary: np.ndarray, partner: np.ndarray,
                   excluded: np.ndarray) -> tuple[np.ndarray, bool, float | None]:
    """Align partner to primary using background pixels only."""
    template = cv2.cvtColor(primary, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    moving = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    mask = (~excluded).astype(np.uint8) * 255
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        score, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_AFFINE, criteria, mask, 5)
        aligned = cv2.warpAffine(
            partner, warp, (primary.shape[1], primary.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT)
        return aligned, True, float(score)
    except cv2.error:
        return partner.copy(), False, None


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    parser.add_argument("--reports-root", default="figures/deliverable_eval")
    parser.add_argument("--rig-profile", default="config/deliverable_rig.yaml")
    parser.add_argument("--out", default="figures/interaction_bakeoff/input")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    manifest_path = Path(args.manifest).resolve()
    reports_root = Path(args.reports_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to replace existing benchmark: {out}")

    manifest = yaml.safe_load(manifest_path.read_text())
    rig = yaml.safe_load(Path(args.rig_profile).read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="interaction_bakeoff_", dir=out.parent))
    try:
        (tmp / "images").mkdir(parents=True)
        (tmp / "pairs").mkdir()
        (tmp / "proposals").mkdir()
        cases = []
        for recording in manifest["recordings"]:
            rec_id = recording["id"]
            report_path = reports_root / rec_id / "selection_report.json"
            if not report_path.exists():
                raise FileNotFoundError(report_path)
            report = json.loads(report_path.read_text())
            selections = {(s["role"], int(s["cycle_idx"])): s
                          for s in report.get("selections", [])}
            trial = (root / recording["trial"]).resolve()
            for declared in recording.get("ground_truth", []):
                role = declared["selector_role"]
                cycle = int(declared["cycle_idx"])
                selected = selections.get((role, cycle))
                if selected is None:
                    raise ValueError(f"missing selection for {rec_id}:{role}:c{cycle}")
                img_id = str(selected["img_id"])
                source = trial / RGB_DIR / f"{img_id}.png"
                if not source.exists():
                    raise FileNotFoundError(source)
                case_id = f"{rec_id}__{role}__c{cycle}"
                image_rel = Path("images") / f"{case_id}.png"
                shutil.copy2(source, tmp / image_rel)

                cache_rel_in_report = selected.get(
                    "proposal_cache_path", f"proposal_cache/{img_id}.npz")
                cache_source = (report_path.parent / cache_rel_in_report).resolve()
                if not cache_source.exists():
                    raise FileNotFoundError(cache_source)
                proposal_rel = Path("proposals") / f"{case_id}.npz"
                shutil.copy2(cache_source, tmp / proposal_rel)

                entry = {
                    "case_id": case_id,
                    "recording_id": rec_id,
                    "independent_group": recording["independent_group"],
                    "role": role,
                    "cycle_idx": cycle,
                    "img_id": img_id,
                    "image": image_rel.as_posix(),
                    "proposal_cache": proposal_rel.as_posix(),
                }

                if role == "grasped":
                    probes = [str(v) for v in selected.get("hold_probe_frames", [])]
                    alternatives = [value for value in probes if value != img_id]
                    if not alternatives:
                        raise ValueError(f"no DistinctNet partner for {case_id}")
                    partner_id = alternatives[0]
                    partner_source = trial / RGB_DIR / f"{partner_id}.png"
                    if not partner_source.exists():
                        raise FileNotFoundError(partner_source)
                    primary = cv2.imread(str(source))
                    partner = cv2.imread(str(partner_source))
                    if primary is None or partner is None or primary.shape != partner.shape:
                        raise ValueError(f"invalid frame pair for {case_id}")
                    excluded = polygon_mask(
                        primary.shape[:2], rig["regions"]["grasped"]["polygon"])
                    aligned, ok, ecc = stable_partner(primary, partner, excluded)
                    primary_rel = Path("pairs") / f"{case_id}__primary.png"
                    raw_rel = Path("pairs") / f"{case_id}__partner_raw.png"
                    stable_rel = Path("pairs") / f"{case_id}__partner_stabilized.png"
                    shutil.copy2(source, tmp / primary_rel)
                    shutil.copy2(partner_source, tmp / raw_rel)
                    if not cv2.imwrite(str(tmp / stable_rel), aligned):
                        raise RuntimeError(f"failed to write stabilized pair for {case_id}")
                    entry["distinctnet"] = {
                        "primary": primary_rel.as_posix(),
                        "partner_raw": raw_rel.as_posix(),
                        "partner_stabilized": stable_rel.as_posix(),
                        "partner_img_id": partner_id,
                        "stabilization_ok": ok,
                        "stabilization_ecc": ecc,
                    }
                cases.append(entry)

        counts = {"grasped": sum(c["role"] == "grasped" for c in cases),
                  "contact_receiver": sum(c["role"] == "contact_receiver"
                                          for c in cases)}
        if counts != {"grasped": 5, "contact_receiver": 7}:
            raise ValueError(f"frozen benchmark changed unexpectedly: {counts}")
        payload = {
            "schema_version": "1.0",
            "purpose": "inference_only_no_reference_masks",
            "source_manifest": os.path.relpath(manifest_path, root),
            "source_manifest_sha256": sha256(manifest_path),
            "counts": counts,
            "cases": cases,
        }
        atomic_json(tmp / "benchmark.json", payload)
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out)
        print(f"[write] {out} ({counts['grasped']} grasp, "
              f"{counts['contact_receiver']} contact)")
        return 0
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
