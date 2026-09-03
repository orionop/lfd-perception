"""Validate and quality-gate a Deliverable A ``objects.json`` sidecar."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

def resolve_path(sidecar_path: str, value: str, path_base: str) -> str:
    if os.path.isabs(value):
        return value
    base = os.path.dirname(os.path.abspath(sidecar_path)) \
        if path_base == "sidecar_directory" else os.getcwd()
    return os.path.normpath(os.path.join(base, value))


def validate(data: Dict, sidecar_path: str, required_roles=(),
             require_paths: bool = True) -> Tuple[List[str], List[str], Dict]:
    errors, warnings = [], []
    required = ("schema_version", "trial_dir", "events", "objects", "frames")
    for key in required:
        if key not in data:
            errors.append(f"missing top-level field: {key}")
    if data.get("schema_version") != "1.0":
        errors.append(f"unsupported schema_version: {data.get('schema_version')!r}")

    objects = data.get("objects", {})
    roles = {str(v.get("role")) for v in objects.values()}
    for role in required_roles:
        if role not in roles:
            errors.append(f"required role missing: {role}")

    frames = data.get("frames", [])
    indices = [f.get("frame_idx") for f in frames]
    if any(not isinstance(i, int) or i < 0 for i in indices):
        errors.append("frame_idx values must be non-negative integers")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        errors.append("frames must have unique, ascending frame_idx values")

    path_base = data.get("path_base", "working_directory")
    image_dir = data.get("image_dir")
    image_dir_abs = resolve_path(sidecar_path, image_dir, path_base) if image_dir else None
    if require_paths:
        for name in ("trial_dir", "csv", "image_dir"):
            value = data.get(name)
            if value and not os.path.exists(resolve_path(sidecar_path, value, path_base)):
                errors.append(f"declared {name} does not exist: {value}")

    area_by_object = defaultdict(list)
    records = 0
    for frame in frames:
        filename = frame.get("img_filename", "")
        frame_overlay = frame.get("overlay_path")
        if require_paths and frame_overlay and not os.path.exists(
                resolve_path(sidecar_path, frame_overlay, path_base)):
            errors.append(f"frame {frame.get('frame_idx')}: overlay_path does not exist")
        if image_dir_abs:
            import cv2
            image = cv2.imread(os.path.join(image_dir_abs, filename))
        else:
            image = None
        height, width = image.shape[:2] if image is not None else (None, None)
        seen = set()
        for obj in frame.get("objects", []):
            records += 1
            oid = str(obj.get("obj_id"))
            if oid in seen:
                errors.append(f"frame {frame.get('frame_idx')}: duplicate obj_id {oid}")
            seen.add(oid)
            if oid not in objects:
                errors.append(f"frame {frame.get('frame_idx')}: undeclared obj_id {oid}")
            elif obj.get("role") != objects[oid].get("role"):
                errors.append(f"frame {frame.get('frame_idx')}: role mismatch for obj_id {oid}")
            area = obj.get("mask_px")
            object_overlay = obj.get("object_overlay_path")
            if require_paths and object_overlay and not os.path.exists(
                    resolve_path(sidecar_path, object_overlay, path_base)):
                errors.append(
                    f"frame {frame.get('frame_idx')}: object overlay does not exist for obj_id {oid}")
            if not isinstance(area, int) or area < 0:
                errors.append(f"frame {frame.get('frame_idx')}: invalid mask_px for obj_id {oid}")
                continue
            area_by_object[oid].append(area)
            box = obj.get("bbox_xyxy")
            if box is None:
                if area > 0:
                    errors.append(f"frame {frame.get('frame_idx')}: positive mask with null bbox")
                continue
            if len(box) != 4 or any(not isinstance(v, int) for v in box):
                errors.append(f"frame {frame.get('frame_idx')}: malformed bbox for obj_id {oid}")
                continue
            x0, y0, x1, y1 = box
            if x0 < 0 or y0 < 0 or x1 < x0 or y1 < y0:
                errors.append(f"frame {frame.get('frame_idx')}: invalid bbox for obj_id {oid}")
            if width and (x1 >= width or y1 >= height):
                errors.append(f"frame {frame.get('frame_idx')}: bbox outside image for obj_id {oid}")
            if width and area > 0.80 * width * height:
                errors.append(f"frame {frame.get('frame_idx')}: obj_id {oid} consumes >80% of image")
            if y1 <= 40 and area <= 2000:
                errors.append(f"frame {frame.get('frame_idx')}: obj_id {oid} resembles caption artifact")

    for oid, areas in area_by_object.items():
        positive = [a for a in areas if a > 0]
        for before, after in zip(positive, positive[1:]):
            ratio = max(before, after) / max(1, min(before, after))
            if ratio > 20:
                warnings.append(f"obj_id {oid}: >20x consecutive positive-area jump")
                break

    summary = {"frames": len(frames), "object_records": records,
               "declared_objects": len(objects), "roles": sorted(roles),
               "errors": len(errors), "warnings": len(warnings)}
    return errors, warnings, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar")
    ap.add_argument("--required-role", action="append", default=[])
    ap.add_argument("--no-path-check", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    with open(args.sidecar) as f:
        data = json.load(f)
    errors, warnings, summary = validate(
        data, args.sidecar, args.required_role, not args.no_path_check)
    report = {"valid": not errors, "summary": summary,
              "errors": errors, "warnings": warnings}
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
