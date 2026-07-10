"""
Custom .mcap -> merged-CSV extractor. Written because ros2_unbag (the lab's
exporter) currently cannot decode the ZED compressedDepth topic, so bags with
depth arrive as raw .mcap with no merged/synchronised export.

No ROS2 install required: .mcap files are self-describing (msg schemas are
embedded in the file), so mcap-ros2-support decodes standard message types
(WrenchStamped, PoseStamped, CompressedImage) without rclpy/rosidl/rosbag2_py.

Output layout matches the documented trial schema (CLAUDE.md "Trial data
layout"): a merged CSV keyed on the highest-rate topic (current_pose), a PNG
folder for RGB, and a 16-bit-PNG folder (millimetres) for depth.

compressedDepth decode: ROS's compressed_depth_image_transport prefixes the
PNG payload with a 12-byte header (3x float32 LE: format flag, depthQuantA,
depthQuantB). For 32FC1 source encoding the PNG stores a 16-bit quantized
code; true depth (metres) = depthQuantA / (code - depthQuantB) for code != 0.
We store the recovered depth as millimetres in a 16-bit PNG (0 = invalid),
which is lossless and matches common RGB-D dataset convention.

Usage:
    .venv_analysis/bin/python Code/mcap_extract.py \
        --bag <path/to/trial_0.mcap> --trial_name lfdws_t001_depth --out Data
"""
import argparse
import io
import os
import struct

import numpy as np
import pandas as pd
from PIL import Image
from mcap_ros2.reader import read_ros2_messages

RGB_TOPIC = "/zed/zed_node/rgb/color/rect/image/compressed"
DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered/compressedDepth"
POSE_TOPIC = "/NS_1/franka_robot_state_broadcaster/current_pose"
IMAGE_TOPICS = {RGB_TOPIC, DEPTH_TOPIC}


def topic_prefix(topic):
    # "/NS_1/franka_robot_state_broadcaster/current_pose" -> "NS_1.franka_robot_state_broadcaster.current_pose"
    return topic.strip("/").replace("/", ".")


def flatten(msg, prefix=""):
    """mcap-ros2-support decodes messages into dynamic slot-based objects
    (no get_fields_and_field_types like real rclpy messages), so recurse on
    __slots__ instead.

    A list of SCALARS (e.g. JointState.position -> [0.029, 0.029]) is
    written as ONE column holding a Python-list-repr string ("[0.029,
    0.029]"), matching ros2_unbag's convention -- the rest of the pipeline
    parses gripper width via ast.literal_eval on exactly that shape. This
    was originally written to explode into name[0]/name[1]/... indexed
    columns instead, which silently broke gripper-width parsing the first
    time a gripper-bearing bag actually went through this extractor (see
    Docs/FAILURE_MODES.md) -- lists of SUB-MESSAGES (not currently present
    in our topic set, but handled for safety) still explode per index,
    since there's no single-column representation for those."""
    out = {}
    for field in msg.__slots__:
        val = getattr(msg, field)
        key = f"{prefix}{field}"
        if hasattr(val, "__slots__"):
            out.update(flatten(val, prefix=f"{key}."))
        elif isinstance(val, (list, tuple, np.ndarray)):
            val_list = list(val)
            if val_list and all(hasattr(v, "__slots__") for v in val_list):
                for i, v in enumerate(val_list):
                    out.update(flatten(v, prefix=f"{key}[{i}]."))
            else:
                out[key] = str(val_list)
        else:
            out[key] = val
    return out


def decode_compressed_depth(data: bytes):
    """Return (depth_mm uint16 array, ok bool)."""
    if len(data) < 12:
        return None, False
    fmt, quant_a, quant_b = struct.unpack("<fff", data[:12])
    try:
        im = Image.open(io.BytesIO(data[12:]))
        code = np.array(im, dtype=np.float32)
    except Exception as e:
        print(f"  [depth-decode-fail] {e}", flush=True)
        return None, False
    depth_m = np.zeros_like(code, dtype=np.float32)
    valid = code != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m[valid] = quant_a / (code[valid] - quant_b)
    depth_m[~np.isfinite(depth_m)] = 0
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    return depth_mm, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--trial_name", required=True)
    ap.add_argument("--out", default="Data")
    args = ap.parse_args()

    trial_dir = os.path.join(args.out, args.trial_name)
    rgb_dir = os.path.join(trial_dir, "zed_zed_node_rgb_color_rect_image_compressed")
    depth_dir = os.path.join(trial_dir, "zed_zed_node_depth_depth_registered_compressedDepth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"[read] {args.bag}", flush=True)

    per_topic_rows = {}   # topic -> list[dict]
    n_msgs = 0
    n_rgb = n_depth = n_depth_fail = 0

    for msg in read_ros2_messages(args.bag):
        topic = msg.channel.topic
        r = msg.ros_msg
        n_msgs += 1
        prefix = topic_prefix(topic)
        ts_ns = msg.log_time_ns if hasattr(msg, "log_time_ns") else msg.publish_time_ns

        if topic in IMAGE_TOPICS:
            img_id = str(ts_ns)
            if topic == RGB_TOPIC:
                png_path = os.path.join(rgb_dir, f"{img_id}.png")
                if not os.path.exists(png_path):
                    Image.open(io.BytesIO(bytes(r.data))).convert("RGB").save(png_path)
                    n_rgb += 1
            else:
                depth_mm, ok = decode_compressed_depth(bytes(r.data))
                if ok:
                    Image.fromarray(depth_mm, mode="I;16").save(
                        os.path.join(depth_dir, f"{img_id}.png"))
                    n_depth += 1
                else:
                    n_depth_fail += 1
            row = {f"{prefix}.timestamp": ts_ns, f"{prefix}": img_id}
        else:
            flat = flatten(r)
            row = {f"{prefix}.timestamp": ts_ns}
            row.update({f"{prefix}.{k}": v for k, v in flat.items()})

        per_topic_rows.setdefault(topic, []).append(row)

        if n_msgs % 20000 == 0:
            print(f"  [progress] {n_msgs} messages  (rgb={n_rgb} depth={n_depth} "
                  f"depth_fail={n_depth_fail})", flush=True)

    print(f"[done reading] {n_msgs} messages total, "
          f"rgb={n_rgb} depth={n_depth} depth_fail={n_depth_fail}", flush=True)

    if POSE_TOPIC not in per_topic_rows:
        print(f"[fatal] base topic {POSE_TOPIC} not found in bag", flush=True)
        return

    base_df = pd.DataFrame(per_topic_rows[POSE_TOPIC]).sort_values(
        f"{topic_prefix(POSE_TOPIC)}.timestamp").reset_index(drop=True)
    ts_col = f"{topic_prefix(POSE_TOPIC)}.timestamp"
    merged = base_df

    for topic, rows in per_topic_rows.items():
        if topic == POSE_TOPIC:
            continue
        df = pd.DataFrame(rows).sort_values(
            f"{topic_prefix(topic)}.timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            merged, df,
            left_on=ts_col, right_on=f"{topic_prefix(topic)}.timestamp",
            direction="nearest")
        print(f"[merge] asof-joined {topic}  ({len(df)} rows)", flush=True)

    out_csv = os.path.join(trial_dir, f"{args.trial_name}_0.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}  ({len(merged)} rows, {len(merged.columns)} cols)", flush=True)
    print(f"[write] {rgb_dir}  ({n_rgb} frames)", flush=True)
    print(f"[write] {depth_dir}  ({n_depth} frames, {n_depth_fail} failed)", flush=True)


if __name__ == "__main__":
    main()
