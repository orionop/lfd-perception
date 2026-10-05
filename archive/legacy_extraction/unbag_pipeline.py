"""
ROS2 Bag to CSV Pipeline — University of Twente / NAKAMA Robotics Lab
Replaces bag_to_csv.py with a modular approach using ros2 bag CLI.

Usage:
    python3 unbag_pipeline.py --path /path/to/bag_folder
    python3 unbag_pipeline.py --path /path/to/bag_folder --output /path/to/output
    python3 unbag_pipeline.py --root /path/to/scan   (scans for ros2_bag folders)

Steps:
    1. extract  — uses rosbag2_py to read MCAP bag, writes per-topic CSVs
    2. images   — image topics saved as PNG, filename reference in CSV
    3. merge    — combines all per-topic CSVs into one combined CSV

Requires:
    - ROS2 sourced
    - MCAP format bag files
    - Message definitions available for non-standard messages
    - PIL (Pillow), numpy
"""

import argparse
import os
import io
import csv
import json
import numpy as np
from PIL import Image
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# =============================================================================
# Constants
# =============================================================================

BAG_WRITE_STAMP = "//bag_write_stamp"
IMAGE_RAW = "sensor_msgs/msg/Image"
IMAGE_COMPRESSED = "sensor_msgs/msg/CompressedImage"
IMAGE_TYPES = {IMAGE_RAW, IMAGE_COMPRESSED}


# =============================================================================
# Message Flattening (preserved from original)
# =============================================================================

def flatten_ros_message(msg, prefix=""):
    """Recursively flatten a ROS message into a flat dict."""
    flat_dict = {}
    for field_name in msg.get_fields_and_field_types().keys():
        value = getattr(msg, field_name)
        field_key = f"{prefix}{field_name}"

        if hasattr(value, "get_fields_and_field_types"):
            flat_dict.update(flatten_ros_message(value, prefix=f"{field_key}/"))
        elif isinstance(value, (list, tuple)):
            for idx, val in enumerate(value):
                if hasattr(val, "get_fields_and_field_types"):
                    flat_dict.update(flatten_ros_message(val, prefix=f"{field_key}[{idx}]/"))
                else:
                    flat_dict[f"{field_key}[{idx}]"] = val
        else:
            flat_dict[field_key] = value
    return flat_dict


# =============================================================================
# Image Handling (preserved from original)
# =============================================================================

def save_image_as_png(msg, msg_type_str, topic_name, output_dir, counter):
    """
    Save a ROS image message as PNG. Returns the filename.
    Handles raw (sensor_msgs/msg/Image) and compressed images.
    """
    topic_safe = topic_name.strip("/").replace("/", "_")
    if msg_type_str == IMAGE_RAW:
        filename = os.path.join(output_dir, f"{topic_safe}_{counter:05d}_r.png")

        if msg.encoding in ["rgb8", "bgr8", "rgba8", "bgra8"]:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if "a" not in msg.encoding:
                img = img.reshape((msg.height, msg.width, 3))
            else:
                img = img.reshape((msg.height, msg.width, 4))
            if "bgr" in msg.encoding:
                img[..., 0:3] = img[..., 2::-1]
            image = Image.fromarray(img, "RGB")
        elif msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape((msg.height, msg.width))
            image = Image.fromarray(img, "L")
        elif msg.encoding == "16UC1":
            img = np.frombuffer(msg.data, dtype=np.uint16)
            img = img.reshape((msg.height, msg.width))
            image = Image.fromarray(img)
        else:
            raise NotImplementedError(f"Unsupported image encoding: {msg.encoding}")

    elif msg_type_str == IMAGE_COMPRESSED:
        filename = os.path.join(output_dir, f"{topic_safe}_{counter:05d}_c.png")
        image = Image.open(io.BytesIO(msg.data))
    else:
        raise ValueError(f"Not an image type: {msg_type_str}")

    image.save(filename, format="PNG")
    return filename


def flatten_image_message(msg, png_filename):
    """Flatten an image message, replacing binary data with the PNG filename."""
    flat_dict = flatten_ros_message(msg)
    for key in flat_dict.keys():
        if "data" in key:
            flat_dict[key] = png_filename
            break
    return flat_dict


# =============================================================================
# Step 1: Extract bag → per-topic CSVs + image PNGs
# =============================================================================

def extract_bag(bag_path, output_dir):
    """
    Read an MCAP bag file and write per-topic CSVs.
    Image topics are saved as PNGs with filename references in the CSV.

    Returns a list of per-topic CSV filenames written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open bag
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic → type mapping
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_dict = {t.name: t.type for t in topics_and_types}

    if not topic_type_dict:
        print("No topics found in bag.")
        return []

    print(f"Found {len(topic_type_dict)} topics:")
    for name, typ in sorted(topic_type_dict.items()):
        tag = " [IMAGE]" if typ in IMAGE_TYPES else ""
        print(f"  {name} ({typ}){tag}")

    # State
    fids = {}
    writers = {}
    image_counters = {}
    csv_files = []

    # Read all messages
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()

        msg_type_str = topic_type_dict[topic_name]
        msg_type = get_message(msg_type_str)
        msg = deserialize_message(data, msg_type)

        # Handle image vs non-image
        if msg_type_str in IMAGE_TYPES:
            if topic_name not in image_counters:
                image_counters[topic_name] = 0
            image_counters[topic_name] += 1

            png_filename = save_image_as_png(
                msg, msg_type_str, topic_name, output_dir, image_counters[topic_name]
            )
            flat_msg = flatten_image_message(msg, png_filename)
        else:
            flat_msg = flatten_ros_message(msg)

        # Add bag write timestamp
        flat_msg[BAG_WRITE_STAMP] = timestamp / 1e9

        # Create CSV writer on first encounter
        if topic_name not in writers:
            csv_name = f'{topic_name.strip("/").replace("/", "_")}.csv'
            csv_path = os.path.join(output_dir, csv_name)
            csv_files.append(csv_path)
            fids[topic_name] = open(csv_path, "w", newline="")
            fieldnames = sorted(flat_msg.keys())
            writers[topic_name] = csv.DictWriter(fids[topic_name], fieldnames=fieldnames)
            writers[topic_name].writeheader()

        writers[topic_name].writerow(flat_msg)

    # Close all files
    for fid in fids.values():
        fid.close()

    del reader

    # Save topic name → csv filename mapping for merge step
    topics_map = {csv_path: topic_name for topic_name, csv_path in
                  zip(writers.keys(), csv_files)}
    map_path = os.path.join(output_dir, "topics_map.json")
    with open(map_path, "w") as f:
        json.dump(topics_map, f, indent=2)

    print(f"\nExtracted {len(csv_files)} per-topic CSVs to {output_dir}")
    return csv_files


# =============================================================================
# Step 2: Merge per-topic CSVs into one combined CSV
# =============================================================================

def merge_csvs(output_dir, csv_files, output_filename=None):
    """
    Merge all per-topic CSVs into one combined CSV.
    Headers are prefixed with the topic name (derived from filename).
    Rows are sorted by bag_write_stamp.
    """
    if not csv_files:
        print("No CSV files to merge.")
        return

    # Load topic name → csv path mapping written during extract
    map_path = os.path.join(output_dir, "topics_map.json")
    topics_map = {}
    if os.path.exists(map_path):
        with open(map_path) as f:
            topics_map = json.load(f)
    # Invert: csv_path → topic_name
    csv_to_topic = {v: k for k, v in topics_map.items()}

    # Collect all rows with topic-prefixed headers
    all_rows = []
    all_headers = set()

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            continue

        # Use original topic name if available, else fall back to filename
        topic_prefix = csv_to_topic.get(csv_path, None)
        if topic_prefix is None:
            basename = os.path.splitext(os.path.basename(csv_path))[0]
            topic_prefix = "/" + basename

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefixed_row = {}
                for key, value in row.items():
                    if key == BAG_WRITE_STAMP:
                        prefixed_row[key] = value
                    else:
                        prefixed_key = f"{topic_prefix}/{key}"
                        prefixed_row[prefixed_key] = value

                all_headers.update(prefixed_row.keys())
                all_rows.append(prefixed_row)

    # Sort by timestamp
    all_rows.sort(key=lambda r: float(r.get(BAG_WRITE_STAMP, 0)))

    # Write combined CSV
    if output_filename is None:
        folder_name = os.path.basename(output_dir.rstrip("/"))
        output_filename = f"{folder_name}.csv"
    combined_path = os.path.join(output_dir, output_filename)

    sorted_headers = sorted(all_headers)
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_headers, restval="")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Merged {len(csv_files)} CSVs → {combined_path} ({len(all_rows)} rows)")
    return combined_path


# =============================================================================
# Main: full pipeline
# =============================================================================

def run_pipeline(bag_path, output_dir=None):
    """Run extract → merge pipeline on a single bag file."""
    if output_dir is None:
        output_dir = os.path.dirname(bag_path)

    print(f"=== Processing: {bag_path} ===\n")

    # Step 1: Extract
    csv_files = extract_bag(bag_path, output_dir)

    # Step 2: Merge
    if csv_files:
        merge_csvs(output_dir, csv_files)

    print(f"\n=== Done ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS2 bag to CSV pipeline with image handling and CSV merging."
    )
    parser.add_argument("--path", help="Path to a specific bag folder")
    parser.add_argument("--root", help="Scan directory for ros2_bag folders")
    parser.add_argument("--output", help="Output directory (default: same as bag folder)")
    args = parser.parse_args()

    if args.path:
        # Process specific bag folder
        bag_path = args.path
        for bag_file in os.listdir(bag_path):
            if ".mcap" in bag_file:
                break
        run_pipeline(os.path.join(bag_path, bag_file), args.output)

    else:
        # Scan root for all bag folders
        root = args.root if args.root else "/"
        content = os.listdir(root)
        bag_paths = [p for p in content if "ros2_bag" in p and "." not in p]

        for bag_folder in bag_paths:
            for bag_file in os.listdir(os.path.join(root, bag_folder)):
                if ".mcap" in bag_file:
                    break
            run_pipeline(
                os.path.join(root, bag_folder, bag_file),
                args.output
            )
