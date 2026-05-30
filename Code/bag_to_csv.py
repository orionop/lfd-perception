"""
Generated with help of ChatGPT on 251110.
Requires ros2 to be sourced.
Requires the bag file to be in MCAP format.
Requires message definitions to be available for non-standard ROS2 messages.
Scans the folder and exports for all folders with ros2_bag substring.
"""

import argparse
import os
import io
import csv
from PIL import Image
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def flatten_ros_message(msg, prefix=""):
    """Recursively flatten a ROS message into a flat dict."""
    flat_dict = {}
    for field_name in msg.get_fields_and_field_types().keys():
        value = getattr(msg, field_name)
        field_key = f"{prefix}{field_name}"

        if hasattr(value, "get_fields_and_field_types"):
            # Nested ROS message
            flat_dict.update(flatten_ros_message(value, prefix=f"{field_key}/"))
        elif isinstance(value, (list, tuple)):
            # Lists or arrays
            for idx, val in enumerate(value):
                if hasattr(val, "get_fields_and_field_types"):
                    flat_dict.update(flatten_ros_message(val, prefix=f"{field_key}[{idx}]/"))
                else:
                    flat_dict[f"{field_key}[{idx}]"] = val
        else:
            flat_dict[field_key] = value
    return flat_dict


class SpecialHeaders(object):
    BAG_WRITE_STAMP = "//bag_write_stamp"



class ImageHandler(object):
    """
    To deal with images that should not be written to csv.
    """

    IMAGE_RAW = "sensor_msgs/msg/Image"
    IMAGE_COMPRESSED = "sensor_msgs/msg/CompressedImage"

    def __init__(self):
        self.filename_counters = {}
        return

    def to_png(self, topic, msg, msg_type, output_dir):
        """
        Only for image types.
        Writes the message to a .png file, while mapping its filename to a timestamp
        in the output dictionary (which gets written to csv).
        """

        if topic not in self.filename_counters.keys():
            self.filename_counters[topic] = 0
        self.filename_counters[topic] += 1

        # Put the image data in a png file, and overwrite the image data in the
        # flattened message with the image filename.
        match msg_type:
            case self.IMAGE_RAW:
                filename = os.path.join(output_dir, f"{self.filename_counters[topic]:05d}_r.png")

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
                    msg = np.frombuffer(msg.data, dypte=np.uint16)
                    msg = msg.reshape((msg.height, msg.width))
                    image = Image.fromarray(msg)
                else:
                    raise NotImplementedError(f"Failed export for image with encoding {msg.encoding}.")
                
            case self.IMAGE_COMPRESSED:
                filename = os.path.join(output_dir, f"{self.filename_counters[topic]:05d}_c.png")

                image = Image.open(io.BytesIO(msg.data))
            case _:
                pass
            
        image.save(filename, format="PNG")

        flat_dict = flatten_ros_message(msg)
        self.replace_image_data(flat_dict, filename)

        return flat_dict
    

    def replace_image_data(self, flat_dict, data):
        """
        Replace the image with the input data

        NOTE: WARNING: assumes only 1 field with "data" exists in the message
        """

        for key in flat_dict.keys():
            if "data" in key:
                break

        flat_dict[key] = data

        return


def discover_headers(bag_path, common_headers=None):
    """
    Discover all headers in flattened ROS2 messages.
    We need these up front for writing to a single csv file.
    Note that the headers are prepended by the topic name, to prevent overlap
    in field names accross different topics.

    =INPUT=
        bag_path - string
            absolute path to bag file
        [common_headers] - list of strings
            names of additional headers to include, common name over all topics
    """

    all_headers = set()

    # Open bag
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic info
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_dict = {tats.name: tats.type for tats in topics_and_types}
    if not len(topic_type_dict.keys()):
        return all_headers

    # Read until we've encountered each topic at least once and got its keys
    all_topic_names = sorted(topic_type_dict.keys())
    topics_read = []
    while topics_read != all_topic_names:
        topic_name, data, _ = reader.read_next()
        if topic_name in topics_read:
            continue
        topics_read.append(topic_name)
        topics_read = sorted(topics_read)
        msg_type = get_message(topic_type_dict[topic_name])
        msg = deserialize_message(data, msg_type)
        flat_msg = flatten_ros_message(msg)
        all_headers.update(["/".join((topic_name, key)) for key in flat_msg.keys()])

    # Force closure of the reader
    del reader

    # Add common header for bagging timestamp
    if common_headers is not None:
        all_headers.update(common_headers)

    return sorted(all_headers)


def bag_to_csv_all_topics(bag_path, output_dir=None):
    """
    Read all topics from an MCAP bag and export each one to a CSV.

    =INPUT= 
        bag_path - string
            Absolute path to bag file.
        [output_dir] - string
            Absolute path to output directory. Will be created if not exists.
    """
    
    fids = {}
    writers = {}
    
    if output_dir is None:
        output_dir = os.path.dirname(bag_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get all headers
    common_headers = [SpecialHeaders.BAG_WRITE_STAMP]
    all_headers = discover_headers(bag_path, common_headers)

    # Special handlers
    image_handler = ImageHandler()

    # Open bag
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic info
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_dict = {tats.name: tats.type for tats in topics_and_types}

    # Create writer for overarching csv: filename same as folder it is in
    filename = output_dir.split("/")[-1]
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    fids["all_data"] = open(csv_path, "w", newline="")
    writers["all_data"] = csv.DictWriter(fids["all_data"], fieldnames=all_headers)
    writers["all_data"].writeheader()

    # Read all data and write to csv immediately
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()

        # Get flattened message
        msg_type = get_message(topic_type_dict[topic_name])
        msg_type_str = topic_type_dict[topic_name]
        msg = deserialize_message(data, msg_type)
        if (msg_type_str == image_handler.IMAGE_RAW or
                msg_type_str == image_handler.IMAGE_COMPRESSED):
            flat_msg = image_handler.to_png(topic_name, msg, msg_type_str, output_dir)
        else:
            flat_msg = flatten_ros_message(msg)
        
        flat_msg[SpecialHeaders.BAG_WRITE_STAMP] = timestamp / 1e9  # nano to sec

        # Create CSV writer on first encounter of topic
        if topic_name not in writers.keys():
            csv_path = os.path.join(output_dir, f'{topic_name.strip("/").replace("/", "_")}.csv')
            fids[topic_name] = open(csv_path, "w", newline="")
            fieldnames = sorted(flat_msg.keys())
            writers[topic_name] = csv.DictWriter(fids[topic_name], fieldnames=fieldnames)
            writers[topic_name].writeheader()
            
        # Write data to correct csv
        writers[topic_name].writerow(flat_msg)

        # Write data to overarching csv. Requires blanks to be filled in data.
        total_msg = dict.fromkeys(all_headers, "")
        total_msg.update({
            ("/".join((topic_name, key)) if key not in common_headers else key): value 
            for key, value in flat_msg.items()})
        writers["all_data"].writerow(total_msg)

    # Close all files
    for fid in fids.values():
        fid.close()
    
    print(f"\nExport done.")



if __name__ == "__main__":
    """
    --root is ignored if --path is specified

    Root path is scanned for all possible bag folders, which are then exported.
    Path is a path to a specific bag folder.
    """

    # Parse root path, if provided.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--path")
    args = parser.parse_args()
    
    if args.path:
        bag_path = args.path

        # Process specific bag folder
        for bag_file in os.listdir(bag_path):
            if ".mcap" in bag_file:
                break
        
        bag_to_csv_all_topics(os.path.join(bag_path, bag_file))

    else:
        # Scan root for all possible bag folders
        root = args.root if args.root else "/"
        content = os.listdir(root)
        bag_paths = [path for path in content if "ros2_bag" in path and "." not in path]

        for bag_path in bag_paths:
            for bag_file in os.listdir(os.path.join(root, bag_path)):
                if ".mcap" in bag_file:
                    break
            
            bag_to_csv_all_topics(os.path.join(root, bag_path, bag_file))
