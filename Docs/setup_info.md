# ROS 2 Pipeline Setup Instructions

These instructions will guide you through running the data extraction pipeline. It extracts data from ROS 2 bag files (MCAP format) into per-topic CSV files, saves images as PNG files, and then merges everything into a single combined CSV file based on writing timestamps.

The pipeline is tested and guaranteed to work on both **ROS 2 Humble (Ubuntu 22.04)** and **ROS 2 Jazzy (Ubuntu 24.04)**.
## Step-by-Step Instructions

### Step 1: Install Required System Dependencies

First, ensure that the system has the tools to read MCAP bag files. Open a terminal and run:

```bash
# If using ROS 2 Humble:
sudo apt install ros-humble-rosbag2-storage-mcap -y

# If using ROS 2 Jazzy:
sudo apt install ros-jazzy-rosbag2-storage-mcap -y
```

> **What this does:** This installs the backend storage plugin needed by ROS 2 to read `.mcap` files properly. It may already be installed on a full ROS 2 setup, but this ensures it is present.

### Step 2: Initialize the ROS 2 Environment

Before running any ROS 2 commands, the environment variables must be loaded:

```bash
# If using ROS 2 Humble:
source /opt/ros/humble/setup.bash

# If using ROS 2 Jazzy:
source /opt/ros/jazzy/setup.bash
```

> **What this does:** This makes ROS 2 available in your current terminal session.

### Step 3: Create a Dedicated Python Environment

It is highly recommended to isolate Python packages using a virtual environment:

```bash
python3 -m venv --system-site-packages venv
```

> **What this does:** This creates a folder named `venv` that holds our Python dependencies. The `--system-site-packages` flag ensures that the environment still has access to all the core ROS 2 Python libraries.

### Step 4: Activate the Environment and Install Libraries

Activate the newly created environment and install the required utility packages:

```bash
source venv/bin/activate
pip install pillow numpy
```

> **What this does:** The `source` command activates the isolated environment. The `pip install` command downloads the libraries required to process and save images as `.png` files.

### Step 5: Run the Extraction Pipeline

Finally, run the Python scripts against your target bag file directory:

```bash
python3 unbag_pipeline.py --path /absolute/or/relative/path/to/bag_folder
```

> **What this does:** This reads the bag file located in the provided folder, exports all the numerical/text data to CSV files, processes and saves image topics as PNGs, and then merges everything into a single synchronized CSV output file inside the same directory.

---

## Important Reminders

*   **For Every New Terminal Session:** Steps 2 and 4 must be re-run whenever you open a new terminal window to process more data.
*   **To Exit the Environment:** Simply type `deactivate` in the terminal when you are done.
