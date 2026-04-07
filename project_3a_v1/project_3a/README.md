# Project 3a: ICP

## Overview

Implement the Iterative Closest Point (ICP) algorithm for 2D scan alignment. Test on real LiDAR data from the Deutsches Museum and build a map by chaining ICP.

## Deadline

**Monday, March 16, 11:59 PM ET.** Submit `icp.py` to Gradescope (10 pts total).

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate amr_p3a
```

### 2. Install the project library

```bash
pip install -e .
```

Verify:

```bash
python -c "import p3_lib; print('p3_lib OK')"
```

### 3. Download and extract the LiDAR dataset

This downloads the rosbag (~500 MB) from the [Google Cartographer](https://google-cartographer-ros.readthedocs.io/en/latest/data.html) public dataset and extracts 529 LiDAR scans:

```bash
python scripts/extract_real_lidar.py
```

This saves `data/real_lidar/deutsches_museum_scans.npz`. The notebook needs this file.

## What to implement

Three functions in `p3_lib/icp.py`:

| Function | Points | Description |
|---|---|---|
| `find_correspondences` | 3 | Nearest-neighbor matching via KDTree + distance filtering |
| `compute_transform` | 1 | Best-fit rigid transform via `gtsam.Pose2.Align()` |
| `icp` | 6 | Full iterative loop combining the above |

The notebook `notebooks/Project_3a_ICP.ipynb` walks you through each task with tests and visualizations.
