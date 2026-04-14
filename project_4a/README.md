# Project 4A: Generalized-ICP

## Overview

Iterative Closest Point (ICP) is a seminal algorithm that forms the basis for much of geometric LiDAR odometry, which has risen in popularity due to the range and accuracy of depth measurements that LiDAR is able to produce over other exteroceptive sensors, like stereo.

In Project 3, you were introduced to point-to-point ICP (traditional), which aims to minimize the distance between point correspondences in two point clouds. Why might this not be ideal? Firstly, it's unlikely we'd ever scan the same point twice. And secondly, it can create a discontinuous error surface when the point-to-point correspondences change significantly between algorithm iterations, leading to unstable optimization.

In part A of this project, you will implement Generalized-ICP (GICP) with GTSAM for real drone data! You will leverage GTSAM's abilities to represent on-manifold states and conduct nonlinear optimization. This modification of ICP reformulates the original algorithm's objective function to achieve a more  robust optimization and avoid some of the nonlinearities associated when using a nearest neighbor  correspondence scheme.

**Project 4A is due on Tuesday, April 14th at 11:59 PM EST.**

## Process

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate amr_p4a
```

### 2. Download the LiDAR dataset

Download the court_of_sciences.bag rosbag from [OneDrive](https://gtvault-my.sharepoint.com/:f:/g/personal/mwoodward8_gatech_edu/IgAfPZcCURS5T6Wbu9TiZbeHAUo6LXxynwLcbDStqHM1K2U?e=kJiTyT). This data is provided by the authors of [DLIO](https://arxiv.org/abs/2203.03749) and contains aerial data of regions of the UCLA campus!

**Place data in `data/`**.

### 3. Deliverables

- Read [Generalized-ICP (Segal, et. al.)](https://www.roboticsproceedings.org/rss05/p21.pdf) and answer reflection questions in [reflection_questions.pptx](./reflection_questions.pptx).
    - **Submit slides in PDF format to the appropriate Gradescope assignment.**

- Navigate to [Project_4a_GICP.ipynb](./Project_4a_GICP.ipynb) for instructions on testing your code implementation.
    - **Submit gicp.py to the appropriate Gradescope assignment.**

## Repository Structure

```
Project_4A/
├── environment.yml                    # conda environment description
│
├── Project_4a_GICP.ipynb              # expore GICP with DLIO dataset
│
├── reflection_questions.md            # reflection questions on Generalized-ICP
│
└── data/
    ├── court_of_sciences.bag
│
└── src/
    ├── gicp.py                        # implementation file
    ├── real_scan_registration.py      # helper file to perform registration on rosbag data
│
└── tests/
    ├── real_scans.npz                 # ground-truth data for tests
    ├── test_data.npz                  # ground-truth data for tests
    ├── test_functions.py              # local tests for functions
    ├── test_registration.py           # local tests for performance on DLIO data
```