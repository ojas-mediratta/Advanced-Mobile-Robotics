# Project 3b & 3c: Pose-Graph SLAM 

## Overview

Build a pose-graph SLAM pipeline using your ICP from P3a. Convert wheel encoder readings to odometry, construct factor graphs with odometry and ICP factors using GTSAM, and extend to multi-robot SLAM with inter-robot range factors. Run your pipeline on both simulated and real Robotarium data to build maps.

## Deadline

**Friday, April 3, 11:59 PM ET.**

- Submit `slam.py` to Gradescope.
- Submit P3c Reflection as PDF on Gradescope.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate amr_p3b
```

On Windows, the environment installs `gtsam` from `conda-forge`. The previous `pip` dependency `gtsam-develop` does not currently provide a usable `win-64` wheel for this setup.

### 2. Install the project library and Robotarium simulator

```bash
pip install -e .
pip install -e ./robotarium_python_simulator
```

Verify:

```bash
python -c "import p3_lib; import rps.robotarium; print('OK')"
```

## What to Implement

Three functions in [p3_lib/slam.py](p3_lib/slam.py):

| Function | Points | Description |
|---|---|---|
| `encoders_to_odometry` | 3 | Wheel encoder ticks -> SE2 poses via differential drive kinematics |
| `build_single_robot_graph` | 6 | Single-robot factor graph: prior + odometry + ICP factors |
| `build_multi_robot_graph` | 6 | Multi-robot factor graph: per-robot subgraphs + UWB range factors |

Note: A working `icp.py` is provided so all students start P3b on equal footing, regardless of P3a results. 

The notebook [Project_3b_Slam.ipynb](notebooks/Project_3b_Slam.ipynb) walks you through each task and section. You will need to have real sensor data from Robotarium for the last section of the notebook.

## Real Robots in Robotarium - Start Early!

You will collect your own robot data from [Robotarium](https://www.robotarium.gatech.edu/). You do **NOT** need to finish the SLAM code first. 

### How to submit

1. Open [`scripts/robotarium_experiment.py`](scripts/robotarium_experiment.py) and review the script. It drives 5 robots through the arena collecting encoder data, (simulated) lidar scans, and (simulated) range measurements.

2. **Local test first**. Set `LOCAL_SIM = True` at the top of the script, then run it:
   ```bash
   cd scripts
   python robotarium_experiment.py
   ```
   Check that it completes without errors and produces `.npy` files.

3. **Submit to the real Robotarium:**
   - Set `LOCAL_SIM = False` in the script.
   - Go to [robotarium.gatech.edu](https://www.robotarium.gatech.edu) and create an account if you don't have one (approval takes 1–2 business days).
   - Create a new experiment and fill the experiment description:
     - Title: `AMR - <gt username>`
     - Estimated Duration: 200 seconds
     - Number of Robots: 5
   - Upload `robotarium_experiment.py` as the main file.
   - Click **Submit**.

4. **Wait for results** (turnaround is typically **~1 day**). You'll get an email when it's done.

5. **Download** the `.npy` data files and place them in a folder, e.g., `real_robotarium/run1/`. You will use these to run your SLAM pipeline locally in Section 5 of the notebook.

Note: You will need 2 complete Robotarium runs to complete 3c. The script auto-randomizes exploration parameters each time, so just submit it twice, each submission produces a unique trajectory and dataset. You may optionally tweak the config at the top of the script (seed, speed, detection distance, turn range) but this is not required or needed.

**Important:**
- **Always test in the simulator first** (`LOCAL_SIM = True`) before submitting to the real Robotarium. This saves you (and everyone else in the queue) time.
- Experiments can occasionally fail (e.g., "Collision Detected! Stopping experiment!" in the log file). This is normal - just resubmit. 


## Extra Credit

### Early Bird Robotarium 

Submit your first Robotarium experiment this week to earn **1% bonus on your final course grade**. This is a flat bonus -- the goal is to get familiar with the Robotarium submission workflow early. 

**To claim:** Follow instruction in P3c Q10. The Robotarium `Created On` timestamp should be **before Friday, March 20, 5 PM ET**.

### Early P3c Submission

Submit P3c early to earn **0.5% per day** before the deadline, **scaled by your P3c score**. You must score **>=70%** on P3c to be eligible. 

>Example 1: You submit 4 days early and score 85%. Your bonus = 4 × 0.5 × 0.85 = 1.7%.

>Example 2: You submit 15 days early and score 69%. Your bonus = 0%.

Note: The best strategy is simply to do your best work as early as you can and submit when done. TAs are happy to discuss concepts, but won't be able to give feedback on individual answers or release grades early. We reserve the right to withhold extra credit for submissions that do not demonstrate a good-faith effort to learn the materials.
