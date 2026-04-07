# Project 1: Swim with Invariant Filters

## Project Structure

This project consists of four distinct components, each with its own submission on Gradescope:
- [Project 1A](project_1a/project_1a.ipynb) should have already been submitted and was about the state of the robot.
- [Project 1B](project_1b/project_1b.ipynb) is the main programming assignment, and will be about state estimation and trajectory following. You will be primarily working and adding code to [brain.py](project_1b/controllers/ROV_controller/brain.py). 
- [Project 1C](project_1c_reflection.pptx) will allow you to reflect on what you did. Feel free to complete that as you're completing the assignment.
- [Project 1D](project_1d_team.md) is where you will submit a deliverable with your team. 

## Environment Setup

### macOS / Linux

1. **Create a Conda environment:**
   ```bash
   conda create -n amr python=3.12 plotly kaleido==0.2.1
   conda activate amr
   ```

2. **Install GTSAM:**
   ```bash
   pip install gtsam-develop
   ```

3. **Install Webots:**
   - Download Webots from the [official website](https://cyberbotics.com/) or their [GitHub releases](https://github.com/cyberbotics/webots/releases)
   - Follow the installation instructions for your operating system

4. **Configure Webots to use your Conda Python:**
   - Activate your Conda environment: `conda activate amr`
   - Find your Python path: `which python`
   - Open Webots and navigate to **Tools → Preferences**
   - Paste the Python path into the **Python command** field
   - Press the refresh button next to the simulation time window for changes to take effect

### Windows (WSL2)

Please refer to the detailed WSL2 setup instructions posted on Piazza:
**https://piazza.com/class/mk62vbxt9lq5dp/post/14**


## Submission

Submit the following as separate assignments to **Gradescope**:

1. Project 1A was already submitted
2. Project 1B, submit your completed `brain.py` (only!)
3. Project 1C, submit your reflection question answers (PDF)
4. Project 1D, submit your team deliverable (one PDF per team)

**Deadline:** *Friday February 6, 11:59 PM ET*

## Resources

- Course lectures on SE(3), state estimation, and control
- GTSAM documentation: https://gtsam.org/
- Webots documentation: https://cyberbotics.com/doc/guide/index


Good luck, and have fun swimming with your underwater robot!
