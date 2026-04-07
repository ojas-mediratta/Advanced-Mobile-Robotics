"""
Dynamically tracks TARGET_CUBE and solves IK to reach it.
"""
from controller import Supervisor
import numpy as np
import gtsam
import brain # Imports your updated brain.py
from robot import M, B_list
from brain import forward_kinematics

def clamp_target_position(target_pos, shoulder_pos, max_reach=0.58):
    """
    Clamps the target position to ensure it is within the maximum physical reach of the leg.

    Args:
        target_pos (np.ndarray): The target position as a numpy array [x, y, z].
        shoulder_pos (np.ndarray): The position of the shoulder joint as a numpy array [x, y, z].
        max_reach (float, optional): The maximum reach distance of the leg. Defaults to 0.58.

    Returns:
        np.ndarray: The clamped target position as a numpy array [x, y, z].
    """
    vector_to_target = target_pos - shoulder_pos
    distance = np.linalg.norm(vector_to_target)
    
    if distance > max_reach:
        # Scale the vector down to the max reach boundary
        vector_to_target = (vector_to_target / distance) * max_reach
        return shoulder_pos + vector_to_target
    return target_pos

def main():
    """
    Main function to dynamically track the TARGET_CUBE and solve inverse kinematics (IK) to reach it.

    This function initializes the Webots Supervisor, retrieves motor devices, and continuously updates
    the joint angles to track the target cube using inverse kinematics.

    Steps:
        1. Retrieve motor devices and the target cube node.
        2. Define the system kinematics and workspace limits.
        3. Continuously compute the local target position, clamp it within the workspace, and solve IK.
        4. Command the motors to move to the computed joint angles.
    """

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Get Motor Devices
    motors = [
        supervisor.getDevice("leg shoulder abduction motor"),
        supervisor.getDevice("leg shoulder rotation motor"),
        supervisor.getDevice("leg elbow motor")
    ]

    # Get Target Cube
    target_node = supervisor.getFromDef("TARGET_CUBE")
    if not target_node:
        print("Error: Could not find node DEF TARGET_CUBE.")
        return
    robot_node = supervisor.getSelf()

    # Approximate shoulder origin to calculate workspace limits
    shoulder_pos = np.array([0.0528, 0.0, 0.0]) 

    # Keep track of current joint angles to "warm start" the IK solver
    current_theta = np.array([0.0, 0.0, 0.0])

    print("--- Starting Interactive IK Tracker ---")

    # Main Control Loop
    while supervisor.step(timestep) != -1:
        # Get GLOBAL positions directly from the Webots physics engine
        global_target_pos = np.array(target_node.getPosition())
        global_leg_pos = np.array(robot_node.getPosition())
        
        # Convert Global Target -> Local Target for the IK 
        local_target_pos = global_target_pos - global_leg_pos
        
        # Clamp
        safe_local_target = clamp_target_position(local_target_pos, shoulder_pos)
        target_pose = gtsam.Pose3(M.rotation(), safe_local_target)
        
        # Run Inverse Kinematics
        final_theta = brain.inverse_kinematics(M, B_list, target_pose, current_theta, max_iters=20)
        current_theta = final_theta 
        
        # Command Motors
        for i, motor in enumerate(motors):
            motor.setPosition(final_theta[i])

if __name__ == "__main__":
    main()