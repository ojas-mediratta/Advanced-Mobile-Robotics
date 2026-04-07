import numpy as np
import gtsam

# Home pose M at theta = [0, 0, 0]
M_rot = gtsam.Rot3(np.array([
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0]
]))
M_trans = gtsam.Point3(0.16604, -0.59098, -0.04271)
M = gtsam.Pose3(M_rot, M_trans)

# Body screw axes B = [omega_x, omega_y, omega_z, v_x, v_y, v_z]^T
B1 = np.array([0.0, -1.0,  0.0, 0.11323, 0.00000, -0.59158]) # Shoulder Abduction
B2 = np.array([0.0,  0.0, -1.0, 0.04270, 0.59105,  0.00000]) # Shoulder Rotation
B3 = np.array([0.0,  0.0, -1.0, 0.22504, 0.27132,  0.00000]) # Elbow/Piston
B_list = [B1, B2, B3]