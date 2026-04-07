"""ROV_1 controller."""

import numpy as np
from brain import Brain
from controller import Keyboard, Supervisor
from logger import Logger
from robot import BEACONS, Robot
from trajectory import Trajectory

# Webots interaction
supervisor: Supervisor = Supervisor()
timestep_ms = int(supervisor.getBasicTimeStep())
keyboard: Keyboard = supervisor.getKeyboard()
keyboard.enable(1000)

# Initialize robot
rov = Robot(supervisor)
brain = Brain(BEACONS)
keyboard_control = False  # Set to False to use trajectory tracking controller
logger = Logger()
trajectory = Trajectory()

# Initial EKF covariance
P0 = np.eye(9) * 0.1  # std ~ 0.316 on each component

k = 0  # step counter
t_prev = 0.0
while supervisor.step(timestep_ms) != -1:
    t = supervisor.getTime()

    # SENSE
    key: int = keyboard.getKey()  # Get the latest key pressed
    f_b = rov.read_accel()
    omega_b = rov.read_gyro()
    measurements = rov.sense(k)
    X_true = measurements.X_true
    dt = t - t_prev
    t_prev = t
    k += 1

    # THINK
    if k == 1:
        assert X_true is not None, "True state is required here."
        brain.initialize_EKF(X_true, P0)  # Initialize with ground-truth state
        continue

    # EKF predict
    X_est, P = brain.EKF_predict(omega_b, f_b, dt)

    # Update with measurements
    brain.EKF_update(measurements)

    # ACT
    if keyboard_control:
        desired_state = None
        control = brain.act_on_command(key, keyboard)
    else:
        desired_state = trajectory.query(t)
        control = brain.follow_step(t, X_true, desired_state)

    rov.set_motor_velocities(control)

    # LOGGING AND PLOTTING
    logger.log_state(t, X_true, X_est, P, desired_state)

    if key == ord("A"):  # Plot Attitude estimation results
        logger.plot_attitude()
    elif key == ord("P"):  # Plot Position estimation results
        logger.plot_position()
    elif key == ord("V"):  # Plot Velocity estimation results
        logger.plot_velocity()
    elif key == ord("T"):  # Plot trajectory tracking results
        logger.plot_trajectory_tracking_3d()
