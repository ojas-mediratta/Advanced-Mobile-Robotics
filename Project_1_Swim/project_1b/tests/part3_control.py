import numpy as np
import gtsam
from controllers.ROV_controller.brain import Brain, Control
from controllers.ROV_controller.robot import BEACONS

def test_follow_step():
    brain = Brain(BEACONS)
    data = np.load("tests/test_data/follow_step.npz", allow_pickle=True)

    for raw in data.values():
        c = raw.item()

        R = gtsam.Rot3(c["X_pose"][:3, :3])
        t = gtsam.Point3(c["X_pose"][:3, 3])
        X = gtsam.NavState(gtsam.Pose3(R, t), gtsam.Point3(c["X_vel"]))

        Rd = gtsam.Rot3(c["Xd_pose"][:3, :3])
        td = gtsam.Point3(c["Xd_pose"][:3, 3])
        Xd = gtsam.NavState(gtsam.Pose3(Rd, td), gtsam.Point3(c["Xd_vel"]))

        u = brain.follow_step(c["t"], X, Xd)

        # type check
        assert isinstance(u, Control)

        # value check
        u_arr = np.array([u.u_lm, u.u_rm, u.u_vlm, u.u_vrm])
        assert np.allclose(u_arr, c["u"], atol=1e-3)