import gtsam
import numpy as np

from controllers.ROV_controller.brain import Brain
from controllers.ROV_controller.robot import BEACONS, Measurements


def test_initialize_EKF():
    brain = Brain(BEACONS)
    brain.initialize_EKF(X0=gtsam.NavState(), P0=np.eye(9) * 0.1)
    assert isinstance(brain.ekf, gtsam.NavStateImuEKF)


def test_EKF_predict():
    brain = Brain(BEACONS)
    inputs = np.load("tests/test_data/ekf_predict_inputs.npz", allow_pickle=True)
    outputs = np.load("tests/test_data/ekf_predict_outputs.npz", allow_pickle=True)

    for _, (inp, out) in enumerate(zip(inputs.values(), outputs.values())):
        inp = inp.item()
        out = out.item()

        R = gtsam.Rot3(inp['X0_pose'][:3, :3])
        t = gtsam.Point3(inp['X0_pose'][:3, 3])
        vel = inp['X0_vel']
        X0 = gtsam.NavState(R=R, t=t, v=vel)

        brain.initialize_EKF(X0=X0, P0=inp['P0'])
        X_pred, P_pred = brain.EKF_predict(
            omega_meas=inp['omega_meas'],
            acc_meas=inp['acc_meas'],
            dt=inp['dt']
        )
        
        # type check
        assert isinstance(X_pred, gtsam.NavState)
        assert isinstance(P_pred, np.ndarray)
        
        # shape check
        assert X_pred.position().shape == (3,)
        assert X_pred.velocity().shape == (3,)
        assert X_pred.attitude().matrix().shape == (3, 3)
        assert P_pred.shape == (9, 9)
        
        # value check
        assert np.allclose(X_pred.position(), out['position'], atol=1e-3)
        assert np.allclose(X_pred.velocity(), out['velocity'], atol=1e-3)
        assert np.allclose(X_pred.attitude().matrix(), out['attitude'], atol=1e-3)
        assert np.allclose(P_pred, out['P_pred'], atol=1e-3)


def test_EKF_predict_update():
    brain = Brain(BEACONS)
    inputs = np.load("tests/test_data/ekf_inputs.npz", allow_pickle=True)
    outputs = np.load("tests/test_data/ekf_outputs.npz", allow_pickle=True)

    for i in range(len(inputs.files)):
        c_in = inputs[f"arr_{i}"].item()
        c_out = outputs[f"arr_{i}"].item()

        R = gtsam.Rot3(c_in["X0_pose"][:3, :3])
        t = gtsam.Point3(c_in["X0_pose"][:3, 3])
        X0 = gtsam.NavState(gtsam.Pose3(R, t), gtsam.Point3(c_in["X0_vel"]))

        brain.initialize_EKF(X0, c_in["P0"])

        X_pred, P_pred = brain.EKF_predict(
            c_in["omega"], c_in["acc"], c_in["dt"]
        )

        meas = Measurements(
            position=c_in["meas"]["position"],
            depth=c_in["meas"]["depth"],
            ranges=c_in["meas"]["ranges"],
            X_true=None,
        )

        brain.EKF_update(meas)

        X_upd = brain.ekf.state()
        P_upd = brain.ekf.covariance()

        # predicted state check
        assert np.allclose(X_pred.position(), c_out["X_pred"]["p"], atol=1e-3)
        assert np.allclose(X_pred.velocity(), c_out["X_pred"]["v"], atol=1e-3)
        assert np.allclose(X_pred.attitude().matrix(), c_out["X_pred"]["R"], atol=1e-3)
        assert np.allclose(P_pred, c_out["P_pred"], atol=1e-2)

        # updated state check
        assert np.allclose(X_upd.position(), c_out["X_upd"]["p"], atol=1e-3)
        assert np.allclose(X_upd.velocity(), c_out["X_upd"]["v"], atol=1e-3)
        assert np.allclose(X_upd.attitude().matrix(), c_out["X_upd"]["R"], atol=1e-3)
        assert np.allclose(P_upd, c_out["P_upd"], atol=5e-2)
