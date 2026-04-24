from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import gtsam
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lio_common import BaseLidarImuSlam
from src.lio_results import build_global_map
from src.lio_types import ImuSample, KeyframeState, LioSlamConfig


def make_keyframe(
    index: int,
    timestamp: float,
    position: tuple[float, float, float],
    cloud: np.ndarray,
) -> KeyframeState:
    pose = gtsam.Pose3(gtsam.Rot3(), np.array(position, dtype=float))
    navstate = gtsam.NavState(pose, np.zeros(3, dtype=float))
    return KeyframeState(
        keyframe_index=index,
        timestamp_sec=timestamp,
        state_key=index,
        bias_key=index,
        navstate=navstate,
        bias=gtsam.imuBias.ConstantBias(),
        cloud=np.asarray(cloud, dtype=float),
    )


def make_imu_sample(
    timestamp_sec: float,
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 9.81),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stream_index: int = 0,
) -> ImuSample:
    return ImuSample(
        timestamp_sec=timestamp_sec,
        linear_acceleration=np.array(linear_acceleration, dtype=float),
        angular_velocity=np.array(angular_velocity, dtype=float),
        stream_index=stream_index,
    )


class DummySlam(BaseLidarImuSlam):
    def initialize_backend(self, state_key: int, bias_key: int, navstate: gtsam.NavState, bias: gtsam.imuBias.ConstantBias) -> None:
        return None

    def process_lidar_keyframe(self, lidar, previous_keyframe, predicted_state, relative_lidar_pose) -> None:
        return None

    def apply_backend_update(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> None:
        return None

    def current_estimate(self) -> gtsam.Values:
        return gtsam.Values()


class ImuPreintegrationTests(unittest.TestCase):
    def test_handle_imu_measurement_skips_integration_for_tiny_dt_but_updates_timestamp(self) -> None:
        slam = DummySlam(LioSlamConfig(use_preintegration=True, min_preintegration_dt_sec=1e-3))
        slam.keyframes.append(make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.zeros((0, 3))))
        slam.latest_imu_timestamp_sec = 1.0
        slam.preintegrator = mock.Mock()

        imu = make_imu_sample(1.001, stream_index=1)

        slam.handle_imu_measurement(imu)

        self.assertEqual(slam.latest_imu_timestamp_sec, 1.001)
        slam.preintegrator.integrateMeasurement.assert_not_called()

    def test_handle_imu_measurement_integrates_with_current_sample_values_and_dt(self) -> None:
        slam = DummySlam(LioSlamConfig(use_preintegration=True))
        slam.keyframes.append(make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.zeros((0, 3))))
        slam.latest_imu_timestamp_sec = 0.0
        slam.preintegrator = mock.Mock()

        imu = make_imu_sample(
            timestamp_sec=0.1,
            linear_acceleration=(1.0, 2.0, 3.0),
            angular_velocity=(0.1, 0.2, 0.3),
            stream_index=1,
        )

        slam.handle_imu_measurement(imu)

        self.assertEqual(slam.latest_imu_timestamp_sec, 0.1)
        slam.preintegrator.integrateMeasurement.assert_called_once()
        call_args = slam.preintegrator.integrateMeasurement.call_args[0]
        np.testing.assert_allclose(call_args[0], imu.linear_acceleration)
        np.testing.assert_allclose(call_args[1], imu.angular_velocity)
        self.assertAlmostEqual(call_args[2], 0.1, places=8)

    def test_add_imu_factor_requires_predicted_state_when_preintegration_disabled(self) -> None:
        slam = DummySlam(LioSlamConfig(use_preintegration=False))
        graph = gtsam.NonlinearFactorGraph()
        previous_keyframe = make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.zeros((0, 3)))

        with self.assertRaisesRegex(ValueError, "predicted_state is required"):
            slam.add_imu_factor(graph, previous_keyframe, current_state_key=1)

    def test_add_imu_factor_uses_previous_keys_and_active_preintegrator(self) -> None:
        slam = DummySlam(LioSlamConfig(use_preintegration=True))
        previous_keyframe = make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.zeros((0, 3)))
        graph = mock.Mock()
        factor = object()

        with mock.patch.object(gtsam, "ImuFactor2", return_value=factor) as imu_factor_ctor:
            slam.add_imu_factor(graph, previous_keyframe, current_state_key=123)

        imu_factor_ctor.assert_called_once_with(
            previous_keyframe.state_key,
            123,
            previous_keyframe.bias_key,
            slam.preintegrator,
        )
        graph.add.assert_called_once_with(factor)

    def test_add_bias_evolution_factor_uses_clamped_random_walk_sigmas(self) -> None:
        slam = DummySlam(
            LioSlamConfig(
                use_preintegration=True,
                accel_bias_rw_sigma=0.5,
                gyro_bias_rw_sigma=0.25,
                bias_between_sigmas=(0.4, 0.4, 0.4, 0.1, 0.1, 0.1),
                min_preintegration_dt_sec=0.2,
            )
        )
        graph = mock.Mock()
        previous_keyframe = make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.zeros((0, 3)))
        noise_model = object()
        factor = object()

        with mock.patch.object(gtsam.noiseModel.Diagonal, "Sigmas", return_value=noise_model) as sigmas_ctor:
            with mock.patch.object(gtsam, "BetweenFactorConstantBias", return_value=factor) as bias_factor_ctor:
                slam.add_bias_evolution_factor(
                    graph,
                    previous_keyframe,
                    current_timestamp_sec=0.5,
                    current_bias_key=7,
                )

        expected_sigmas = np.array(
            [
                0.4,
                0.4,
                0.4,
                max(0.1, 0.25 * np.sqrt(0.5)),
                max(0.1, 0.25 * np.sqrt(0.5)),
                max(0.1, 0.25 * np.sqrt(0.5)),
            ],
            dtype=float,
        )
        np.testing.assert_allclose(sigmas_ctor.call_args[0][0], expected_sigmas)
        bias_factor_args = bias_factor_ctor.call_args[0]
        self.assertEqual(bias_factor_args[0], previous_keyframe.bias_key)
        self.assertEqual(bias_factor_args[1], 7)
        self.assertIs(bias_factor_args[3], noise_model)
        graph.add.assert_called_once_with(factor)

    def test_add_bias_evolution_factor_clamps_dt_to_minimum(self) -> None:
        slam = DummySlam(
            LioSlamConfig(
                use_preintegration=True,
                accel_bias_rw_sigma=0.3,
                gyro_bias_rw_sigma=0.15,
                bias_between_sigmas=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                min_preintegration_dt_sec=0.25,
            )
        )
        graph = mock.Mock()
        previous_keyframe = make_keyframe(0, 1.0, (0.0, 0.0, 0.0), np.zeros((0, 3)))

        with mock.patch.object(gtsam.noiseModel.Diagonal, "Sigmas") as sigmas_ctor:
            with mock.patch.object(gtsam, "BetweenFactorConstantBias", return_value=object()):
                slam.add_bias_evolution_factor(
                    graph,
                    previous_keyframe,
                    current_timestamp_sec=1.05,
                    current_bias_key=1,
                )

        expected_dt_sec = 0.25
        expected_sigmas = np.array(
            [
                0.3 * np.sqrt(expected_dt_sec),
                0.3 * np.sqrt(expected_dt_sec),
                0.3 * np.sqrt(expected_dt_sec),
                0.15 * np.sqrt(expected_dt_sec),
                0.15 * np.sqrt(expected_dt_sec),
                0.15 * np.sqrt(expected_dt_sec),
            ],
            dtype=float,
        )
        np.testing.assert_allclose(sigmas_ctor.call_args[0][0], expected_sigmas)

    def test_predict_navstate_from_preintegration_uses_previous_state_and_bias(self) -> None:
        slam = DummySlam(LioSlamConfig(use_preintegration=True))
        previous_keyframe = make_keyframe(0, 0.0, (1.0, 2.0, 3.0), np.zeros((0, 3)))
        expected_state = gtsam.NavState(
            gtsam.Pose3(gtsam.Rot3(), np.array([4.0, 5.0, 6.0], dtype=float)),
            np.array([0.1, 0.2, 0.3], dtype=float),
        )
        preintegrator = mock.Mock()
        preintegrator.predict.return_value = expected_state
        slam.preintegrator = preintegrator

        predicted_state = slam.predict_navstate_from_preintegration(previous_keyframe)

        self.assertIs(predicted_state, expected_state)
        preintegrator.predict.assert_called_once_with(
            previous_keyframe.navstate,
            previous_keyframe.bias,
        )


class MappingTests(unittest.TestCase):
    def test_build_global_map_returns_empty_array_for_no_keyframes(self) -> None:
        points = build_global_map([], lambda navstate: navstate.pose())

        self.assertEqual(points.shape, (0, 3))

    def test_build_global_map_applies_rotation_and_translation_to_each_cloud(self) -> None:
        rotated_pose = gtsam.Pose3(
            gtsam.Rot3.Rz(np.pi / 2.0),
            np.array([1.0, 2.0, 0.0], dtype=float),
        )
        keyframes = [
            make_keyframe(0, 0.0, (0.0, 0.0, 0.0), np.array([[0.0, 0.0, 0.0]])),
            KeyframeState(
                keyframe_index=1,
                timestamp_sec=1.0,
                state_key=1,
                bias_key=1,
                navstate=gtsam.NavState(rotated_pose, np.zeros(3, dtype=float)),
                bias=gtsam.imuBias.ConstantBias(),
                cloud=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
            ),
            make_keyframe(2, 2.0, (5.0, 0.0, 0.0), np.zeros((0, 3))),
        ]

        points = build_global_map(keyframes, lambda navstate: navstate.pose())
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0],
                [-1.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        self.assertEqual(points.shape, expected.shape)
        self.assertTrue(np.allclose(points, expected), msg=f"Expected {expected}, got {points}")


if __name__ == "__main__":
    unittest.main(verbosity=2)