from __future__ import annotations

"""Shared front-end logic for the LIO backends.
"""

from typing import Iterator
from tqdm.auto import tqdm

import gtsam
import numpy as np
from rosbags.highlevel import AnyReader

from .lio_math import (
    nearest_timestamp_indices,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_between_vectors,
)
from .lio_loop_closure import find_loop_closure_candidate_pairs
from .lio_open3d import GICPConfig, gicp
from .lio_results import build_global_map, evaluate_trajectory, load_ground_truth_tum
from .lio_types import (
    CombinedMeasurement,
    ImuSample,
    KeyframeState,
    LidarFrame,
    LioSlamConfig,
    LioSlamResult,
)


X = gtsam.symbol_shorthand.X
B = gtsam.symbol_shorthand.B


class BaseLidarImuSlam:
    """Shared front-end utilities for the batch and ISAM2 implementations.

    This class owns the shared runtime state used by both backends.

    Important examples:
    - self.keyframes is a list[KeyframeState] storing solved keyframe states.
    - self.initial_imu_samples is a list[ImuSample] used before initialization.
    - self.update_times_sec is a list[float]; append one elapsed update time per backend solve.
    - self.current_bias is a gtsam.imuBias.ConstantBias for the latest estimate.
    - self.preintegrator is a gtsam.PreintegratedImuMeasurements object.
    """

    def __init__(self, config: LioSlamConfig | None = None) -> None:
        self.config = config or LioSlamConfig()
        self.current_bias = gtsam.imuBias.ConstantBias()
        self.preintegration_params = self._build_preintegration_params()
        self.preintegrator = gtsam.PreintegratedImuMeasurements(
            self.preintegration_params,
            self.current_bias,
        )

        self.keyframes: list[KeyframeState] = []
        self.initial_imu_samples: list[ImuSample] = []
        self.update_times_sec: list[float] = []
        self.latest_imu_timestamp_sec: float | None = None
        self.ground_truth_tum: np.ndarray | None = None
        self.loop_closure_count: int = 0

    def run(self) -> LioSlamResult:
        """Template method shared by both optimization backends."""
        self.ground_truth_tum = load_ground_truth_tum(self.config.ground_truth_path)

        measurements_iter = self.iterate_measurements()
        expected_count = self.config.stream_end_index - self.config.stream_start_index + 1
        measurements_iter = tqdm(measurements_iter, desc="messages", unit="msg", total=expected_count)

        for measurement in measurements_iter:
            if measurement.kind == "imu":
                assert measurement.imu is not None
                self.handle_imu_measurement(measurement.imu)
            else:
                assert measurement.lidar is not None
                self.handle_lidar_measurement(measurement.lidar)

        return LioSlamResult(
            keyframes=self.keyframes,
            update_times_sec=list(self.update_times_sec),
            ground_truth_tum=self.ground_truth_tum,
        )

    def iterate_measurements(self) -> Iterator[CombinedMeasurement]:
        """Yield IMU and LiDAR messages from the requested combined stream slice."""
        with AnyReader([self.config.bag_path]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in {self.config.imu_topic, self.config.lidar_topic}
            ]

            for stream_index, (connection, _timestamp_ns, rawdata) in enumerate(
                reader.messages(connections=connections)
            ):
                if stream_index < self.config.stream_start_index:
                    continue
                if stream_index > self.config.stream_end_index:
                    break

                message = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic == self.config.imu_topic:
                    imu = self.parse_imu_message(message, stream_index)
                    yield CombinedMeasurement(
                        kind="imu",
                        timestamp_sec=imu.timestamp_sec,
                        stream_index=stream_index,
                        imu=imu,
                    )
                elif connection.topic == self.config.lidar_topic:
                    lidar = self.parse_lidar_message(message, stream_index)
                    yield CombinedMeasurement(
                        kind="lidar",
                        timestamp_sec=lidar.timestamp_sec,
                        stream_index=stream_index,
                        lidar=lidar,
                    )

    def _build_preintegration_params(self) -> gtsam.PreintegrationParams:
        params = gtsam.PreintegrationParams.MakeSharedU(self.config.gravity_mps2)
        accel_cov = (self.config.accel_noise_sigma**2) * np.eye(3)
        gyro_cov = (self.config.gyro_noise_sigma**2) * np.eye(3)
        integration_cov = (self.config.integration_noise_sigma**2) * np.eye(3)

        params.setAccelerometerCovariance(accel_cov)
        params.setGyroscopeCovariance(gyro_cov)
        params.setIntegrationCovariance(integration_cov)
        params.setBodyPSensor(
            self.pose3_from_rt(
                self.config.baselink2imu_R,
                self.config.baselink2imu_t,
            )
        )
        return params

    def parse_imu_message(self, msg, stream_index: int) -> ImuSample:
        return ImuSample(
            timestamp_sec=(
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
                + self.config.imu_time_offset_sec
            ),
            linear_acceleration=np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ],
                dtype=float,
            ),
            angular_velocity=np.array(
                [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ],
                dtype=float,
            ),
            stream_index=stream_index,
        )

    def handle_imu_measurement(self, imu: ImuSample) -> None:
        """Accumulate IMU data for initialization or preintegrate it between keyframes."""
        if not self.config.use_preintegration:
            return

        if not self.keyframes:
            self.initial_imu_samples.append(imu)
            if len(self.initial_imu_samples) > self.config.initialization_imu_sample_count:
                self.initial_imu_samples.pop(0)
            self.latest_imu_timestamp_sec = imu.timestamp_sec
            return

        if self.latest_imu_timestamp_sec is None:
            self.latest_imu_timestamp_sec = imu.timestamp_sec
            return

        latest_imu_timestamp_sec = self.latest_imu_timestamp_sec
        min_preintegration_dt_sec = self.config.min_preintegration_dt_sec
        preintegrator = self.preintegrator

        # STUDENT TODO START: implement the IMU-handling.
        # Suggested steps:
        # 1. Compute dt_sec using the current IMU timestamp and the previous IMU
        #    timestamp.
        # 2. Update self.latest_imu_timestamp_sec to the current IMU timestamp
        #    so the next sample has the correct reference time.
        # 3. If dt_sec is too small, return without integrating.
        # 4. Otherwise, call integrateMeasurement(...) using the current IMU
        #    linear acceleration, angular velocity, and dt_sec.
        dt_sec = imu.timestamp_sec - latest_imu_timestamp_sec
        self.latest_imu_timestamp_sec = imu.timestamp_sec
        if dt_sec < min_preintegration_dt_sec:
            return
        preintegrator.integrateMeasurement(
            imu.linear_acceleration,
            imu.angular_velocity,
            dt_sec,
        )
        # STUDENT TODO END: implement the IMU-handling.
    
    def add_imu_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        previous_keyframe: KeyframeState,
        current_state_key: int,
        predicted_state: gtsam.NavState | None = None,
    ) -> None:
        """Connect consecutive keyframes with an IMU or fallback motion constraint."""
        if not self.config.use_preintegration:
            if predicted_state is None:
                raise ValueError("predicted_state is required when IMU preintegration is disabled")

            # Add a prior to keep velocity constrained when not using IMU preintegration.
            pose_sigmas = np.repeat(1e6, 6)
            velocity_sigmas = np.asarray(self.config.prior_velocity_sigmas, dtype=float)
            navstate_sigmas = np.concatenate([pose_sigmas, velocity_sigmas])
            graph.add(
                gtsam.PriorFactorNavState(
                    current_state_key,
                    predicted_state,
                    gtsam.noiseModel.Diagonal.Sigmas(navstate_sigmas),
                )
            )
            return

        preintegrator = self.preintegrator

        # STUDENT TODO START: add a gtsam.ImuFactor2 to the factor graph between 
        # the previous and current state. Use the previous keyframe's bias and the 
        # active preintegrator.
        imu_factor = gtsam.ImuFactor2(
            previous_keyframe.state_key,
            current_state_key,
            previous_keyframe.bias_key,
            preintegrator,
        )
        graph.add(imu_factor)
        # STUDENT TODO END: implement IMU-factor insertion into the factor graph.
    
    def add_bias_evolution_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        previous_keyframe: KeyframeState,
        current_timestamp_sec: float,
        current_bias_key: int,
    ) -> None:
        if not self.config.use_preintegration:
            return
        
        accel_rw_sigma = self.config.accel_bias_rw_sigma
        gyro_rw_sigma = self.config.gyro_bias_rw_sigma
        min_preintegration_dt_sec = self.config.min_preintegration_dt_sec
        bias_between_sigmas = self.config.bias_between_sigmas
        
        # STUDENT TODO START: implement the bias-evolution factor body.
        # This is a standard random-walk model: the expected change in bias is zero, but the 
        # uncertainty grows with elapsed time.
        #
        # Suggested steps:
        # 1. Compute dt_sec. Clamp it to at least min_preintegration_dt_sec to avoid a 
        #    zero-noise model.
        # 2. Convert the accelerometer and gyroscope bias random-walk sigmas
        #    into per-interval sigmas by multiplying each by sqrt(dt_sec).
        # 3. Build a 6-vector of sigmas in the order:
        #    [ax, ay, az, gx, gy, gz].
        #    Clamp it to bias_between_sigmas to keep it from becoming unrealistically small.
        # 4. Create a GTSAM diagonal noise model from those sigmas.
        # 5. Add a BetweenFactorConstantBias from the previous bias key to the
        #    current bias key.
        # 6. Use a zero-valued ConstantBias as the measurement, because this
        #    factor models "bias_j is close to bias_i" rather than a known
        #    nonzero change.
        dt_sec = current_timestamp_sec - previous_keyframe.timestamp_sec
        dt_sec = max(dt_sec, min_preintegration_dt_sec)

        accel_interval_sigma = accel_rw_sigma * np.sqrt(dt_sec)
        gyro_interval_sigma = gyro_rw_sigma * np.sqrt(dt_sec)

        rw_sigmas = np.array([
                accel_interval_sigma,accel_interval_sigma,accel_interval_sigma,
                gyro_interval_sigma,gyro_interval_sigma,gyro_interval_sigma,],
            dtype=float,
        )
        minimum_sigmas = np.asarray(bias_between_sigmas, dtype=float)
        sigmas = np.maximum(rw_sigmas, minimum_sigmas)

        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        zero_bias_delta = gtsam.imuBias.ConstantBias()
        graph.add(
            gtsam.BetweenFactorConstantBias(
                previous_keyframe.bias_key,
                current_bias_key,
                zero_bias_delta,
                noise_model,
            )
        )
        # STUDENT TODO END: implement the bias-evolution factor body.
    
    def predict_navstate_from_preintegration(
        self,
        previous_keyframe: KeyframeState,
    ) -> gtsam.NavState:
        preintegrator = self.preintegrator
        # STUDENT TODO START: implement preintegration prediction.
        # Use the current preintegrator and the previous keyframe to predict the next gtsam.NavState.
        return preintegrator.predict(
            previous_keyframe.navstate,
            previous_keyframe.bias,
        )
        # STUDENT TODO END: implement preintegration prediction.
    
    def pointcloud2_to_xyz(self, msg) -> np.ndarray:
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [0, 4, 8],
                "itemsize": msg.point_step,
            }
        )
        count = msg.width * msg.height
        structured = np.frombuffer(msg.data, dtype=dtype, count=count)
        points = np.column_stack([structured["x"], structured["y"], structured["z"]])
        return points[np.all(np.isfinite(points), axis=1)]

    def parse_lidar_message(self, msg, stream_index: int) -> LidarFrame:
        return LidarFrame(
            timestamp_sec=(
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
                + self.config.lidar_time_offset_sec
            ),
            points=self.pointcloud2_to_xyz(msg),
            stream_index=stream_index,
        )
    
    def voxel_downsample(self, points: np.ndarray, voxel_size_m: float) -> np.ndarray:
        if voxel_size_m <= 0.0 or len(points) == 0:
            return points

        voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        return points[np.sort(unique_indices)]
    
    def preprocess_lidar_points(self, points: np.ndarray) -> np.ndarray:
        """Voxel-filter the scan and cap point count for predictable runtime."""
        downsampled = self.voxel_downsample(points, self.config.voxel_size_m)
        if len(downsampled) <= self.config.max_points_per_scan:
            return downsampled

        indices = np.random.default_rng(7).choice(
            len(downsampled),
            size=self.config.max_points_per_scan,
            replace=False,
        )
        return downsampled[np.sort(indices)]

    def estimate_lidar_relative_pose(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
        predicted_state: gtsam.NavState,
    ) -> gtsam.Pose3 | None:
        if not self.config.enable_gicp:
            return None

        if self.keyframes:
            previous_lidar_pose = self.navstate_lidar_pose(self.keyframes[-1].navstate)
            predicted_lidar_pose = self.navstate_lidar_pose(predicted_state)
            initial_guess = previous_lidar_pose.between(predicted_lidar_pose)
        else:
            initial_guess = None

        result = gicp(
            source_points=current_points,
            target_points=previous_points,
            initial_bTa=initial_guess,
            config=GICPConfig(),
        )
        return result.bTa

    def handle_lidar_measurement(self, lidar: LidarFrame) -> None:
        """Preprocess a LiDAR scan and either bootstrap or extend the factor graph."""
        lidar = LidarFrame(
            timestamp_sec=lidar.timestamp_sec,
            points=self.preprocess_lidar_points(lidar.points),
            stream_index=lidar.stream_index,
        )

        if not self.keyframes:
            self.bootstrap_with_first_lidar_frame(lidar)
            return

        previous_keyframe = self.keyframes[-1]
        predicted_state = self.predict_navstate_from_preintegration(previous_keyframe)
        relative_lidar_pose = self.estimate_lidar_relative_pose(
            previous_points=previous_keyframe.cloud,
            current_points=lidar.points,
            predicted_state=predicted_state,
        )

        self.process_lidar_keyframe(
            lidar=lidar,
            previous_keyframe=previous_keyframe,
            predicted_state=predicted_state,
            relative_lidar_pose=relative_lidar_pose,
        ) 
    
    def make_navstate_lidar_factor(
        self,
        state_key_i: int,
        state_key_j: int,
        measured_relative_lidar_pose: gtsam.Pose3,
    ) -> gtsam.CustomFactor:
        """Create the custom LiDAR relative-pose factor between two NavStates."""
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(self.config.lidar_pose_sigmas, dtype=float)
        )

        def error_function(
            _factor: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians: list[np.ndarray] | None,
        ) -> np.ndarray:
            state_i = values.atNavState(state_key_i)
            state_j = values.atNavState(state_key_j)
            error = self.compute_lidar_pose_error(
                state_i,
                state_j,
                measured_relative_lidar_pose,
            )
            if jacobians is not None:
                jacobians[0] = self.numerical_navstate_jacobian(
                    lambda candidate: self.compute_lidar_pose_error(
                        candidate,
                        state_j,
                        measured_relative_lidar_pose,
                    ),
                    state_i,
                )
                jacobians[1] = self.numerical_navstate_jacobian(
                    lambda candidate: self.compute_lidar_pose_error(
                        state_i,
                        candidate,
                        measured_relative_lidar_pose,
                    ),
                    state_j,
                )
            return error

        return gtsam.CustomFactor(noise_model, [state_key_i, state_key_j], error_function)

    def add_lidar_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        previous_keyframe: KeyframeState,
        current_state_key: int,
        relative_lidar_pose: gtsam.Pose3 | None,
    ) -> None:
        if relative_lidar_pose is None:
            return
        graph.add(
            self.make_navstate_lidar_factor(
                previous_keyframe.state_key,
                current_state_key,
                relative_lidar_pose,
            )
        )

    def bootstrap_with_first_lidar_frame(self, lidar: LidarFrame) -> None:
        """Initialize the graph from the first LiDAR frame in the selected slice."""
        initial_state_key = X(0)
        initial_bias_key = B(0)
        initial_pose, initial_velocity, initial_bias = self.estimate_initial_state(
            lidar.timestamp_sec
        )
        initial_navstate = gtsam.NavState(initial_pose, initial_velocity)

        self.initialize_backend(
            state_key=initial_state_key,
            bias_key=initial_bias_key,
            navstate=initial_navstate,
            bias=initial_bias,
        )

        self.current_bias = initial_bias
        self.preintegrator.resetIntegrationAndSetBias(initial_bias)
        self.latest_imu_timestamp_sec = lidar.timestamp_sec
        self.initial_imu_samples.clear()
        self.keyframes.append(
            KeyframeState(
                keyframe_index=0,
                timestamp_sec=lidar.timestamp_sec,
                state_key=initial_state_key,
                bias_key=initial_bias_key,
                navstate=initial_navstate,
                bias=initial_bias,
                cloud=lidar.points,
            )
        )

    def add_initial_priors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        state_key: int,
        bias_key: int,
        navstate: gtsam.NavState,
        bias: gtsam.imuBias.ConstantBias,
    ) -> None:
        """Add priors for the first state and IMU bias."""
        pose_sigmas = np.asarray(self.config.prior_pose_sigmas, dtype=float)
        velocity_sigmas = np.asarray(self.config.prior_velocity_sigmas, dtype=float)
        navstate_sigmas = np.concatenate([pose_sigmas, velocity_sigmas])
        bias_sigmas = np.asarray(self.config.prior_bias_sigmas, dtype=float)

        graph.add(
            gtsam.PriorFactorNavState(
                state_key,
                navstate,
                gtsam.noiseModel.Diagonal.Sigmas(navstate_sigmas),
            )
        )
        graph.add(
            gtsam.PriorFactorConstantBias(
                bias_key,
                bias,
                gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas),
            )
        )

    def estimate_initial_state(
        self,
        timestamp_sec: float,
    ) -> tuple[gtsam.Pose3, np.ndarray, gtsam.imuBias.ConstantBias]:
        ground_truth_state = self.initial_state_from_ground_truth(timestamp_sec)
        if ground_truth_state is not None:
            return ground_truth_state
        return self.estimate_initial_state_from_imu()

    def initial_state_from_ground_truth(
        self,
        timestamp_sec: float,
    ) -> tuple[gtsam.Pose3, np.ndarray, gtsam.imuBias.ConstantBias] | None:
        if not self.config.use_ground_truth_slice_initialization:
            return None
        if self.ground_truth_tum is None or len(self.ground_truth_tum) == 0:
            return None

        gt_timestamps = self.ground_truth_tum[:, 0]
        nearest_index = int(
            nearest_timestamp_indices(
                gt_timestamps,
                np.array([timestamp_sec], dtype=float),
            )[0]
        )
        gt_position = self.ground_truth_tum[nearest_index, 1:4]
        gt_rotation = quaternion_xyzw_to_rotation_matrix(
            self.ground_truth_tum[nearest_index, 4:8]
        )

        previous_index = max(nearest_index - 1, 0)
        next_index = min(nearest_index + 1, len(self.ground_truth_tum) - 1)
        dt_sec = self.ground_truth_tum[next_index, 0] - self.ground_truth_tum[previous_index, 0]
        if dt_sec <= self.config.min_preintegration_dt_sec:
            gt_velocity = np.zeros(3, dtype=float)
        else:
            gt_velocity = (
                self.ground_truth_tum[next_index, 1:4]
                - self.ground_truth_tum[previous_index, 1:4]
            ) / dt_sec

        pose = gtsam.Pose3(gtsam.Rot3(gt_rotation), gt_position)
        bias = gtsam.imuBias.ConstantBias()
        return pose, np.asarray(gt_velocity, dtype=float), bias

    def estimate_initial_state_from_imu(
        self,
    ) -> tuple[gtsam.Pose3, np.ndarray, gtsam.imuBias.ConstantBias]:
        initial_pose = gtsam.Pose3.Identity()
        initial_velocity = np.zeros(3, dtype=float)
        initial_bias = gtsam.imuBias.ConstantBias()

        if not self.config.use_preintegration or not self.initial_imu_samples:
            return initial_pose, initial_velocity, initial_bias

        accelerations = np.vstack([sample.linear_acceleration for sample in self.initial_imu_samples])
        angular_velocities = np.vstack([sample.angular_velocity for sample in self.initial_imu_samples])

        mean_acceleration = accelerations.mean(axis=0)
        mean_angular_velocity = angular_velocities.mean(axis=0)
        accel_norm = float(np.linalg.norm(mean_acceleration))
        gyro_norm = float(np.linalg.norm(mean_angular_velocity))

        is_stationary = (
            abs(accel_norm - self.config.gravity_mps2) <= self.config.stationary_accel_tolerance_mps2
            and gyro_norm <= self.config.stationary_gyro_tolerance_rps
        )
        if not is_stationary or accel_norm <= self.config.min_preintegration_dt_sec:
            return initial_pose, initial_velocity, initial_bias

        rotation = rotation_matrix_between_vectors(
            mean_acceleration / accel_norm,
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        initial_pose = gtsam.Pose3(gtsam.Rot3(rotation), np.zeros(3, dtype=float))
        expected_gravity_body = rotation.T @ np.array([0.0, 0.0, self.config.gravity_mps2], dtype=float)
        accel_bias = mean_acceleration - expected_gravity_body
        initial_bias = gtsam.imuBias.ConstantBias(accel_bias, mean_angular_velocity)
        return initial_pose, initial_velocity, initial_bias

    def should_attempt_loop_closure(self) -> bool:
        keyframe_count = len(self.keyframes)
        loop_closure_interval = self.config.loop_closure_interval

        if keyframe_count <= 1:
            return False
        if loop_closure_interval <= 0:
            return False

        return keyframe_count % loop_closure_interval == 0
    
    def find_loop_closure_candidates(self) -> list[tuple[int, int]]:
        return find_loop_closure_candidate_pairs(
            self.keyframes,
            min_keyframe_separation=self.config.loop_closure_min_keyframe_separation,
            search_radius_m=self.config.loop_closure_search_radius_m,
            max_candidates_per_keyframe=self.config.loop_closure_max_candidates_per_keyframe,
        )
    
    def estimate_relative_pose_between_keyframes(
        self,
        reference_keyframe: KeyframeState,
        current_keyframe: KeyframeState,
    ) -> gtsam.Pose3 | None:
        if not self.config.enable_gicp:
            return None

        initial_guess = self.navstate_lidar_pose(reference_keyframe.navstate).between(
            self.navstate_lidar_pose(current_keyframe.navstate)
        )
        result = gicp(
            source_points=current_keyframe.cloud,
            target_points=reference_keyframe.cloud,
            initial_bTa=initial_guess,
            config=GICPConfig(),
        )
        return result.bTa

    def add_loop_closure_constraints(self) -> None:
        if not self.should_attempt_loop_closure():
            return None

        candidates = self.find_loop_closure_candidates()
        if not candidates:
            return None

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        added = 0

        for reference_index, current_index in candidates:
            reference_keyframe = self.keyframes[reference_index]
            current_keyframe = self.keyframes[current_index]
            relative_lidar_pose = self.estimate_relative_pose_between_keyframes(
                reference_keyframe,
                current_keyframe,
            )
            if relative_lidar_pose is None:
                continue

            self.add_lidar_factor(
                graph,
                reference_keyframe,
                current_keyframe.state_key,
                relative_lidar_pose,
            )
            added += 1

        if added == 0:
            return None

        self.apply_backend_update(graph, values)
        self.refresh_keyframe_estimates()
        self.loop_closure_count += added
        return None

    def build_global_map(self) -> np.ndarray:
        return build_global_map(self.keyframes, self.navstate_lidar_pose)

    def evaluate_trajectory(self) -> dict:
        return evaluate_trajectory(self.keyframes, self.ground_truth_tum)

    def navstate_lidar_pose(self, navstate: gtsam.NavState) -> gtsam.Pose3:
        return navstate.pose().compose(
            self.pose3_from_rt(
                self.config.baselink2lidar_R,
                self.config.baselink2lidar_t,
            )
        )

    def pose3_from_rt(
        self,
        rotation_flat: tuple[float, ...],
        translation: tuple[float, float, float],
    ) -> gtsam.Pose3:
        rotation = np.asarray(rotation_flat, dtype=float).reshape(3, 3)
        return gtsam.Pose3(gtsam.Rot3(rotation), np.asarray(translation, dtype=float))

    def compute_lidar_pose_error(
        self,
        state_i: gtsam.NavState,
        state_j: gtsam.NavState,
        measured_relative_lidar_pose: gtsam.Pose3,
    ) -> np.ndarray:
        predicted_relative_pose = self.navstate_lidar_pose(state_i).between(
            self.navstate_lidar_pose(state_j)
        )
        return measured_relative_lidar_pose.localCoordinates(predicted_relative_pose)

    def numerical_navstate_jacobian(
        self,
        error_fn,
        state: gtsam.NavState,
    ) -> np.ndarray:
        """Approximate a factor Jacobian by central differences on NavState.retract."""
        error_dim = len(np.asarray(error_fn(state), dtype=float))
        jacobian = np.zeros((error_dim, 9), order="F")
        epsilon = self.config.numerical_jacobian_eps

        for column in range(9):
            delta = np.zeros(9, dtype=float)
            delta[column] = epsilon
            error_plus = np.asarray(error_fn(state.retract(delta)), dtype=float)
            error_minus = np.asarray(error_fn(state.retract(-delta)), dtype=float)
            jacobian[:, column] = (error_plus - error_minus) / (2.0 * epsilon)

        return jacobian

    def refresh_keyframe_estimates(self) -> None:
        estimate = self.current_estimate()
        for keyframe in self.keyframes:
            keyframe.navstate = estimate.atNavState(keyframe.state_key)
            keyframe.bias = estimate.atConstantBias(keyframe.bias_key)
        if self.keyframes:
            self.current_bias = self.keyframes[-1].bias
            self.preintegrator.resetIntegrationAndSetBias(self.current_bias)


    def initialize_backend(
        self,
        state_key: int,
        bias_key: int,
        navstate: gtsam.NavState,
        bias: gtsam.imuBias.ConstantBias,
    ) -> None:
        raise NotImplementedError

    def process_lidar_keyframe(
        self,
        lidar: LidarFrame,
        previous_keyframe: KeyframeState,
        predicted_state: gtsam.NavState,
        relative_lidar_pose: gtsam.Pose3 | None,
    ) -> None:
        raise NotImplementedError

    def apply_backend_update(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
    ) -> None:
        raise NotImplementedError

    def current_estimate(self) -> gtsam.Values:
        raise NotImplementedError