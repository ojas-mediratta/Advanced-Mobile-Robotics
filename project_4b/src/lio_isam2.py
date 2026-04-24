from __future__ import annotations

from time import perf_counter

"""ISAM2 backend for the LIO implementation.
"""

import gtsam

from .lio_common import B, X, BaseLidarImuSlam
from .lio_types import KeyframeState, LidarFrame, LioSlamConfig, LioSlamResult


class Isam2LidarImuSlam(BaseLidarImuSlam):
    """Incremental optimization with ISAM2."""

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self.isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_values = gtsam.Values()

    def initialize_backend(
        self,
        state_key: int,
        bias_key: int,
        navstate: gtsam.NavState,
        bias: gtsam.imuBias.ConstantBias,
    ) -> None:
        # STUDENT TODO START: implement ISAM2 initialization.
        # Use self.add_initial_priors() to add the initial priors to 
        # self.pending_graph, insert the initial state and bias into 
        # self.pending_values, then call self.update_isam2().
        self.add_initial_priors(self.pending_graph, state_key, bias_key, navstate, bias)
        self.pending_values.insert(state_key, navstate)
        self.pending_values.insert(bias_key, bias)
        self.update_isam2()
        # STUDENT TODO END: implement ISAM2 initialization.

    def run(self) -> LioSlamResult:
        result = super().run()
        self.refresh_keyframe_estimates()
        return result

    def process_lidar_keyframe(
        self,
        lidar: LidarFrame,
        previous_keyframe: KeyframeState,
        predicted_state: gtsam.NavState,
        relative_lidar_pose: gtsam.Pose3 | None,
    ) -> None:
        current_index = len(self.keyframes)
        # STUDENT TODO START: implement the core ISAM2 update for one new keyframe.
        # Suggested steps:
        # 1. Allocate the current state and bias keys.
        # 2. Add IMU, LiDAR, and bias-evolution factors (self.add_imu_factor(), 
        #    self.add_lidar_factor(), self.add_bias_evolution_factor()).
        # 3. Insert predicted values in self.pending_values.
        # 4. Update ISAM2 with self.update_isam2().
        current_state_key = X(current_index)
        current_bias_key = B(current_index)

        self.add_imu_factor(
            self.pending_graph,
            previous_keyframe,
            current_state_key,
            predicted_state=predicted_state,
        )
        self.add_lidar_factor(
            self.pending_graph,
            previous_keyframe,
            current_state_key,
            relative_lidar_pose,
        )
        self.add_bias_evolution_factor(
            self.pending_graph,
            previous_keyframe,
            lidar.timestamp_sec,
            current_bias_key,
        )

        self.pending_values.insert_or_assign(current_state_key, predicted_state)
        self.pending_values.insert_or_assign(current_bias_key, previous_keyframe.bias)
        self.update_isam2()
        # STUDENT TODO END: implement the core ISAM2 update for one new keyframe.

        estimate = self.isam2.calculateEstimate()
        navstate = estimate.atNavState(current_state_key)
        bias = estimate.atConstantBias(current_bias_key)
        self.current_bias = bias
        self.preintegrator.resetIntegrationAndSetBias(bias)

        self.keyframes.append(
            KeyframeState(
                keyframe_index=current_index,
                timestamp_sec=lidar.timestamp_sec,
                state_key=current_state_key,
                bias_key=current_bias_key,
                navstate=navstate,
                bias=bias,
                cloud=lidar.points,
            )
        )

        print(f"[ISAM2] Added keyframe {current_index} (total={len(self.keyframes)})")

        previous_loop_count = self.loop_closure_count
        self.add_loop_closure_constraints()
        if self.loop_closure_count > previous_loop_count:
            added_closures = self.loop_closure_count - previous_loop_count
            print(f"[ISAM2] Applied {added_closures} loop-closure constraint(s)")

    def update_isam2(self) -> None:
        # STUDENT TODO START: implement ISAM2 update.
        # Update ISAM2 with self.pending_graph and self.pending_values,
        # append the elapsed time in seconds to self.update_times_sec 
        # (use perf_counter()), then clear both self.pending_graph and 
        # self.pending_values for next update.
        start_time = perf_counter()
        self.isam2.update(self.pending_graph, self.pending_values)
        self.update_times_sec.append(perf_counter() - start_time)
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_values = gtsam.Values()
        # STUDENT TODO END: implement ISAM2 update.

    def apply_backend_update(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
    ) -> None:
        start_time = perf_counter()
        self.isam2.update(graph, values)
        self.update_times_sec.append(perf_counter() - start_time)

    def current_estimate(self) -> gtsam.Values:
        return self.isam2.calculateEstimate()


def main(config: LioSlamConfig | None = None) -> LioSlamResult:
    slam = Isam2LidarImuSlam(config)
    return slam.run()


if __name__ == "__main__":
    result = main()
    print(f"Processed {len(result.keyframes)} LiDAR keyframes with the ISAM2 backend.")