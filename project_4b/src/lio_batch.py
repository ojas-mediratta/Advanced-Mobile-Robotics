from __future__ import annotations

from time import perf_counter

"""Batch backend for the LIO implementation.
"""

import gtsam

from .lio_common import B, X, BaseLidarImuSlam
from .lio_types import KeyframeState, LidarFrame, LioSlamConfig, LioSlamResult


class BatchLidarImuSlam(BaseLidarImuSlam):
    """Batch graph construction with repeated full solves."""

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

    def run(self) -> LioSlamResult:
        result = super().run()
        self.refresh_keyframe_estimates()
        return result

    def initialize_backend(
        self,
        state_key: int,
        bias_key: int,
        navstate: gtsam.NavState,
        bias: gtsam.imuBias.ConstantBias,
    ) -> None:
        # STUDENT TODO START: implement initialization for a nonlinear factor graph.
        # Use self.add_initial_priors() to add the initial priors to self.graph and insert 
        # the initial state and bias into self.values.
        self.add_initial_priors(self.graph, state_key, bias_key, navstate, bias)
        self.values.insert(state_key, navstate)
        self.values.insert(bias_key, bias)
        # STUDENT TODO END: implement initialization for a nonlinear factor graph.

    def process_lidar_keyframe(
        self,
        lidar: LidarFrame,
        previous_keyframe: KeyframeState,
        predicted_state: gtsam.NavState,
        relative_lidar_pose: gtsam.Pose3 | None,
    ) -> None:
        current_index = len(self.keyframes)
        # STUDENT TODO START: implement the core batch update for one new keyframe.
        # Suggested steps:
        # 1. Allocate the current state and bias keys.
        # 2. Add IMU, LiDAR, and bias-evolution factors to self.graph (self.add_imu_factor(), 
        #    self.add_lidar_factor(), self.add_bias_evolution_factor()).
        # 3. Insert predicted values into self.values.
        # 4. Run batch optimization with self.optimize_batch_graph() and record the update time
        #    by appending a float number of seconds to self.update_times_sec (use perf_counter()).
        current_state_key = X(current_index)
        current_bias_key = B(current_index)

        self.add_imu_factor(
            self.graph,
            previous_keyframe,
            current_state_key,
            predicted_state=predicted_state,
        )
        self.add_lidar_factor(
            self.graph,
            previous_keyframe,
            current_state_key,
            relative_lidar_pose,
        )
        self.add_bias_evolution_factor(
            self.graph,
            previous_keyframe,
            lidar.timestamp_sec,
            current_bias_key,
        )

        self.values.insert_or_assign(current_state_key, predicted_state)
        self.values.insert_or_assign(current_bias_key, previous_keyframe.bias)

        start_time = perf_counter()
        self.values = self.optimize_batch_graph(self.graph, self.values)
        self.update_times_sec.append(perf_counter() - start_time)
        # STUDENT TODO END: implement the core batch update for one new keyframe.

        self.refresh_keyframe_estimates()

        estimate = self.values
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

        print(f"[Batch] Added keyframe {current_index} (total={len(self.keyframes)})")

        previous_loop_count = self.loop_closure_count
        self.add_loop_closure_constraints()
        if self.loop_closure_count > previous_loop_count:
            added_closures = self.loop_closure_count - previous_loop_count
            print(f"[Batch] Applied {added_closures} loop-closure constraint(s)")

    def optimize_batch_graph(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
    ) -> gtsam.Values:
        max_iterations = self.config.batch_max_iterations
        # STUDENT TODO START: implement optimization over a factor graph.
        # Configure a LevenbergMarquardtOptimizer with max_iterations and 
        # return the optimized Values.
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        return optimizer.optimize()
        # STUDENT TODO END: implement optimization over a factor graph.

    def apply_backend_update(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
    ) -> None:
        self.graph.push_back(graph)
        if not values.empty():
            self.values.insert_or_assign(values)

        start_time = perf_counter()
        self.values = self.optimize_batch_graph(self.graph, self.values)
        self.update_times_sec.append(perf_counter() - start_time)

    def current_estimate(self) -> gtsam.Values:
        return self.values


def main(config: LioSlamConfig | None = None) -> LioSlamResult:
    slam = BatchLidarImuSlam(config)
    return slam.run()


if __name__ == "__main__":
    result = main()
    print(f"Processed {len(result.keyframes)} LiDAR keyframes with the batch backend.")