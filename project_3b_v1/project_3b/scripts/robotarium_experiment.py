import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import *
import numpy as np

LOCAL_SIM = True # Set to False before submitting to real Robotarium

class RandomConfig:
    def __init__(self):
        _cfg_rng = np.random.default_rng()

        self.seed = int(_cfg_rng.integers(0, 2**31))
        self.forward_speed = float(_cfg_rng.uniform(0.15, 0.20))
        self.turn_omega = float(_cfg_rng.uniform(1.5, 2.0))
        self.detect_dist = float(_cfg_rng.uniform(0.20, 0.30))
        self.turn_steps_range = (int(_cfg_rng.integers(10, 20)), int(_cfg_rng.integers(40, 60)))

ALL_SEGMENTS = [
    ((-1.6, -1.0), ( 1.6, -1.0)),
    (( 1.6, -1.0), ( 1.6,  1.0)),
    (( 1.6,  1.0), (-1.6,  1.0)),
    ((-1.6,  1.0), (-1.6, -1.0)),
    ((-1.16, -0.50), (-0.95,  0.50)),
    ((-0.74, -0.50), (-0.95,  0.50)),
    ((-1.02,  0.00), (-0.88,  0.00)),
    ((-0.25, -0.50), (-0.25,  0.50)),
    ((-0.25,  0.50), ( 0.00,  0.00)),
    (( 0.00,  0.00), ( 0.25,  0.50)),
    (( 0.25, -0.50), ( 0.25,  0.50)),
    (( 0.77, -0.50), ( 0.77,  0.50)),
    (( 0.77,  0.50), ( 1.12,  0.50)),
    (( 1.12,  0.50), ( 1.12,  0.10)),
    (( 0.94,  0.10), ( 1.12,  0.10)),
    (( 0.94,  0.10), ( 1.22, -0.50)),
]

INIT_POSES = np.array([
    [-1.45, -0.49,  0.00,  0.51,  1.45],
    [-0.80,  0.70, -0.80,  0.70, -0.80],
    [ 0.0,   0.0,   0.0,   0.0,   0.0 ],
])

# ============================================================
# INLINED HELPERS (from p3_lib/lidar.py, icp.py, navigation.py)
# ============================================================
def simulate_lidar(pose, obstacles, n_rays=60, max_range=1.5, fov=2*np.pi):
    rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])
    p = np.array([rx, ry])
    include_endpoint = (fov < 2*np.pi - 1e-9)
    angles = np.linspace(-fov/2, fov/2, n_rays, endpoint=include_endpoint)
    world_angles = angles + rtheta
    ranges = np.full(n_rays, np.nan)
    points = np.full((n_rays, 2), np.nan)
    for i, wangle in enumerate(world_angles):
        d = np.array([np.cos(wangle), np.sin(wangle)])
        best_t = max_range
        hit = False
        for seg in obstacles:
            a = np.asarray(seg[0], dtype=float)
            b = np.asarray(seg[1], dtype=float)
            ba = b - a
            d_x_ba = d[0]*ba[1] - d[1]*ba[0]
            if abs(d_x_ba) < 1e-9:
                continue
            ap = a - p
            t = (ap[0]*ba[1] - ap[1]*ba[0]) / d_x_ba
            s = (ap[0]*d[1] - ap[1]*d[0]) / d_x_ba
            if t >= 0 and t < best_t and 0 <= s <= 1:
                best_t = t
                hit = True
        if hit:
            ranges[i] = best_t
            points[i] = p + best_t * d
    return angles, ranges, points

def world_to_local(points_world, pose):
    rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(rtheta), np.sin(rtheta)
    R = np.array([[c, -s], [s, c]])
    return (points_world - np.array([rx, ry])) @ R

class RandomWalkController:
    def __init__(self, n_robots, forward_speed=0.15, turn_omega=1.8,
                 detect_dist=0.35, turn_steps_range=(10, 45), seed=99):
        self.n_robots = n_robots
        self.forward_speed = forward_speed
        self.turn_omega = turn_omega
        self.detect_dist = detect_dist
        self.turn_steps_range = turn_steps_range
        self.front_n_rays = 9
        self.front_max_range = 0.5
        self.front_fov = np.pi / 2
        self.clear_margin = 0.05
        self.clear_steps = 3
        self.min_turn_steps = 5
        self.stuck_window = 40
        self.stuck_dist = 0.03
        self.recovery_turn_steps_range = (35, 85)
        self.stuck_cooldown_steps = 60
        np.random.seed(seed)
        self.turn_remaining = np.zeros(n_robots, dtype=int)
        self.turn_dir = np.ones(n_robots)
        self.turn_elapsed = np.zeros(n_robots, dtype=int)
        self.clear_count = np.zeros(n_robots, dtype=int)
        self.pos_hist = np.zeros((n_robots, self.stuck_window, 2), dtype=float)
        self.hist_idx = np.zeros(n_robots, dtype=int)
        self.hist_len = np.zeros(n_robots, dtype=int)
        self.stuck_cooldown = np.zeros(n_robots, dtype=int)

    def _pick_turn_dir_biased(self, angles, ranges, max_range):
        rr = np.where(np.isnan(ranges), max_range, ranges)
        left_mask = angles > 1e-9
        right_mask = angles < -1e-9
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return float(np.random.choice([-1, 1]))
        left_clear = float(rr[left_mask].mean())
        right_clear = float(rr[right_mask].mean())
        if abs(left_clear - right_clear) < 1e-6:
            return float(np.random.choice([-1, 1]))
        return 1.0 if left_clear > right_clear else -1.0

    def _is_stuck(self, ri, pose_xy):
        self.pos_hist[ri, self.hist_idx[ri]] = pose_xy
        self.hist_idx[ri] = (self.hist_idx[ri] + 1) % self.stuck_window
        self.hist_len[ri] = min(self.hist_len[ri] + 1, self.stuck_window)
        if self.stuck_cooldown[ri] > 0:
            self.stuck_cooldown[ri] -= 1
            return False
        if self.hist_len[ri] < self.stuck_window:
            return False
        old_xy = self.pos_hist[ri, self.hist_idx[ri]]
        moved = float(np.linalg.norm(pose_xy - old_xy))
        return moved < self.stuck_dist

    def step(self, poses, all_segs):
        dxu = np.zeros((2, self.n_robots))
        for ri in range(self.n_robots):
            pose_ri = poses[:, ri]
            pose_xy = pose_ri[:2].astype(float)
            stuck = self._is_stuck(ri, pose_xy)
            angles, front_ranges, _ = simulate_lidar(
                pose_ri, all_segs,
                n_rays=self.front_n_rays, max_range=self.front_max_range,
                fov=self.front_fov)
            has_hit = np.any(~np.isnan(front_ranges))
            front_min = float(np.nanmin(front_ranges)) if has_hit else np.inf
            if self.turn_remaining[ri] > 0:
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega
                self.turn_remaining[ri] -= 1
                self.turn_elapsed[ri] += 1
                if front_min > (self.detect_dist + self.clear_margin):
                    self.clear_count[ri] += 1
                else:
                    self.clear_count[ri] = 0
                if (self.turn_elapsed[ri] >= self.min_turn_steps
                        and self.clear_count[ri] >= self.clear_steps):
                    self.turn_remaining[ri] = 0
            elif stuck:
                self.turn_remaining[ri] = np.random.randint(*self.recovery_turn_steps_range)
                self.turn_dir[ri] = self._pick_turn_dir_biased(angles, front_ranges, self.front_max_range)
                self.turn_elapsed[ri] = 0
                self.clear_count[ri] = 0
                self.stuck_cooldown[ri] = self.stuck_cooldown_steps
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega
            elif has_hit and front_min < self.detect_dist:
                self.turn_remaining[ri] = np.random.randint(*self.turn_steps_range)
                self.turn_dir[ri] = self._pick_turn_dir_biased(angles, front_ranges, self.front_max_range)
                self.turn_elapsed[ri] = 0
                self.clear_count[ri] = 0
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega
            else:
                dxu[0, ri] = self.forward_speed
                dxu[1, ri] = 0.0
        return dxu

def main():
    import time

    cfg = RandomConfig()
    print(f"= Experiment Config =")
    print(f"seed = {cfg.seed}")
    print(f"forward_speed = {cfg.forward_speed:.3f}")
    print(f"turn_omega = {cfg.turn_omega:.3f}")
    print(f"detect_dist = {cfg.detect_dist:.3f}")
    print(f"turn_steps_range = {cfg.turn_steps_range}")

    # Experiment parameters
    N_ROBOTS = 5
    SIM_STEPS = 6000      # ~200s of robot time at 33ms/step
    LIDAR_EVERY = 30      # keyframe interval -> ~100 keyframes
    LIDAR_RAYS = 180
    RANGE_TRIGGER = 0.60  # max inter-robot dist for UWB range (m)
    RANGE_SIGMA = 0.02    # UWB range noise std dev (m)

    init_conds = INIT_POSES.copy()
    # Robotarium may modify init_conds in-place (shared ref with self.poses).
    # Save a copy before creating the Robotarium to preserve actual initial poses.
    init_conds_save = init_conds.copy()

    r = robotarium.Robotarium(
        number_of_robots=N_ROBOTS,
        show_figure=not LOCAL_SIM,
        sim_in_real_time=not LOCAL_SIM,
        initial_conditions=init_conds,
    )

    uni_barrier = create_unicycle_barrier_certificate_with_boundary()

    controller = RandomWalkController(
        n_robots=N_ROBOTS,
        forward_speed=cfg.forward_speed,
        turn_omega=cfg.turn_omega,
        detect_dist=cfg.detect_dist,
        turn_steps_range=cfg.turn_steps_range,
        seed=cfg.seed,
    )

    # Pre-allocate data arrays
    gt_poses_all = np.zeros((SIM_STEPS, 3, N_ROBOTS))
    encoder_data_all = np.zeros((SIM_STEPS, 2, N_ROBOTS))

    n_keyframes_max = SIM_STEPS // LIDAR_EVERY + 1
    lidar_scans_local = np.full((n_keyframes_max, N_ROBOTS, LIDAR_RAYS, 2), np.nan)
    keyframe_steps = []
    range_edges = [] # list of (r, s, kf_idx, dist_noisy)

    rng = np.random.default_rng(0)  # for UWB noise

    t0 = time.time()

    for step in range(SIM_STEPS):
        # Get current state
        poses = r.get_poses()       # (3, N_ROBOTS)
        encoders = r.get_encoders() # (2, N_ROBOTS)

        # Store every-step data
        gt_poses_all[step] = poses
        encoder_data_all[step] = encoders

        # Keyframe: lidar + UWB
        if step % LIDAR_EVERY == 0:
            kf_idx = len(keyframe_steps)
            keyframe_steps.append(step)

            for ri in range(N_ROBOTS):
                pose_ri = poses[:, ri]
                _, _, pts_world = simulate_lidar(pose_ri, ALL_SEGMENTS, n_rays=LIDAR_RAYS)
                pts_local = world_to_local(pts_world, pose_ri)
                lidar_scans_local[kf_idx, ri] = pts_local

            # UWB range measurements between robot pairs
            for ri in range(N_ROBOTS):
                for si in range(ri + 1, N_ROBOTS):
                    dist = np.linalg.norm(poses[:2, ri] - poses[:2, si])
                    if dist < RANGE_TRIGGER:
                        noisy_dist = float(dist + rng.normal(0.0, RANGE_SIGMA))
                        range_edges.append((ri, si, kf_idx, noisy_dist))

        # Compute velocities
        dxu = controller.step(poses, ALL_SEGMENTS)
        dxu = uni_barrier(dxu, poses)
        r.set_velocities(np.arange(N_ROBOTS), dxu)
        r.step()

        # Progress logging
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"[step {step:4d}/{SIM_STEPS}] "
                  f"keyframes={len(keyframe_steps)}, "
                  f"range_edges={len(range_edges)}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0

    # Trim lidar array to actual keyframe count
    n_kf = len(keyframe_steps)
    lidar_scans_local = lidar_scans_local[:n_kf]

    # Build range_edges array
    if range_edges:
        range_edges_arr = np.array(range_edges)
    else:
        range_edges_arr = np.zeros((0, 4))

    # Save data
    np.save('gt_poses', gt_poses_all)                     # (SIM_STEPS, 3, N_ROBOTS)
    np.save('encoder_data', encoder_data_all)             # (SIM_STEPS, 2, N_ROBOTS)
    np.save('lidar_scans', lidar_scans_local)             # (n_kf, N_ROBOTS, 180, 2)
    np.save('keyframe_steps', np.array(keyframe_steps))   # (n_kf,)
    np.save('range_edges', range_edges_arr)               # (M, 4) or (0, 4)
    np.save('initial_poses', init_conds_save)             # (3, N_ROBOTS)

    print(f"\n= EXPERIMENT COMPLETE =")
    print(f"Total steps: {SIM_STEPS}")
    print(f"Keyframes: {n_kf}")
    print(f"Range edges: {len(range_edges)}")
    print(f"Wall-clock time: {elapsed:.1f}s")
    print(f"Saved: gt_poses.npy, encoder_data.npy, lidar_scans.npy, "
          f"keyframe_steps.npy, range_edges.npy, initial_poses.npy")

    r.call_at_scripts_end()

if __name__ == "__main__":
    main()
