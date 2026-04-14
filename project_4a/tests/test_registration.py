import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gicp import GICPConfig, gicp
from src.real_scan_registration import preprocess_scan, evaluate_transform

class TestGICPRegistration(unittest.TestCase):
    def setUp(self):
        """Set up the data and configuration for the specific scan pair.
        """
        self.voxel_size = 0.25
        self.max_points = 5000
        self.seed = 21
        
        # Initialize RNG for reproducibility
        self.rng = np.random.default_rng(self.seed)

    def test_rmse_reduction(self):
        """Verify that GICP reduces RMSE for the scan pair.
        """
        # 1. Load Scans
        data_path = Path(__file__).resolve().parent / "real_scans.npz"
        data = np.load(data_path)
        source_raw = data['start_scan_points']
        target_raw = data['end_scan_points']

        # 2. Preprocess
        source_points = preprocess_scan(source_raw, self.voxel_size, self.max_points, self.rng)
        target_points = preprocess_scan(target_raw, self.voxel_size, self.max_points, self.rng)

        # 3. Run GICP
        config = GICPConfig(
            correspondence_distance_threshold=2.0,
            covariance_neighbor_count=12,
            min_valid_correspondences=40,
            inner_max_iterations=30,
            prior_sigmas=(1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
        )
        
        result = gicp(source_points, target_points, config=config)
        
        # 4. Evaluate RMSE
        before_rmse, after_rmse = evaluate_transform(
            source_points, target_points, result.bTa_matrix
        )

        # 5. Assertions
        # Ensure the algorithm actually converged to a better result
        self.assertLess(after_rmse, before_rmse, 
                        f"RMSE did not decrease! Before: {before_rmse:.4f}, After: {after_rmse:.4f}")
        print(f'RMSE Before: {before_rmse:.4f}, After: {after_rmse:.4f}')
        
        self.assertLess(after_rmse, 0.95, 
                        f"GICP precision is lower than expected. RMSE: {after_rmse:.4f}")