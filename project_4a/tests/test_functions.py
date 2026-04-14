import unittest
import sys
import gtsam
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gicp import compute_covariance_matrix_single_point, transform_covariances, build_information_matrices, diag_cov

class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.reg = np.linalg.eigvalsh(diag_cov)

    def test_compute_function_eigenvalues(self):
        """Verify compute_covariance_matrix_single_point() produces the exact eigenvalues provided 
        in regularization.
        """
        # Create a random neighborhood
        np.random.seed(42)
        neighbors = np.random.rand(20, 3)
        
        cov = compute_covariance_matrix_single_point(neighbors)
        
        # Get eigenvalues of the resulting matrix
        evals = np.linalg.eigvalsh(cov)
        
        # Sort because eigh returns in ascending order
        self.assertIsNone(np.testing.assert_allclose(np.sort(evals), np.sort(self.reg), atol=1e-8),
                          'Failed to find chosen variances. Did you use the provided variances?')
    
    def test_compute_function_with_data(self):
        """ Verify compute_covariance_matrix_single_point() works with synthetic test data.
        """        
        # Load the stored data
        data_path = Path(__file__).resolve().parent / "test_data.npz"
        data = np.load(data_path)
        
        # Extract inputs and expected output
        neighbors = data['neighbors']
        expected_cov = data['expected_cov']

        # Check that the diagonal covariance contains expected eigenvalues
        if np.any(np.array([1e-3, 1.0, 1.0]) != self.reg):
            self.fail(f'Please change your diag_cov matrix back to contain the eigenvalues {[1e-3, 1.0, 1.0]}')
        
        # Run the function
        actual_cov = compute_covariance_matrix_single_point(neighbors)
        
        # Assert equality within a small numerical tolerance
        self.assertIsNone(np.testing.assert_allclose(
            actual_cov, 
            expected_cov, 
            atol=1e-7, 
            err_msg="The computed covariance does not match the stored ground truth."
        ))
    
    def test_transformation_alignment(self):
        """Verify that a transformation is correctly applied to a covariance matrix.
        """
        # Create a covariance matrix skewed heavily in X
        cov = np.array([[[10.0, 0.0, 0.0],
                         [0.0,  1.0, 0.0],
                         [0.0,  0.0, 1.0]]]) # Shape (1, 3, 3)
        
        rotation = gtsam.Rot3.Yaw(np.pi / 2)
        translation = gtsam.Point3(5, 10, 15)
        transform = gtsam.Pose3(rotation, translation)
        
        output = transform_covariances(cov, transform)
        
        expected = np.array([[[1.0,  0.0, 0.0],
                              [0.0, 10.0, 0.0],
                              [0.0,  0.0, 1.0]]])
        
        self.assertIsNone(np.testing.assert_allclose(output, expected, atol=1e-8))
    
    def test_information_sum_and_inverse(self):
        """Verify that Inf = inv(Cov_src + Cov_tgt) for valid correspondences.
        """
        # Create two simple diagonal covariances
        # Source: Sigma = diag(1, 1, 1)
        # Target: Sigma = diag(2, 2, 2)
        # Sum = diag(3, 3, 3) -> Inverse = diag(1/3, 1/3, 1/3)
        
        src_covs = np.array([np.eye(3), np.eye(3) * 2])
        tgt_covs = np.array([np.eye(3) * 2, np.eye(3) * 4])
        
        # Correspondence indices: source matches target
        indices = np.array([[0], [1]])
        mask = np.array([[True], [True]])
        
        info_mats = build_information_matrices(src_covs, tgt_covs, indices, mask)
        
        expected = np.array([np.eye(3) * (1.0 / 3.0), np.eye(3) * (1.0 / 6.0)])
        self.assertIsNone(np.testing.assert_allclose(info_mats, expected, atol=1e-5))
    
    def test_information_valid_mask_handling(self):
        """Verify that invalid correspondences result in zero/null matrices.
        """
        src_covs = np.array([np.eye(3), np.eye(3)])
        tgt_covs = np.array([np.eye(3), np.eye(3)])
        
        indices = np.array([[0], [1]])
        # Only the first one is valid
        mask = np.array([True, False])
        
        info_mats = build_information_matrices(src_covs, tgt_covs, indices, mask)
        
        # First should be inv(I + I) = 0.5 * I
        self.assertFalse(np.all(info_mats == 0))
        # Second should be all zeros (as initialized)
        self.assertIsNone(np.testing.assert_array_equal(info_mats[1], np.zeros((3, 3))),
                          'Associate invalid correspondences with a zero covariance matrix.')