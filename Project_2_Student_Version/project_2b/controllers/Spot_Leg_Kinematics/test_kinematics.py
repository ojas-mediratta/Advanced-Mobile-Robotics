import unittest
import numpy as np
import gtsam
from brain import forward_kinematics, jacobian_body, inverse_kinematics
from robot import M, B_list

class TestBrainFunctions(unittest.TestCase):

    def setUp(self):
        self.M = M
        self.B_list = B_list

    def test_forward_kinematics(self):
        theta = np.array([0.0, 0.0, 0.0])
        result = forward_kinematics(self.M, self.B_list, theta)
        self.assertIsInstance(result, gtsam.Pose3)
        self.assertTrue(np.allclose(result.translation(), [0.16604, -0.59098, -0.04271], atol=1e-2))

    def test_jacobian_body(self):
        delta = 1e-6  # Small perturbation for numerical differentiation

        inputs = [
            np.array([0.71089257, -0.24801771, 0.30830328]),
            np.array([0.66591222, -0.5939644, 0.70885446]),
            np.array([0.4453684, -0.02515904, -0.17632338]),
            np.array([-0.1929157, 0.58678236, -0.76513543]),
            np.array([0.69847322, 0.70261703, 0.17043766]),
            np.array([0.35135767, 0.75693013, 0.47448235]),
            np.array([-0.55973997, -0.67041943, 0.27985106]),
            np.array([-0.34108073, 0.36943876, 0.70079703]),
            np.array([-0.34823398, 0.12524976, 0.56803332]),
            np.array([-0.33203877, 0.21085682, -0.46760393]),
        ]

        for theta in inputs:  # Run the test 10 times
            # Compute the analytical Jacobian
            analytical_jacobian = jacobian_body(self.B_list, theta)

            # Compute the numerical Jacobian
            numerical_jacobian = np.zeros((6, 3))
            for i in range(3):  # Loop over each joint
                theta_perturbed = np.copy(theta)
                theta_perturbed[i] += delta  # Perturb the i-th joint angle

                # Compute the forward kinematics for perturbed and unperturbed angles
                T_original = forward_kinematics(self.M, self.B_list, theta)
                T_perturbed = forward_kinematics(self.M, self.B_list, theta_perturbed)

                # Compute the twist (change in pose) and divide by delta
                twist = gtsam.Pose3.Logmap(T_original.inverse().compose(T_perturbed))
                numerical_jacobian[:, i] = twist / delta

            # Compute the Frobenius norm of the difference
            norm_difference = np.linalg.norm(analytical_jacobian - numerical_jacobian, ord='fro')

            # Assert that the norm of the difference is within acceptable tolerance
            self.assertTrue(norm_difference < 0.5)

    def test_inverse_kinematics(self):
        inputs = [
            np.array([0.71089257, -0.24801771, 0.30830328]),
            np.array([0.66591222, -0.5939644, 0.70885446]),
            np.array([0.4453684, -0.02515904, -0.17632338]),
            np.array([-0.1929157, 0.58678236, -0.76513543]),
            np.array([0.69847322, 0.70261703, 0.17043766]),
            np.array([0.35135767, 0.75693013, 0.47448235]),
            np.array([-0.55973997, -0.67041943, 0.27985106]),
            np.array([-0.34108073, 0.36943876, 0.70079703]),
            np.array([-0.34823398, 0.12524976, 0.56803332]),
            np.array([-0.33203877, 0.21085682, -0.46760393]),
        ]
        for theta in inputs:
            # Compute the target pose using forward kinematics
            target_pose = forward_kinematics(self.M, self.B_list, theta)

            # Solve inverse kinematics starting from zero initial guess
            initial_guess = np.array([0.0, 0.0, 0.0])
            result_theta = inverse_kinematics(self.M, self.B_list, target_pose, initial_guess)

            # Compute the resulting pose using forward kinematics
            result_pose = forward_kinematics(self.M, self.B_list, result_theta)

            # Check the error between the resulting pose and the target pose
            error = gtsam.Pose3.Logmap(result_pose.inverse().compose(target_pose))

            # Assert that the error is within acceptable tolerance
            self.assertTrue(np.linalg.norm(error) < 0.5)

if __name__ == "__main__":
    unittest.main()