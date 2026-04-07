import numpy as np
import gtsam
from robot import M, B_list

def forward_kinematics(M: gtsam.Pose3, B_list: list, theta: np.ndarray) -> gtsam.Pose3:
    """
    Computes the forward kinematics using the Product of Exponentials (POE) formula in the body frame.

    Args:
        M (gtsam.Pose3): The home configuration of the end-effector (pose at zero joint angles).
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        theta (np.ndarray): A numpy array of joint angles (in radians).

    Returns:
        gtsam.Pose3: The pose of the end-effector in the body frame after applying the joint angles.
    """
    T_out = None
    # ------- start solution ---------------------------------
    T_out = M
    for i in range(len(B_list)):
        xi = B_list[i] * theta[i]
        T_out = T_out.compose(gtsam.Pose3.Expmap(xi))
    # ------- end solution -----------------------------------
    return T_out

def jacobian_body(B_list: list, theta: np.ndarray) -> np.ndarray:
    """
    Computes the 6x3 body Jacobian for a 3-joint leg.

    Args:
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        theta (np.ndarray): A numpy array of joint angles (in radians).

    Returns:
        np.ndarray: A 6x3 numpy array representing the body Jacobian matrix.
    """
    out = None
    # ------- start solution ---------------------------------
    n = len(B_list)
    out = np.zeros((6, n))
    
    for i in range(n):
        # ccompose inverse exponentials after joint i
        T_adj_inv = gtsam.Pose3()
        for j in range(n - 1, i, -1):
            xi_j = -B_list[j] * theta[j]
            T_adj_inv = T_adj_inv.compose(gtsam.Pose3.Expmap(xi_j))
        
        out[:, i] = T_adj_inv.AdjointMap() @ B_list[i]
    # ------- end solution -----------------------------------

    return out

def inverse_kinematics(M: gtsam.Pose3, B_list: list, T_d: gtsam.Pose3, theta_init: np.ndarray, 
                       alpha: float = 0.1, max_iters: int = 100, tol: float = 1e-4) -> np.ndarray:
    """
    Solves the Inverse Kinematics (IK) problem using gradient descent on the body-frame error twist.

    Args:
        M (gtsam.Pose3): The home configuration of the end-effector (pose at zero joint angles).
        B_list (list): A list of 6D screw axes in the body frame, one for each joint.
        T_d (gtsam.Pose3): The desired pose of the end-effector in the body frame.
        theta_init (np.ndarray): Initial guess for the joint angles (in radians).
        alpha (float, optional): Step size for the gradient descent. Defaults to 0.1.
        max_iters (int, optional): Maximum number of iterations for the solver. Defaults to 100.
        tol (float, optional): Convergence tolerance for the error twist norm. Defaults to 1e-4.

    Returns:
        np.ndarray: A numpy array of joint angles (in radians) that achieve the desired pose.
    """
    theta = None
    # ------- start solution ---------------------------------
    theta = theta_init.copy()
    W = np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # weight position only, not orientation
    
    for k in range(max_iters):
        T_current = forward_kinematics(M, B_list, theta)
        error_twist = gtsam.Pose3.Logmap(T_current.inverse().compose(T_d))
        
        if np.linalg.norm(error_twist) < tol:
            break
        
        J_b = jacobian_body(B_list, theta)
        theta = theta + alpha * J_b.T @ W @ error_twist
    # ------- end solution -----------------------------------
    return theta