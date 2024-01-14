"""
Author: Eckert ZHANG
Date: 2021-12-20 19:30:04
LastEditTime: 2022-01-12 22:34:17
LastEditors: Eckert ZHANG
Description: 
"""
import torch


def skew(w):
    """
    Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
        w: (B, 3) A 3-vector
    Returns:
        W: (B, 3, 3) A skew matrix such that W @ v == w x v
    """
    assert w.shape[1] == 3
    W = torch.zeros([w.shape[0], 3, 3]).to(w.device)
    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] = w[:, 1]
    W[:, 1, 0] = w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] = w[:, 0]
    return W


def rp_to_se3(R, p):
    """
    Rotation and translation to homogeneous transform.

    Args:
        R (B, 3, 3): An orthonormal rotation matrix.
        p (B, 3, 1): A 3-vector representing an offset.
    Returns:
        X: (4, 4) The homogeneous transformation matrix described by rotating by R and translating by p.
    """
    X = torch.eye(4, dtype=torch.float32).reshape(1, 4, \
                4).expand(R.shape[0], -1, -1).to(R.device)
    X[:, :3, :] = torch.cat([R, p], dim=-1)
    return X


def exp_so3(w, theta):
    """
    Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
        w (B,3): An axis of rotation. This is assumed to be a unit-vector.
        theta (B,): An angle of rotation.
    Returns:
        R: (B, 3, 3) An orthonormal rotation matrix representing a 
            rotation of magnitude theta about axis w.
    """
    W = skew(w)
    theta = theta[:, None, None]
    R = torch.eye(3).reshape(1, 3, 3).expand(w.shape[0],-1,-1).to(theta.device) + torch.sin(\
        theta) * W + (1.0 - torch.cos(theta)) * torch.matmul(W, W)
    return R


def exp_se3(S, theta):
    """
    Exponential map from Lie algebra so3 to Lie group SO3.

    Args:
        S (B, 6): A screw axis of motion. B:batchsize
        theta (float): Magnitude of motion.
    Returns:
        a_X_b: (4, 4) The homogeneous transformation matrix attained by 
            integrating motion of magnitude theta about S for one second.
    """
    w, v = torch.chunk(S, 2, dim=1)
    W = skew(w)
    R = exp_so3(w, theta)
    theta = theta[:, None, None]
    p = torch.eye(3).reshape(1, 3, 3).expand(w.shape[0],-1,-1).to(theta.device) * theta + \
        (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * torch.matmul(W, W)
    p = torch.matmul(p, v[..., None])
    return rp_to_se3(R, p)


def to_homogenous(v):
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v):
    return v[..., :3] / v[..., -1:]


def rotation2quaternion(R):
    """
    translate the rotation matrix to the quanternion.

    Args:
        R (B,3,3): rotation matrix
    """
    w = torch.sqrt(R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] +
                   torch.ones_like(R[:, 0, 0])) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    q = torch.stack([x, y, z, w], dim=-1)
    return q


def map_xyz_2_normalized(xyz, point_c, width):
    """
    Args:
        xyz (B,3): [description]
        point_c (3): [description]
        width (int): [description]
    """
    B, _ = xyz.shape
    point_c = point_c
    point_c = point_c.unsqueeze(0).expand(B, -1)
    point = (xyz - point_c) / width
    return point
