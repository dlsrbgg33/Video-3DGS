#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from scipy.spatial.transform import Rotation
import copy

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def compute_transformation_matrix(pose_a, pose_b):
    # Assuming pose is a dictionary with 'R' and 't' keys
    R_a, t_a = pose_a.R, pose_a.T
    R_b, t_b = pose_b.R, pose_b.T
    
    R = np.dot(R_b, np.linalg.inv(R_a))
    t = t_b - np.dot(R, t_a)
    
    transformation_matrix = np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0, 0, 0, 1]])
    return transformation_matrix

def pose_to_matrix(rotation, translation):
    """Converts a camera pose represented by a rotation matrix and a translation vector to a 4x4 matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def compute_transformation_matrix_re(pose_a, pose_b):
    # Assuming pose is a dictionary with 'R' and 'T' keys
    # transformation matrix from pose b to pose a
    R_a, t_a = pose_a.R, pose_a.T
    R_b, t_b = pose_b.R, pose_b.T

    pose_a_matx = pose_to_matrix(R_a, t_a)
    pose_b_matx = pose_to_matrix(R_b, t_b)
    
    b_inv = np.linalg.inv(pose_b_matx)

    return np.dot(b_inv, pose_a_matx)


def apply_transformation(pose, transformation_matrix):
    
    R, t = pose.R, pose.T
    pose_matx = np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0, 0, 0, 1]])
    new_pose = np.dot(pose_matx, transformation_matrix)
    pose.R = new_pose[:3, :3]
    pose.T = new_pose[:3, 3]
    # Assuming points is a Nx4 matrix (with homogeneous coordinates)
    return pose


def average_transformations(transformations):
    # Sum of quaternions
    quaternion_sum = None
    
    # Sum of translation vectors
    translation_sum = np.zeros(3)
    
    for transformation in transformations:
        # Extract rotation matrix and translation vector
        R, t = transformation[:3, :3], transformation[:3, 3]
        
        # Convert rotation matrix to quaternion
        quaternion = Rotation.from_matrix(R).as_quat()
        
        # Add quaternion to sum
        if quaternion_sum is None:
            quaternion_sum = quaternion
        else:
            # Ensure quaternion consistency
            if np.dot(quaternion_sum, quaternion) < 0:
                quaternion = -quaternion
            quaternion_sum += quaternion
        
        # Add translation to sum
        translation_sum += t
    
    # Normalize quaternion to average it
    quaternion_avg = quaternion_sum / np.linalg.norm(quaternion_sum)
    
    # Convert average quaternion back to rotation matrix
    R_avg = Rotation.from_quat(quaternion_avg).as_matrix()
    
    # Average translation
    t_avg = translation_sum / len(transformations)
    
    # Compose average transformation matrix
    transformation_avg = np.eye(4)
    transformation_avg[:3, :3] = R_avg
    transformation_avg[:3, 3] = t_avg
    
    return transformation_avg


def normalize_3d_points(points):
    # Translate the points to the center
    mean = np.mean(points, axis=0)
    points_centered = points - mean
    
    # Find the scaling factor
    max_abs_value = np.max(np.abs(points_centered))
    scale_factor = 1 / max_abs_value
    
    # Scale the points
    points_normalized = points_centered * scale_factor
    
    return points_normalized


def transform_camera_pose(pose, transformation_matrix):
    R, t = pose.R, pose.T
    R_new = np.dot(transformation_matrix[:3, :3], R)
    t_new = np.dot(transformation_matrix[:3, :3], t) + transformation_matrix[:3, 3]
    pose.R = R_new
    pose.T = t_new
    return pose

def transform_point(scene, transformation_matrix):
    try:
        point = copy.deepcopy(scene.point_cloud[0])
        colors = scene.point_cloud[1]
        normals = scene.point_cloud[2]
    except:
        point = copy.deepcopy(scene[0])
        colors = scene[1]
        normals = scene[2]

    ones = np.ones((point.shape[0], 1))

    point_homo = np.hstack((point, ones))
    point_transformed_homo = point_homo.dot(transformation_matrix.T)
    point_transformed = point_transformed_homo[:, :3] / point_transformed_homo[:, 3:]

    return BasicPointCloud(
        points=point_transformed, colors=colors, normals=normals)


def transform_point_normalize(scene, transformation_matrix):
    point = copy.deepcopy(scene.point_cloud[0])
    ones = np.ones((point.shape[0], 1))

    point_homo = np.hstack((point, ones))
    point_transformed_homo = np.dot(point_homo, transformation_matrix.T)
    point_transformed = point_transformed_homo[:, :3] / point_transformed_homo[:, 3:]

    point_transformed_norm = normalize_3d_points(point_transformed)

    return BasicPointCloud(
        points=point_transformed_norm, colors=scene.point_cloud[1], normals=scene.point_cloud[2])
