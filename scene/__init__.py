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

import os
import random
import json

from numpy.random.mtrand import f
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
import copy
import numpy as np
from scene.dataset_readers import storePly
import math

from scene.shs_updater import ShsUpdater

np.random.seed(1234)


class Scene_fore_w_rand_group:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None,
                 shuffle=False, resolution_scales=[1.0], source_path=None, total_frame=100, deform_type="multi",
                 load_after_diff=False, fsize=15,
                 random_pts_num=50000, random_style='nordn', radius=[5,], ext_scale=4, group_idx=0, fore=False,
                 use_alpha=False, depth_use=False, init_edit_path=None, update_idx=0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.loaded_iter = None
        self.model_path = args.model_path
        self.gaussians = gaussians
        self.use_alpha = use_alpha
        
        # define current saved_folder name for Rendering
        if fore:
            base_saved_folder_name = "point_cloud_f_edit{}".format(str(update_idx))
        else:
            base_saved_folder_name = "point_cloud_b_edit{}".format(str(update_idx))

        if init_edit_path is not None:
            # Video-3DGS-Edit, 
            load_iteration = -1
            if not load_after_diff:
                # We need to load pre-trained 3DGS for SH update
                base_saved_folder_name = base_saved_folder_name.replace(
                    "edit{}".format(str(update_idx)),
                    "edit{}".format(str(update_idx-1)))
            
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(
                        self.model_path, base_saved_folder_name, 'group_{}'.format(group_idx)))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # load multiple clips
        self.train_list = []
        self.test_list = []
        scene_info_list = []
        self.camera_extent_list = []

        clip_start = 1000
        clip_end = 0
        self.clip_overlap_dict = {}
        self.full_frame_name = set([])

        # below aims to know the first, last, and overlapping frame when multiple clips come
        # ex) source_path = ["clip_from011_to030", "clip_from030_to041"]
        # first: 11, overlap: 30, last: 41
        for clip_idx, clip_source_path in enumerate(source_path):
            clip_source_path = clip_source_path.split('/')[-1][len('clip_from'):]
            cur_clip_start = int(clip_source_path.split('_')[0])
            cur_clip_end = int(clip_source_path.split('_')[-1][2:])
            clip_range = set(list(range(cur_clip_start, cur_clip_end+1)))
            overlapping_elements = clip_range & self.full_frame_name
            if len(overlapping_elements) > 0:
                for overlap_ele in list(overlapping_elements):
                    self.clip_overlap_dict[overlap_ele] = [clip_idx-1, clip_idx]
            self.full_frame_name = self.full_frame_name.union(clip_range)
            clip_start = min(int(clip_source_path.split('_')[0]), clip_start)
            clip_end = max(int(clip_source_path.split('_')[-1][2:]), clip_end)
        frame_number = clip_end - clip_start + 1

        for sub_source_path in source_path:
            self.train_cameras = {}
            self.test_cameras = {}
            scene_info = sceneLoadTypeCallbacks["ColmapFB"](
                sub_source_path, args.eval,
                n_frames=frame_number, f_size=fsize, deform_type=deform_type, first_idx=clip_start,
                depth_use=depth_use, init_edit_path=init_edit_path)

            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                
                camera_json_path = os.path.join(self.model_path, 'cam_fore')
                os.makedirs(camera_json_path, exist_ok=True)
                source_clip_name = sub_source_path.split('/')[-1]
                with open(os.path.join(camera_json_path,
                            "cameras_{}.json".format(str(source_clip_name))), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.camera_extent_list.append(scene_info.nerf_normalization["radius"] * ext_scale)
            scene_info_list.append(scene_info)

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            self.train_list.append(self.train_cameras)
            self.test_list.append(self.test_cameras)

        # select the key clip according to the number of points
        max_idx = 0
        max_points_num = 0
        self.cameras_extent = self.camera_extent_list[0]
        for idx, scene_info_ in enumerate(scene_info_list):
            current_max_points_num = len(scene_info_.point_cloud[0])
            if current_max_points_num > max_points_num:
                max_idx = idx
                max_points_num = current_max_points_num
                self.cameras_extent = self.camera_extent_list[idx]

        scene_info = scene_info_list[max_idx]
        single_camera = self.train_list[0][1.0]
        ref_camera = single_camera[0]
        width, height = ref_camera.image_width, ref_camera.image_height

        # mask generation
        num_frames = total_frame
        camera_mask_list = []
        for i in range(num_frames):
            for train_cams in self.train_list:
                img_list = [int(x.image_name) for x in train_cams[1.0]]
                if i in img_list:
                    idx = img_list.index(i)
                    camera_mask_list.append(train_cams[1.0][idx].mask[0])
                    break

        # add random points
        if random_style != "nordn":
            scene_pts = self.generate_rdn_pts(
                scene_info.point_cloud, num_points=random_pts_num,
                random_style=random_style, distance=radius, group_idx=group_idx)
        else:
            scene_pts = scene_info.point_cloud


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, base_saved_folder_name, 'group_{}'.format(group_idx),
                        "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
            if fore and use_alpha:
                self.gaussians.load_alpha(os.path.join(self.model_path, base_saved_folder_name, 'group_{}'.format(group_idx),
                        "iteration_" + str(self.loaded_iter), "blend.npy"))
        else: 
            self.gaussians.create_from_pcd(scene_pts, self.cameras_extent / ext_scale,
                    width, height, camera_mask_list)


    def generate_rdn_pts(self, original_pts, num_points, random_style, distance, group_idx):
        point_max_coordinate = np.max(original_pts[0], axis=0)
        point_min_coordinate = np.min(original_pts[0], axis=0)
        scene_center = (point_max_coordinate + point_min_coordinate) / 2
        scene_size = np.max(point_max_coordinate - point_min_coordinate)
        # build unit sphere points
        n_points = num_points / len(distance)
        points_list = []
        colors_list = []
        for dist in distance:
            print(f"Generating Spherical Random point cloud ({n_points})...")
            samples = np.arange(n_points)
            y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
            theta = phi * samples  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
            # build background sphere
            print(f"Spherical Points are Aligned")
            background_sphere_point_xyz = (unit_sphere_points * scene_size * dist) + scene_center
            background_sphere_point_rgb = np.asarray(np.random.random(background_sphere_point_xyz.shape) * 255, dtype=np.uint8)

            points_list.append(background_sphere_point_xyz)
            colors_list.append(background_sphere_point_rgb)

        background_sphere_point_xyz = np.concatenate(points_list, axis=0)
        background_sphere_point_rgb = np.concatenate(colors_list, axis=0)

        if random_style == "rdn_merged_w_fore":
            print(f"Spherical Points are merged with existing fore points")
            background_sphere_point_xyz = np.concatenate([original_pts[0], background_sphere_point_xyz], axis=0)
            background_sphere_point_rgb = np.concatenate([original_pts[1]*255, background_sphere_point_rgb], axis=0)
            
        new_pts = BasicPointCloud(
            points=background_sphere_point_xyz,
            colors=background_sphere_point_rgb/255.,
            normals=np.zeros_like(background_sphere_point_rgb)
        )
        storePly(os.path.join(self.model_path, 'new_points_{}.ply'.format(group_idx)),
            background_sphere_point_xyz, background_sphere_point_rgb)

        return new_pts


    def save_clip_f(self, iteration, group, update_idx):
        point_cloud_path = os.path.join(
            self.model_path,
            "point_cloud_f_edit{}/group_{}/iteration_{}".format(
                update_idx, group, iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.use_alpha:
            self.gaussians.save_trans(os.path.join(point_cloud_path, "blend.npy"))
  

    def save_clip_b(self, iteration, group, update_idx):
        point_cloud_path = os.path.join(
            self.model_path,
            "point_cloud_b_edit{}/group_{}/iteration_{}".format(
                update_idx, group, iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self):
        return self.train_list

    def getTestCameras(self):
        return self.test_list

    def getOverlappingFrames(self):
        return self.clip_overlap_dict

    def getFramesInClip(self):
        return self.full_frame_name



