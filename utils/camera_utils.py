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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorchMask, PILtoTorchDepth
from utils.graphics_utils import fov2focal
import os

from PIL import Image
import math

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]


    if cam_info.detail_image is not None:
        resized_detail_image_rgb = PILtoTorch(cam_info.detail_image, resolution)
        gt_detail_image = resized_detail_image_rgb[:3, ...]
    else:
        gt_detail_image = None

    mask_path = cam_info.image_path.replace('input', 'mask').replace('.jpg', '.jpg.png')
    if os.path.isfile(mask_path):
        mask = Image.open(mask_path)
        resized_mask = PILtoTorchMask(mask, resolution)
        loaded_mask = resized_mask
    else:
        loaded_mask = None

    if cam_info.edited_image is not None:
        if isinstance(cam_info.edited_image, list):
            gt_edited_image = []
            for idx in range(len(cam_info.edited_image)):
                gt_edited_image.append(PILtoTorch(cam_info.edited_image[idx], resolution)[:3, ...])
        else:
            resized_edited_image = PILtoTorch(cam_info.edited_image, resolution)
            gt_edited_image = resized_edited_image[:3, ...]
    else:
        gt_edited_image = None


    if cam_info.img_edited_image is not None:
        if isinstance(cam_info.img_edited_image, list):
            img_edited_image = []
            for idx in range(len(cam_info.img_edited_image)):
                img_edited_image.append(PILtoTorch(cam_info.img_edited_image[idx], resolution)[:3, ...])
        else:
            resized_img_edited_image = PILtoTorch(cam_info.img_edited_image, resolution)
            img_edited_image = resized_img_edited_image[:3, ...]
    else:
        img_edited_image = None

    if cam_info.depth is not None:
        depth = cam_info.depth / (cam_info.depth.max() + 1e-5)
        depth = PILtoTorchDepth(depth)
    else:
        depth = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, fid=cam_info.fid,
                  gt_detail_image=gt_detail_image, depth=depth, edited_image=gt_edited_image, img_edited_image=img_edited_image)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )

