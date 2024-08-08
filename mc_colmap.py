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
import logging
from argparse import ArgumentParser
import shutil
import glob
from tqdm import tqdm

from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
import struct
import time

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
colmap_command = "colmap"

use_gpu = 1
camera = "SIMPLE_PINHOLE"


def run_colmap(source_path, logging):

    input_path = os.path.join(source_path, "local", "full")
    source_path = os.path.join(source_path, "local", "full", "all") 
    print("Running colmap on full images")
    start_time = time.time()
    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)    
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + input_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.num_threads -1 \
        --SiftExtraction.estimate_affine_shape=true \
        --SiftExtraction.domain_size_pooling=true \
        --SiftExtraction.use_gpu " + str(use_gpu)

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.num_threads 1 \
        --SiftMatching.guided_matching=true \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + input_path + "/input \
        --output_path "  + source_path + "/distorted/sparse \
        --Mapper.num_threads=1 \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + input_path + "/input \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    colmap_end_time = time.time()
    total_colmap_time = colmap_end_time - start_time
    logging.info(f"COLMAP generated in {total_colmap_time:.2f} seconds")
    print("Done.")


def run_colmap_w_mask(source_path, logging):

    print("Running colmap on full images")
    start_time = time.time()
    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)    
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --ImageReader.mask_path " + source_path + "/mask \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.num_threads -1 \
        --SiftExtraction.estimate_affine_shape=true \
        --SiftExtraction.domain_size_pooling=true \
        --SiftExtraction.use_gpu " + str(use_gpu)

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.num_threads 1 \
        --SiftMatching.guided_matching=true \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + source_path + "/input \
        --output_path "  + source_path + "/distorted/sparse \
        --Mapper.num_threads=1 \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + source_path + "/input \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    colmap_end_time = time.time()
    total_colmap_time = colmap_end_time - start_time
    logging.info(f"COLMAP generated in {total_colmap_time:.2f} seconds")
    print("Done.")


def run_colmap_mask_clip(source_path):

    subtree_sources = glob.glob(source_path+'/*')
    for sub_source in subtree_sources:
        if "local" in sub_source.split('/')[-1]:
            print("Running colmap on local")
            source_path_list = glob.glob(sub_source+'/*')
            size_of_clip = 10 # initial default size of clip
        else:
            print("Running colmap on global")
            source_path_list = [sub_source]
            size_of_clip = 10
            # we are not running COLMAP on background
            ## placeholder for future work
            continue
            
        for source_path in tqdm(source_path_list):
            if 'full' not in source_path:
                # Consider all the moving objects as one instance
                continue
            total_num_frames = len(glob.glob(source_path+'/input/*'))
            start_frame = 0
            clip_size = size_of_clip
            # all the cameras should share same intrinsics (monocular video)
            intrinsic_shared = None 
            while clip_size <= total_num_frames:
                clip_fail = True
                error_test = False
                if start_frame > 0:
                    # each clip can have one overlapping frame
                    start_frame -= 1
                while clip_fail:
                    # we use zfill(3), indicating maximum number of frames used are less than 1000.
                    # users can change this number based on their dataset.
                    start_frame_str = str(start_frame).zfill(3)
                    clip_size_str = str(clip_size-1).zfill(3) 
                    new_clip_path = os.path.join(source_path, 'clip_from{}_to{}'.format(start_frame_str, clip_size_str))
                    os.makedirs(new_clip_path, exist_ok=True)
                    if not os.path.exists(new_clip_path+'/mask'):
                        shutil.copytree(source_path+'/mask', new_clip_path+'/mask')
                        new_clip_image_path = os.path.join(new_clip_path, 'input')
                        original_image_path = os.path.join(source_path,'input')
                        os.makedirs(new_clip_image_path, exist_ok=True)
                        for idx in range(start_frame, clip_size, 1):
                            prefix = "{:05}.jpg".format(idx)
                            shutil.copy(os.path.join(original_image_path, prefix), new_clip_image_path)
                    # "expect_num" is expected number of images generated from COLMAP
                    expect_num = clip_size - start_frame
                    try:
                        clip_fail = colmap_command_func(new_clip_path, expect_num, intrinsic_shared)
                        if clip_fail:
                            shutil.rmtree(new_clip_path)
                            if error_test:
                                if clip_size == total_num_frames:
                                    start_frame -= 1
                                else:
                                    clip_size += 1
                            else:
                                if clip_size == total_num_frames:
                                    start_frame -= 1
                                else:
                                    clip_size -= 1
                        else:
                            # succeed with COLMAP processsing on current clip
                            if intrinsic_shared is None:
                                # store the intrinsic parameters generated from first clip COLMAP processing
                                cameras_intrinsic_file = os.path.join(new_clip_path, "sparse/0", "cameras.bin")
                                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)[1].params
                                intrinsic_shared = ", ".join(map(str, cam_intrinsics))
                            error_test = False
                            start_frame = clip_size
                            clip_size += size_of_clip
                        if clip_size > total_num_frames and start_frame < total_num_frames:
                            clip_size = total_num_frames
                    except:
                        shutil.rmtree(new_clip_path)
                        if clip_size >= total_num_frames:
                            start_frame -= 1
                        else:
                            clip_size += 1
                        error_test = True


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def colmap_command_func(source_path, clip_size, intrinsic_shared):

    if os.path.exists(source_path + "/distorted"):
        shutil.rmtree(source_path + "/distorted")
    if os.path.exists(source_path + "/images"):
        shutil.rmtree(source_path + "/images")
    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)
    
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.mask_path " + source_path + "/mask \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.num_threads -1 \
        --SiftExtraction.estimate_affine_shape=true \
        --SiftExtraction.domain_size_pooling=true \
        --SiftExtraction.use_gpu " + str(use_gpu)
        

    if intrinsic_shared is not None:
        feat_extracton_cmd += f' --ImageReader.camera_params "{intrinsic_shared}"'

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.num_threads 1 \
        --SiftMatching.guided_matching=true \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + source_path + "/input \
        --output_path "  + source_path + "/distorted/sparse \
        --Mapper.num_threads=1 \
        --Mapper.ba_global_function_tolerance=0.000001")

    if intrinsic_shared is not None:
        mapper_cmd += " --Mapper.ba_refine_focal_length 0"
        mapper_cmd += " --Mapper.ba_refine_extra_params 0"
    
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + source_path + "/input \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        if "points3D" in file:
            with open(destination_file, "rb") as fid:
                num_points = read_next_bytes(fid, 8, "Q")[0]
                if num_points == 0:
                    return True

    extracted_frame_num = len(os.listdir(source_path+"/images"))
    if clip_size < 20:
        return extracted_frame_num != clip_size
    else:
        return extracted_frame_num <= clip_size * 0.1