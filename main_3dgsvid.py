"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

3DGS Reference: 3D Gaussian Splatting for Real-Time Radiance Field Rendering [1]
            (https://github.com/graphdeco-inria/gaussian-splatting)
Open VOS Reference: DEVA: Tracking Anything with Decoupled Video Segmentation [2]
            (https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

Video Editors References
- Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators (ICCV 2023) [3]
            (https://github.com/Picsart-AI-Research/Text2Video-Zero)
- TokenFlow: TokenFlow: Consistent Diffusion Features for Consistent Video Editing (ICLR 2024) [4]
            (https://github.com/omerbt/TokenFlow)
- RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models [5]
            (https://github.com/rehg-lab/RAVE)
"""


import os
import shutil
from numpy.lib.arraysetops import isin
import torch
from random import randint
from tqdm import tqdm
import sys
import copy
import glob
from PIL import Image
import logging
import time
import shutil

from gaussian_renderer import render_deform
from scene import Scene_fore_w_rand_group, GaussianModel, DeformModel, Scene_fore_w_rand_group_unified
from argparse import ArgumentParser, Namespace
from arguments import ModelParamsForeBack, PipelineParams
from arguments import OptimizationClipMaskParams3k, OptimizationClipMaskParams5k, OptimizationClipMaskParams30k 
from arguments import OptimizationClipMaskParams10k, OptimizationClipMaskParamsEdit

from utils.loss_utils import loss_func_w_bilateral, generate_bi_image
from utils.mask_gen_utils import gen_mask
from utils.cmd_utils import edit_cmd_run
from utils.general_utils import check_n_groups_clips
from mc_colmap import run_colmap_mask_clip, run_colmap

from main_3dgsvid_render import render_sets_dual
from metrics import evaluate, evaluate_edit
from clip_metrics import clipscore

from utils.gif_utils import create_gif_from_frames
from datetime import datetime



def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations,
             random_pts_num, radius, f_size, fore_random,
             tfloss_weight=2.0, group_size=1, deform_type="multi", frame_num=0,
             use_dual=True, model_path=None, logging=None, edited_path=None, update_idx=0):
    
    # check the number of groups and clips for each group
    sub_source_path = sorted(glob.glob(os.path.join(dataset.lsource_path, '*')))
    clip_dict, clip_num_in_g = check_n_groups_clips(group_size, sub_source_path)
    logging.info(
        f"This video has {clip_num_in_g} clips in total, divided into {len(clip_dict)} group(s)")

    # Here, we set multiple gaussians for multiple clips in video
    gaussians_f_list, scene_f_list = dict(), dict()
    gaussians_b_list, scene_b_list = dict(), dict()
    fore_random_style = "nordn" if use_dual else fore_random
    for group_idx, clip_path in clip_dict.items():
        # create gaussian model and scene for each group - fore / back
        gaussians_f = create_gaussian_model(opt, dataset, use_alpha=use_dual)
        scene_fore = create_scene(dataset, gaussians_f, clip_path, frame_num, deform_type,
                                   fore_random_style, random_pts_num, radius, f_size,
                                   group_idx, fore=True, use_alpha=use_dual,
                                   edited_path=edited_path, update_idx=update_idx)
        gaussians_f.training_setup(opt, edited_path)
        gaussians_f_list[group_idx] = gaussians_f
        scene_f_list[group_idx] = scene_fore
        if use_dual:
            gaussians_b = create_gaussian_model(opt, dataset)
            scene_back = create_scene(dataset, gaussians_b, clip_path, frame_num, deform_type,
                                        "only_s_random", random_pts_num, radius, f_size,
                                        group_idx, fore=False,
                                        edited_path=edited_path, update_idx=update_idx)
            gaussians_b.training_setup(opt, edited_path)
            gaussians_b_list[group_idx] = gaussians_b
            scene_b_list[group_idx] = scene_back
            # sanity check
            assert len(gaussians_f_list) == len(gaussians_b_list), \
             "number of group should be same for fore and back"

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Define deformation networks for each clips
    # we have multiple deformation networks in the same group
    deform_fore_all_dict = {}
    for group_idx in range(len(scene_f_list)):
        if deform_type == "multi":
            deform_fore_all_dict[group_idx] = {clip_idx: create_deform_model(
                                                    opt, edited_path=edited_path, group_idx=group_idx,
                                                    clip_idx=clip_idx, fore=True, model_path=dataset.model_path)
                                                    for clip_idx in range(len(scene_f_list[group_idx].getTrainCameras()))}
        elif deform_type == "single":
            deform_fore_all_dict[group_idx] = create_deform_model(
                                                    opt, edited_path=edited_path, group_idx=group_idx,
                                                    clip_idx=0, model_path=dataset.model_path)
        else:
            raise NotImplementedError("current deformation type supports multi / single")

    deform_back_all_dict = {}
    if use_dual:
        for group_idx in range(len(scene_b_list)):
            if deform_type == "multi":
                deform_back_all_dict[group_idx] = {clip_idx: create_deform_model(
                                                        opt, edited_path=edited_path, group_idx=group_idx,
                                                        clip_idx=clip_idx, model_path=dataset.model_path)
                                                        for clip_idx in range(len(scene_b_list[group_idx].getTrainCameras()))}
            elif deform_type == "single":
                deform_back_all_dict[group_idx] = create_deform_model(
                                                        opt, edited_path=edited_path, group_idx=group_idx,
                                                        clip_idx=0, model_path=dataset.model_path)
            else:
                raise NotImplementedError("current deformation type supports multi / single")

               
    # sequential training starts for each groups
    for group_int in range(len(gaussians_f_list)):
        first_iter = 0
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        clip_idx = 0
        gaussians_f = gaussians_f_list[group_int]
        scene_f = scene_f_list[group_int]
        deform_fore_dict = deform_fore_all_dict[group_int]
        if use_dual:
            gaussians_b = gaussians_b_list[group_int]
            scene_b = scene_b_list[group_int]
            deform_back_dict = deform_back_all_dict[group_int]
        else:
            gaussians_b, scene_b, deform_back_dict = None, None, None

        viewpoint_stack_fore = None
        viewpoint_stack_back = None
        deform_back = None

        # unified training for the clips in the same group with single fore / back 3dg
        for iteration in range(first_iter, opt.iterations + 1):  
            iter_start.record()

            # learning rate and SH degree update for each gaussians
            # (only during video-3dgs-recon)
            if edited_path is None:
                gaussians_f.update_learning_rate(iteration)
                if iteration % 1000 == 0:
                    gaussians_f.oneupSHdegree()
                if use_dual:
                    gaussians_b.update_learning_rate(iteration)
                    if iteration % 1000 == 0:
                        gaussians_b.oneupSHdegree()

            # check the number of clips in current group
            if not viewpoint_stack_fore:
                cur_clip_num = len(scene_f.getTrainCameras().copy())
                clip_idx = (clip_idx + 1) % cur_clip_num
                scene_fore_cams = scene_f.getTrainCameras().copy()[clip_idx]
                viewpoint_stack_fore = copy.deepcopy(scene_fore_cams[1.0])
                if use_dual:
                    scene_back_cams = scene_b.getTrainCameras().copy()[clip_idx]
                    viewpoint_stack_back = copy.deepcopy(scene_back_cams[1.0])

            random_camera = randint(0, len(viewpoint_stack_fore)-1)
            viewpoint_cam_fore = viewpoint_stack_fore.pop(random_camera)
            
            cur_clip_frames = sorted(list(scene_f.getFramesInClip()))
            cam_index = int(viewpoint_cam_fore.image_name) - cur_clip_frames[0]
            fid_f = viewpoint_cam_fore.fid
        
            if use_dual:
                # number of bleding = number of frames in current groups
                bled_alpha = gaussians_f.get_alpha()
                viewpoint_cam_back = viewpoint_stack_back.pop(random_camera)
                fid_b = viewpoint_cam_back.fid
                assert fid_f == fid_b, \
                    "same frame id for foreground and background"
            else:
                bled_alpha, viewpoint_cam_back, fid_b = None, None, None

            # set the multi deformation field for fore/back
            if iteration < opt.warm_up:
                if iteration == 0:
                    print('warming up before deformation field generated')
                d_xyz_f, d_rotation_f, d_scaling_f = 0.0, 0.0, 0.0
                d_xyz_b, d_rotation_b, d_scaling_b = 0.0, 0.0, 0.0
                deform_fore, deform_back = None, None
            else:
                deform_fore, d_xyz_f, d_rotation_f, d_scaling_f = process_fore_deformation(
                    deform_type, gaussians_f, deform_fore_dict, clip_idx, fid_f)
                if use_dual:
                    deform_back, d_xyz_b, d_rotation_b, d_scaling_b = process_back_deformation(
                        deform_type, gaussians_b, deform_back_dict, clip_idx, fid_b)
                else:
                    d_xyz_b, d_rotation_b, d_scaling_b = 0, 0, 0

            gt_image = []
            if edited_path is not None:
                for idx in range(len(viewpoint_cam_fore.edited_image)):
                    gt_image.append(viewpoint_cam_fore.edited_image[idx])
            else:
                gt_image.append(viewpoint_cam_fore.original_image_nomask.cuda())
            detail_gt_image = viewpoint_cam_fore.detail_image.cuda()
            
            # Rendering for foreground and background w/ deformation network
            render_pkg_f = render_deform(viewpoint_cam_fore, gaussians_f, pipe, background,
                                         d_xyz_f, d_rotation_f, d_scaling_f, edited_path=edited_path)
            image_f, viewspace_point_tensor_f, visibility_filter_f, radii_f, _ = (
                    render_pkg_f["render"], render_pkg_f["viewspace_points"],
                    render_pkg_f["visibility_filter"], render_pkg_f["radii"], render_pkg_f["depth"])
            # loss for fore and back
            loss = 0

            for gt_ in gt_image:  
                loss += loss_func_w_bilateral(opt, image_f, gt_, tfloss_weight=0.0)

            if use_dual:
                render_pkg_b = render_deform(viewpoint_cam_back, gaussians_b, pipe, background,
                            d_xyz_b, d_rotation_b, d_scaling_b, edited_path=edited_path)
                image_b, viewspace_point_tensor_b, visibility_filter_b, radii_b, _ = (
                        render_pkg_b["render"], render_pkg_b["viewspace_points"],
                        render_pkg_b["visibility_filter"], render_pkg_b["radii"], render_pkg_b["depth"])
                loss_b = 0
                for gt_ in gt_image:
                    loss_b += loss_func_w_bilateral(opt, image_b, gt_, tfloss_weight=0.0)

                # loss for blended output
                if edited_path is not None:
                    with torch.no_grad():
                        bled_alpha_frame = bled_alpha[cam_index]
                else:
                    bled_alpha_frame = bled_alpha[cam_index]
                image_fuse = (bled_alpha_frame) * image_f + (1.-bled_alpha_frame) * image_b
                
                loss_fuse = 0
                for gt_ in gt_image:
                    loss_fuse += loss_func_w_bilateral(opt, image_fuse, gt_, tfloss_weight, detail_gt_image)
                loss += loss_b + loss_fuse
            else:
                viewspace_point_tensor_b, visibility_filter_b, radii_b = None, None, None

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # save
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians and Deformation Field".format(iteration))
                    scene_f.save_clip_f(iteration, group=group_int, update_idx=update_idx)
                    if use_dual:
                        scene_b.save_clip_b(iteration, group=group_int, update_idx=update_idx)
                    if edited_path is None:
                        if deform_type == "multi":
                            for deform_clip_idx in range(len(deform_fore_dict)):
                                deform_fore_dict[deform_clip_idx].save_weightsf_group(
                                    model_path, iteration, group=group_int, clip=deform_clip_idx)
                            if use_dual:
                                # number of deform_fore_dict == number of deform_back_dict
                                for deform_clip_idx in range(len(deform_fore_dict)):
                                    deform_back_dict[deform_clip_idx].save_weightsb_group(
                                        model_path, iteration, group=group_int, clip=deform_clip_idx)
                        else:
                            deform_fore_dict.save_weightsf_group(
                                model_path, iteration, group=group_int, clip=0)
                            deform_back_dict.save_weightsb_group(
                                model_path, iteration, group=group_int, clip=0)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians_f.max_radii2D[visibility_filter_f] = torch.max(
                        gaussians_f.max_radii2D[visibility_filter_f], radii_f[visibility_filter_f])
                    gaussians_f.add_densification_stats(viewspace_point_tensor_f, visibility_filter_f)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians_f.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_f.cameras_extent, size_threshold)
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians_f.reset_opacity()

                    # back 3dgs Densification
                    if use_dual:
                        # Keep track of max radii in image-space for pruning
                        gaussians_b.max_radii2D[visibility_filter_b] = torch.max(gaussians_b.max_radii2D[visibility_filter_b], radii_b[visibility_filter_b])
                        gaussians_b.add_densification_stats(viewspace_point_tensor_b, visibility_filter_b)
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians_b.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_b.cameras_extent, size_threshold)
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians_b.reset_opacity()

                # tracking the number of generated gaussians for foreground and background
                if iteration == opt.densify_until_iter:
                    logging.info(f"number of foreground gaussians: {gaussians_f.get_xyz.shape[0]}")
                    if use_dual:
                        logging.info(f"number of background gaussians: {gaussians_b.get_xyz.shape[0]}")

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians_f.optimizer.step()
                    if iteration >= opt.warm_up and edited_path is None:
                        deform_fore.optimizer.step()
                    gaussians_f.optimizer.zero_grad(set_to_none = True)
                    if iteration >= opt.warm_up and edited_path is None:
                        deform_fore.optimizer.zero_grad()
                        deform_fore.update_learning_rate(iteration)
                    
                    if use_dual:
                        gaussians_b.optimizer.step()
                        if iteration >= opt.warm_up and edited_path is None:
                            deform_back.optimizer.step()
                        gaussians_b.optimizer.zero_grad(set_to_none = True)
                        if iteration >= opt.warm_up and edited_path is None:
                            deform_back.optimizer.zero_grad()
                            deform_back.update_learning_rate(iteration)     


                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians_f.capture(), iteration), scene_f_list[0].model_path + "/chkpnt" + str(iteration) + ".pth")
                    if use_dual:
                        torch.save((gaussians_b.capture(), iteration), scene_b_list[0].model_path + "/chkpnt" + str(iteration) + ".pth")


def create_gaussian_model(opt, dataset, use_alpha=False):
    gaussian_model = GaussianModel(dataset.sh_degree, use_alpha=use_alpha)
    return gaussian_model

def create_scene(dataset, gaussian_model, clip_path, frame_num, deform_type,
                 random_style, random_pts_num, radius, fsize,
                 group_idx, fore, use_alpha=False, edited_path=None, update_idx=0):
    scene = Scene_fore_w_rand_group(dataset, gaussian_model, source_path=clip_path,
                                    total_frame=frame_num, deform_type=deform_type, fsize=fsize,
                                    random_style=random_style, random_pts_num=random_pts_num,
                                    radius=radius, group_idx=group_idx, fore=fore, use_alpha=use_alpha,
                                    init_edit_path=edited_path, update_idx=update_idx)
    return scene

def create_deform_model(opt, edited_path=None,
    group_idx=0, clip_idx=0, fore=False, model_path=None):
    
    deform_model = DeformModel(use_hash=True)
    if edited_path is None:
        deform_model.train_setting(opt)
    else:
        # load pretrained deform model for Video-3DGS-Edit
        if fore:
            deform_model.load_weightsf_group(
                model_path, group=group_idx, clip=clip_idx)
        else:
            deform_model.load_weightsb_group(
                model_path, group=group_idx, clip=clip_idx)
            
    return deform_model


def process_fore_deformation(deform_type, gaussians, deform_dict, clip_idx, fid):
    N = gaussians.get_xyz.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)
    deform = deform_dict[clip_idx] if deform_type == "multi" else deform_dict
    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
    return deform, d_xyz, d_rotation, d_scaling

def process_back_deformation(deform_type, gaussians, deform_dict, clip_idx, fid):
    return process_fore_deformation(deform_type, gaussians, deform_dict, clip_idx, fid)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParamsForeBack(parser)
    pp = PipelineParams(parser)
    # basic setup
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 10_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--small", action='store_true', default=False)
    parser.add_argument("--mixed_precision", action='store_true', default=False)
    # setup for Video-3DGS (Reconstruction)
    parser.add_argument('--random_pts_num', type=int, default=60000,
        help="number of random sphere points, currently fixed to 60k")
    parser.add_argument('--radius', nargs="+", type=int, default=[1,3])
    parser.add_argument('--tfloss_weight', type=float, default=2.0)
    parser.add_argument("--use_dual", action='store_true', default=False)
    parser.add_argument('--iteration', type=int, default=10000)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--f_size', type=int, default=15)
    parser.add_argument('--fore_random', type=str, default="nordn")
    parser.add_argument('--deform_type', type=str, default="multi")
    parser.add_argument("--use_pre_obtained_mask", type=bool, default=True)
    parser.add_argument('--mask_prompt', type=str, default="person.dog",
        help="prompt for DEVA mask generation")
    parser.add_argument("--only_eval", action='store_true', default=False)
    # setup for Video-3DGS (Editing)
    parser.add_argument("--skip_recon", action='store_true', default=False)
    parser.add_argument("--editing_method", type=str, default=None,
        help="[None, noprogress_single, noprogress_multi, progress_single, progress_multi, progress_accu_multi, progress_accu_all_multi]")
    parser.add_argument("--initial_editor", type=int, default=1,
        help="choose the initial video editor. We currently support three video editors."
             "text2video-zero[3]: 1 / TokenFlow[4]: 2 / RAVE[5]: 3")
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--ori_text', type=str, default="")
    parser.add_argument('--cate', type=str, default="style",
        help="[style, object, back, multi]")
    parser.add_argument('--progressive_num', type=int, default=0)
    parser.add_argument('--cuda_num', type=int, default=2)
    parser.add_argument('--delete_weights', type=bool, default=True)



    args = parser.parse_args(sys.argv[1:])
    edit_parser = copy.deepcopy(parser)
    edit_args = edit_parser.parse_args(sys.argv[1:])

    # You can choose the different iteration options
    # For reconstruction, you can choose among 3k/5k/10k.
    if args.iteration == 3000:
        op = OptimizationClipMaskParams3k(parser)
        iter_size = '3k'
    elif args.iteration == 5000:
        op = OptimizationClipMaskParams5k(parser)
        iter_size = '5k'
    elif args.iteration == 10000:
        op = OptimizationClipMaskParams10k(parser)
        iter_size = '10k'
    elif args.iteration == 30000:
        op = OptimizationClipMaskParams30k(parser)
        iter_size = '30k'
    else:
        NotImplementedError("iteration should in [3k, 5k, 10k]")
        op, iter_size = None, None

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    edit_op = OptimizationClipMaskParamsEdit(edit_parser)
    edit_args = edit_parser.parse_args(sys.argv[1:])
    edit_args.save_iterations.append(edit_args.iterations)
    
    # Name output path
    s_path = args.source_path
    if len(args.radius) > 1:
        radius = 'multi'
    else:
        radius = args.radius[0]
    frame_number = len(os.listdir(s_path))

    # initilized with None for editing setup
    edited_path = None
    editor = None
    ve_model = None
    # currently, only text2vid-zero is modulelized.
    # For other editors, load them by python command
    if args.initial_editor == 1:
        from models.video_editors.text2vidzero.model import Model
        editor="vid-pix2pix"
        ve_model = Model(device="cuda", dtype=torch.float16)
    elif args.initial_editor == 2:
        editor="tokenflow"
    elif args.initial_editor == 3:
        editor="RAVE"
    else:
        editor="noedit"

    prompt = args.prompt
    if args.editing_method is not None:
        if "noprogress" in args.editing_method:
            assert args.progressive_num == 0

    '''
     - Settings to be considered (recon)
       0) number of iterations
            - 3000 / 5000 / 10000
       1) use dual branch 
            - If True, dual branch (fore3DGS w fore points / back3DGS w random sphere back points) 
            - otherwise, single branch (3DGS w {fore points and random sphere back points})
       2) group size: how many clips in each group (the more, the faster but weaker)
            - 1/2/4/-1{all clips in same group}
            - we currently use group size 1 in our main paper.
       3) deformation type: 
            - "multi": each clips in group have their own deformation field
            - "single": integrated deformation field for whole clips in group
       4) random points: the number of random sphere points and radius distance for sphere
            - number of points: currently hard coded 
            - rad_dist: 
                - single: 1/3
                - multi: {1,3} - total points above are divided equally for each spheres
       5) tf_weight: the weight of tf loss
            - 0.0 / 2.0
     - Settings to be considered (edit)
       0) initial_editor
       1) editing method
    '''
    scene_name=s_path.split('/')[-1]
    today = datetime.today()
    # formatted_date = today.strftime("%Y-%m-%d")  # For example, "2024-05-06"
    formatted_date = "vv_yt"

    args.model_path = "output/{}/vid3dgs_output/recon/dual{}_fore{}_def{}_group{}_tf{}_fsize{}_iter{}/{}_frame{}".format(
        formatted_date, args.use_dual, args.fore_random, args.deform_type, str(args.group_size),
        args.tfloss_weight, args.f_size, args.iteration, scene_name, str(frame_number))
    if args.editing_method is not None:
        args.model_path = args.model_path.replace("recon", "edit_{}/{}_{}".format(
            args.cate, editor, args.editing_method))
    os.makedirs(args.model_path, exist_ok=True)
    print("Saving all into " + args.model_path)

    result_path = os.path.join(args.model_path, 'train_edit2', 'results_clip.json')
    if os.path.exists(result_path):
        exit()

    logging.basicConfig(
        filename=os.path.join(args.model_path, 'end-to-end-time.log'),
        level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f"Training configuration: {args}")

    # 1. Generate the 3D points w/ corresponding camera poses.
    '''
        Here, we obtain the 3D points from video sequence.
        For better 3D points aqcuistion in dynamic scene where moving objects exist,
        we obtain 3D points of dynamic objects using MC-COLMAP.
    '''
    # 1-1. Path for outputs of MC-COLMAP.
    pts_folder = os.path.join(s_path+"_pts_camera_from_deva")
    fore_folder = os.path.join(pts_folder, 'mask', 'local', 'full')
    os.makedirs(fore_folder, exist_ok=True)
    subfolders_fore = [f.name for f in os.scandir(fore_folder) if f.is_dir()]
    clip_folders_fore = [folder for folder in subfolders_fore if folder.startswith("clip_")]

    # 1-2. Generate Mask (DEVA[2])
    start_time = time.time()
    logging.info(f"Generating Mask for Frg-3DGS")
    input_path = os.path.join(pts_folder, 'input')
    if not os.path.exists(input_path):
        shutil.copytree(s_path, input_path)
    saved_mask_path = os.path.join(pts_folder, 'mask')
    if not args.use_pre_obtained_mask:
        cmd = (
            f'python3 models/video_segmentor/DEVA/demo/demo_with_text.py '
            f'--chunk_size 4 --img_path {args.source_path} --amp --temporal_setting semionline '
            f'--size 480 --output {args.source_path}_deva --prompt {args.mask_prompt}'
        )
        mask_path = args.source_path+'_deva'
        mask_path = os.path.join(mask_path, 'Annotations')
        gen_mask(mask_path, saved_mask_path)
        mask_end_time = time.time()
        total_mask_time = mask_end_time - start_time
        logging.info(f"Mask generated in {total_mask_time:.2f} seconds")

    # 1-3. MC-COLMAP
    logging.info(f"Generating COLMAP for Frg-3DGS")
    if len(clip_folders_fore) == 0:
        run_colmap_mask_clip(saved_mask_path)
    colmap_end_time = time.time()
    total_colmap_time = colmap_end_time - start_time
    logging.info(f"COLMAP generated in {total_colmap_time:.2f} seconds")
    
    # 2. training options
    args.lsource_path = fore_folder
    clip_num = len([folder for folder in subfolders_fore if folder.startswith("clip_")])
    if args.group_size == -1:
        args.group_size = clip_num

    # 2-1. options for bilateral loss
    logging.info(f"Generating bilateral filter images")
    if args.tfloss_weight > 0.0:
        generate_bi_image(pts_folder, args.f_size)
    bi_end_time = time.time()
    total_bi_time = bi_end_time - start_time
    logging.info(f"Bilateral images generated in {total_bi_time:.2f} seconds")

    if not args.only_eval:
        # Optimizing 3DGS
        # Video-3DGS-Recon: Rendering and Eval (PSNR/SSIM) /
        # Video-3DGS-Edit: Editing and Eval (WarpSSIMxCLIP=Qedit)
        for update_idx in range(args.progressive_num+1):
            v_op = op if update_idx == 0 else edit_op
            v_args = args if update_idx == 0 else edit_args
            # you can skip reconstruction and move onto Video-3DGS-Edit
            if edited_path is not None or not args.skip_recon:
                torch.autograd.set_detect_anomaly(args.detect_anomaly)    
                logging.info(f"Training start")
                training(lp.extract(args), v_op.extract(v_args), pp.extract(args),
                        args.save_iterations, args.checkpoint_iterations,
                        args.random_pts_num, args.radius, args.f_size, args.fore_random,
                        args.tfloss_weight, args.group_size, args.deform_type, frame_number,
                        args.use_dual, args.model_path, logging, edited_path, update_idx)
                train_end_time = time.time()
                total_train_time = train_end_time - start_time
                logging.info(f"Training finished in {total_train_time:.2f} seconds")
            
                # 3. Rendering
                load_after_diff = edited_path is not None
                logging.info(f"Rendering start")
                render_sets_dual(lp.extract(args), -1, pp.extract(args),
                    group_size=args.group_size, deform_type=args.deform_type,
                    load_after_diff=load_after_diff, init_edit_path=edited_path,
                    update_idx=update_idx)
                render_end_time = time.time()
                total_render_time = render_end_time - start_time
                logging.info(f"Rendering finished in {total_render_time:.2f} seconds")

                # 4. Metric (PSNR / SSIM / WarpSSIM)
                # PSNR / SSIM / Warp SSIM for video recon
                # Warp SSIM / CLIP Score / Pick Score for video editing
                if edited_path is not None:
                    evaluate_edit([args.model_path], args, edited_path, update_idx)
                else:
                    evaluate([args.model_path])
                
                # 5. Generate the GIF for visualization
                create_gif_from_frames(args.model_path, args.iteration, edited_path, update_idx=update_idx)
                if edited_path is not None:
                    clipscore(prompt, args.model_path, edited_path, update_idx)

            # 6 (option). Run video-3dgs-edit if you want to proceed to editing.
            #       skip editing by setting "editing_method" as None. 
            if args.editing_method is not None:
                editing_folder = "output/{}/initial_edit/{}".format(formatted_date, str(args.editing_method))
                joined_prompt = prompt.replace(" ", "_")
                edited_path = os.path.join(editing_folder, editor, s_path.split('/')[-1], joined_prompt)

                if update_idx == 0:
                    # If single update or first update for progressive update,
                    # load original video (we currently use original images for fair comparison) for editing 
                    prev_vid_path = "train_edit0/ours_{}".format(args.iteration)
                    prev_vid_path = os.path.join(args.model_path,
                        prev_vid_path, "gt")
                else:
                    prev_vid_path = "train_edit{}/ours_1000".format(update_idx)
                    prev_vid_path = os.path.join(args.model_path,
                        prev_vid_path, "refined_edited")
                    if args.editing_method == "progress_accu_multi":
                        prv_edited_path = os.path.join(
                            edited_path, "vid_output_prv")
                        os.makedirs(prv_edited_path, exist_ok=True)
                        for filename in os.listdir(prev_vid_path):
                            file_path = os.path.join(prev_vid_path, filename)
                            if os.path.isfile(file_path):
                                shutil.copy(file_path, prv_edited_path)
                    elif args.editing_method == "progress_accu_all_multi":
                        for fld_idx, foldername in enumerate(sorted(os.listdir(edited_path))):
                            prv_edited_path = os.path.join(
                                edited_path, "vid_output_prv_{}".format(fld_idx))
                            os.makedirs(prv_edited_path, exist_ok=True)
                            prev_vid_path_sub = prev_vid_path.replace("refined_edited", foldername)
                            for filename in os.listdir(prev_vid_path_sub):
                                file_path = os.path.join(prev_vid_path_sub, filename)
                                if os.path.isfile(file_path):
                                    shutil.copy(file_path, prv_edited_path)

                # Run video editor (here includes "vid-pix2pix, tokenflow, RAVE")
                ig_strategy = args.editing_method.split('_')[-1]
                if editor == "vid-pix2pix":
                    ve_model.process_pix2pix(
                        prev_vid_path, prompt, edited_path,
                        update_num=args.progressive_num+1, ig_strategy=ig_strategy)
                else:
                    edit_cmd_run(
                        editor, args.cuda_num, prev_vid=prev_vid_path,
                        text=args.prompt, edited_path=edited_path,
                        progressive=args.progressive_num+1,
                        ori_text=args.ori_text, ig_strategy=ig_strategy)

                # Final update
                if update_idx == args.progressive_num:
                    update_idx += 1
                    load_after_diff = edited_path is not None
                    torch.autograd.set_detect_anomaly(args.detect_anomaly)
                    training(lp.extract(args), edit_op.extract(edit_args), pp.extract(args),
                            args.save_iterations, args.checkpoint_iterations,
                            args.random_pts_num, args.radius, args.f_size, args.fore_random,
                            args.tfloss_weight, args.group_size, args.deform_type, frame_number,
                            args.use_dual, args.model_path, logging, edited_path, update_idx)
                    render_sets_dual(lp.extract(args), -1, pp.extract(args),
                        group_size=args.group_size, deform_type=args.deform_type,
                        load_after_diff=load_after_diff, init_edit_path=edited_path,
                        update_idx=update_idx)
                    evaluate_edit([args.model_path], args, edited_path, update_idx)
                    create_gif_from_frames(args.model_path, args.iteration, edited_path, update_idx=update_idx)
                    clipscore(prompt, args.model_path, edited_path, update_idx)
            else:
                # if not progressive editing, only run once for rendering
                break
        
        # delete heavy files
        if args.delete_weights:
            for source_folder in sorted(os.listdir(args.model_path)):
                if "point_cloud" in source_folder or "deform" in source_folder:
                    shutil.rmtree(os.path.join(args.model_path, source_folder))

        print("\nTraining and Rendering complete.")
            
    else:
        # Recon
        render_sets_dual(lp.extract(args), -1, pp.extract(args),
            group_size=args.group_size, deform_type=args.deform_type,
            load_after_diff=False, init_edit_path=edited_path,
            update_idx=0)
            
        # TODO) add evaluation pipeline for only_eval
        print("\nEvaluation complete.")
    