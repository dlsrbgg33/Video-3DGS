
import torch
from scene import Scene_fore_w_rand_group, GaussianModel, DeformModel

import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_deform
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParamsForeBack, PipelineParams, get_combined_args, OptimizationClipMaskParams
from gaussian_renderer import GaussianModel
import glob
from utils.group_clips import check_n_groups_clips


def merge_and_generate(view_f, gaussians_f, pipeline, background,
            d_xyz_f, d_rotation_f, d_scaling_f, view_b, gaussians_b,
            d_xyz_b, d_rotation_b, d_scaling_b,
            render_path, gts_path, initial_edit_path, idx, bled_alpha_frame, load_after_diff):


    rendering_f = render_deform(view_f, gaussians_f, pipeline, background, d_xyz_f, d_rotation_f, d_scaling_f)["render"]
    rendering_b = render_deform(view_b, gaussians_b, pipeline, background, d_xyz_b, d_rotation_b, d_scaling_b)["render"]

    rendering_fusion = (bled_alpha_frame)*rendering_f + (1.-bled_alpha_frame)*rendering_b
    rendered_file_name = '{0:05d}'.format(idx) + ".png"

    gt = view_f.original_image_nomask[0:3, :, :]
    if load_after_diff:
        for idx, init_edit in enumerate(initial_edit_path):
            initial = view_f.edited_image[idx][0:3, :, :]
            torchvision.utils.save_image(initial, os.path.join(init_edit, rendered_file_name))

    torchvision.utils.save_image(gt, os.path.join(gts_path, rendered_file_name))
    torchvision.utils.save_image(rendering_fusion, os.path.join(render_path, rendered_file_name))


def render_func(model_path, iteration, views_f_list,
                gaussians_f_list, deform_fore_dict, views_b_list,
                gaussians_b_list, deform_back_dict, pipeline, background, deform_type,
                load_after_diff=False, init_edit_path=None, update_idx=0):

    # train_edit0 contains reconstructed video (no editing results are included)
    render_path = os.path.join(
        model_path, "train_edit{}".format(str(update_idx)), "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(
        model_path, "train_edit{}".format(str(update_idx)), "ours_{}".format(iteration), "gt")

    if load_after_diff:
        # load initial edited frames and refined ones using Video-3DGS
        initial_edit_path = []
        render_path = render_path.replace('renders', 'refined_edited')
        for init_edit_pt in sorted(os.listdir(init_edit_path)):
            initial_edit = render_path.replace('refined_edited', init_edit_pt)
            makedirs(initial_edit, exist_ok=True)
            initial_edit_path.append(initial_edit)
    else:
        initial_edit_path = None

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, views_f in views_f_list.items():
        views_f_group = views_f.getTrainCameras().copy()
        views_b_group = views_b_list[idx].getTrainCameras().copy()

        deform_fore_list = deform_fore_dict[idx]
        deform_back_list = deform_back_dict[idx]
    
        cur_clip_frames = sorted(list(views_f.getFramesInClip()))

        for b_idx, views_f in enumerate(views_f_group):
            views_b = views_b_group[b_idx]
            for view_idx, view_f in enumerate(tqdm(views_f[1.0], desc="Rendering progress")):
                
                cam_index = int(view_f.image_name)
                cam_offset_index = cam_index - cur_clip_frames[0]
                score_fore_int = b_idx
                view_b = views_b[1.0][view_idx]
                fid = view_f.fid
                time_inputf = fid.unsqueeze(0).expand(gaussians_f_list[idx].get_xyz.shape[0], -1)
                # bid = fid
                time_inputb = fid.unsqueeze(0).expand(gaussians_b_list[idx].get_xyz.shape[0], -1)
                
                if deform_type == "multi":
                    deform_fore = deform_fore_list[score_fore_int]
                    deform_back = deform_back_list[score_fore_int]
                else:
                    deform_fore = deform_fore_list
                    deform_back = deform_back_list
                    
                d_xyz_f, d_rotation_f, d_scaling_f = deform_fore.step(
                        gaussians_f_list[idx].get_xyz.detach(), time_inputf)
                d_xyz_b, d_rotation_b, d_scaling_b = deform_back.step(
                        gaussians_b_list[idx].get_xyz.detach(), time_inputb)

                fuse_param = gaussians_f_list[idx].get_alpha()
            
                bled_alpha_frame = fuse_param[cam_offset_index]

                merge_and_generate(view_f, gaussians_f_list[idx], pipeline, background,
                    d_xyz_f, d_rotation_f, d_scaling_f, view_b,
                    gaussians_b_list[idx],
                    d_xyz_b, d_rotation_b, d_scaling_b,
                    render_path, gts_path, initial_edit_path, cam_index, bled_alpha_frame, load_after_diff)


def render_sets_dual(dataset : ModelParamsForeBack, iteration : int,
                pipeline : PipelineParams, group_size : int = 1,
                deform_type : str = "multi",
                load_after_diff : bool = False,
                init_edit_path : str = None, update_idx : int = 0):

    with torch.no_grad():
        
        sub_source_path = sorted(glob.glob(os.path.join(dataset.lsource_path, '*')))
        clip_dict, _= check_n_groups_clips(group_size, sub_source_path)

        gaussians_f_list, scene_f_list = dict(), dict()
        gaussians_b_list, scene_b_list = dict(), dict()

        for group_idx, clip_path in clip_dict.items():
            gaussians_f = GaussianModel(dataset.sh_degree)
            scene_fore = Scene_fore_w_rand_group(dataset, gaussians_f, load_iteration=iteration, shuffle=False,
                    source_path=clip_path, deform_type=deform_type, load_after_diff=load_after_diff,
                    group_idx=group_idx, fore=True, use_alpha=True,
                    init_edit_path=init_edit_path, update_idx=update_idx)
            gaussians_b = GaussianModel(dataset.sh_degree)
            scene_back = Scene_fore_w_rand_group(dataset, gaussians_b, load_iteration=iteration, shuffle=False,
                    source_path=clip_path, deform_type=deform_type, load_after_diff=load_after_diff,
                    group_idx=group_idx, fore=False,
                    init_edit_path=init_edit_path, update_idx=update_idx)
            
            gaussians_f_list[group_idx] = gaussians_f
            scene_f_list[group_idx] = scene_fore
            gaussians_b_list[group_idx] = gaussians_b
            scene_b_list[group_idx] = scene_back

        loaded_iter = scene_f_list[0].loaded_iter

        deform_fore_all_dict = {}
        for group_idx in range(len(scene_f_list)):
            deform_fore_dict = {}
            if deform_type == "multi":
                for clip_idx in range(len(scene_f_list[group_idx].getTrainCameras())):
                    deform_fore = DeformModel(use_hash=True)
                    deform_fore.load_weightsf_group(
                        dataset.model_path, iteration=iteration, group=group_idx, clip=clip_idx)
                    deform_fore_dict[clip_idx] = deform_fore
                deform_fore_all_dict[group_idx] = deform_fore_dict
            else:
                deform_fore = DeformModel(use_hash=True)
                deform_fore.load_weightsf_group(dataset.model_path, iteration=iteration, group=group_idx, clip=0)
                deform_fore_all_dict[group_idx] = deform_fore

        deform_back_all_dict = {}
        for group_idx in range(len(scene_b_list)):
            deform_back_dict = {}
            if deform_type == "multi":
                for clip_idx in range(len(scene_b_list[group_idx].getTrainCameras())):
                    deform_back = DeformModel(use_hash=True)
                    deform_back.load_weightsb_group(dataset.model_path, iteration=iteration, group=group_idx, clip=clip_idx)
                    deform_back_dict[clip_idx] = deform_back
                deform_back_all_dict[group_idx] = deform_back_dict
            else:
                deform_back = DeformModel(use_hash=True)
                deform_back.load_weightsb_group(dataset.model_path, iteration=iteration, group=group_idx, clip=0)
                deform_back_all_dict[group_idx] = deform_back


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
        render_func(
            dataset.model_path, loaded_iter, scene_f_list,
            gaussians_f_list, deform_fore_all_dict, scene_b_list,
            gaussians_b_list, deform_back_all_dict, pipeline, background, deform_type,
            load_after_diff, init_edit_path, update_idx)

