
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
            d_xyz_f, d_rotation_f, d_scaling_f,
            render_path, gts_path, initial_edit_path, idx, load_after_diff):


    rendering_f = render_deform(view_f, gaussians_f, pipeline, background, d_xyz_f, d_rotation_f, d_scaling_f)["render"]

    rendering_fusion = rendering_f

    rendered_file_name = '{0:05d}'.format(idx) + ".png"


    gt = view_f.original_image_nomask[0:3, :, :]
    if load_after_diff:
        for idx, init_edit in enumerate(initial_edit_path):
            initial = view_f.edited_image[idx][0:3, :, :]
            torchvision.utils.save_image(initial, os.path.join(init_edit, rendered_file_name))

    
    torchvision.utils.save_image(gt, os.path.join(gts_path, rendered_file_name))
    torchvision.utils.save_image(rendering_fusion, os.path.join(render_path, rendered_file_name))


def render_func(model_path, iteration, views_f_list,
                gaussians_f_list,
                pipeline, background, deform_type,
                load_after_diff=False, init_edit_path=None, update_idx=0):

    render_path = os.path.join(
        model_path, "train_edit{}".format(str(update_idx)), "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(
        model_path, "train_edit{}".format(str(update_idx)), "ours_{}".format(iteration), "gt")

    if load_after_diff:
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

        cur_clip_frames = sorted(list(views_f.getFramesInClip()))

        for b_idx, views_f in enumerate(views_f_group):
            for view_idx, view_f in enumerate(tqdm(views_f[1.0], desc="Rendering progress")):
                
                cam_index = int(view_f.image_name)
                cam_offset_index = cam_index - cur_clip_frames[0]
                score_fore_int = b_idx
                fid = view_f.fid
                    
                d_xyz_f, d_rotation_f, d_scaling_f = 0, 0, 0

                fuse_param = gaussians_f_list[idx].get_alpha()

                merge_and_generate(view_f, gaussians_f_list[idx], pipeline, background,
                    d_xyz_f, d_rotation_f, d_scaling_f, 
                    render_path, gts_path, initial_edit_path, cam_index, load_after_diff)


def render_sets_single(dataset : ModelParamsForeBack, iteration : int,
                pipeline : PipelineParams, group_size : int = 1,
                deform_type : str = "multi",
                load_after_diff : bool = False,
                init_edit_path : str = None, update_idx : int = 0):

    with torch.no_grad():
        
        sub_source_path = sorted(glob.glob(os.path.join(dataset.lsource_path, '*')))
        clip_dict, _= check_n_groups_clips(group_size, sub_source_path)

        gaussians_f_list, scene_f_list = dict(), dict()

        for group_idx, clip_path in clip_dict.items():
            gaussians_f = GaussianModel(dataset.sh_degree)
            scene_fore = Scene_fore_w_rand_group(dataset, gaussians_f, load_iteration=iteration, shuffle=False,
                    source_path=clip_path, deform_type=deform_type, load_after_diff=load_after_diff,
                    group_idx=group_idx, fore=True, use_alpha=False,
                    init_edit_path=init_edit_path, update_idx=update_idx)

            
            gaussians_f_list[group_idx] = gaussians_f
            scene_f_list[group_idx] = scene_fore
        loaded_iter = scene_f_list[0].loaded_iter


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
        render_func(
            dataset.model_path, loaded_iter, scene_f_list,
            gaussians_f_list, pipeline, background, deform_type,
            load_after_diff, init_edit_path, update_idx)

