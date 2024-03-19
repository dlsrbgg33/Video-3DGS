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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

# using raft_large to measure warping-error
from torchvision.models.optical_flow import raft_large
import numpy as np
import cv2

from models.optical_flow.RAFT.raft import RAFT
from models.optical_flow.RAFT.utils import flow_viz
from models.optical_flow.RAFT.utils.utils import InputPadder

from utils.warp_utils import SaveWarpingImage



def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        # gt = Image.open(gt_dir / fname.replace('png', 'jpg'))
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names



def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    ### SSIM and PSNR
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "train_edit0"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                ssims = []
                psnrs = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}})

            with open(scene_dir + "/recon_results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/recon_per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


def evaluate_videorecon(model_paths, args, edited_path=None, update_idx=0):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")


    ### WarpSSIM
    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "train_edit{}".format(str(update_idx))

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            rendered_dir = method_dir / "codef"

            warped_rendered_dir = os.path.join(method_dir, "warped_rendered")
            next_rendered_dir = os.path.join(method_dir, "next_rendered")
            os.makedirs(warped_rendered_dir, exist_ok=True)
            os.makedirs(next_rendered_dir, exist_ok=True)

            SaveWarpingImage(
                warped_rendered_dir, next_rendered_dir, rendered_dir, gt_dir, args)

            renders, gts, image_names = readImages(
                        method_dir / "warped_rendered",
                        method_dir / "next_rendered")
            ssims_init = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress on VideoRecon"):
                ssims_init.append(ssim(renders[idx], gts[idx]))
            ssims_mean = torch.tensor(ssims_init).mean()


            print("  WarpSSIM (Video Recon): {:>12.7f}".format(ssims_mean, ".5"))
            print("")

            full_dict[scene_dir][method].update({"WarpSSIM (Video Recon)": torch.tensor(ssims_mean).mean().item()})

        with open(scene_dir + "/results_codef.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
 

def evaluate_edit(model_paths, args, edited_path=None, update_idx=0):

    full_dict = {}
    print("")


    ### WarpSSIM
    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "train_edit{}".format(str(update_idx))

        for method in os.listdir(test_dir):
            # currently, hard-coded for refining iteration
            if "ours" not in method:
                continue

            print("Method:", method)

            full_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"

            # WarpSSIM for refined
            refined_dir = method_dir / "refined_edited"
            warped_refined_dir = os.path.join(method_dir, "warped_refined")
            next_refined_gt_dir = os.path.join(method_dir, "next_refined")
            os.makedirs(warped_refined_dir, exist_ok=True)
            os.makedirs(next_refined_gt_dir, exist_ok=True)
            SaveWarpingImage(
                warped_refined_dir, next_refined_gt_dir,
                refined_dir, gt_dir, args)
            renders, gts, image_names = readImages(
                        method_dir / "warped_refined",
                        method_dir / "next_refined")
            ssims_refined = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress on Refined"):
                ssims_refined.append(ssim(renders[idx], gts[idx]))
            ssims_mean_refined = torch.tensor(ssims_refined).mean()

            full_dict[scene_dir][method].update({"SSIM REFINED": ssims_mean_refined.item()})

            # initial(s)
            for init_pth in sorted(os.listdir(edited_path)):
                if "vid_output" not in init_pth:
                    continue
                init_dir = method_dir / init_pth
                warped_init_dir = os.path.join(method_dir, "warped_{}".format(init_pth))
                next_init_gt_dir = os.path.join(method_dir, "next_{}".format(init_pth))
                os.makedirs(warped_init_dir, exist_ok=True)
                os.makedirs(next_init_gt_dir, exist_ok=True)

                SaveWarpingImage(
                    warped_init_dir, next_init_gt_dir,
                    init_dir, gt_dir, args)

                renders, gts, image_names = readImages(
                            method_dir / "warped_{}".format(init_pth),
                            method_dir / "next_{}".format(init_pth))
                ssims_init = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress on Init"):
                    ssims_init.append(ssim(renders[idx], gts[idx]))
                ssims_mean_init = torch.tensor(ssims_init).mean()
                full_dict[scene_dir][method].update(
                    {"SSIM {}".format(init_pth): ssims_mean_init.item()})

                print("  SSIM {}: {:>12.7f}".format(init_pth, ssims_mean_init, ".5"))

            print("  SSIM REFINED: {:>12.7f}".format(ssims_mean_refined, ".5"))
            print("")

        with open(scene_dir + "/train_edit{}/results_warpssim.json".format(str(update_idx)), 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
            

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])


    args = parser.parse_args()
    evaluate(args.model_paths)
