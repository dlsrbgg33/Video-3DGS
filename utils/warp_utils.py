
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



def load_image(imfile):
    long_dim = 768
    img = np.array(Image.open(imfile)).astype(np.uint8)

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round(float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim
        input_h = int(round(float(input_w) / img.shape[1] * img.shape[0]))

    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )
    return fwd_mask, bwd_mask


def SaveWarpingImage(warped_dir, next_gt_dir,
                     original_dir, gt_dir, args):
    
    model = torch.nn.DataParallel(RAFT(args))
    # hard-coded to use "raft-things" model for optical flow
    model.load_state_dict(torch.load(
        "models/optical_flow/RAFT/models/raft-things.pth"))
    model = model.module
    model.cuda()
    model.eval()

    original_file_list = sorted(os.listdir(original_dir))
    gt_file_list = sorted(os.listdir(gt_dir))
    img_train = cv2.imread(os.path.join(original_dir, original_file_list[0]))
    img_h, img_w = img_train.shape[0], img_train.shape[1]

    for idx, fname in enumerate(original_file_list):

        original_frame = load_image(original_dir / fname)
        gt = load_image(gt_dir / fname.replace('jpg', 'png'))

        if idx != len(original_file_list) -1:
            next_idx = idx+1
        else:
            continue

        original_next_frame = load_image(original_dir / original_file_list[next_idx])
        gt_next = load_image(gt_dir / gt_file_list[next_idx])
        
        padder = InputPadder(gt.shape)
        gt, gt_next = padder.pad(gt, gt_next)
        original_frame, original_next_frame = padder.pad(
                        original_frame, original_next_frame)

        _, flow_fwd = model(gt, gt_next, iters=20, test_mode=True)
        _, flow_bwd = model(gt_next, gt, iters=20, test_mode=True)
        flow_fwd = padder.unpad(flow_fwd[0]).detach().cpu().numpy().transpose(1, 2, 0)
        flow_bwd = padder.unpad(flow_bwd[0]).detach().cpu().numpy().transpose(1, 2, 0)

        original_frame = padder.unpad(original_frame[0]).detach().cpu().numpy().transpose(1, 2, 0)
        original_next_frame = padder.unpad(original_next_frame[0]).detach().cpu().numpy().transpose(1, 2, 0)

        flow_fwd = resize_flow(flow_fwd, img_h, img_w)
        flow_bwd = resize_flow(flow_bwd, img_h, img_w)

        original_frame = resize_flow(original_frame, img_h, img_w)
        original_next_frame = resize_flow(original_next_frame, img_h, img_w)

        _, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

        warped_frame = warp_flow(original_frame, flow_bwd)
        warped_zero = np.zeros_like(warped_frame)

        # warping the mask
        mask_bwd = np.expand_dims(mask_bwd, axis=-1)
        warped_frame = np.where(mask_bwd, warped_frame, warped_zero)
        original_next_frame = np.where(mask_bwd, original_next_frame, warped_zero)

        warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
        original_next_frame = cv2.cvtColor(original_next_frame, cv2.COLOR_BGR2RGB)

        # saved warped / warped_gt
        cv2.imwrite(os.path.join(warped_dir, original_file_list[next_idx]), warped_frame)
        cv2.imwrite(os.path.join(next_gt_dir, original_file_list[next_idx]), original_next_frame)
