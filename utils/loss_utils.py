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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import os
import glob
import cv2
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_nomean(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def generate_bi_image(source_path, f_size=15):
    source_video = sorted(glob.glob(os.path.join(
        source_path, 'input', '*.jpg'
    )))
    output_video = os.path.join(source_path, 'bi_filter_images_{}'.format(str(f_size)))
    output_video_sub = os.path.join(source_path, 'sub_bi_filter_images_{}'.format(str(f_size)))
    os.makedirs(output_video, exist_ok=True)
    os.makedirs(output_video_sub, exist_ok=True)
    
    for source_image in source_video:
        img = cv2.imread(source_image) 
        bilateral = cv2.bilateralFilter(img, f_size, 75, 75) 
        sub = np.abs(np.subtract(np.asarray(img, np.int16), np.asarray(bilateral, np.int16)))
        sub_mean = np.mean(sub, axis=-1)
        sub_mean = (sub_mean - sub_mean.min()) / (sub_mean.max() - sub_mean.min())
        sub_bilateral = np.asarray(255*sub_mean, np.uint8)
        output_image_path = os.path.join(output_video, source_image.split('/')[-1])
        sub_output_image_path = os.path.join(
            output_video_sub, source_image.split('/')[-1])
        cv2.imwrite(output_image_path, bilateral) 
        cv2.imwrite(sub_output_image_path, sub_bilateral) 


def loss_func_w_bilateral(opt, image_l, gt_image, tfloss_weight, detail_gt_image=None):
    if tfloss_weight > 0.0:
        Ll1_f = l1_loss_nomean(image_l, gt_image)
        W = detail_gt_image
        Ll1_f = (1+tfloss_weight*W) * Ll1_f
        Ll1_f = torch.mean(Ll1_f)
        loss = (1.0 - opt.lambda_dssim) * Ll1_f + opt.lambda_dssim * (1.0 - ssim(image_l, gt_image))
    else:
        Ll1_f = l1_loss(image_l, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1_f + opt.lambda_dssim * (1.0 - ssim(image_l, gt_image))
    return loss