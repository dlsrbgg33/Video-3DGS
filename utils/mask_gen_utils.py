import os
import cv2
import glob
import numpy as np
from PIL import Image
import shutil



def per_mask_gen(orig_mask_files, saved_path, fore_full=False, mask_id=0, rescale=False, image_path=None):
    mask_vis_path = os.path.join(saved_path, 'vis')
    os.makedirs(mask_vis_path, exist_ok=True)

    mask_num = 0
    get_full_image = False
    for mask_path in orig_mask_files:
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        fore_mask = mask != 0
        mask_num += int(fore_mask.sum())
    
    if (mask_num / len(orig_mask_files)) < (mask.shape[0]*mask.shape[1]*0.1):
        get_full_image = True

    for mask_path in orig_mask_files:
        mask_postfix = mask_path.split('/')[-1].split('.')[0]
        mask_postfix += '.jpg.png'
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        if fore_full:
            mask = mask != 0
            if get_full_image:
                mask = np.ones_like(mask)
        else:
            mask = mask == mask_id

        mask = mask.astype(np.uint8)
        mask_saved_name = os.path.join(saved_path, mask_postfix)
        cv2.imwrite(mask_saved_name, mask)

        mask_vis_saved_name = os.path.join(mask_vis_path, mask_postfix)
        cv2.imwrite(mask_vis_saved_name, mask*255)


def mask_generation(image_path, origin_mask_path, saved_path, rescale=False, local=True):
    mask_file_path = glob.glob(origin_mask_path+'/*')
    mask = np.array(Image.open(mask_file_path[0])).astype(np.uint8)
    # first, need to discover how many instances are in that video
    # use first frame to determine the number of instances
    if local:
        num_instances = len(np.unique(mask)) - 1 # -1 to eliminate background
        if num_instances == 0:
            full_path = os.path.join(saved_path, 'full')
            os.makedirs(full_path, exist_ok=True)
            full_mask_path = os.path.join(full_path, 'mask')
            full_image_path = os.path.join(full_path, 'input')
            os.makedirs(full_mask_path, exist_ok=True)
            if not os.path.exists(full_image_path):
                shutil.copytree(image_path, full_image_path)
            per_mask_gen(mask_file_path, full_mask_path, fore_full=True)   
        else: 
            for i in range(num_instances):
                if i == 0:
                    full_path = os.path.join(saved_path, 'full')
                    os.makedirs(full_path, exist_ok=True)
                    full_mask_path = os.path.join(full_path, 'mask')
                    full_image_path = os.path.join(full_path, 'input')
                    os.makedirs(full_mask_path, exist_ok=True)
                    if not os.path.exists(full_image_path):
                        shutil.copytree(image_path, full_image_path)
                    per_mask_gen(mask_file_path, full_mask_path, fore_full=True)     
                else:
                    inst_path = os.path.join(saved_path, 'inst_{}'.format(str(i)))
                    os.makedirs(inst_path, exist_ok=True)
                    inst_mask_path = os.path.join(inst_path, 'mask')
                    inst_image_path = os.path.join(inst_path, 'input')
                    os.makedirs(inst_mask_path, exist_ok=True)
                    if not os.path.exists(inst_image_path):
                        shutil.copytree(image_path, inst_image_path)
                    per_mask_gen(mask_file_path, inst_mask_path, mask_id=i+1, rescale=rescale, image_path=inst_image_path)
    else:
        back_mask_path = os.path.join(saved_path, 'mask')
        back_image_path = os.path.join(saved_path, 'input')
        os.makedirs(back_mask_path, exist_ok=True)
        if not os.path.exists(back_image_path):
            shutil.copytree(image_path, back_image_path)
        per_mask_gen(mask_file_path, back_mask_path, mask_id=0)


def gen_mask(mask_path, new_mask_path, rescale=False):
    image_path = new_mask_path.replace('mask', 'input')
    local_path = os.path.join(new_mask_path, 'local')
    global_path = os.path.join(new_mask_path, 'global')
    os.makedirs(local_path, exist_ok=True)
    
    mask_generation(image_path, mask_path, local_path, local=True)
    mask_generation(image_path, mask_path, global_path, local=False)
