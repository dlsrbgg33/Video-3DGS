import os
import glob
from PIL import Image


def create_gif_from_frames(frames_folder, iteration, edited_path,
                           duration=100, update_idx=0):
    
    if edited_path is not None:
        iteration = 1000

    saved_path = os.path.join(
            frames_folder, 'train_edit{}'.format(update_idx), 'ours_{}'.format(iteration))

    # Update the extension if your frames have a different format
    gt_files_list = []
    gt_gif_path_list = []
    if edited_path is None:
        frame_files = sorted(glob.glob(os.path.join(saved_path, 'renders', '*.png')))
        gt_files_list.append(sorted(glob.glob(os.path.join(saved_path, 'gt', '*.png'))))
        render_gif_path = os.path.join(saved_path, 'renders.gif')
        gt_gif_path_list.append(os.path.join(saved_path, 'gt.gif'))
    else:
        frame_files = sorted(glob.glob(os.path.join(saved_path, 'refined_edited', '*.png')))
        for edited_p in sorted(os.listdir(edited_path)):
            gt_files_list.append(sorted(glob.glob(os.path.join(saved_path, edited_p, '*.png'))))
            gt_gif_path_list.append(os.path.join(saved_path, 'gt.gif'))
        render_gif_path = os.path.join(saved_path, 'refined_edited.gif')
    
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    images[0].save(render_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

    # save gt (or initial edited) output
    for idx, gt_files in enumerate(gt_files_list):
        gt_images = []
        for frame_file in gt_files:
            img = Image.open(frame_file)
            gt_images.append(img)
        gt_images[0].save(
            gt_gif_path_list[idx], save_all=True, append_images=gt_images[1:], optimize=False, duration=duration, loop=0)


def create_gif_from_frames_inter(frames_folder, iteration, edited_path,
                           duration=100, update_idx=0):
    
    if edited_path is not None:
        iteration = 1000

    saved_path = os.path.join(
            frames_folder, 'train_inter_edit{}'.format(update_idx), 'ours_{}'.format(iteration))

    # Update the extension if your frames have a different format
    gt_files_list = []
    gt_gif_path_list = []
    if edited_path is None:
        frame_files = sorted(glob.glob(os.path.join(saved_path, 'renders', '*.png')))
        render_gif_path = os.path.join(saved_path, 'renders.gif')
    else:
        frame_files = sorted(glob.glob(os.path.join(saved_path, 'refined_edited', '*.png')))
        render_gif_path = os.path.join(saved_path, 'refined_edited.gif')
    
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    images[0].save(render_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
