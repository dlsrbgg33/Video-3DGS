import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from models.video_editors.text2vidzero.annotator.util import resize_image, HWC3
from models.video_editors.text2vidzero.annotator.canny import CannyDetector
from models.video_editors.text2vidzero.annotator.openpose import OpenposeDetector
from models.video_editors.text2vidzero.annotator.midas import MidasDetector
import decord

import glob



def prepare_vidframes(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):


    video = load_frames_from_video_folder(video_path)
    _, h, w, _ = video.shape
    
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def load_frames_from_video_folder(folder_path):
    """
    Load all frames from a video folder and convert them into a numpy array of shape (T, H, W, 3),
    where T is the number of frames, H is the height, W is the width, and 3 represents the color channels.

    Parameters:
    folder_path (str): Path to the folder containing the video frames.

    Returns:
    numpy.ndarray: An array of frames loaded from the folder.
    """
    frames = []
    # Sort the files to maintain the frame order
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                frames.append(img)

    # Convert list of frames to numpy array
    frame_array = np.array(frames)

    return frame_array


def create_gif_from_frames(frames_folder, duration=100):
    
    frame_files = sorted(glob.glob(
        os.path.join(frames_folder, '*.jpg')))  # Update the extension if your frames have a different format
    if len(frame_files) == 0:
        frame_files = sorted(glob.glob(
            os.path.join(frames_folder, '*.png')))  # Update the extension if your frames have a different format

    images = []
    gif_path = os.path.join(frames_folder, 'init_edited.gif')
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)


def create_gif(frames, rescale=False,
               edited_path=None, original_res=None,
               video_path=None, update_idx=None):

    image_save_path = os.path.join(
        edited_path, "vid_output_{}".format(str(update_idx)))

    os.makedirs(image_save_path, exist_ok=True)
    outputs = []

    for i, video_source in enumerate(sorted(os.listdir(video_path))):
        file_postpix = video_source.split('/')[-1]
        x = frames[i]
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)

        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = cv2.resize(x, original_res).astype(np.uint8)

        outputs.append(x)
        imageio.imsave(os.path.join(image_save_path, file_postpix), x)

    create_gif_from_frames(image_save_path)
    return image_save_path



class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
