from enum import Enum
import gc
import numpy as np
import tomesd
import torch

from models.video_editors.text2vidzero.diffusers.src.diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionInstructPix2PixProgPipeline, StableDiffusionControlNetPipeline
# from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler
from models.video_editors.text2vidzero.text_to_video_pipeline import TextToVideoPipeline

import models.video_editors.text2vidzero.utils as utils
import models.video_editors.text2vidzero.gradio_utils as gradio_utils
import os

import argparse


on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


class ModelType(Enum):
    Pix2Pix = 1,
    Text2Video = 2,
    ControlNetCanny = 3,
    ControlNetCannyDB = 4,
    ControlNetPose = 5,
    ControlNetDepth = 6,


class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.pipe_dict = {
            ModelType.Pix2Pix: StableDiffusionInstructPix2PixPipeline,
            ModelType.Text2Video: TextToVideoPipeline,
            ModelType.ControlNetCanny: StableDiffusionControlNetPipeline,
            ModelType.ControlNetCannyDB: StableDiffusionControlNetPipeline,
            ModelType.ControlNetPose: StableDiffusionControlNetPipeline,
            ModelType.ControlNetDepth: StableDiffusionControlNetPipeline,
        }
        self.controlnet_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)
        self.pix2pix_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=3)
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)

        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

    def set_model(self, model_type: ModelType, model_id: str, model_path: str = None, **kwargs):
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        safety_checker = kwargs.pop('safety_checker', None)
        requires_safety_checker = kwargs.pop('requires_safety_checker', False)
        if model_path is None:
            self.pipe = self.pipe_dict[model_type].from_pretrained(
                model_id, safety_checker=None,
                requires_safety_checker=requires_safety_checker, **kwargs).to(self.device).to(self.dtype)
        else:
            self.pipe = self.pipe_dict[model_type].from_pretrained(
                model_path).to(self.device).to(self.dtype)
        self.model_type = model_type
        self.model_name = model_id

    def inference_chunk(self, frame_ids, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        prompt = np.array(kwargs.pop('prompt'))
        negative_prompt = np.array(kwargs.pop('negative_prompt', ''))
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(prompt=prompt[frame_ids].tolist(),
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)
        seed = kwargs.pop('seed', 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop('generator', '')

        if 'image' in kwargs:
            f = kwargs['image'].shape[0]
        else:
            f = kwargs['video_length']

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)
            result = []
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                result.append(self.inference_chunk(frame_ids=frame_ids,
                                                   prompt=prompt,
                                                   negative_prompt=negative_prompt,
                                                   **kwargs).images[1:])
                frames_counter += len(chunk_ids)-1
                if on_huggingspace and frames_counter >= 80:
                    break
            result = np.concatenate(result)
            return result
        else:
            self.generator.manual_seed(seed)
            return self.pipe(prompt=prompt, negative_prompt=negative_prompt, generator=self.generator, **kwargs).images


    def process_pix2pix(self,
                        video,
                        prompt,
                        edited_path,
                        update_num=1,
                        resolution=512,
                        seed=0,
                        image_guidance_scale=1.0,
                        start_t=0,
                        end_t=-1,
                        out_fps=-1,
                        chunk_size=8,
                        merging_ratio=0.0,
                        use_cf_attn=True,
                        ig_strategy="single"):
        print("Module Pix2Pix")
        if self.model_type != ModelType.Pix2Pix:
            self.set_model(ModelType.Pix2Pix,
                        model_id="timbrooks/instruct-pix2pix")

            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.pix2pix_attn_proc)

        if ig_strategy == "single":
            image_guidance_scale_list = [image_guidance_scale]
        elif ig_strategy == "multi":
            image_guidance_scale_list = [1.0, 1.5, 2.0]
        else:
            raise ValueError("Invalid image guidance strategy")

        video_path = video
        video, _ = utils.prepare_vidframes(
            video, resolution, self.device, self.dtype, True, start_t, end_t, out_fps)

        self.generator.manual_seed(seed)
        for ig_idx, image_guidance_scale in enumerate(image_guidance_scale_list):
            result = self.inference(image_orig=video,
                                    image=video,
                                    prompt=prompt,
                                    seed=seed,
                                    output_type='numpy',
                                    num_inference_steps=30 // update_num,
                                    image_guidance_scale=image_guidance_scale,
                                    split_to_chunks=True,
                                    chunk_size=chunk_size,
                                    merging_ratio=merging_ratio)

            utils.create_gif_idx(result, edited_path=edited_path,
                original_res=[480,480], video_path=video_path, ig_idx=ig_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="")
    parser.add_argument('--prompt',
                        type=str,
                        default="Several goldfish swim in a tank.",
                        help='The input path to video.')
    parser.add_argument('--dataset', type=str, default="davis")
    parser.add_argument('--edited_path', type=str, default="davis")
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--guidance', type=float, default=1.0)

    args = parser.parse_args()

    model = Model(device = "cuda", dtype = torch.float16)
    model.process_pix2pix(args.source, args.prompt, args.dataset, args.edited_path, seed=args.seed, image_guidance_scale=args.guidance)