import random
import os
import PIL
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import set_seed
from tqdm import tqdm
from transformers import logging
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler

import torch.nn as nn
import numpy as np
import models.video_editors.RAVE.utils.feature_utils as fu
import models.video_editors.RAVE.utils.preprocesser_utils as pu
import models.video_editors.RAVE.utils.image_process_utils as ipu


logging.set_verbosity_error()

def set_seed_lib(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

@torch.no_grad()
class RAVE(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.dtype = torch.float

    @torch.no_grad()
    def __init_pipe(self, hf_cn_path, hf_path):
        controlnet = ControlNetModel.from_pretrained(hf_cn_path, torch_dtype=self.dtype).to(self.device, self.dtype)

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(hf_path, controlnet=controlnet, torch_dtype=self.dtype).to(self.device, self.dtype) 
        pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention()
        return pipe
        
    @torch.no_grad()
    def init_models(self, hf_cn_path, hf_path, preprocess_name, model_id=None):
        if model_id is None or model_id == "None":
            pipe = self.__init_pipe(hf_cn_path, hf_path)  
        else:
            pipe = self.__init_pipe(hf_cn_path, model_id)  
        self.preprocess_name = preprocess_name
        
        
        self._prepare_control_image = pipe.prepare_control_image
        self.run_safety_checker = pipe.run_safety_checker
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        self.vae = pipe.vae
        self.unet = pipe.unet    

        self.controlnet = pipe.controlnet
        self.scheduler_config = pipe.scheduler.config        
        
        del pipe
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):

        cond_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        cond_embeddings = self.text_encoder(cond_input.input_ids.to(self.device))[0]


        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]


        return cond_embeddings, uncond_embeddings

    @torch.no_grad()
    def prepare_control_image(self, control_pil, width, height):

        control_image = self._prepare_control_image(
        image=control_pil,
        width=width,
        height=height,
        device=self.device,
        dtype=self.controlnet.dtype,
        batch_size=1,
        num_images_per_prompt=1
    )

        return control_image
    
    @torch.no_grad()
    def pred_controlnet_sampling(self, current_sampling_percent, latent_model_input, t, text_embeddings, control_image):
        if (current_sampling_percent < self.controlnet_guidance_start or current_sampling_percent > self.controlnet_guidance_end):
            down_block_res_samples = None
            mid_block_res_sample = None
        else:
            
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                conditioning_scale=self.controlnet_conditioning_scale,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,                    
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample)['sample']
        return noise_pred
    
    
    @torch.no_grad()
    def denoising_step(self, latents, control_image, text_embeddings, t, guidance_scale, current_sampling_percent):
        
        latent_model_input = torch.cat([latents] * 2)
        control_image = torch.cat([control_image] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


        noise_pred = self.pred_controlnet_sampling(current_sampling_percent, latent_model_input, t, text_embeddings, control_image)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        
        latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents


    @torch.no_grad()
    def preprocess_control_grid(self, image_pil):

        list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=self.grid) # List[C, W, H] -> len = num_frames
        list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), self.preprocess_name) for frame_pil in list_of_image_pils]
        control_images = np.array(list_of_pils)
        control_img = ipu.create_grid_from_numpy(control_images, grid_size=self.grid)
        control_img = PIL.Image.fromarray(control_img).convert("L")

        return control_img
    
    @torch.no_grad()
    def shuffle_latents(self, latents, control_image, indices):
        rand_i = torch.randperm(self.total_frame_number).tolist()
        
        latents_l, controls_l, randx = [], [], []
        for j in range(self.sample_size):
            rand_indices = rand_i[j*self.grid_frame_number:(j+1)*self.grid_frame_number]

            latents_keyframe, _ = fu.prepare_key_grid_latents(latents, self.grid, self.grid, rand_indices)
            control_keyframe, _ = fu.prepare_key_grid_latents(control_image, self.grid, self.grid, rand_indices)
            latents_l.append(latents_keyframe)
            controls_l.append(control_keyframe)
            randx.extend(rand_indices)
        rand_i = randx.copy()
        latents = torch.cat(latents_l, dim=0)
        control_image = torch.cat(controls_l, dim=0)
        indices = [indices[i] for i in rand_i]
        return latents, indices, control_image
    
    @torch.no_grad()
    def batch_denoise(self, latents, control_image, indices, t, guidance_scale, current_sampling_percent):
        latents_l, controls_l = [], []
        control_split = control_image.split(self.batch_size, dim=0)
        latents_split = latents.split(self.batch_size, dim=0)
        for idx in range(len(control_split)):
            txt_embed = torch.cat([self.uncond_embeddings] * len(latents_split[idx]) + [self.cond_embeddings] * len(latents_split[idx])) 


            latents = self.denoising_step(latents_split[idx], control_split[idx], txt_embed, t, guidance_scale, current_sampling_percent)
            
            latents_l.append(latents)
            controls_l.append(control_split[idx])
        latents = torch.cat(latents_l, dim=0)
        controls = torch.cat(controls_l, dim=0)
        return latents, indices, controls
    
    @torch.no_grad()
    def reverse_diffusion(self, latents=None, control_image=None, guidance_scale=7.5, indices=None):
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        with torch.autocast('cuda'):

            for i, t in tqdm(enumerate(self.scheduler.timesteps), desc='reverse_diffusion'):
                indices = list(indices)
                current_sampling_percent = i / len(self.scheduler.timesteps)

                if self.is_shuffle:
                    latents, indices, control_image = self.shuffle_latents(latents, control_image, indices)
                    
                if self.cond_step_start < current_sampling_percent:
                    latents, indices, controls = self.batch_denoise(latents, control_image, indices, t, guidance_scale, current_sampling_percent)
                else:
                    latents, indices, controls = self.batch_denoise(latents, control_image, indices, t, 0.0, current_sampling_percent)

        return latents, indices, controls

    @torch.no_grad()
    def encode_imgs(self, img_torch):
        latents_l = []
        splits = img_torch.split(self.batch_size_vae, dim=0)
        for split in splits:
            image = 2 * split - 1
            posterior = self.vae.encode(image).latent_dist
            latents = posterior.mean * self.vae.config.scaling_factor
            latents_l.append(latents)
        

        return torch.cat(latents_l, dim=0)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        image_l = []
        splits = latents.split(self.batch_size_vae, dim=0)
        for split in splits:
            image = self.vae.decode(split / self.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image_l.append(image)
        return torch.cat(image_l, dim=0)




    @torch.no_grad()
    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1,
            return_dict=False,
        )
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def ddim_inversion(self, latents, control_batch, indices):
        k = None
        els = os.listdir(self.inverse_path) 
        els = [el for el in els if el.endswith('.pt')]

        for k,inv_path in enumerate(sorted(els, key=lambda x: int(x.split('.')[0]))):
            latents[k] = torch.load(os.path.join(self.inverse_path, inv_path)).to(device=self.device)

        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.inverse_scheduler.set_timesteps(self.num_inversion_step, device=self.device)
        self.timesteps = reversed(self.inverse_scheduler.timesteps)

        # if k == (latents.shape[0]-1):
        #     return latents, indices, control_batch
        inv_cond = torch.cat([self.inv_uncond_embeddings] * 1 + [self.inv_cond_embeddings] * 1)[1].unsqueeze(0)
        for i, t in enumerate(tqdm(self.timesteps)):
            
            alpha_prod_t = self.inverse_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (self.inverse_scheduler.alphas_cumprod[self.timesteps[i - 1]] if i > 0 else self.inverse_scheduler.final_alpha_cumprod)
            
            if k is not None:
                if len(latents[:k+1].shape) == 3:
                    latents[:k+1] = latents[:k+1].unsqueeze(0)
            latents_l = [] if k is None else [latents[:k+1]]
            latents_split = latents.split(self.inv_batch_size, dim=0) if k is None else latents[k+1:].split(self.inv_batch_size, dim=0)
            control_batch_split = control_batch.split(self.inv_batch_size, dim=0) if k is None else control_batch[k+1:].split(self.inv_batch_size, dim=0)
            for idx in range(len(latents_split)):
                cond_batch = inv_cond.repeat(latents_split[idx].shape[0], 1, 1)
                latents = self.ddim_step(latents_split[idx], t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch_split[idx])
                latents_l.append(latents)
            latents = torch.cat(latents_l, dim=0)
            
        for k,i in enumerate(latents):
            torch.save(i.detach().cpu(), f'{self.inverse_path}/{str(k).zfill(5)}.pt')        

        return latents, indices, control_batch
    
   
    def ddim_step(self, latent_frames, t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch):
        mu = alpha_prod_t ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
        if self.give_control_inversion:
            eps = self.controlnet_pred(latent_frames, t, text_embed_input=cond_batch, controlnet_cond=control_batch)
        else:
            eps = self.unet(latent_frames, t, encoder_hidden_states=cond_batch, return_dict=False)[0]
        pred_x0 = (latent_frames - sigma_prev * eps) / mu_prev
        latent_frames = mu * pred_x0 + sigma * eps
        return latent_frames
    

    def process_image_batch(self, image_pil_list):
        # if len(os.listdir(self.controls_path)) > 0:
        #     control_torch = torch.load(os.path.join(self.controls_path, 'control.pt')).to(self.device)
        #     img_torch = torch.load(os.path.join(self.controls_path, 'img.pt')).to(self.device)
        # else:
        image_torch_list = []
        control_torch_list = []
        for image_pil in image_pil_list:
            width, height = image_pil.size
            control_pil = self.preprocess_control_grid(image_pil)
            control_image = self.prepare_control_image(control_pil, width, height)
            control_torch_list.append(control_image)
            image_torch_list.append(ipu.pil_img_to_torch_tensor(image_pil))
        control_torch = torch.cat(control_torch_list, dim=0).to(self.device)
        img_torch = torch.cat(image_torch_list, dim=0).to(self.device)
        torch.save(control_torch, os.path.join(self.controls_path, 'control.pt'))
        torch.save(img_torch, os.path.join(self.controls_path, 'img.pt'))
            
        return img_torch, control_torch
        
    def order_grids(self, list_of_pils, indices):
        k = []
        for i in range(len(list_of_pils)):
            k.extend(fu.pil_grid_to_frames(list_of_pils[i], self.grid))
            
        frames = [k[indices.index(i)] for i in np.arange(len(indices))]    
        return frames


    @torch.autocast(dtype=torch.float16, device_type='cuda')  
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config["batch_size"]
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0,len(x),batch_size) 
        
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        for i, b in enumerate(range(0, len(x), batch_size)):
            denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    @torch.no_grad()
    def __preprocess_inversion_input(self, init_latents, control_batch):
        list_of_flattens = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in init_latents]
        init_latents = torch.cat(list_of_flattens, dim=-1)
        init_latents = torch.cat(torch.chunk(init_latents, self.total_frame_number, dim=-1), dim=0)
        control_batch_flattens = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in control_batch]
        control_batch = torch.cat(control_batch_flattens, dim=-1)
        control_batch = torch.cat(torch.chunk(control_batch, self.total_frame_number, dim=-1), dim=0)
        return init_latents, control_batch
    
    @torch.no_grad()
    def __postprocess_inversion_input(self, latents_inverted, control_batch):
            latents_inverted = torch.cat([fu.unflatten_grid(torch.cat([a for a in latents_inverted[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            control_batch = torch.cat([fu.unflatten_grid(torch.cat([a for a in control_batch[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            return latents_inverted, control_batch
    
    
    @torch.no_grad()
    def __call__(self, input_dict):
        set_seed_lib(input_dict['seed'])
        
        self.grid_size = input_dict['grid_size']
        self.sample_size = input_dict['sample_size']
        
        self.grid_frame_number = self.grid_size * self.grid_size
        self.total_frame_number = (self.grid_frame_number) * self.sample_size
        self.grid = [self.grid_size, self.grid_size]
        
        self.cond_step_start = input_dict['cond_step_start']
        
        self.controlnet_guidance_start = input_dict['controlnet_guidance_start']
        self.controlnet_guidance_end = input_dict['controlnet_guidance_end']
        self.controlnet_conditioning_scale = input_dict['controlnet_conditioning_scale']
        
        self.positive_prompts = input_dict['positive_prompts']
        self.negative_prompts = input_dict['negative_prompts']
        self.inversion_prompt = input_dict['inversion_prompt']
        
        self.batch_size = input_dict['batch_size']
        self.inv_batch_size = self.batch_size * self.grid_size * self.grid_size
        self.batch_size_vae = input_dict['batch_size_vae']

        self.num_inference_steps = input_dict['num_inference_steps']
        self.num_inversion_step = input_dict['num_inversion_step']
        self.inverse_path = input_dict['inverse_path']
        self.controls_path = input_dict['control_path']
        
        self.is_ddim_inversion = input_dict['is_ddim_inversion']
        self.is_shuffle = input_dict['is_shuffle']
        self.give_control_inversion = input_dict['give_control_inversion']
        
        self.guidance_scale = input_dict['guidance_scale']
        
        indices = list(np.arange(self.total_frame_number))
        
        img_batch, control_batch = self.process_image_batch(input_dict['image_pil_list'])
        init_latents_pre = self.encode_imgs(img_batch)
        
        self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.inv_cond_embeddings, self.inv_uncond_embeddings = self.get_text_embeds(self.inversion_prompt, "")
        if self.is_ddim_inversion:
            init_latents, control_batch = self.__preprocess_inversion_input(init_latents_pre, control_batch)
            latents_inverted, indices, control_batch = self.ddim_inversion(init_latents, control_batch, indices)
            latents_inverted, control_batch = self.__postprocess_inversion_input(latents_inverted, control_batch)
        else:
            init_latents_pre = torch.cat([init_latents_pre], dim=0) 
            noise = torch.randn_like(init_latents_pre)
            latents_inverted = self.scheduler.add_noise(init_latents_pre, noise, self.scheduler.timesteps[:1])

        self.cond_embeddings, self.uncond_embeddings = self.get_text_embeds(self.positive_prompts, self.negative_prompts)
        latents_denoised, indices, controls = self.reverse_diffusion(latents_inverted, control_batch, self.guidance_scale, indices=indices)
    
        image_torch = self.decode_latents(latents_denoised)
        ordered_img_frames = self.order_grids(ipu.torch_to_pil_img_batch(image_torch), indices)
        ordered_control_frames = self.order_grids(ipu.torch_to_pil_img_batch(controls), indices)
        return ordered_img_frames, ordered_control_frames
    
