import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)

# prompt = "Make it Swan look like Robot Bird"
# prompt = "Snowy day"
# prompt = "oil painting of a horse running on snowy weather, a high-quality, detailed, and professional photo"
# prompt = "Transform the video to mimic the impressionist style of Claude Monet, with soft brush strokes and a pastel color palette that gives a dreamlike quality to the horse jumping scene."
prompt = "Several goldfish swim in a tank."

motion_path = '/opt/tiger/inkyunasus/loveu-tgve-2023/gold-fish'

model.process_pix2pix(motion_path, prompt=prompt, scene="gold-fish", dataset="davis", seed=1001)

# prompt = 'an astronaut dancing in outer space'
# motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
# out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
# model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path)