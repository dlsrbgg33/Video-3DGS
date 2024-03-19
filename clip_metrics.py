import os
import argparse
import numpy as np
import torch
# import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, CLIPModel, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import json

# Clip score for text alignment
def clip_score_text(frames, prompt, model, processor):
    inputs = processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image.detach().cpu().numpy()
    score = logits_per_image.mean()

    return score


# Clip score for frame consistency
def clip_score_frame(frames, prompt, model, processor):

    inputs = processor(images=frames, return_tensors="pt").to("cuda")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).detach().cpu().numpy()

    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    score = cosine_sim_matrix.sum() / (len(frames) * (len(frames)-1))

    return score


# PickScore: https://github.com/yuvalkirstain/PickScore
def pick_score(frames, prompt, model, processor):

    image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        score_per_image = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        score_per_image = score_per_image.detach().cpu().numpy()
        score = score_per_image.mean()

    return score


eval_functions = {
    "clip_score_text": clip_score_text,
    "clip_score_frame": clip_score_frame,
    "pick_score": pick_score,
}


def clipscore(prompt, path, edited_path, update_idx):
    full_dict = {}
    with open(path + "/train_edit{}/results_clip.json".format(update_idx), 'w') as fp:
        clipscore_func("clip_score_text", prompt, path, edited_path, full_dict, update_idx)
        clipscore_func("clip_score_frame", prompt, path, edited_path, full_dict, update_idx)
        json.dump(full_dict, fp, indent=True)


def clipscore_func(metric, prompt, video_path,
                            edited_path=None, metric_dict=None, update_idx=None):
    # hard-coded for LOVEU dataset
    width, height = 480, 480

    if metric == "clip_score_text" or metric == "clip_score_frame":
        preatrained_model_path = "models/clipscore/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(preatrained_model_path).cuda()
        processor = AutoProcessor.from_pretrained(preatrained_model_path)
    elif metric == "pick_score":
        preatrained_model_path = "models/clipscore/PickScore_v1"
        model = AutoModel.from_pretrained(preatrained_model_path).cuda()
        processor_path = "models/clipscore/CLIP-ViT-H-14-laion2B-s32B-b79K"
        processor = AutoProcessor.from_pretrained(processor_path)
    else:
        raise NotImplementedError(args.metric)

    scores = []
    eval_function = eval_functions[metric]


    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    for init_pth in sorted(os.listdir(edited_path)):
        if "vid_output" not in init_pth:
            continue
        sub_folder = os.path.join(
            video_path, "train_edit{}/ours_1000".format(update_idx))
        initial_edit_path = os.path.join(sub_folder, init_pth)
        init_frames = [Image.open(x) for x in sorted(glob(f"{initial_edit_path}/*.png"))]
        init_frames = [i.resize((width, height)) for i in init_frames]
        init_scores = eval_function(init_frames, prompt, model, processor)
        print("{} of init for {}: {:.3f}".format(metric, init_pth, np.mean(init_scores)))
        metric_dict.update({"{} for {}".format(metric, init_pth): np.mean(init_scores).item()})

    refined_edit_path = os.path.join(sub_folder, "refined_edited")
    refined_frames = [Image.open(x) for x in sorted(glob(f"{refined_edit_path}/*.png"))]
    refined_frames = [i.resize((width, height)) for i in refined_frames]
    refine_scores = eval_function(refined_frames, prompt, model, processor)
    print("{} of refined: {:.3f}".format(metric, np.mean(refine_scores)))
    metric_dict.update({"{} for Refined".format(metric): np.mean(refine_scores).item()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_path", type=str, required=True, help="path to submission folder")
    parser.add_argument("--data_path", type=str, default="./data/loveu-tgve-2023", help="path to data folder")
    parser.add_argument("--metric", type=str, default="clip_score_text",
                        choices=['clip_score_text', 'clip_score_frame', 'pick_score'])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    width, height = 480, 480
    device = args.device

    if args.metric == "clip_score_text" or args.metric == "clip_score_frame":
        preatrained_model_path = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(preatrained_model_path).to(device)
        processor = AutoProcessor.from_pretrained(preatrained_model_path)
    elif args.metric == "pick_score":
        preatrained_model_path = "pickapic-anonymous/PickScore_v1"
        model = AutoModel.from_pretrained(preatrained_model_path).to(device)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        processor = AutoProcessor.from_pretrained(processor_path)
    else:
        raise NotImplementedError(args.metric)

    df = pd.read_csv(f"{args.data_path}/LOVEU-TGVE-2023_Dataset.csv")
    sub_dfs = {
        'DAVIS_480p': df[1:17],
        'youtube_480p': df[19:42],
        'videvo_480p': df[44:82],
    }

    scores = []
    eval_function = eval_functions[args.metric]
    for sub_name, sub_df in sub_dfs.items():
        print(f"Processing {sub_name} ..")
        for index, row in tqdm(sub_df.iterrows(), total=sub_df.shape[0]):
            video_name = row['Video name']
            edited_prompts = {x.split(" ")[0].lower(): str(row[x]).strip() for x in [
                "Style Change Caption",
                "Object Change Caption",
                "Background Change Caption",
                "Multiple Changes Caption"
            ]}

            for edited_type, edited_prompt in edited_prompts.items():
                video_path = f"{args.submission_path}/{sub_name}/{video_name}/{edited_type}"
                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)
                frames = [Image.open(x) for x in sorted(glob(f"{video_path}/*.jpg"))]
                frames = [i.resize((width, height)) for i in frames]

                scores.append(eval_function(frames, edited_prompt))

    print("{}: {:.3f}".format(args.metric, np.mean(scores)))


