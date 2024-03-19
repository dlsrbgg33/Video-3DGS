
import numpy as np
from PIL import Image
import shutil
from collections import defaultdict

def check_n_groups_clips(group_size, sub_source_path):
    clip_dict = defaultdict(list)
    clip_num = 0

    for sub_source in sub_source_path:
        if 'clip_from' in sub_source:
            group_num = clip_num // group_size
            clip_dict[group_num].append(sub_source)
            clip_num += 1

    return dict(clip_dict), clip_num