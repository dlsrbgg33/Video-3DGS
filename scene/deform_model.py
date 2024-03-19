import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, DeformNetwork_hash
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, n_mlp=3, use_hash=False):
        if use_hash:
            self.deform = DeformNetwork_hash().cuda()
        else:
            self.deform = DeformNetwork(D=n_mlp).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weightsf_group(self, model_path, iteration, group=None, clip=None, clip_deform=False):
        out_weights_path = os.path.join(model_path, "deform_fore/group_{}/clip_{}/iteration_{}".format(group, clip, iteration))
        if clip_deform:
            out_weights_path = out_weights_path.replace('deform_fore', 'deform_fore_clip')
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def save_weightsb_group(self, model_path, iteration, group=None, clip=None, clip_deform=False):
        out_weights_path = os.path.join(model_path, "deform_back/group_{}/clip_{}/iteration_{}".format(group, clip, iteration))
        if clip_deform:
            out_weights_path = out_weights_path.replace('deform_back', 'deform_back_clip')
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weightsf_group(self, model_path, iteration=-1, group=None, clip=None, clip_deform=False):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_fore", "group_{}".format(group), "clip_{}".format(clip)))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "deform_fore/group_{}/clip_{}/iteration_{}/deform.pth".format(group, clip, loaded_iter))
        if clip_deform:
            weights_path = weights_path.replace('deform_fore', 'deform_fore_clip')
        self.deform.load_state_dict(torch.load(weights_path), strict=False)

    def load_weightsb_group(self, model_path, iteration=-1, group=None, clip=None, clip_deform=False):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_back", "group_{}".format(group), "clip_{}".format(clip)))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "deform_back/group_{}/clip_{}/iteration_{}/deform.pth".format(group, clip, loaded_iter))
        if clip_deform:
            weights_path = weights_path.replace('deform_back', 'deform_back_clip')
        self.deform.load_state_dict(torch.load(weights_path), strict=False)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
