import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
import tinycudann as tcnn
import json

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)
        
        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling


class DeformNetwork_hash(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10):
        super(DeformNetwork_hash, self).__init__()

        with open('utils/hash.json') as f:
            config = json.load(f)

        self.encoder = tcnn.Encoding(n_input_dims=4,
                                     encoding_config=config["encoding_deform3d"])

        self.gaussian_warp = nn.Linear(self.encoder.n_output_dims + 4, 3)
        self.gaussian_rotation = nn.Linear(self.encoder.n_output_dims + 4, 4)
        self.gaussian_scaling = nn.Linear(self.encoder.n_output_dims + 4, 3)


    def forward(self, x, t, clip=None):
        # normalize x and t to be arranged between 0 and 1
        x_emb = normalize_input(x)
        t_emb = t
        h = torch.cat([x_emb, t_emb], dim=-1)
        input = self.encoder(h)
        input = torch.cat([h, input], dim=-1)

        d_xyz = self.gaussian_warp(input)
        scaling = self.gaussian_scaling(input)
        rotation = self.gaussian_rotation(input)

        return d_xyz, rotation, scaling


def normalize_input(input):
    # step 1. Translate
    min_point = torch.min(input, dim=0).values
    points_translated = input - min_point

    # step 2. Scale
    max_dist = torch.max(torch.sqrt(torch.sum(points_translated**2, dim=1)))
    points_normalized = points_translated / max_dist

    return points_normalized
