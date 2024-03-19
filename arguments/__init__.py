#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class ModelParamsForeBack(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        # temporary mapped
        self._fsource_path_fore = "datasets/DAVIS/JPEGImages/480p/blackswan_colmap_covsift_clip10_overlap_intrinsic/mask/fore/inst_0"
        self._bsource_path_back = "/opt/tiger/gaussian-splatting/datasets/DAVIS/JPEGImages/480p/horsejump-high_colmap_covsift_full_onback_full_noma/mask/back"
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path_fore = os.path.abspath(g.fsource_path_fore)
        g.source_path_back = os.path.abspath(g.bsource_path_back)
        return g

class ModelParamsClipMask(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._fsource_path_fore = "/opt/tiger/gaussian-splatting/datasets/DAVIS/JPEGImages/480p/car-turn_colmap_covsiftx/mask/fore/inst_0"
        self._bsource_path_back = "/opt/tiger/gaussian-splatting/datasets/DAVIS/JPEGImages/480p/car-turn_colmap_covsiftx/mask/back"

        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path_fore = os.path.abspath(g.fsource_path_fore)
        g.source_path_back = os.path.abspath(g.bsource_path_back)
        return g


class ModelParamsForeBack(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        # temporary testing on "blackswan clip"
        self._source_path = ""
        self._lsource_path = ""
        self._gsource_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.lsource_path = os.path.abspath(g.lsource_path)
        g.gsource_path = os.path.abspath(g.gsource_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # self.iterations = 30_000
        self.iterations = 10_000
        self.warm_up = 1_000
        # self.warm_up = 0
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000
        self.feature_lr = 0.01
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 5_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

class OptimizationClipMaskParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 5_000
        self.warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 3_000
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")



class OptimizationClipMaskParams3k(ParamGroup):
    def __init__(self, parser):
        self.iterations = 3_000
        self.warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 2_000
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")


class OptimizationClipMaskParams5k(ParamGroup):
    def __init__(self, parser):
        self.iterations = 5_000
        self.warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 3_000
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")

class OptimizationClipMaskParams10k(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 5_000
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")


class OptimizationClipMaskParams30k(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.transp_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")


class OptimizationClipMaskParamsEdit(ParamGroup):
    def __init__(self, parser):
        self.iterations = 1_000
        self.warm_up = 0
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 3_000
        self.deform_lr_max_steps = 3_000

        self.feature_lr = 0.025
        self.opacity_lr = 0.05
        self.transp_lr = 0.0
        self.scaling_lr = 0
        self.rotation_lr = 0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 2000
        self.densify_from_iter = 0
        self.densify_until_iter = 0
        self.densify_grad_threshold = 0.0002
        super().__init__(
            parser, "Optimization(local/global) Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
