# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import json
import math
import tyro
import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib

from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
   
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    point_output_path: Optional[Path] = None
    """Path to save the converted gaussian splatting file"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""

class ConvertRep(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg


        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        

    def init_gaussians(self):
        # load gaussians
        if (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=None, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')
        
        point_cloud_path = os.path.join(cfg.point_output_path, "point_cloud.ply")

        self.gaussians.save_ply_as_3dgs(point_cloud_path)

        print(f"Converted point cloud saved to {point_cloud_path}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = ConvertRep(cfg)
