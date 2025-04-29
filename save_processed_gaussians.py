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
from uuid import uuid4
from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render

@dataclass
class Config(Mini3DViewerConfig):
    gaussians_path: Optional[Path] = None
    """Path to folder containing multiple id/point_cloud.ply and id/flame_param.npz"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    save_folder: Path = Path("./viewer_output")

def process_gaussians(cfg):
    for id_path in cfg.gaussians_path.iterdir():
        if (id_path / "flame_param.npz").exists():
            gaussians = FlameGaussianModel(cfg.sh_degree)
        else:
            gaussians = GaussianModel(cfg.sh_degree)

        unselected_fid = []
            
        ply_path = id_path / "point_cloud.ply"  
        if ply_path is not None:
            if ply_path.exists():
                    gaussians.load_ply(ply_path, has_target=False, motion_path=None, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{ply_path} does not exist.')
            
        save_path = cfg.save_folder / id_path.name
            
        if (id_path / "flame_param.npz").exists():
            gaussians.save_ply_3dgs_format(f'{save_path}/point_cloud.ply')
            np.save(f"{save_path}/lmk.npy",
                gaussians.get_landmarks().squeeze(0).cpu().numpy())
            np.save(f"{save_path}/mesh.npy",
                gaussians.get_meshes().squeeze(0).cpu().numpy())
            np.save(f"{save_path}/selected_lmk.npy",
                gaussians.get_landmarks_from_meshes())


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    process_gaussians(cfg)
