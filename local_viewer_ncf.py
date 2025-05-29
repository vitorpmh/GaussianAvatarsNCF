# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import copy
import json
import math
import threading
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
from ifmorph.util import warp_points_ncf
from ifmorph.model import from_pth
from ifmorph.neural_odes import NeuralODE
from ifmorph.diff_operators import jacobian_flat, jacobian
from utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_build_rotation
from ifmorph.fd_models import convert_to_fd

NO_MESH_VIEWER = True

if not NO_MESH_VIEWER:
    from mesh_renderer import MeshRenderer


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path_1: Optional[Path] = None
    """Path to the gaussian splatting file"""
    point_path_2: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float, float, float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""
    warp_file_checkpoint: Optional[Path] = None
    """Path to the warp file checkpoint"""



class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False
        self.playing_morph = False
        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians_1.binding is not None and not NO_MESH_VIEWER:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        if self.gaussians_1.binding is not None and NO_MESH_VIEWER:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        if self.gaussians_1.binding is not None:
            print("Initializing FLAME parameters...")
            self.reset_flame_param()
        
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians_1.binding is not None:
            self.num_timesteps = self.gaussians_1.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)

            self.gaussians_1.select_mesh_by_timestep(self.timestep)

    def init_gaussians(self):
        # load gaussians
        if (Path(self.cfg.point_path_1).parent / "flame_param.npz").exists():
            self.gaussians_1 = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians_1 = GaussianModel(self.cfg.sh_degree)
            self.gaussians_2 = GaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians_1.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians_1.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians_1.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        
        if self.cfg.point_path_1 is not None:
            if self.cfg.point_path_1.exists():
                self.gaussians_1.load_ply(self.cfg.point_path_1, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
                if self.cfg.point_path_2 is not None:
                    self.gaussians_2.load_ply(self.cfg.point_path_2, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
                    self.gaussians_mid = copy.deepcopy(self.gaussians_1)
                    weights_file = self.cfg.warp_file_checkpoint
                    warp_net = torch.load(weights_file, map_location="cuda:0", weights_only=False)
                    self.warp_net = warp_net.to("cuda:0").eval().to(dtype=torch.float32)
                    self.warp_net = convert_to_fd(self.warp_net)

                    self.precalculate_warping(self.warp_net)
                    try:
                        self.warp_net.method = "dopri5"
                    except:
                        print("No ODE solver found, using default method.")
                    self.join_gaussians(0)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path_1} does not exist.')
        
        if (Path(self.cfg.point_path_1).parent / "flame_param.npz").exists():
            id = Path(self.cfg.point_path_1).parent.name
            self.gaussians_1.save_ply_3dgs_format(f'media/output_{id}.ply')
            np.save(f"media/lmk_{id}.npy",
                self.gaussians_1.get_landmarks().squeeze(0).cpu().numpy())
            np.save(f"media/mesh_{id}.npy",
                self.gaussians_1.get_meshes().squeeze(0).cpu().numpy())
            np.save(f"media/selected_lmk_{id}.npy",
                self.gaussians_1.get_landmarks_from_meshes())
    
    def precalculate_warping(self,warp_net, batch = 400000):
        # Precalculate the warping for the two gaussians
        self.times = torch.arange(0, 1.0, 0.5).to(self.gaussians_1._xyz.device).float()
        xyz_1 = self.gaussians_1.get_xyz
        xyz_2 = self.gaussians_2.get_xyz


        with torch.no_grad():
            if isinstance(warp_net, NeuralODE):
                self.xyz_1_warp = warp_net(self.times, xyz_1)
                self.xyz_2_warp = warp_net(-self.times, xyz_2)
            else:
                
                self.xyz_1_warp = []
                self.xyz_2_warp = []
                self.jacobian_1_warp = []
                self.jacobian_2_warp = []

                batches_xyz_1 = xyz_1.split(batch)
                batches_xyz_2 = xyz_2.split(batch)    
                for t in self.times:
                    print(f"Precalculating warping for t={t:.3f}...")
                    start = time.time()
                    batches_jacobian_1_warp = []
                    batches_jacobian_2_warp = []

                    batches_warped_1 = []
                    batches_warped_2 = []
                    for xyz_1, xyz_2 in zip(batches_xyz_1, batches_xyz_2):
                        # warp the points
                        with torch.enable_grad():
                            xyz_1.requires_grad_(True)
                            xyz_2.requires_grad_(True)

                            xyz_1_warp = warp_points_ncf(warp_net, xyz_1, t)
                            batches_warped_1.append(xyz_1_warp.clone().detach())

                            xyz_2_warp = warp_points_ncf(warp_net, xyz_2, t-1)
                            batches_warped_2.append(xyz_2_warp.clone().detach())

                            jac_1 = jacobian(xyz_1_warp.unsqueeze(0), xyz_1)[0].squeeze(0).detach()
                            jac_2 = jacobian(xyz_2_warp.unsqueeze(0), xyz_2)[0].squeeze(0).detach()

                            batches_jacobian_1_warp.append(jac_1)
                            batches_jacobian_2_warp.append(jac_2)

                    self.jacobian_1_warp.append(torch.concat(batches_jacobian_1_warp, dim=0))
                    self.jacobian_2_warp.append(torch.concat(batches_jacobian_2_warp, dim=0))
                    self.xyz_1_warp.append(torch.concat(batches_warped_1, dim=0))
                    self.xyz_2_warp.append(torch.concat(batches_warped_2, dim=0))

                    end = time.time()
                    print(f"Precalculated warping for t={t:.3f} in {end - start:.2f} seconds.")


    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance

    def join_gaussians(self, t, filter_by_scaling= 0.3):
        # t is between 0.000 and 1.000
        # clamp within a range 0.01
        idx_t = np.abs(self.times.cpu().numpy() - t).argmin()



        xyz_1_warp = self.xyz_1_warp[idx_t]
        if isinstance(self.warp_net, NeuralODE):
            xyz_2_warp = self.xyz_2_warp[-1-idx_t]
        else:
            xyz_2_warp = self.xyz_2_warp[idx_t]

        features_dc_1 = self.gaussians_1._features_dc
        features_dc_2 = self.gaussians_2._features_dc

        features_rest_1 = self.gaussians_1._features_rest
        features_rest_2 = self.gaussians_2._features_rest



        scaling_1 = self.gaussians_1._scaling
        scaling_2 = self.gaussians_2._scaling

        rotation_1 = self.gaussians_1._rotation
        rotation_2 = self.gaussians_2._rotation

        scaling_1_warped = scaling_1
        scaling_2_warped = scaling_2

        rotation_1_warped = rotation_1
        rotation_2_warped = rotation_2

        jacobians_1 = self.jacobian_1_warp[idx_t]
        jacobians_2 = self.jacobian_2_warp[idx_t]

        opacity_1_blend = self.gaussians_1.get_opacity * (1-t)
        opacity_2_blend = self.gaussians_2.get_opacity * t

        new_opacity_feat_1 = self.gaussians_1.inverse_opacity_activation(opacity_1_blend)
        new_opacity_feat_2 = self.gaussians_2.inverse_opacity_activation(opacity_2_blend)

        self.gaussians_mid._features_dc = torch.concat([features_dc_1, features_dc_2], dim=0)
        self.gaussians_mid._features_rest = torch.concat([features_rest_1, features_rest_2], dim=0)
        self.gaussians_mid._scaling = torch.concat([scaling_1_warped, scaling_2_warped], dim=0)
        self.gaussians_mid._rotation = torch.concat([rotation_1_warped, rotation_2_warped], dim=0)
        self.gaussians_mid._opacity = torch.concat([new_opacity_feat_1, new_opacity_feat_2], dim=0)

        self.gaussians_mid.jacobian = torch.concat([jacobians_1, jacobians_2], dim=0)

        self.gaussians_mid._xyz = torch.concat([xyz_1_warp, xyz_2_warp], dim=0)


    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def update_record_timeline(self):
        cycles = dpg.get_value("_input_cycles")
        if cycles == 0:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes[:-1]])
        else:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes]) * cycles

        dpg.configure_item("_slider_record_timestep", min_value=0, max_value=self.num_record_timeline-1)

        if len(self.keyframes) <= 0:
            self.all_frames = {}
            return
        else:
            k_x = []

            keyframes = self.keyframes.copy()
            if cycles > 0:
                # pad a cycle at the beginning and the end to ensure smooth transition
                keyframes = self.keyframes * (cycles + 2)
                t_couter = -sum([keyframe['interval'] for keyframe in self.keyframes])
            else:
                t_couter = 0

            for keyframe in keyframes:
                k_x.append(t_couter)
                t_couter += keyframe['interval']
            
            x = np.arange(self.num_record_timeline)
            self.all_frames = {}

            if len(keyframes) <= 1:
                for k in keyframes[0]:
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                    self.all_frames[k] = np.tile(k_y, (self.num_record_timeline, 1))
            else:
                kind = 'linear' if len(keyframes) <= 3 else 'cubic'
            
                for k in keyframes[0]:
                    if k == 'interval':
                        continue
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                  
                    interp_funcs = [interp1d(k_x, k_y[:, i], kind=kind, fill_value='extrapolate') for i in range(k_y.shape[1])]

                    y = np.array([interp_func(x) for interp_func in interp_funcs]).transpose(1, 0)
                    self.all_frames[k] = y

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps*self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        record_timestep = dpg.get_value("_slider_record_timestep")
        state_dict = {k: self.all_frames[k][record_timestep] for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius'].item()
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy'].item()
    
    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths
    
    def export_trajectory(self):
        tid2paths = self.parse_ref_json()

        if self.num_record_timeline <= 0:
            return
        
        timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        traj_dict = {'frames': []}
        timestep_indices = []
        camera_indices = []
        for i in range(self.num_record_timeline):
            # update
            dpg.set_value("_slider_record_timestep", i)
            state_dict = self.get_state_dict_record()
            self.apply_state_dict(state_dict)

            self.need_update = True
            while self.need_update:
                time.sleep(0.001)

            # save image
            save_folder = self.cfg.save_folder / timestamp
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            path = save_folder / f"{i:05d}.png"
            print(f"Saving image to {path}")
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

            # cache camera parameters
            cx = self.cam.intrinsics[2]
            cy = self.cam.intrinsics[3]
            fl_x = self.cam.intrinsics[0].item() if isinstance(self.cam.intrinsics[0], np.ndarray) else self.cam.intrinsics[0]
            fl_y = self.cam.intrinsics[1].item() if isinstance(self.cam.intrinsics[1], np.ndarray) else self.cam.intrinsics[1]
            h = self.cam.image_height
            w = self.cam.image_width
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            c2w = self.cam.pose.copy()  # opencv convention
            c2w[:, [1, 2]] *= -1  # opencv to opengl
            # transform_matrix = np.linalg.inv(c2w).tolist()  # world2cam
            
            timestep_index = self.timestep
            camera_indx = i
            timestep_indices.append(timestep_index)
            camera_indices.append(camera_indx)
            
            tid2paths[timestep_index]['file_path']

            frame = {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": c2w.tolist(),
                'timestep_index': timestep_index,
                'camera_indx': camera_indx,
            }
            if timestep_index in tid2paths:
                frame['file_path'] = tid2paths[timestep_index]['file_path']
                frame['fg_mask_path'] = tid2paths[timestep_index]['fg_mask_path']
                frame['flame_param_path'] = tid2paths[timestep_index]['flame_param_path']
            traj_dict['frames'].append(frame)

            # update timestep
            if dpg.get_value("_checkbox_dynamic_record"):
                self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians_1.select_mesh_by_timestep(self.timestep)
        
        traj_dict['timestep_indices'] = sorted(list(set(timestep_indices)))
        traj_dict['camera_indices'] = sorted(list(set(camera_indices)))
        
        # save camera parameters
        path = save_folder / f"trajectory.json"
        print(f"Saving trajectory to {path}")
        with open(path, 'w') as f:
            json.dump(traj_dict, f, indent=4)

    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians_1.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }
    def callback_update_time(self, sender, app_data):
        t = app_data  # t in [0.0, 1.0]
        with torch.no_grad():
            self.join_gaussians(t)  # assumes self.warp_net is set
        self.need_update = True

    def play_slider(self):
        if self.playing_morph:
            return
        self.playing_morph = True

        def _play():
            t = dpg.get_value("_slider_time_t")
            while t <1.0 and self.playing_morph:
                t += 0.001
                if t>1: break
                dpg.set_value("_slider_time_t", t)
                self.callback_update_time("_slider_time_t", t)
                time.sleep(0.001)  # adjust speed here
            self.playing_morph = False

        threading.Thread(target=_play, daemon=True).start()

    def stop_slider(self):
        self.playing_morph = False

    def save_move_morph_top(self):
        self.keyframes = []
        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
        self.update_record_timeline()
        half = len(self.manual_key_frames)//2
        for idx,keyframe in enumerate(self.manual_key_frames[half:]):
            self.keyframes.insert(idx, keyframe)
            dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
            dpg.set_value("_listbox_keyframes", idx)
            self.update_record_timeline()

        times = np.linspace(0, 1, self.num_record_timeline)
        
        self.apply_state_dict({k: self.all_frames[k][0] for k in self.all_frames})
        self.callback_update_time("_slider_time_t",0)
        self.need_update = True
        time.sleep(0.1)
        
        for idx in range(self.num_record_timeline):
            state_dict = {k: self.all_frames[k][idx] for k in self.all_frames}
            self.apply_state_dict(state_dict)
            self.callback_update_time("_slider_time_t",times[idx])
            self.need_update = True
            time.sleep(0.1)

            
            name_1 = Path(self.cfg.point_path_1).parent.name
            name_2 = Path(self.cfg.point_path_2).parent.name
            folder_name = f"{name_1}-{name_2}"
            folder_cameras = "top"
            folder_path = self.cfg.save_folder / folder_name / folder_cameras
            path = folder_path / f"{idx:04d}.png"
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

    
    def save_move_morph_bottom(self):
        self.keyframes = []
        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
        self.update_record_timeline()
        half = len(self.manual_key_frames)//2
        for idx,keyframe in enumerate(self.manual_key_frames[:half]):
            self.keyframes.insert(idx, keyframe)
            dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
            dpg.set_value("_listbox_keyframes", idx)
            self.update_record_timeline()
        
        times = np.linspace(0, 1, self.num_record_timeline)
        
        self.apply_state_dict({k: self.all_frames[k][0] for k in self.all_frames})
        self.callback_update_time("_slider_time_t",0)
        self.need_update = True
        time.sleep(0.1)
        for idx in range(self.num_record_timeline):

            state_dict = {k: self.all_frames[k][idx] for k in self.all_frames}
            self.apply_state_dict(state_dict)
            self.callback_update_time("_slider_time_t",times[idx])
            self.need_update = True
            time.sleep(0.1)

            
            name_1 = Path(self.cfg.point_path_1).parent.name
            name_2 = Path(self.cfg.point_path_2).parent.name
            folder_name = f"{name_1}-{name_2}"
            folder_cameras = "bottom"
            folder_path = self.cfg.save_folder / folder_name / folder_cameras
            path = folder_path / f"{idx:04d}.png"
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
    
    def save_move_morph(self):
        self.keyframes = []
        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
        self.update_record_timeline()
        for idx,keyframe in enumerate(self.manual_key_frames):
            self.keyframes.insert(idx, keyframe)
            dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
            dpg.set_value("_listbox_keyframes", idx)
            self.update_record_timeline()

        times = np.linspace(0, 1, self.num_record_timeline)
        
        self.apply_state_dict({k: self.all_frames[k][0] for k in self.all_frames})
        self.callback_update_time("_slider_time_t",0)
        self.need_update = True
        time.sleep(0.1)
        for idx in range(self.num_record_timeline):
            state_dict = {k: self.all_frames[k][idx] for k in self.all_frames}
            self.apply_state_dict(state_dict)
            self.callback_update_time("_slider_time_t",times[idx])
            self.need_update = True
            time.sleep(0.05)

            
            name_1 = Path(self.cfg.point_path_1).parent.name
            name_2 = Path(self.cfg.point_path_2).parent.name
            folder_name = f"{name_1}-{name_2}"
            folder_cameras = "all"
            folder_path = self.cfg.save_folder / folder_name / folder_cameras
            path = folder_path / f"{idx:04d}.png"
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)


    def callback_reset_morph(self):
        self.callback_update_time("_slider_time_t",0)
        dpg.set_value("_slider_time_t", 0)

    def callback_interval(self, sender, app_data):
        self.interval = app_data
        self.keyframes = []
        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
        self.update_record_timeline()
        for idx,keyframe in enumerate(self.manual_key_frames):
            keyframe['interval'] = self.interval
            self.keyframes.insert(idx, keyframe)
            dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
            dpg.set_value("_listbox_keyframes", idx)
            self.update_record_timeline()

    def save_zero_five_one(self):
        self.callback_update_time("_slider_time_t",0)
        dpg.set_value("_slider_time_t", 0)
        self.need_update = True
        for slider_time in [0.0, 0.5, 1.0]:
            dpg.set_value("_slider_time_t", slider_time)
            self.callback_update_time("_slider_time_t", slider_time)
            self.need_update = True
            time.sleep(0.1)
            for idx in range(self.num_record_timeline):
                state_dict = {k: self.all_frames[k][idx] for k in self.all_frames}
                self.apply_state_dict(state_dict)
                self.need_update = True
                time.sleep(0.05)

                
                name_1 = Path(self.cfg.point_path_1).parent.name
                name_2 = Path(self.cfg.point_path_2).parent.name
                folder = "benchmark"
                slider_time_str = str(slider_time).replace(".", "")
                folder_name = f"{name_1}-{name_2}_{slider_time_str}"
                folder_cameras = "all"
                folder_path = self.cfg.save_folder / folder / folder_name / folder_cameras
                path = folder_path / f"{idx:04d}.png"
                if not folder_path.exists():
                    folder_path.mkdir(parents=True)
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

    def save_unblended_five(self):
        self.callback_update_time("_slider_time_t",0.5)
        dpg.set_value("_slider_time_t", 0.5)
        self.need_update = True
        time.sleep(0.1)
        for idx in range(self.num_record_timeline):
            state_dict = {k: self.all_frames[k][idx] for k in self.all_frames}
            self.apply_state_dict(state_dict)
            self.need_update = True
            time.sleep(0.05)

            for i in range(2):
                opacity_1_blend = self.gaussians_1.get_opacity * (1-i)
                opacity_2_blend = self.gaussians_2.get_opacity * i
                new_opacity_feat_1 = self.gaussians_1.inverse_opacity_activation(opacity_1_blend)
                new_opacity_feat_2 = self.gaussians_2.inverse_opacity_activation(opacity_2_blend)
 
                self.gaussians_mid._opacity = torch.concat([new_opacity_feat_1, new_opacity_feat_2], dim=0)
                self.need_update = True
                time.sleep(0.05)
                
                
                name_1 = Path(self.cfg.point_path_1).parent.name
                name_2 = Path(self.cfg.point_path_2).parent.name
                
                if i==0: 
                    file_name = name_1 + "_unblended"
                else:
                    file_name = name_2 + "_unblended"
            
                
                folder = "benchmark"
                folder_name = f"{name_1}-{name_2}_unblended"
                folder_cameras = "all"
                folder_path = self.cfg.save_folder / folder / folder_name / folder_cameras
                path = folder_path / f"{idx:04d}_{file_name}.png"
                if not folder_path.exists():
                    folder_path.mkdir(parents=True)
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
    
    def update_needupdate(self, appdata):
        self.need_update = True
    

    def define_gui(self):
        super().define_gui()
        with dpg.window(label="Blend", tag="_Blend_window", autosize=True, pos=(self.W//2, 0)):
            
            dpg.add_slider_float(
                label="Time (t)", tag="_slider_time_t",
                min_value=0.0, max_value=1.0, default_value=0.0,
                callback=self.callback_update_time
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=lambda: self.play_slider())
                dpg.add_button(label="Stop", callback=lambda: self.stop_slider())
                dpg.add_button(label="Reset", callback=lambda: self.callback_reset_morph())
            dpg.add_button(label="SaveMove and morph (bottom cameras)", callback=lambda: self.save_move_morph_bottom())   
            dpg.add_button(label="Save-Move and morph (top cameras)", callback=lambda: self.save_move_morph_top())   
            dpg.add_button(label="Save-Move and morph (all cameras)", callback=lambda: self.save_move_morph())   
            dpg.add_slider_int(label="Set Interval", tag="_slider_set_interval", min_value=1, max_value=100, default_value=1, callback=self.callback_interval)
            dpg.add_button(label="Save 0.0, 0.5, 1.0", callback=lambda: self.save_zero_five_one())
            dpg.add_button(label="Save unblended 0.5", callback=lambda: self.save_unblended_five())

            dpg.add_checkbox(label="use jacobian", default_value=False, tag="_checkbox_use_jacobian", callback=self.update_needupdate)



        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

            dpg.add_text(f"number of points: {self.gaussians_1._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # show splatting
                def callback_show_splatting(sender, app_data):
                    self.need_update = True

                dpg.add_checkbox(label="show splatting", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians_1.binding is not None:
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")

                    # # show original mesh
                    # def callback_original_mesh(sender, app_data):
                    #     self.original_mesh = app_data
                    #     self.need_update = True
                    # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
            
            # timestep slider and buttons
            if self.num_timesteps != None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.timestep = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians_1.select_mesh_by_timestep(self.timestep)

                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200, default_value=1, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            if self.gaussians_1.binding is not None:
                # visualization options
                def callback_visual_options(sender, app_data):
                    if app_data == 'number of points per face':
                        value, ct = self.gaussians_1.binding.unique(return_counts=True)
                        ct = torch.log10(ct + 1)
                        ct = ct.float() / ct.max()
                        cmap = matplotlib.colormaps["plasma"]
                        self.face_colors = torch.from_numpy(cmap(ct.cpu())[None, :, :3]).to(self.gaussians_1.verts)
                    else:
                        self.face_colors = self.mesh_color[:3].to(self.gaussians_1.verts)[None, None, :].repeat(1, self.gaussians_1.face_center.shape[0], 1)  # (1, F, 3)
                    
                    dpg.set_value('_checkbox_show_mesh', True)
                    self.need_update = True
                dpg.add_combo(["none", "number of points per face"], default_value="none", label='visualization', width=200, callback=callback_visual_options, tag="_visual_options")

                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    if dpg.get_value("_visual_options") == 'none':
                        self.face_colors = self.mesh_color[:3].to(self.gaussians_1.verts)[None, None, :].repeat(1, self.gaussians_1.face_center.shape[0], 1)
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

            # # near slider
            # def callback_set_near(sender, app_data):
            #     self.cam.znear = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

            # # far slider
            # def callback_set_far(sender, app_data):
            #     self.cam.zfar = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")
            
            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose", callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache", callback=callback_clear_cache, show=not self.cfg.demo_mode)
                
        # window: recording ==================================================================================================
        with dpg.window(label="Record", tag="_record_window", autosize=True, pos=(0, self.H//2)):
            dpg.add_text("Keyframes")
            with dpg.group(horizontal=True):
                # list keyframes
                def callback_set_current_keyframe(sender, app_data):
                    idx = int(dpg.get_value("_listbox_keyframes"))
                    self.apply_state_dict(self.keyframes[idx])

                    record_timestep = sum([keyframe['interval'] for keyframe in self.keyframes[:idx]])
                    dpg.set_value("_slider_record_timestep", record_timestep)

                    self.need_update = True
                dpg.add_listbox(self.keyframes, width=200, tag="_listbox_keyframes", callback=callback_set_current_keyframe)
                # edit keyframes
                with dpg.group():
                    # add
                    def callback_add_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            new_idx = 0
                        else:
                            new_idx = int(dpg.get_value("_listbox_keyframes")) + 1

                        states = self.get_state_dict()
                        
                        self.keyframes.insert(new_idx, states)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", new_idx)

                        self.update_record_timeline()
                    dpg.add_button(label="add", tag="_button_add_keyframe", callback=callback_add_keyframe)

                    # delete
                    def callback_delete_keyframe(sender, app_data):
                        idx = int(dpg.get_value("_listbox_keyframes"))
                        self.keyframes.pop(idx)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", idx-1)

                        self.update_record_timeline()
                    dpg.add_button(label="delete", tag="_button_delete_keyframe", callback=callback_delete_keyframe)

                    # update
                    def callback_update_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            return
                        else:
                            idx = int(dpg.get_value("_listbox_keyframes"))

                        states = self.get_state_dict()
                        states['interval'] = self.cfg.fps*self.cfg.keyframe_interval

                        self.keyframes[idx] = states
                    dpg.add_button(label="update", tag="_button_update_keyframe", callback=callback_update_keyframe)

            with dpg.group(horizontal=True):
                def callback_set_record_cycles(sender, app_data):
                    self.update_record_timeline()
                dpg.add_input_int(label="cycles", tag="_input_cycles", default_value=1, width=70, callback=callback_set_record_cycles)

                def callback_set_keyframe_interval(sender, app_data):
                    self.cfg.keyframe_interval = app_data
                    for keyframe in self.keyframes:
                        keyframe['interval'] = self.cfg.fps*self.cfg.keyframe_interval
                    self.update_record_timeline()
                dpg.add_input_int(label="interval", tag="_input_interval", default_value=self.cfg.keyframe_interval, width=70, callback=callback_set_keyframe_interval)
            
            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()
                
                self.apply_state_dict(state_dict)
                self.need_update = True
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0, max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="dynamic", default_value=False, tag="_checkbox_dynamic_record")
                dpg.add_checkbox(label="loop", default_value=True, tag="_checkbox_loop_record")
            
            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    self.playing = not self.playing
                    self.need_update = True
                dpg.add_button(label="play", tag="_button_play", callback=callback_play)

                def callback_export_trajectory(sender, app_data):
                    self.export_trajectory()
                dpg.add_button(label="export traj", tag="_button_export_traj", callback=callback_export_trajectory)
            
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)
        # add manual key frames
        self.interval = 1
        self.manual_key_frames = [
            {
                'rot': np.array([0.12845670052713626, 0.3911218873012749, 0.04349290594446551, 0.9102916634222662], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.1241163190942145, 0.22927181121348672, 0.022238832291733228, 0.9651606137092886], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.11771155320349946, 0.12924004000118156, 0.007559415927074113, 0.9845729315463402], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.1273665316891493, 0.04522232790923264, 0.006163521461862363, 0.9908050861128721], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.12523362491555481, -0.047280686384868494, -0.00471256756918486, 0.990988833232944], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.11963654985759396, -0.12226587629601156, -0.010977756829316329, 0.9851992896296344], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.13349822398986533, -0.22328017572370945, -0.023279273355937696, 0.9652886939938543], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([0.13509485099150081, -0.38241998691201334, -0.032130069655336144, 0.9134943861183511], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.11687478737901108, -0.38658575044642157, 0.05522862005676153, 0.9131492436362819], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.11749332239910888, -0.23122632613661667, 0.04562386454794905, 0.9647010771615695], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.13896990148361024, -0.12601667572576697, 0.01859872343436828, 0.9820698811221503], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.1495199668326914, -0.0445583248040299, -0.0006177727061756095, 0.9877539944570677], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.1291004374356289, 0.04658480881616941, -0.013753495479365427, 0.9904412016892217], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.1344806380543836, 0.12426444908253852, -0.012944467828555125, 0.9830085174785286], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.12885595556691845, 0.22656075736373854, -0.035817828226532906, 0.9647711900335955], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
            {
                'rot': np.array([-0.11919636281810188, 0.3723500763715023, -0.06138207092653921, 0.9183571685819418], dtype=np.float32),
                'look_at': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'radius': np.array([1.0], dtype=np.float32),
                'fovy': np.array([20.0], dtype=np.float32),
                'interval': self.interval,
            },
        ]
        for idx,keyframe in enumerate(self.manual_key_frames):
            self.keyframes.insert(idx, keyframe)
            dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
            dpg.set_value("_listbox_keyframes", idx)
            self.update_record_timeline()
        self.apply_state_dict(self.manual_key_frames[0])
        # window: FLAME ==================================================================================================
        if self.gaussians_1.binding is not None:
            with dpg.window(label="FLAME parameters", tag="_flame_window", autosize=True, pos=(self.W-300, 0)):
                def callback_enable_control(sender, app_data):
                    if app_data:
                        self.gaussians_1.update_mesh_by_param_dict(self.flame_param)
                    else:
                        self.gaussians_1.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
                dpg.add_checkbox(label="enable control", default_value=False, tag="_checkbox_enable_control", callback=callback_enable_control)

                dpg.add_separator()

                def callback_set_pose(sender, app_data):
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    self.flame_param[joint][0, axis_idx] = app_data
                    if joint == 'eyes':
                        self.flame_param[joint][0, 3+axis_idx] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians_1.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                dpg.add_text(f'Joints')
                self.pose_sliders = []
                max_rot = 0.5
                for joint in ['neck', 'jaw', 'eyes']:
                    if joint in self.flame_param:
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=70)
                            self.pose_sliders.append(f"_slider-{joint}-x")
                            self.pose_sliders.append(f"_slider-{joint}-y")
                            self.pose_sliders.append(f"_slider-{joint}-z")
                            dpg.add_text(f'{joint:4s}')
                dpg.add_text('   roll       pitch      yaw')
                
                dpg.add_separator()
                
                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.flame_param['expr'][0, expr_i] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians_1.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                self.expr_sliders = []
                dpg.add_text(f'Expressions')
                for i in range(5):
                    dpg.add_slider_float(label=f"{i}", min_value=-3, max_value=3, format="%.2f", default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=250)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.reset_flame_param()
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians_1.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                    for slider in self.pose_sliders + self.expr_sliders:
                        dpg.set_value(slider, 0)
                dpg.add_button(label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)

        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians_1.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")

        while dpg.is_dearpygui_running():

            if self.need_update or self.playing:
                cam = self.prepare_camera()

                if dpg.get_value("_checkbox_show_splatting"):
                    # rgb
                    use_jacobian = dpg.get_value("_checkbox_use_jacobian")
                    if self.cfg.point_path_2 is None:
                        rgb_splatting = render(cam, self.gaussians_1, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"),use_jacobian=use_jacobian)["render"].permute(1, 2, 0).contiguous()
                    elif self.cfg.point_path_2 is not None:
                        rgb_splatting = render(
                            cam,
                            self.gaussians_mid,
                            self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"),use_jacobian=use_jacobian
                        )["render"].permute(1, 2, 0).contiguous()
                        
                    # opacity
                    # override_color = torch.ones_like(self.gaussians_1._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # rgb_splatting = render(cam, self.gaussians_1, self.cfg.pipeline, background_color, scaling_modifier=dpg.get_value("_slider_scaling_modifier"), override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians_1.binding is not None and dpg.get_value("_checkbox_show_mesh") and not NO_MESH_VIEWER:
                    out_dict = self.mesh_renderer.render_from_camera(self.gaussians_1.verts, self.gaussians_1.faces, cam, face_colors=self.face_colors)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]
                    mesh_opacity = self.mesh_color[3:].cuda()

                if dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif dpg.get_value("_checkbox_show_splatting") and not dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_splatting
                elif not dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                self.render_buffer = rgb.cpu().numpy()
                if self.render_buffer.shape[0] != self.H or self.render_buffer.shape[1] != self.W:
                    continue
                dpg.set_value("_texture", self.render_buffer)

                self.refresh_stat()
                self.need_update = False

                if self.playing:
                    record_timestep = dpg.get_value("_slider_record_timestep")
                    if record_timestep >= self.num_record_timeline - 1:
                        if not dpg.get_value("_checkbox_loop_record"):
                            self.playing = False
                        dpg.set_value("_slider_record_timestep", 0)
                    else:
                        dpg.set_value("_slider_record_timestep", record_timestep + 1)
                        if dpg.get_value("_checkbox_dynamic_record"):
                            self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                            dpg.set_value("_slider_timestep", self.timestep)
                            self.gaussians_1.select_mesh_by_timestep(self.timestep)

                        state_dict = self.get_state_dict_record()
                        self.apply_state_dict(state_dict)

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()
