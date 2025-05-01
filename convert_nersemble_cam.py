from nersemble_data.data.nersemble_data import NeRSembleDataManager, NeRSembleParticipantDataManager

import numpy as np
from scipy.spatial.transform import Rotation as R
nersemble_folder = "/home/vitor/Documents/doc/nersemble_raw_data"
data_folder = NeRSembleDataManager(nersemble_folder)
downloaded_participant_ids = data_folder.list_participants()    # <- List of all participants that were downloaded
participant_id = downloaded_participant_ids[0]                  # <- Use first available participant

data_manager = NeRSembleParticipantDataManager(nersemble_folder, participant_id)
downloaded_sequences = data_manager.list_sequences()            # <- List of all sequences that were downloaded for that participant
sequence_name = downloaded_sequences[0]                         # <- Use first available sequence

downloaded_cameras = data_manager.list_cameras(sequence_name)   # <- List of all cameras that were downloaded for that sequence
serial = downloaded_cameras[0]                                  # <- Use first available camera


# %%
camera_calibration = data_manager.load_camera_calibration()
world_2_cam_poses = camera_calibration.world_2_cam  # <- For each camera: 4x4 Extrinsic matrices in W2C direction and OpenCV camera coordinate convention
intrinsics = camera_calibration.intrinsics          # <- 3x3 intrinsic matrix (shared across all 16 cameras) for 3208x2200 images



import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

def convert_world2cam_to_nerf_format(pose: np.ndarray, intrinsics: np.ndarray, image_height: int = 2200, interval: int = 25):
    """
    Convert a world-to-camera extrinsic matrix and intrinsic matrix into NeRF-style camera parameters.
    This version applies the correct conversion from COLMAP's OpenCV-style to NeRF's OpenGL-style coordinates.

    Args:
        world_2_cam (np.ndarray): 4x4 OpenCV-style world-to-camera matrix.
        intrinsics (np.ndarray): 3x3 intrinsic matrix.
        image_height (int): Height of the image in pixels. Default is 2200.
        interval (int): Frame interval value to include in the result.

    Returns:
        dict: NeRF-style camera parameters with fixed values for 'look_at', 'radius', and 'fovy'.
    """
    # Extract rotation matrix and translation vector from pose
    rotation_matrix = pose[:3, :3]
    translation_vector = pose[:3, 3]

    # Invert the rotation matrix for OpenCV to match your desired convention
    # This assumes the OpenCV convention has Z-forward, Y-down, X-right.
    rotation_matrix = rotation_matrix @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Convert the adjusted rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    rot_quat = rotation.as_quat()  # Quaternion (x, y, z, w)

    # Camera's position (in world space)
    camera_position = translation_vector

    # The look_at point is typically the target point the camera is oriented toward.
    # Assuming it's still pointing towards the origin
    look_at = np.array([0., 0., 0.], dtype=np.float32)

    # Fixed values for radius, fovy, and interval
    radius = np.array([1.], dtype=np.float32)
    fovy = np.array([20.], dtype=np.float32)

    # Return fixed values as per the user's request
    return {
        'rot': rot_quat,
        'look_at': look_at,  # Fixed look_at
        'radius': radius,  # Fixed radius
        'fovy':fovy,  # Fixed fovy
        'interval': interval
    }

id_order = [
    '222200042',
    '222200046',
    '222200036',
    '220700191',
    '222200037',
    '222200047',
    '222200049',
    '221501007',
    '222200045',
    '222200039',
    '222200043',
    '222200038',
    '222200041',
    '222200048',
    '222200040',
    '222200044',
]


# Process all camera poses
results = []
for cam_id in id_order:
    print(f"Processing camera {cam_id}...")
    cam_pose = np.array(world_2_cam_poses[cam_id].tolist())  # Assuming Pose(...) object has .matrix attribute
    camera_dict = convert_world2cam_to_nerf_format(cam_pose, intrinsics)
    results.append(camera_dict)
#%%
# Output the results as requested
print("camera_configs = [")
for entry in results:
    print("    {")
    for key, val in entry.items():
        if isinstance(val, np.ndarray):
            print(f"        '{key}': np.array({val.tolist()}, dtype=np.float32),")
        else:
            print(f"        '{key}': {val},")
    print("    },")
print("]")

# #%%
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # Camera pose (OpenCV-style, [R, T])
# pose = np.array([[ 0.7054452 ,  0.04462468,  0.70735806, -0.02603636],
#                  [-0.16202722, -0.961434  ,  0.22224241, -0.00370569],
#                  [ 0.68999594, -0.27139112, -0.6710087 ,  1.1310539 ],
#                  [ 0.        ,  0.        ,  0.        ,  1.        ]])

# # Intrinsics matrix (camera parameters)
# intrinsics = np.array([[8185.07715, 0, 1099.70068],
#                        [0, 8185.32178, 1603.50024],
#                        [0, 0, 1]], dtype=np.float32)

# # Extract rotation matrix and translation vector from pose
# rotation_matrix = pose[:3, :3]
# translation_vector = pose[:3, 3]

# # Invert the rotation matrix for OpenCV to match your desired convention
# # This assumes the OpenCV convention has Z-forward, Y-down, X-right.
# rotation_matrix = rotation_matrix @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# # Convert the adjusted rotation matrix to a quaternion
# rotation = R.from_matrix(rotation_matrix)
# rot_quat = rotation.as_quat()  # Quaternion (x, y, z, w)

# # Camera's position (in world space)
# camera_position = translation_vector

# # The look_at point is typically the target point the camera is oriented toward.
# # Assuming it's still pointing towards the origin
# look_at = np.array([0., 0., 0.], dtype=np.float32)

# # Fixed values for radius, fovy, and interval
# radius = np.array([1.], dtype=np.float32)
# fovy = np.array([20.], dtype=np.float32)
# interval = 25

# # Create the dictionary with the adjusted parameters
# camera_params = {
#     'rot': np.array(rot_quat, dtype=np.float32),
#     'look_at': look_at,
#     'radius': radius,
#     'fovy': fovy,
#     'interval': interval
# }

# print(camera_params)


