To visualize a morphing of 3d head avatar gaussians you can use the following command

`python local_viewer_flowing.py --point_path_1 <ply_path_1> --point_path_2 <ply_path_2> --warp-file-checkpoint <warping_checkpoint_pth>`

PS: If you have a folder with multiple gaussian avatars point cloud, you can rewrite `launch_gui.py` to access them and then you have a nice GUI to select what morphing and method you want.


The viewer has a time step slider at the top, this is where you can visualize the morphing, either by sliding manually or clicking the play button.


Feel free to request a PR or create a issue.