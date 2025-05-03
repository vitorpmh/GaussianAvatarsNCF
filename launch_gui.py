import tkinter as tk
from tkinter import ttk
import subprocess

ids = ["074", "104", "218", "253", "264", "302", "304", "306"]
modes = {
    "Sel-LMK": "selected_lmk",
    "MESH": "mesh"
}
# do not call me racist please
# 074 - "indian" guy
# 104 - white guy with big jawline
# 218 - bald
# 253 - jake peralta
# 264 - indian woman
# 302 - white woman
# 304 - white old woman
# 306 - example guy (semi - bald)

def run_viewer():
    id1 = var1.get()
    id2 = var2.get()
    mode = mode_var.get()  # either 'selected_lmk' or 'mesh'

    point_path_1 = f"/home/vitor/Documents/doc/GaussianAvatars/viewer_output/{id1}/point_cloud.ply"
    point_path_2 = f"/home/vitor/Documents/doc/GaussianAvatars/viewer_output/{id2}/point_cloud.ply"
    if mode == "selected_lmk":
        warp_checkpoint = f"/home/vitor/Documents/doc/conjugate-morphing/results_ncf/3d_experiments_gaussian_avatar_NCF_selected_lmk_{id1}_{id2}/best.pth"
    elif mode == "mesh":
        warp_checkpoint = f"/home/vitor/Documents/doc/conjugate-morphing/results_NCF_MESH/3d_experiments_gaussian_avatar_NCF_mesh_{id1}_{id2}/best.pth"
    
    command = [
        "/home/vitor/anaconda3/envs/gaussian-avatars/bin/python",
        "local_viewer_ncf.py",
        "--point_path_1", point_path_1,
        "--point_path_2", point_path_2,
        "--warp-file-checkpoint", warp_checkpoint
    ]

    subprocess.run(command)

# GUI setup
root = tk.Tk()
root.title("Select Point Clouds and Mode")

# ID selectors
tk.Label(root, text="First ID:").grid(row=0, column=0, padx=10, pady=5)
var1 = tk.StringVar(value=ids[0])
dropdown1 = ttk.Combobox(root, textvariable=var1, values=ids, state="readonly")
dropdown1.grid(row=0, column=1)

tk.Label(root, text="Second ID:").grid(row=1, column=0, padx=10, pady=5)
var2 = tk.StringVar(value=ids[1])
dropdown2 = ttk.Combobox(root, textvariable=var2, values=ids, state="readonly")
dropdown2.grid(row=1, column=1)

# Mode selector
tk.Label(root, text="Mode:").grid(row=2, column=0, padx=10, pady=5)
mode_var = tk.StringVar(value="selected_lmk")
mode_frame = tk.Frame(root)
mode_frame.grid(row=2, column=1)

for mode_label, mode_value in modes.items():
    tk.Radiobutton(mode_frame, text=mode_label, variable=mode_var, value=mode_value).pack(side=tk.LEFT)

# Run button
run_button = tk.Button(root, text="Run Viewer", command=run_viewer)
run_button.grid(row=3, column=0, columnspan=2, pady=15)

root.mainloop()
