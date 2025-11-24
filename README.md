This repository is a refactored version of the [Gaussian Avatars splat viewer](https://github.com/ShenhanQian/GaussianAvatars), adapted for **3D Gaussian Splatting (3DGS) morphing**.
Follow the [Original Flow](FORK_README.md) to create your avatars.

To return to our original project git page, visit: [FLOWING](https://schardong.github.io/flowing/)

---

### 🔄 Convert Gaussian Avatars to the Standard 3DGS Representation

Our method assumes the **standard 3DGS `.ply` format**.
To convert a trained Gaussian Avatar into the traditional 3DGS representation, run:

```bash
python convert_representation.py \
    --point_path output/UNION10EMOEXP_104_eval_600k/point_cloud/iteration_600000/point_cloud.ply \
    --point_output_path test_104/
```

You can apply this conversion to any subject you want.

---

### 🎭 Visualizing a Morphing

You will have to create the python venv of gaussian avatars, and then install the flowing package inside it.

To visualize a morph between two 3D head Gaussian avatars, use:

```bash
python local_viewer_flowing.py \
    --point_path_1 <ply_path_1> \
    --point_path_2 <ply_path_2> \
    --warp-file-checkpoint <warping_checkpoint.pth>
```

**Tip:**
If you have a folder containing multiple Gaussian Avatar point clouds, you can modify `launch_gui.py` to load them automatically. This gives you a convenient GUI to select the avatars and morphing method.

The viewer includes a **time-step slider** at the top, allowing you to explore the morphing by dragging manually or pressing the play button.

---

### 🤝 Contributions

Feel free to open a PR or create an issue if you encounter problems or want to suggest improvements.

