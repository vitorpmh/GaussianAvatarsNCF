import os
import cv2
import numpy as np
from typing import Optional
def concat_ordered_images(folder_path: str, ext: str = '.png', resize_height: Optional[int] = None, skip: int = 1) -> np.ndarray:
    """
    Concatenates images in a folder ordered by zero-padded 4-digit filenames (%04d) side by side.
    Always includes the last image even if skip would skip it.

    Args:
        folder_path (str): Path to the folder containing the images.
        ext (str): File extension of the images (default: '.png').
        resize_height (Optional[int]): If specified, all images will be resized to this height (preserving aspect ratio).
        skip (int): Skip interval (e.g., 2 means use every 2nd image).

    Returns:
        np.ndarray: The final concatenated image.
    """
    # Get sorted list of %04d-style files
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(ext) and f[:4].isdigit()])

    if not filenames:
        raise ValueError("No valid image files found in the folder.")

    # Compute selected indices, ensuring the last one is included
    indices = list(range(0, len(filenames), skip))
    if indices[-1] != len(filenames) - 1:
        indices.append(len(filenames) - 1)  # Always include last image

    images = []
    for idx in indices:
        fname = filenames[idx]
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        if resize_height is not None:
            h, w = img.shape[:2]
            scale = resize_height / h
            img = cv2.resize(img, (int(w * scale), resize_height), interpolation=cv2.INTER_AREA)

        images.append(img)

    return cv2.hconcat(images)


#%%

id1 = "074"
id2 = "264"

for skip in [1,2]:
    for type in ["top", "bottom",'all']:
        try:
            result = concat_ordered_images(
                f"/home/vitor/Documents/doc/GaussianAvatars/viewer_output/{id1}-{id2}/{type}",
                resize_height=960,
                skip=2)
            cv2.imwrite(f"concatenated_{type}_{skip}.png", result)
        except:
            print(f"Failed to concatenate images for {id1}-{id2} with skip {skip} and type {type}")