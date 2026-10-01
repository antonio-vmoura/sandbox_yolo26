"""Convert ISIC images from RGB to the HED (Haematoxylin–Eosin–DAB) colour space.

This utility mirrors the structure of :mod:`rgb_to_lab`: it walks the source
dataset tree, copies every auxiliary file (label ``.txt``, ``data.yaml``,
``README``, etc.) unchanged and converts only the images that live inside a
folder named ``images``.

Implementation:
    1. Read the image with OpenCV (BGR ``uint8``) and convert to RGB.
    2. Apply :func:`skimage.color.rgb2hed` (Ruifrok–Johnston colour
       deconvolution), which returns a float array with three channels
       ``(H, E, D)`` in optical-density units.
    3. Per-channel min–max rescale to ``uint8 [0, 255]`` using
       :func:`skimage.exposure.rescale_intensity`, producing a viewable
       "RGB-like" image where:

           * R (cv2 BGR=2) ← H (Haematoxylin)
           * G (cv2 BGR=1) ← E (Eosin)
           * B (cv2 BGR=0) ← D (DAB)

    4. Write to disk with :func:`cv2.imwrite` (expects BGR ``uint8``).

Notes:
    * ISIC is dermoscopy, not an H&E histology slide — the original HED
      hypothesis (separating haematoxylin / eosin / DAB stains) does not
      apply literally here. The absolute channel values therefore have no
      physical meaning, but the transformation is still deterministic and
      decorrelates the channels, which can act as an alternative input
      representation for fine-tuning (similar to what we did with LAB).
    * Normalisation is per image (min–max) to preserve visual contrast. An
      alternative would be a fixed global range in ``[0, 1]`` straight out
      of ``rgb2hed``, but that typically yields very dark/over-saturated
      images for non-histological inputs.

Usage:
    python utils/scripts/rgb_to_hed.py

Paths are hard-coded inside :func:`main` for repeatability of the dataset
conversion that produced ``isic_2018_task1_yolo26_hed`` — edit them in
place if you want to re-run on a different layout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity


def convert_rgb_to_hed_uint8(img_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ``uint8`` image to a HED-encoded BGR ``uint8`` image.

    Each HED channel is independently min–max rescaled to ``[0, 255]``
    within the same image. The output is encoded as BGR so that
    :func:`cv2.imwrite` writes a sensible 3-channel image where
    ``B=DAB``, ``G=Eosin``, ``R=Haematoxylin``.

    Args:
        img_bgr: HxWx3 ``uint8`` array in OpenCV BGR order.

    Returns:
        HxWx3 ``uint8`` array in BGR order with the encoded HED channels.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hed = rgb2hed(img_rgb)  # float, shape (H, W, 3); channels = H, E, D

    h_u8 = rescale_intensity(hed[:, :, 0], out_range=(0, 255)).astype(np.uint8)
    e_u8 = rescale_intensity(hed[:, :, 1], out_range=(0, 255)).astype(np.uint8)
    d_u8 = rescale_intensity(hed[:, :, 2], out_range=(0, 255)).astype(np.uint8)

    # cv2 expects BGR -> stack (D, E, H) so that B=D, G=E, R=H.
    return np.dstack([d_u8, e_u8, h_u8])


def main() -> None:
    """Convert the full ISIC dataset tree from RGB to HED-encoded RGB.

    Side effects:
        * Mirrors the input tree under ``output_base``.
        * Every image inside an ``images`` folder is replaced by its HED
          encoding; all other files are copied verbatim with
          :func:`shutil.copy2`.
    """
    # Source and destination dataset roots
    input_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26")
    output_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26_hed")

    # Extensions we want to convert; everything else is copied as-is.
    image_extensions = {".jpg", ".jpeg", ".png"}

    print(f"Starting HED conversion...\nSource: {input_base}\nTarget: {output_base}\n")

    # Walk the full tree, replicating the directory structure under the target.
    for root, _dirs, files in os.walk(input_base):
        rel_path = Path(root).relative_to(input_base)
        dest_dir = output_base / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            # Only convert image files that live inside an "images" folder.
            if src_file.suffix.lower() in image_extensions and "images" in src_file.parts:
                img_bgr = cv2.imread(str(src_file))
                if img_bgr is not None:
                    img_hed = convert_rgb_to_hed_uint8(img_bgr)
                    cv2.imwrite(str(dest_file), img_hed)
                else:
                    print(f"Warning: could not read image {src_file}")
            else:
                # Auxiliary files (labels .txt, data.yaml, README, ...) are copied verbatim.
                shutil.copy2(src_file, dest_file)

    print("HED dataset conversion and copy completed successfully!")


if __name__ == "__main__":
    main()
