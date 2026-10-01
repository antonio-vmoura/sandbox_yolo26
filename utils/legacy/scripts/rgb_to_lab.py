"""Convert ISIC dataset images from RGB to the CIE LAB colour space.

Walks the source dataset tree, copies every auxiliary file (label ``.txt``,
``data.yaml``, ``README``, etc.) unchanged and converts only the images that
live inside a folder named ``images``.

The conversion uses :func:`cv2.cvtColor` with the ``COLOR_BGR2LAB`` flag and
writes the result as a 3-channel ``uint8`` image where the L*, a* and b*
channels live in the OpenCV-encoded ``[0, 255]`` range (L scaled to
``L * 255/100``, a* and b* shifted by ``+128``). This matches what YOLO will
read when training/evaluating against the converted dataset.

Usage:
    python utils/scripts/rgb_to_lab.py

Paths are hard-coded inside :func:`main` for repeatability of the dataset
conversion that produced ``isic_2018_task1_yolo26_lab`` — edit them in
place if you want to re-run on a different layout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2


def main() -> None:
    """Convert the full ISIC dataset tree from RGB to OpenCV-encoded LAB.

    Side effects:
        * Mirrors the input tree under ``output_base``.
        * Every image inside an ``images`` folder is replaced by its LAB
          encoding; all other files are copied verbatim with
          :func:`shutil.copy2`.
    """
    # Source and destination dataset roots
    input_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26")
    output_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26_lab")

    # Extensions we want to convert; everything else is copied as-is.
    image_extensions = {".jpg", ".jpeg", ".png"}

    print(f"Starting LAB conversion...\nSource: {input_base}\nTarget: {output_base}\n")

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
                    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    cv2.imwrite(str(dest_file), img_lab)
                else:
                    print(f"Warning: could not read image {src_file}")
            else:
                # Auxiliary files (labels .txt, data.yaml, README, ...) are copied verbatim.
                shutil.copy2(src_file, dest_file)

    print("LAB dataset conversion and copy completed successfully!")


if __name__ == "__main__":
    main()
