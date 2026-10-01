"""Clean a YOLO segmentation dataset by removing bounding-box-only samples.

The script scans the labels folder and, whenever an annotation in plain
bounding-box format (exactly five whitespace-separated values per line:
``class x_center y_center width height``) is found, it deletes the offending
``.txt`` together with the matching image. Segmentation labels are expected
to be polygons (class id followed by an arbitrary number of ``x y`` pairs),
so any 5-value entry is taken as evidence of a misconverted sample.

The historical run that produced the current dataset reported the following
removals (kept here as a reference, since they are now committed to the
dataset):

* ``ISIC_0014833_jpg.rf.6e13ae48f44299c82717e5c9db1cf54e``
* ``ISIC_0015216_jpg.rf.6cdfb7ade14de92ba679b4fd46aac6d7``
* ``ISIC_0004337_jpg.rf.f1d77fd17316045a74d7af093a58c8d8``
* ``ISIC_0015078_jpg.rf.e70c3379f35ea5727231564dc281a504``
* ``ISIC_0013196_jpg.rf.2d8f42d40e48be199d21a86a664ec342``
* ``ISIC_0015020_jpg.rf.46d42440c0f404634a09345a23cb0a62``
* ``ISIC_0004346_jpg.rf.828c20dd3782ca643cdfb0ae23120b3b``
* ``ISIC_0015559_jpg.rf.9a0cd2a4af23e4e46ff9e2b4d1f3a1a0``

Total: 8 files removed.

Usage:
    python utils/scripts/yolo_label_cleaner.py
"""

from __future__ import annotations

import glob
import os

#: Image extensions paired with YOLO label files in this dataset.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".png", ".jpeg", ".JPG")


def remove_invalid_labels(label_dir: str, image_dir: str) -> None:
    """Remove ``(label, image)`` pairs whose label is in bounding-box format.

    Any ``.txt`` label file that contains at least one line with exactly five
    values is considered a bounding box (class + x, y, w, h) and is deleted
    together with its matching image (matched by basename across the
    extensions in :data:`IMAGE_EXTENSIONS`).

    Args:
        label_dir: Directory containing the YOLO ``.txt`` label files.
        image_dir: Directory containing the matching image files.

    Side effects:
        Deletes files from disk in both ``label_dir`` and ``image_dir``.
    """
    deleted_count = 0

    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        should_delete = False

        for line in lines:
            values = line.strip().split()

            # 5 values per line == bounding box (class, x, y, w, h).
            # Segmentation labels contain class + many polygon (x, y) pairs.
            if len(values) == 5:
                should_delete = True
                break

        if should_delete:
            os.remove(txt_file)

            base_name = os.path.splitext(os.path.basename(txt_file))[0]

            # Delete the matching image (whichever extension it uses).
            for ext in IMAGE_EXTENSIONS:
                img_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    break

            deleted_count += 1
            print(f"Removed problematic pair: {base_name}")

    print(f"\nCleanup complete. {deleted_count} files were deleted.")


def main() -> None:
    """Run :func:`remove_invalid_labels` against the default train split."""
    label_directory = "datasets/isic_2018_task1_yolo26/train/labels"
    image_directory = "datasets/isic_2018_task1_yolo26/train/images"

    print("Starting dataset scan for bounding-box annotations...\n")
    remove_invalid_labels(label_directory, image_directory)


if __name__ == "__main__":
    main()
