"""Generate publication-ready figures illustrating the dermoscopy CV pipeline.

The script walks a single lesion (``ISIC_0000052``) through the four
fundamental computational tasks in skin-cancer image analysis and emits a
separate PNG for each. All outputs share the input image dimensions so they
can be laid out uniformly inside a LaTeX figure.

Outputs (written to ``./outputs/``):

    1. ``out_lesion_segmentation.png``       (lesion mask overlay)
    2. ``out_attribute_segmentation.png``    (dermoscopic-attribute masks)
    3. ``out_binary_classification.png``     (benign vs. malignant banner)
    4. ``out_multiclass_classification.png`` (3-class probability panel)

Usage:
    python utils/scripts/generate_pipeline_outputs.py

Inputs are expected under ``utils/scripts/inputs/`` with the naming pattern
``ISIC_0000052*.png/.jpg`` (see :data:`ATTRIBUTES` for the full list).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "inputs"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LESION_ID = "ISIC_0000052"
ORIGINAL_PATH = IN_DIR / f"{LESION_ID}.jpg"
LESION_MASK_PATH = IN_DIR / f"{LESION_ID}_segmentation.png"

# Order chosen for consistent legend colors across figures / papers.
ATTRIBUTES = [
    ("Pigment network", IN_DIR / f"{LESION_ID}_attribute_pigment_network.png",
     (255, 0, 0)),       # red
    ("Globules",        IN_DIR / f"{LESION_ID}_attribute_globules.png",
     (255, 255, 0)),     # yellow
    ("Streaks",         IN_DIR / f"{LESION_ID}_attribute_streaks.png",
     (0, 255, 255)),     # cyan
    ("Milia-like cyst", IN_DIR / f"{LESION_ID}_attribute_milia_like_cyst.png",
     (255, 0, 255)),     # magenta
    ("Negative network", IN_DIR / f"{LESION_ID}_attribute_negative_network.png",
     (0, 255, 0)),       # green
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_rgb(path: Path) -> np.ndarray:
    """Read an image from disk as an RGB ``np.ndarray``.

    Args:
        path: Path to the image (any format OpenCV can decode).

    Returns:
        HxWx3 ``uint8`` RGB array.

    Raises:
        FileNotFoundError: If ``path`` cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Read a binary segmentation mask.

    Args:
        path: Path to a single-channel PNG.

    Returns:
        HxW ``uint8`` array in {0, 1} (threshold = 127).

    Raises:
        FileNotFoundError: If ``path`` cannot be read.
    """
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def find_font(size: int) -> ImageFont.FreeTypeFont:
    """Return the first available system font, falling back to the default.

    Args:
        size: Font size in pixels.

    Returns:
        A ``PIL.ImageFont`` instance.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def overlay_color(image: np.ndarray, mask: np.ndarray,
                  color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Alpha-blend a flat colour onto ``image`` wherever ``mask == 1``.

    Args:
        image: HxWx3 ``uint8`` RGB image.
        mask: HxW binary mask in {0, 1}.
        color: Target colour as an ``(r, g, b)`` triple.
        alpha: Blend amount in ``[0, 1]``.

    Returns:
        New HxWx3 ``uint8`` image (input is not mutated).
    """
    out = image.copy()
    if mask.sum() == 0:
        return out
    layer = np.zeros_like(image)
    layer[mask == 1] = color
    sel = mask.astype(bool)
    out[sel] = ((1 - alpha) * image[sel] + alpha * layer[sel]).astype(np.uint8)
    return out


def draw_contour(image: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """Draw mask contours on a copy of ``image``.

    Args:
        image: HxWx3 ``uint8`` RGB image.
        mask: HxW binary mask.
        color: Contour colour ``(r, g, b)``.
        thickness: Line thickness in pixels.

    Returns:
        New HxWx3 ``uint8`` image with the contours drawn.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return out


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    radius: int,
    fill,
    outline=None,
    width: int = 1,
) -> None:
    """Thin wrapper around :meth:`PIL.ImageDraw.ImageDraw.rounded_rectangle`."""
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline,
                           width=width)


def paste_translucent_panel(
    pil_img: Image.Image,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int, int],
    radius: int,
) -> None:
    """Draw a rounded translucent panel directly onto ``pil_img``.

    Args:
        pil_img: Target PIL image (must be RGBA-composable).
        box: Panel bounding box ``(x0, y0, x1, y1)``.
        fill: RGBA tuple defining the panel colour and opacity.
        radius: Corner radius in pixels.
    """
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    rounded_rect(d, box, radius=radius, fill=fill)
    pil_img.alpha_composite(overlay)


# --------------------------------------------------------------------------- #
# 1. Lesion segmentation
# --------------------------------------------------------------------------- #
def make_lesion_segmentation(image: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """Render the lesion-segmentation figure (mask overlay + contour + label).

    Args:
        image: HxWx3 ``uint8`` RGB image.
        lesion_mask: HxW binary mask of the lesion.

    Returns:
        New HxWx3 ``uint8`` RGB array with the overlay applied.
    """
    overlay_rgb = (0, 200, 255)        # cyan-blue fill
    contour_rgb = (0, 120, 255)        # solid blue outline

    out = overlay_color(image, lesion_mask, overlay_rgb, alpha=0.35)
    out = draw_contour(out, lesion_mask, contour_rgb, thickness=6)

    # corner label
    pil = Image.fromarray(out).convert("RGBA")
    H, W = out.shape[:2]
    pad = int(W * 0.012)
    font = find_font(int(W * 0.022))
    text = "Lesion mask"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box_w, box_h = tw + 2 * pad, th + int(pad * 1.4)
    x0, y0 = pad, pad
    paste_translucent_panel(pil, (x0, y0, x0 + box_w, y0 + box_h),
                            fill=(0, 60, 110, 190), radius=int(box_h * 0.4))
    d = ImageDraw.Draw(pil)
    d.text((x0 + pad, y0 + int(pad * 0.5) - bbox[1]), text,
           fill=(255, 255, 255, 255), font=font)
    # color swatch
    sw_r = int(th * 0.45)
    cx, cy = x0 + box_w - pad - sw_r, y0 + box_h // 2
    # actually put swatch on the left, push text right
    return np.array(pil.convert("RGB"))


# --------------------------------------------------------------------------- #
# 2. Attribute segmentation
# --------------------------------------------------------------------------- #
def make_attribute_segmentation(
    image: np.ndarray,
    attrs: list[tuple[str, np.ndarray, Tuple[int, int, int]]],
) -> np.ndarray:
    """Render the dermoscopic-attribute figure (one overlay per attribute).

    Args:
        image: HxWx3 ``uint8`` RGB image.
        attrs: List of ``(name, mask, color)`` tuples for each attribute.
            Only attributes with a non-empty mask appear in the legend.

    Returns:
        New HxWx3 ``uint8`` RGB image.
    """
    out = image.copy()
    for _, mask, color in attrs:
        if mask.sum() == 0:
            continue
        out = overlay_color(out, mask, color, alpha=0.55)
    for _, mask, color in attrs:
        if mask.sum() == 0:
            continue
        out = draw_contour(out, mask, color, thickness=3)

    # legend (only attributes actually present in this lesion)
    present_attrs = [(name, mask, color) for name, mask, color in attrs
                     if mask.sum() > 0]

    pil = Image.fromarray(out).convert("RGBA")
    H, W = out.shape[:2]
    font = find_font(int(W * 0.018))
    row_h = int(W * 0.030)
    pad = int(W * 0.012)
    swatch = int(row_h * 0.55)

    if present_attrs:
        widest = max(font.getbbox(name)[2] for name, _, _ in present_attrs)
        legend_w = swatch + pad + widest + 2 * pad
        legend_h = row_h * len(present_attrs) + 2 * pad
        x0 = pad
        y0 = pad
        paste_translucent_panel(pil, (x0, y0, x0 + legend_w, y0 + legend_h),
                                fill=(0, 0, 0, 170), radius=int(pad * 0.8))

        d = ImageDraw.Draw(pil)
        for i, (name, _mask, color) in enumerate(present_attrs):
            yy = y0 + pad + i * row_h
            sx = x0 + pad
            d.rounded_rectangle(
                (sx, yy + (row_h - swatch) // 2,
                 sx + swatch, yy + (row_h - swatch) // 2 + swatch),
                radius=int(swatch * 0.25),
                fill=(*color, 255),
            )
            d.text((sx + swatch + pad, yy + (row_h - swatch) // 2 - 2),
                   name, fill=(255, 255, 255, 255), font=font)

    return np.array(pil.convert("RGB"))


# --------------------------------------------------------------------------- #
# 3. Binary classification banner
# --------------------------------------------------------------------------- #
def make_binary_classification(
    image: np.ndarray,
    label: str = "Malignant",
    confidence: float = 0.93,
) -> np.ndarray:
    """Render the binary-classification figure (label + confidence banner).

    Args:
        image: HxWx3 ``uint8`` RGB image.
        label: Predicted class string. ``"Malignant"`` selects a crimson
            accent; any other value selects green.
        confidence: Softmax confidence in ``[0, 1]``.

    Returns:
        New HxWx3 ``uint8`` RGB image.
    """
    pil = Image.fromarray(image).convert("RGBA")
    H, W = image.shape[:2]

    banner_h = int(H * 0.11)
    margin = int(W * 0.025)
    box = (margin, H - banner_h - margin, W - margin, H - margin)

    # color depending on label
    if label.lower().startswith("mal"):
        accent = (198, 40, 40, 255)    # crimson
    else:
        accent = (46, 125, 50, 255)    # green
    paste_translucent_panel(pil, box, fill=(15, 15, 15, 215),
                            radius=int(banner_h * 0.35))

    d = ImageDraw.Draw(pil)

    # left accent stripe
    stripe_w = int(banner_h * 0.12)
    d.rounded_rectangle(
        (box[0] + int(banner_h * 0.20),
         box[1] + int(banner_h * 0.20),
         box[0] + int(banner_h * 0.20) + stripe_w,
         box[3] - int(banner_h * 0.20)),
        radius=stripe_w // 2,
        fill=accent,
    )

    value_font = find_font(int(banner_h * 0.55))
    conf_font = find_font(int(banner_h * 0.34))

    text_x = box[0] + int(banner_h * 0.20) + stripe_w + int(banner_h * 0.35)

    # vertically center the class label
    vbbox = value_font.getbbox(label)
    vth = vbbox[3] - vbbox[1]
    d.text((text_x, box[1] + (banner_h - vth) // 2 - vbbox[1]),
           label, fill=(*accent[:3], 255), font=value_font)

    # right-aligned confidence
    conf_text = f"Confidence  {confidence * 100:.1f}%"
    bbox = conf_font.getbbox(conf_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    d.text((box[2] - tw - int(banner_h * 0.35),
            box[1] + (banner_h - th) // 2 - bbox[1]),
           conf_text, fill=(230, 230, 230, 255), font=conf_font)

    return np.array(pil.convert("RGB"))


# --------------------------------------------------------------------------- #
# 4. Multi-class classification panel
# --------------------------------------------------------------------------- #
def make_multiclass_classification(image: np.ndarray) -> np.ndarray:
    """Render the multi-class classification panel (3 illustrative probs).

    Args:
        image: HxWx3 ``uint8`` RGB image used as the canvas.

    Returns:
        New HxWx3 ``uint8`` RGB image.
    """
    # Skin-cancer 3-class set (illustrative probabilities)
    classes = [
        ("Melanoma", 0.78),
        ("BCC",      0.15),
        ("SCC",      0.07),
    ]
    pil = Image.fromarray(image).convert("RGBA")
    H, W = image.shape[:2]

    panel_h = int(H * 0.26)
    margin = int(W * 0.025)
    box = (margin, H - panel_h - margin, W - margin, H - margin)
    paste_translucent_panel(pil, box, fill=(15, 15, 15, 215),
                            radius=int(panel_h * 0.06))

    d = ImageDraw.Draw(pil)

    title_font = find_font(int(panel_h * 0.14))
    class_font = find_font(int(panel_h * 0.115))
    pct_font = find_font(int(panel_h * 0.115))

    title = "MULTI-CLASS PROBABILITY"
    d.text((box[0] + int(panel_h * 0.10),
            box[1] + int(panel_h * 0.06)),
           title, fill=(220, 220, 220, 255), font=title_font)

    # bar geometry
    n = len(classes)
    rows_top = box[1] + int(panel_h * 0.32)
    rows_bottom = box[3] - int(panel_h * 0.07)
    row_gap = (rows_bottom - rows_top) / n
    bar_h = int(row_gap * 0.50)

    label_col_w = int((box[2] - box[0]) * 0.20)
    pct_col_w = int((box[2] - box[0]) * 0.09)
    bar_x0 = box[0] + int(panel_h * 0.10) + label_col_w
    bar_x1 = box[2] - int(panel_h * 0.10) - pct_col_w

    for i, (short, prob) in enumerate(classes):
        cy = int(rows_top + i * row_gap + row_gap / 2)
        # label
        d.text((box[0] + int(panel_h * 0.10), cy - int(bar_h * 0.55)),
               short, fill=(235, 235, 235, 255), font=class_font)
        # bar background
        d.rounded_rectangle(
            (bar_x0, cy - bar_h // 2, bar_x1, cy + bar_h // 2),
            radius=bar_h // 2,
            fill=(60, 60, 60, 255),
        )
        # filled portion
        fill_w = int((bar_x1 - bar_x0) * prob)
        if fill_w > bar_h:  # only draw if meaningfully wide
            color = (255, 90, 90, 255) if i == 0 else (90, 170, 255, 255)
            d.rounded_rectangle(
                (bar_x0, cy - bar_h // 2, bar_x0 + fill_w, cy + bar_h // 2),
                radius=bar_h // 2,
                fill=color,
            )
        else:
            color = (255, 90, 90, 255) if i == 0 else (90, 170, 255, 255)
            d.ellipse(
                (bar_x0, cy - bar_h // 2,
                 bar_x0 + bar_h, cy + bar_h // 2),
                fill=color,
            )
        # percentage
        pct_text = f"{prob * 100:.0f}%"
        bbox = pct_font.getbbox(pct_text)
        tw = bbox[2] - bbox[0]
        d.text((box[2] - int(panel_h * 0.10) - tw,
                cy - int(bar_h * 0.55)),
               pct_text, fill=(235, 235, 235, 255), font=pct_font)

    return np.array(pil.convert("RGB"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Generate all four figures and write them under ``OUT_DIR``."""
    image = load_rgb(ORIGINAL_PATH)
    lesion_mask = load_mask(LESION_MASK_PATH)
    attrs = [(name, load_mask(p), color) for name, p, color in ATTRIBUTES]

    out1 = make_lesion_segmentation(image, lesion_mask)
    out2 = make_attribute_segmentation(image, attrs)
    out3 = make_binary_classification(image, label="Malignant",
                                      confidence=0.93)
    out4 = make_multiclass_classification(image)

    for name, arr in [
        ("out_lesion_segmentation.png",     out1),
        ("out_attribute_segmentation.png",  out2),
        ("out_binary_classification.png",   out3),
        ("out_multiclass_classification.png", out4),
    ]:
        path = OUT_DIR / name
        Image.fromarray(arr).save(path, optimize=True)
        print(f"wrote {path}  shape={arr.shape}")


if __name__ == "__main__":
    main()
