#!/usr/bin/env python3
"""
Gemini Watermark Remover - Python Implementation

A Python port of the reverse alpha blending algorithm for removing
Gemini AI watermarks from generated images.

Based on: https://github.com/journey-ad/gemini-watermark-remover

Usage:
    python gemini_watermark_remover.py <image_path> [output_path]

Requirements:
    pip install pillow numpy
"""

import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO

# Assets directory (same folder as this script)
ASSETS_DIR = Path(__file__).parent


def load_alpha_map(size):
    """Load and calculate alpha map from watermark capture image."""
    bg_path = ASSETS_DIR / f"bg_{size}.png"
    if not bg_path.exists():
        raise FileNotFoundError(f"Alpha map file not found: {bg_path}")

    bg_img = Image.open(bg_path).convert('RGB')
    bg_array = np.array(bg_img, dtype=np.float32)

    # Take max of RGB channels and normalize to [0, 1]
    alpha_map = np.max(bg_array, axis=2) / 255.0
    return alpha_map


# Pre-computed alpha maps (lazy loaded)
_ALPHA_MAPS = {}


def get_alpha_map(size):
    """Get cached alpha map for given size."""
    if size not in _ALPHA_MAPS:
        _ALPHA_MAPS[size] = load_alpha_map(size)
    return _ALPHA_MAPS[size]


def detect_watermark_config(width, height):
    """
    Detect watermark configuration based on image dimensions.

    Gemini's watermark rules:
    - If both width and height > 1024: 96x96 logo, 64px margin
    - Otherwise: 48x48 logo, 32px margin
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin": 64}
    else:
        return {"logo_size": 48, "margin": 32}


def remove_watermark(image, verbose=False, alpha_scale=1.0):
    """
    Remove Gemini watermark from image using reverse alpha blending.

    The algorithm reverses Gemini's watermark application:
        watermarked = alpha * logo + (1 - alpha) * original
    To recover:
        original = (watermarked - alpha * logo) / (1 - alpha)

    If the result shows a faint gray logo (under-subtraction), increase alpha_scale
    above 1.0 (e.g. 1.5 or 2.0) to subtract more aggressively.

    Args:
        image:       PIL Image, file path, or bytes
        verbose:     Print debug information
        alpha_scale: Scale factor applied to the alpha map (default 1.0).
                     Increase (> 1.0) if a gray logo残留 still appears.
                     Decrease (< 1.0) if the result looks too dark in the watermark area.

    Returns:
        PIL Image with watermark removed
    """
    # Handle different input types, preserving format info for JPEG detection
    is_lossy = False
    if isinstance(image, (str, Path)):
        is_lossy = Path(image).suffix.lower() in ('.jpg', '.jpeg', '.webp')
        img = Image.open(image).convert('RGB')
    elif isinstance(image, bytes):
        raw = Image.open(BytesIO(image))
        is_lossy = getattr(raw, 'format', '').upper() in ('JPEG', 'WEBP')
        img = raw.convert('RGB')
    elif isinstance(image, Image.Image):
        is_lossy = getattr(image, 'format', '').upper() in ('JPEG', 'WEBP')
        img = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    width, height = img.size
    config = detect_watermark_config(width, height)
    logo_size = config["logo_size"]
    margin = config["margin"]

    if verbose:
        print(f"Image size: {width}x{height}")
        print(f"Watermark config: {logo_size}x{logo_size}, margin={margin}px")
        print(f"Lossy source (JPEG/WebP): {is_lossy}, alpha_scale: {alpha_scale}")

    # Calculate watermark position (bottom-right corner)
    x = width - margin - logo_size
    y = height - margin - logo_size

    if x < 0 or y < 0:
        if verbose:
            print("Image too small for watermark, returning unchanged")
        return img

    if verbose:
        print(f"Watermark position: ({x}, {y})")

    # Load alpha map
    alpha_map = get_alpha_map(logo_size)

    # Use float64 for better numerical precision
    img_array = np.array(img, dtype=np.float64)

    # Constants
    ALPHA_THRESHOLD = 0.002  # Ignore near-zero alpha (background noise)
    MAX_ALPHA = 0.99         # Cap to avoid division by zero
    LOGO_VALUE = 255.0       # Gemini watermark is white

    # Apply alpha_scale: scaling up makes the formula subtract more aggressively,
    # which fixes gray-logo artifacts caused by the bg_*.png alpha being too low
    # relative to the actual watermark opacity in a given image.
    alpha_clipped = np.clip(alpha_map * alpha_scale, 0.0, MAX_ALPHA)
    valid_mask = alpha_clipped > ALPHA_THRESHOLD  # where watermark is present

    region = img_array[y:y + logo_size, x:x + logo_size]  # view into img_array
    alpha3 = alpha_clipped[:, :, np.newaxis]               # (H, W, 1) for broadcasting

    # Vectorized reverse alpha blending for all valid pixels
    recovered = np.where(
        valid_mask[:, :, np.newaxis],
        (region - alpha3 * LOGO_VALUE) / (1.0 - alpha3),
        region,
    )
    recovered = np.clip(recovered, 0.0, 255.0)

    if verbose:
        print(f"Pixels processed by formula: {int(np.sum(valid_mask))}")

    # For JPEG/WebP: apply a 3×3 median filter to the watermark region.
    # JPEG quantization introduces random noise; after the reverse formula divides
    # by (1 - alpha), that noise is amplified ~2× — a median pass removes it
    # without blurring the surrounding image.
    if is_lossy and np.any(valid_mask):
        img_array[y:y + logo_size, x:x + logo_size] = np.clip(np.round(recovered), 0, 255)
        wm_crop = Image.fromarray(
            img_array[y:y + logo_size, x:x + logo_size].astype(np.uint8)
        )
        wm_crop = wm_crop.filter(ImageFilter.MedianFilter(size=3))
        img_array[y:y + logo_size, x:x + logo_size] = np.array(wm_crop, dtype=np.float64)
        if verbose:
            print("Applied 3×3 median denoising to watermark region (JPEG mode).")
    else:
        img_array[y:y + logo_size, x:x + logo_size] = np.clip(np.round(recovered), 0, 255)

    if verbose:
        print("Done.")

    return Image.fromarray(img_array.astype(np.uint8), 'RGB')


def remove_watermark_bytes(image_bytes, output_format='PNG', quality=95):
    """
    Remove watermark and return as bytes.

    Args:
        image_bytes: Input image as bytes
        output_format: Output format (PNG, JPEG, etc.)
        quality: JPEG quality (1-100)

    Returns:
        Processed image as bytes
    """
    result = remove_watermark(image_bytes)
    output = BytesIO()

    save_kwargs = {"format": output_format}
    if output_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    result.save(output, **save_kwargs)
    return output.getvalue()


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = image_path.parent / f"{image_path.stem}_clean{image_path.suffix}"

    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")

    result = remove_watermark(image_path, verbose=True)
    result.save(output_path)

    print(f"Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
