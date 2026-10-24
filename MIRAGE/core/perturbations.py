"""
MIRAGE - Perturbation Engine
Provides deterministic, reproducible, physically motivated image perturbations:
- Brightness (underexposure and overexposure)
- Blur (Gaussian blur and motion blur)
- Synthetic Shadows (polygonal shadow masks)
- Occlusion (solid patch obstruction)
- Contrast adjustments (low and high contrast)
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


PERTURBATION_REGISTRY = {
    "brightness_dark": "Dark Exposure (Underexposed)",
    "brightness_bright": "Bright Exposure (Overexposed)",
    "gaussian_blur": "Gaussian Blur",
    "synthetic_shadow": "Synthetic Shadow",
    "patch_occlusion": "Patch Occlusion",
    "low_contrast": "Low Contrast (Haze)",
    "high_contrast": "High Contrast",
}


def apply_brightness_dark(img: np.ndarray, severity: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates underexposure / dim lighting.
    Severity 1-5 maps to scale factors from 0.80 down to 0.15.
    """
    severity = max(1, min(5, int(severity)))
    # Factors: 1 -> 0.75, 2 -> 0.60, 3 -> 0.45, 4 -> 0.30, 5 -> 0.15
    factor = 0.90 - (severity * 0.15)
    factor = max(0.10, factor)
    
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, {"factor": float(factor), "severity": severity}


def apply_brightness_bright(img: np.ndarray, severity: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates overexposure / harsh glare.
    Severity 1-5 maps to scale factors from 1.25 up to 2.25.
    """
    severity = max(1, min(5, int(severity)))
    factor = 1.0 + (severity * 0.25)
    
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, {"factor": float(factor), "severity": severity}


def apply_gaussian_blur(img: np.ndarray, severity: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates lens defocus or camera motion blur.
    Severity 1-5 maps to kernel sizes [5, 11, 19, 29, 41] and sigma.
    """
    severity = max(1, min(5, int(severity)))
    k_sizes = [5, 11, 19, 29, 41]
    k = k_sizes[severity - 1]
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8 + (severity * 1.5)
    
    out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    return out, {"kernel_size": k, "sigma": float(sigma), "severity": severity}


def apply_synthetic_shadow(img: np.ndarray, severity: int = 3, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates realistic outdoor shadows using random polygonal masks with feathered edges.
    Deterministic random state ensures identical masks for the same seed.
    Severity 1-5 controls shadow opacity (darkness) and polygon coverage.
    """
    severity = max(1, min(5, int(severity)))
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    
    # Shadow darkening factor: lower means darker shadow
    # 1 -> 0.70, 2 -> 0.58, 3 -> 0.46, 4 -> 0.34, 5 -> 0.22
    shadow_ratio = max(0.15, 0.82 - (severity * 0.12))
    
    # Generate 1 to 2 random shadow polygons
    num_polys = 1 if severity <= 2 else 2
    mask = np.zeros((h, w), dtype=np.float32)
    
    polygon_points = []
    for _ in range(num_polys):
        num_vertices = rng.randint(3, 7)
        xs = rng.randint(0, w, size=num_vertices)
        ys = rng.randint(0, h, size=num_vertices)
        pts = np.stack([xs, ys], axis=1)
        polygon_points.append(pts.tolist())
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        mask = np.maximum(mask, poly_mask.astype(np.float32) / 255.0)
    
    # Feather shadow boundary using Gaussian blur to mimic soft shadow penumbra
    feather_k = max(7, (min(h, w) // 30) | 1)
    mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)
    
    # Apply darkening to the shadow area
    # pixel = pixel * (1 - mask * (1 - shadow_ratio))
    shadow_map = 1.0 - (mask * (1.0 - shadow_ratio))
    shadow_map = np.expand_dims(shadow_map, axis=2)
    
    out = (img.astype(np.float32) * shadow_map)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out, {
        "shadow_ratio": float(shadow_ratio),
        "num_polys": num_polys,
        "feather_k": feather_k,
        "seed": seed,
        "severity": severity,
    }


def apply_patch_occlusion(img: np.ndarray, severity: int = 3, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates physical obstruction / partial occlusion by placing rectangular blocking patches.
    Deterministic random state ensures identical placement for the same seed.
    Severity 1-5 scales the occlusion area from ~5% up to ~35% of total image area.
    """
    severity = max(1, min(5, int(severity)))
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    
    # Occlusion area fraction: 1 -> 0.06, 2 -> 0.12, 3 -> 0.18, 4 -> 0.25, 5 -> 0.35
    area_fraction = 0.03 + (severity * 0.065)
    total_area = h * w * area_fraction
    
    # Aspect ratio between 0.5 and 2.0
    aspect = rng.uniform(0.5, 2.0)
    occ_w = int(np.sqrt(total_area * aspect))
    occ_h = int(np.sqrt(total_area / aspect))
    
    occ_w = max(10, min(w - 10, occ_w))
    occ_h = max(10, min(h - 10, occ_h))
    
    # Pick top-left coordinate, centered biased or random
    x1 = rng.randint(0, max(1, w - occ_w))
    y1 = rng.randint(0, max(1, h - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h
    
    out = img.copy()
    # Fill occlusion patch with neutral gray or noise
    patch_color = (60, 60, 60)  # dark slate gray
    cv2.rectangle(out, (x1, y1), (x2, y2), patch_color, -1)
    
    return out, {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "area_fraction": float(area_fraction),
        "seed": seed,
        "severity": severity,
    }


def apply_low_contrast(img: np.ndarray, severity: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates atmospheric haze / low contrast foggy conditions.
    Severity 1-5 maps to contrast scaling factors from 0.80 down to 0.25.
    """
    severity = max(1, min(5, int(severity)))
    alpha = max(0.20, 0.90 - (severity * 0.14))
    beta = 128.0 * (1.0 - alpha)  # shift towards mid-gray
    
    out = (img.astype(np.float32) * alpha) + beta
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, {"alpha": float(alpha), "beta": float(beta), "severity": severity}


def apply_high_contrast(img: np.ndarray, severity: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulates harsh high-contrast lighting clipping shadows and highlights.
    Severity 1-5 maps to alpha 1.25 up to 2.25.
    """
    severity = max(1, min(5, int(severity)))
    alpha = 1.0 + (severity * 0.25)
    beta = -64.0 * (alpha - 1.0)
    
    out = (img.astype(np.float32) * alpha) + beta
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, {"alpha": float(alpha), "beta": float(beta), "severity": severity}


def apply_perturbation(
    img: np.ndarray,
    perturbation_type: str,
    severity: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified entry point for applying image perturbations.
    Guarantees:
    - Original array is not mutated.
    - Output image has identical shape and dtype uint8.
    - Values are clipped to [0, 255].
    - Detailed metadata dictionary is returned.
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid input image: cannot be None or empty.")
    
    img_copy = img.copy()
    
    if perturbation_type == "brightness_dark":
        out, params = apply_brightness_dark(img_copy, severity)
    elif perturbation_type == "brightness_bright":
        out, params = apply_brightness_bright(img_copy, severity)
    elif perturbation_type == "gaussian_blur":
        out, params = apply_gaussian_blur(img_copy, severity)
    elif perturbation_type == "synthetic_shadow":
        out, params = apply_synthetic_shadow(img_copy, severity, seed)
    elif perturbation_type == "patch_occlusion":
        out, params = apply_patch_occlusion(img_copy, severity, seed)
    elif perturbation_type == "low_contrast":
        out, params = apply_low_contrast(img_copy, severity)
    elif perturbation_type == "high_contrast":
        out, params = apply_high_contrast(img_copy, severity)
    else:
        raise ValueError(f"Unknown perturbation type: '{perturbation_type}'. Valid types: {list(PERTURBATION_REGISTRY.keys())}")
    
    metadata = {
        "perturbation_type": perturbation_type,
        "label": PERTURBATION_REGISTRY.get(perturbation_type, perturbation_type),
        "severity": int(severity),
        "seed": int(seed),
        "details": params,
    }
    return out, metadata
