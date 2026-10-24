"""
Unit tests for MIRAGE Perturbation Engine
Tests:
- Output shape and uint8 dtype preservation
- Pixel value clamping within [0, 255]
- Determinism with identical seeds
- Valid behavior across all registered perturbation types
"""

import pytest
import numpy as np
from core.perturbations import (
    apply_perturbation,
    PERTURBATION_REGISTRY
)


@pytest.fixture
def sample_image():
    """Generates a standard 3-channel 100x100 test image with color patterns."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [200, 100, 50]
    img[50:, 50:] = [50, 180, 220]
    return img


def test_perturbation_registry_completeness():
    expected_keys = [
        "brightness_dark", "brightness_bright", "gaussian_blur",
        "synthetic_shadow", "patch_occlusion", "low_contrast", "high_contrast"
    ]
    for k in expected_keys:
        assert k in PERTURBATION_REGISTRY


def test_perturbations_output_shape_and_dtype(sample_image):
    for p_type in PERTURBATION_REGISTRY.keys():
        out, meta = apply_perturbation(sample_image, p_type, severity=3, seed=42)
        assert out.shape == sample_image.shape, f"{p_type} altered image shape!"
        assert out.dtype == np.uint8, f"{p_type} did not return uint8!"
        assert out.min() >= 0, f"{p_type} produced negative values!"
        assert out.max() <= 255, f"{p_type} exceeded max pixel value 255!"
        assert meta["perturbation_type"] == p_type
        assert meta["severity"] == 3


def test_perturbation_preserves_original(sample_image):
    orig_copy = sample_image.copy()
    _ = apply_perturbation(sample_image, "brightness_dark", severity=5)
    assert np.array_equal(sample_image, orig_copy), "Original image was mutated in-place!"


def test_reproducibility_with_seed(sample_image):
    # Stochastic perturbations: synthetic_shadow and patch_occlusion
    for p_type in ["synthetic_shadow", "patch_occlusion"]:
        out1, _ = apply_perturbation(sample_image, p_type, severity=3, seed=1234)
        out2, _ = apply_perturbation(sample_image, p_type, severity=3, seed=1234)
        assert np.array_equal(out1, out2), f"{p_type} was not deterministic with same seed!"

        # Different seed should yield different result
        out3, _ = apply_perturbation(sample_image, p_type, severity=3, seed=9999)
        assert not np.array_equal(out1, out3), f"{p_type} produced identical output with different seed!"


def test_invalid_perturbation_raises():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = apply_perturbation(img, "non_existent_type")
