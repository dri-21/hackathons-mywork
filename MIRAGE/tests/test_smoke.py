"""
Smoke test for MIRAGE Object Detector and Dataset Pipeline
Verifies:
- Detector initialization and real smoke inference
- Dataset preparation, loading, and annotation integrity
- End-to-end baseline evaluation on a real dataset sample
"""

import os
import pytest
import numpy as np
from core.detector import ObjectDetector
from core.dataset import setup_dataset, DatasetReport
from core.evaluator import evaluate_single_image
from core.perturbations import apply_perturbation


def test_detector_smoke_test():
    detector = ObjectDetector(model_name="yolo11n.pt", device="cpu")
    smoke_result = detector.smoke_test()
    
    assert smoke_result["success"] is True, f"Smoke test failed: {smoke_result.get('error')}"
    assert smoke_result["classes_count"] == 80
    assert smoke_result["device"] == "cpu"
    assert "smoke_latency_ms" in smoke_result


def test_dataset_setup_and_validation(tmp_path):
    # Setup dataset in temporary path
    report = setup_dataset(data_dir=str(tmp_path), preferred_dataset="coco8")
    
    assert isinstance(report, DatasetReport)
    assert report.valid_images > 0, "No valid images found in dataset!"
    assert report.total_ground_truth_boxes > 0, "No ground truth annotations found!"
    
    sample = report.samples[0]
    assert os.path.exists(sample.image_path)
    assert sample.width > 0 and sample.height > 0
    assert len(sample.ground_truths) > 0
    for gt in sample.ground_truths:
        assert gt.bbox[2] > gt.bbox[0]
        assert gt.bbox[3] > gt.bbox[1]


def test_end_to_end_pipeline_single_sample(tmp_path):
    report = setup_dataset(data_dir=str(tmp_path), preferred_dataset="coco8")
    sample = report.samples[0]
    
    detector = ObjectDetector(model_name="yolo11n.pt", device="cpu")
    
    # 1. Baseline prediction
    baseline_preds = detector.predict(sample.image_path, confidence_threshold=0.25)
    base_eval = evaluate_single_image(
        sample.image_id, sample.ground_truths, baseline_preds, iou_threshold=0.5
    )
    
    assert 0.0 <= base_eval.recall <= 1.0
    assert 0.0 <= base_eval.precision <= 1.0
    
    # 2. Perturbation
    import cv2
    img = cv2.imread(sample.image_path)
    pert_img, meta = apply_perturbation(img, "gaussian_blur", severity=4)
    
    # 3. Perturbed prediction
    pert_preds = detector.predict(pert_img, confidence_threshold=0.25)
    pert_eval = evaluate_single_image(
        sample.image_id, sample.ground_truths, pert_preds, iou_threshold=0.5
    )
    
    assert 0.0 <= pert_eval.recall <= 1.0
