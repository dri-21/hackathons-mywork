"""
Unit tests for MIRAGE Replay and Serialization Engine
Tests:
- Exporting BenchmarkSummary to JSON manifest
- Loading and validating manifest format
- Deterministic replay verification logic
"""

import json
import os
import pytest
from core.evaluator import (
    GroundTruthBox,
    PredictionBox,
    evaluate_single_image,
    ComparisonEvaluation,
    aggregate_comparisons
)
from core.replay import (
    export_experiment_report,
    load_experiment_manifest,
    replay_experiment
)


def test_export_and_load_manifest(tmp_path):
    gt = GroundTruthBox(2, "car", (20, 20, 80, 80), id="c1")
    pred = PredictionBox(2, "car", 0.92, (22, 22, 78, 78))
    
    base_eval = evaluate_single_image("test_img_01", [gt], [pred])
    pert_eval = evaluate_single_image("test_img_01", [gt], [])
    
    comp = ComparisonEvaluation("test_img_01", base_eval, pert_eval)
    summary = aggregate_comparisons([comp])
    
    model_info = {"model_name": "yolo11n.pt", "device": "cpu"}
    dataset_info = {"source_name": "COCO8", "license_info": "CC BY 4.0", "is_fallback": False}
    pert_info = {"perturbation_type": "gaussian_blur", "severity": 3, "seed": 42}
    
    manifest, file_path = export_experiment_report(
        summary, model_info, dataset_info, pert_info, output_dir=str(tmp_path)
    )
    
    assert os.path.exists(file_path)
    
    loaded = load_experiment_manifest(file_path)
    assert loaded["model"]["name"] == "yolo11n.pt"
    assert loaded["perturbation"]["type"] == "gaussian_blur"
    assert loaded["metrics_summary"]["baseline"]["tp"] == 1
    assert loaded["metrics_summary"]["perturbed"]["tp"] == 0
    assert loaded["metrics_summary"]["total_regressions"] == 1
