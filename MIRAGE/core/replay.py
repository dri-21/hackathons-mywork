"""
MIRAGE - Experiment Replay & Reporting Engine
Saves complete, self-contained experiment manifests to reproducible JSON,
and validates re-execution against saved benchmarks:
- Compact, non-binary JSON storage
- Full provenance (model, dataset, thresholds, perturbation seed, software versions)
- Automatic drift/reproducibility verification on replay
"""

import os
import json
import time
import sys
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union, List

import cv2
import numpy as np

from core.evaluator import (
    BenchmarkSummary,
    evaluate_single_image,
    ComparisonEvaluation,
    aggregate_comparisons,
    GroundTruthBox,
    PredictionBox
)
from core.perturbations import apply_perturbation


def get_environment_info() -> Dict[str, str]:
    """Captures runtime software versions for reproducibility auditing."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "unavailable"
        
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = "unavailable"
        
    return info


def export_experiment_report(
    summary: BenchmarkSummary,
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    perturbation_info: Dict[str, Any],
    diagnostic_info: Optional[Dict[str, Any]] = None,
    output_dir: str = "reports"
) -> Tuple[Dict[str, Any], str]:
    """
    Serializes a complete experiment benchmark run into a structured JSON manifest.
    Does not serialize raw pixel arrays; references compact bounding boxes and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    p_type = perturbation_info.get("perturbation_type", "unknown")
    sev = perturbation_info.get("severity", 1)
    
    per_image_data: List[Dict[str, Any]] = []
    image_ids: List[str] = []
    
    for comp in summary.comparisons:
        image_ids.append(comp.image_id)
        
        per_image_data.append({
            "image_id": comp.image_id,
            "ground_truths": [
                {"class_name": gt.class_name, "class_id": gt.class_id, "bbox": [round(c, 2) for c in gt.bbox], "size": gt.size_category}
                for gt in comp.baseline.ground_truths
            ],
            "baseline_predictions": [
                {"class_name": p.class_name, "class_id": p.class_id, "confidence": round(p.confidence, 4), "bbox": [round(c, 2) for c in p.bbox]}
                for p in comp.baseline.predictions
            ],
            "perturbed_predictions": [
                {"class_name": p.class_name, "class_id": p.class_id, "confidence": round(p.confidence, 4), "bbox": [round(c, 2) for c in p.bbox]}
                for p in comp.perturbed.predictions
            ],
            "baseline_recall": round(comp.baseline.recall, 4),
            "perturbed_recall": round(comp.perturbed.recall, 4),
            "delta_recall": round(comp.delta_recall, 4),
            "regressions_count": len(comp.regressions),
            "regressed_classes": [r.class_name for r in comp.regressions],
            "false_alarms_count": len(comp.new_false_alarms),
        })
        
    manifest = {
        "manifest_version": "1.0.0",
        "timestamp": timestamp,
        "environment": get_environment_info(),
        "model": {
            "name": model_info.get("model_name", "unknown"),
            "device": model_info.get("device", "cpu"),
            "weights_dir": model_info.get("weights_dir", ""),
        },
        "dataset": {
            "source_name": dataset_info.get("source_name", "unknown"),
            "license": dataset_info.get("license_info", "unknown"),
            "is_fallback": dataset_info.get("is_fallback", False),
            "evaluated_images_count": len(summary.comparisons),
            "image_ids": image_ids,
        },
        "evaluation_parameters": {
            "iou_threshold": summary.iou_threshold,
            "confidence_threshold": summary.confidence_threshold,
        },
        "perturbation": {
            "type": p_type,
            "label": perturbation_info.get("label", p_type),
            "severity": sev,
            "seed": perturbation_info.get("seed", 42),
            "details": perturbation_info.get("details", {}),
        },
        "metrics_summary": summary.to_dict(),
        "diagnostics": diagnostic_info or {},
        "per_image_results": per_image_data,
    }
    
    clean_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mirage_report_{p_type}_sev{sev}_{clean_timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest, file_path


def load_experiment_manifest(source: Union[str, bytes]) -> Dict[str, Any]:
    """
    Parses and validates a saved experiment JSON manifest.
    """
    if isinstance(source, bytes):
        manifest = json.loads(source.decode("utf-8"))
    elif os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = json.loads(source)
        
    required_keys = ["model", "dataset", "evaluation_parameters", "perturbation", "metrics_summary"]
    for k in required_keys:
        if k not in manifest:
            raise ValueError(f"Invalid experiment manifest: missing required section '{k}'.")
            
    return manifest


def replay_experiment(
    manifest: Dict[str, Any],
    detector: Any,
    dataset_samples: List[Any]
) -> Dict[str, Any]:
    """
    Re-executes an experiment using the exact configuration stored in a saved manifest.
    Compares original metrics with replayed metrics to verify complete deterministic reproducibility.
    """
    eval_params = manifest.get("evaluation_parameters", {})
    iou_thr = float(eval_params.get("iou_threshold", 0.50))
    conf_thr = float(eval_params.get("confidence_threshold", 0.25))
    
    pert_info = manifest.get("perturbation", {})
    p_type = pert_info.get("type", "gaussian_blur")
    severity = int(pert_info.get("severity", 3))
    seed = int(pert_info.get("seed", 42))
    
    saved_img_ids = set(manifest.get("dataset", {}).get("image_ids", []))
    matched_samples = [s for s in dataset_samples if s.image_id in saved_img_ids]
    
    if len(matched_samples) == 0:
        return {
            "status": "FAILED",
            "reason": "None of the original images from the saved manifest were found in the current dataset.",
            "replayed_count": 0,
            "target_count": len(saved_img_ids),
        }
        
    comparisons: List[ComparisonEvaluation] = []
    
    for sample in matched_samples:
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            continue
            
        # 1. Baseline inference
        base_preds = detector.predict(img_bgr, confidence_threshold=conf_thr, iou_threshold=iou_thr)
        base_eval = evaluate_single_image(
            sample.image_id, sample.ground_truths, base_preds, iou_thr, conf_thr
        )
        
        # 2. Perturbation with exact seed
        pert_img, _ = apply_perturbation(img_bgr, p_type, severity=severity, seed=seed)
        
        # 3. Perturbed inference
        pert_preds = detector.predict(pert_img, confidence_threshold=conf_thr, iou_threshold=iou_thr)
        pert_eval = evaluate_single_image(
            sample.image_id, sample.ground_truths, pert_preds, iou_thr, conf_thr
        )
        
        comp = ComparisonEvaluation(
            image_id=sample.image_id,
            baseline=base_eval,
            perturbed=pert_eval
        )
        comparisons.append(comp)
        
    replayed_summary = aggregate_comparisons(comparisons)
    
    # Compare with saved original metrics
    saved_metrics = manifest.get("metrics_summary", {})
    saved_base_rec = saved_metrics.get("baseline", {}).get("recall", 0.0)
    saved_pert_rec = saved_metrics.get("perturbed", {}).get("recall", 0.0)
    saved_reg = saved_metrics.get("total_regressions", 0)
    
    rec_diff = abs(replayed_summary.perturbed_recall - saved_pert_rec)
    reg_diff = abs(replayed_summary.total_regressions - saved_reg)
    
    is_exact_match = (rec_diff < 1e-4 and reg_diff == 0)
    
    return {
        "status": "SUCCESS",
        "match_verdict": "EXACT_REPRODUCIBILITY" if is_exact_match else "DIVERGENCE_DETECTED",
        "is_exact_match": is_exact_match,
        "replayed_images_count": len(comparisons),
        "target_images_count": len(saved_img_ids),
        "original_baseline_recall": saved_base_rec,
        "replayed_baseline_recall": round(replayed_summary.baseline_recall, 4),
        "original_perturbed_recall": saved_pert_rec,
        "replayed_perturbed_recall": round(replayed_summary.perturbed_recall, 4),
        "original_regressions": saved_reg,
        "replayed_regressions": replayed_summary.total_regressions,
        "replayed_summary": replayed_summary.to_dict(),
    }
