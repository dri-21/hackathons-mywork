"""
MIRAGE - Evaluation Engine
Implements mathematically rigorous, deterministic evaluation of object detections:
- Accurate IoU calculation (xyxy format)
- Ground-truth to prediction one-to-one matching with configurable IoU threshold
- Strict classification into Matched (TP), Missed GT (FN), and Unmatched Prediction (FP)
- Comprehensive baseline vs perturbed delta comparison
- Breakdown by object class and physical bounding-box scale
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    Handles degenerate boxes safely.
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    
    # Check for invalid coordinates
    if x2_a <= x1_a or y2_a <= y1_a or x2_b <= x1_b or y2_b <= y1_b:
        return 0.0
    
    # Intersection coordinates
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    if inter_area <= 0.0:
        return 0.0
    
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    
    return float(inter_area / union_area)


@dataclass
class GroundTruthBox:
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    area: float = 0.0
    size_category: str = "medium"  # "small" (<32^2), "medium" (32^2..96^2), "large" (>=96^2)
    id: Optional[str] = None
    
    def __post_init__(self):
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        self.area = w * h
        if self.area < 32.0 * 32.0:
            self.size_category = "small"
        elif self.area < 96.0 * 96.0:
            self.size_category = "medium"
        else:
            self.size_category = "large"


@dataclass
class PredictionBox:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]


@dataclass
class MatchedPair:
    ground_truth: GroundTruthBox
    prediction: PredictionBox
    iou: float


@dataclass
class ImageEvaluation:
    image_id: str
    ground_truths: List[GroundTruthBox]
    predictions: List[PredictionBox]
    matched_pairs: List[MatchedPair]
    missed_ground_truths: List[GroundTruthBox]
    unmatched_predictions: List[PredictionBox]
    iou_threshold: float
    confidence_threshold: float
    recall: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    
    def __post_init__(self):
        tp = len(self.matched_pairs)
        fn = len(self.missed_ground_truths)
        fp = len(self.unmatched_predictions)
        
        self.recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        self.precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        if (self.precision + self.recall) > 0.0:
            self.f1_score = float(2.0 * self.precision * self.recall / (self.precision + self.recall))
        else:
            self.f1_score = 0.0


def evaluate_single_image(
    image_id: str,
    ground_truths: List[GroundTruthBox],
    predictions: List[PredictionBox],
    iou_threshold: float = 0.50,
    confidence_threshold: float = 0.25
) -> ImageEvaluation:
    """
    Evaluates predictions against ground truths for one image using 1-to-1 matching.
    Predictions are sorted by confidence descending.
    Matches are formed greedily based on max IoU >= iou_threshold among remaining GTs of the same class.
    """
    filtered_preds = [p for p in predictions if p.confidence >= confidence_threshold]
    sorted_preds = sorted(filtered_preds, key=lambda x: x.confidence, reverse=True)
    
    unmatched_gt_indices = set(range(len(ground_truths)))
    matched_pairs: List[MatchedPair] = []
    unmatched_preds: List[PredictionBox] = []
    
    for pred in sorted_preds:
        best_iou = -1.0
        best_gt_idx = -1
        
        for gt_idx in unmatched_gt_indices:
            gt = ground_truths[gt_idx]
            # Match by class (either name match or ID match)
            if gt.class_name.lower() == pred.class_name.lower():
                iou = compute_iou(gt.bbox, pred.bbox)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            unmatched_gt_indices.remove(best_gt_idx)
            matched_pairs.append(MatchedPair(
                ground_truth=ground_truths[best_gt_idx],
                prediction=pred,
                iou=best_iou
            ))
        else:
            unmatched_preds.append(pred)
            
    missed_gts = [ground_truths[idx] for idx in unmatched_gt_indices]
    
    return ImageEvaluation(
        image_id=image_id,
        ground_truths=ground_truths,
        predictions=filtered_preds,
        matched_pairs=matched_pairs,
        missed_ground_truths=missed_gts,
        unmatched_predictions=unmatched_preds,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )


@dataclass
class ComparisonEvaluation:
    image_id: str
    baseline: ImageEvaluation
    perturbed: ImageEvaluation
    regressions: List[GroundTruthBox] = field(default_factory=list)
    persistent_misses: List[GroundTruthBox] = field(default_factory=list)
    recovered: List[GroundTruthBox] = field(default_factory=list)
    new_false_alarms: List[PredictionBox] = field(default_factory=list)
    confidence_deltas: List[Dict[str, Any]] = field(default_factory=list)
    delta_recall: float = 0.0
    delta_precision: float = 0.0
    
    def __post_init__(self):
        self.delta_recall = self.perturbed.recall - self.baseline.recall
        self.delta_precision = self.perturbed.precision - self.baseline.precision
        
        # Ground truths matched at baseline
        baseline_matched_gts = {m.ground_truth.id or str(m.ground_truth.bbox): m for m in self.baseline.matched_pairs}
        perturbed_matched_gts = {m.ground_truth.id or str(m.ground_truth.bbox): m for m in self.perturbed.matched_pairs}
        
        # Check for regressions (detected at baseline, missed in perturbed)
        for gt in self.baseline.ground_truths:
            key = gt.id or str(gt.bbox)
            in_base = key in baseline_matched_gts
            in_pert = key in perturbed_matched_gts
            
            if in_base and not in_pert:
                self.regressions.append(gt)
            elif not in_base and not in_pert:
                self.persistent_misses.append(gt)
            elif not in_base and in_pert:
                self.recovered.append(gt)
                
            if in_base and in_pert:
                base_match = baseline_matched_gts[key]
                pert_match = perturbed_matched_gts[key]
                self.confidence_deltas.append({
                    "class_name": gt.class_name,
                    "bbox": gt.bbox,
                    "baseline_conf": base_match.prediction.confidence,
                    "perturbed_conf": pert_match.prediction.confidence,
                    "delta_conf": pert_match.prediction.confidence - base_match.prediction.confidence,
                    "baseline_iou": base_match.iou,
                    "perturbed_iou": pert_match.iou,
                })
                
        self.new_false_alarms = list(self.perturbed.unmatched_predictions)


@dataclass
class BenchmarkSummary:
    total_images: int
    total_ground_truth: int
    iou_threshold: float
    confidence_threshold: float
    baseline_tp: int
    baseline_fn: int
    baseline_fp: int
    baseline_recall: float
    baseline_precision: float
    perturbed_tp: int
    perturbed_fn: int
    perturbed_fp: int
    perturbed_recall: float
    perturbed_precision: float
    delta_recall: float
    total_regressions: int
    total_recovered: int
    class_metrics: Dict[str, Dict[str, Any]]
    size_metrics: Dict[str, Dict[str, Any]]
    comparisons: List[ComparisonEvaluation]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "total_ground_truth": self.total_ground_truth,
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "baseline": {
                "tp": self.baseline_tp,
                "fn": self.baseline_fn,
                "fp": self.baseline_fp,
                "recall": round(self.baseline_recall, 4),
                "precision": round(self.baseline_precision, 4),
            },
            "perturbed": {
                "tp": self.perturbed_tp,
                "fn": self.perturbed_fn,
                "fp": self.perturbed_fp,
                "recall": round(self.perturbed_recall, 4),
                "precision": round(self.perturbed_precision, 4),
            },
            "delta_recall": round(self.delta_recall, 4),
            "total_regressions": self.total_regressions,
            "total_recovered": self.total_recovered,
            "class_metrics": self.class_metrics,
            "size_metrics": self.size_metrics,
        }


def aggregate_comparisons(comparisons: List[ComparisonEvaluation]) -> BenchmarkSummary:
    """
    Computes global macro/micro benchmark metrics from per-image comparison evaluations.
    """
    total_images = len(comparisons)
    if total_images == 0:
        return BenchmarkSummary(
            total_images=0, total_ground_truth=0, iou_threshold=0.5, confidence_threshold=0.25,
            baseline_tp=0, baseline_fn=0, baseline_fp=0, baseline_recall=0.0, baseline_precision=0.0,
            perturbed_tp=0, perturbed_fn=0, perturbed_fp=0, perturbed_recall=0.0, perturbed_precision=0.0,
            delta_recall=0.0, total_regressions=0, total_recovered=0,
            class_metrics={}, size_metrics={}, comparisons=[]
        )
        
    iou_thr = comparisons[0].baseline.iou_threshold
    conf_thr = comparisons[0].baseline.confidence_threshold
    
    total_gt = sum(len(c.baseline.ground_truths) for c in comparisons)
    base_tp = sum(len(c.baseline.matched_pairs) for c in comparisons)
    base_fn = sum(len(c.baseline.missed_ground_truths) for c in comparisons)
    base_fp = sum(len(c.baseline.unmatched_predictions) for c in comparisons)
    
    pert_tp = sum(len(c.perturbed.matched_pairs) for c in comparisons)
    pert_fn = sum(len(c.perturbed.missed_ground_truths) for c in comparisons)
    pert_fp = sum(len(c.perturbed.unmatched_predictions) for c in comparisons)
    
    base_recall = float(base_tp / (base_tp + base_fn)) if (base_tp + base_fn) > 0 else 0.0
    base_precision = float(base_tp / (base_tp + base_fp)) if (base_tp + base_fp) > 0 else 0.0
    
    pert_recall = float(pert_tp / (pert_tp + pert_fn)) if (pert_tp + pert_fn) > 0 else 0.0
    pert_precision = float(pert_tp / (pert_tp + pert_fp)) if (pert_tp + pert_fp) > 0 else 0.0
    
    total_regressions = sum(len(c.regressions) for c in comparisons)
    total_recovered = sum(len(c.recovered) for c in comparisons)
    
    # Per-class breakdown
    class_stats: Dict[str, Dict[str, int]] = {}
    for comp in comparisons:
        for gt in comp.baseline.ground_truths:
            cname = gt.class_name
            if cname not in class_stats:
                class_stats[cname] = {
                    "total": 0, "base_detected": 0, "pert_detected": 0, "regressed": 0
                }
            class_stats[cname]["total"] += 1
            
        for m in comp.baseline.matched_pairs:
            class_stats[m.ground_truth.class_name]["base_detected"] += 1
            
        for m in comp.perturbed.matched_pairs:
            class_stats[m.ground_truth.class_name]["pert_detected"] += 1
            
        for r in comp.regressions:
            class_stats[r.class_name]["regressed"] += 1
            
    class_metrics = {}
    for cname, st in class_stats.items():
        tot = st["total"]
        b_rec = st["base_detected"] / tot if tot > 0 else 0.0
        p_rec = st["pert_detected"] / tot if tot > 0 else 0.0
        class_metrics[cname] = {
            "total_instances": tot,
            "baseline_recall": round(b_rec, 3),
            "perturbed_recall": round(p_rec, 3),
            "regressions": st["regressed"],
            "delta_recall": round(p_rec - b_rec, 3)
        }
        
    # Scale / Size breakdown: small, medium, large
    size_stats = {
        "small": {"total": 0, "base_detected": 0, "pert_detected": 0, "regressed": 0},
        "medium": {"total": 0, "base_detected": 0, "pert_detected": 0, "regressed": 0},
        "large": {"total": 0, "base_detected": 0, "pert_detected": 0, "regressed": 0},
    }
    for comp in comparisons:
        for gt in comp.baseline.ground_truths:
            sz = gt.size_category
            size_stats[sz]["total"] += 1
            
        for m in comp.baseline.matched_pairs:
            size_stats[m.ground_truth.size_category]["base_detected"] += 1
            
        for m in comp.perturbed.matched_pairs:
            size_stats[m.ground_truth.size_category]["pert_detected"] += 1
            
        for r in comp.regressions:
            size_stats[r.size_category]["regressed"] += 1
            
    size_metrics = {}
    for sz, st in size_stats.items():
        tot = st["total"]
        b_rec = st["base_detected"] / tot if tot > 0 else 0.0
        p_rec = st["pert_detected"] / tot if tot > 0 else 0.0
        size_metrics[sz] = {
            "total_instances": tot,
            "baseline_recall": round(b_rec, 3),
            "perturbed_recall": round(p_rec, 3),
            "regressions": st["regressed"],
            "delta_recall": round(p_rec - b_rec, 3)
        }
        
    return BenchmarkSummary(
        total_images=total_images,
        total_ground_truth=total_gt,
        iou_threshold=iou_thr,
        confidence_threshold=conf_thr,
        baseline_tp=base_tp,
        baseline_fn=base_fn,
        baseline_fp=base_fp,
        baseline_recall=base_recall,
        baseline_precision=base_precision,
        perturbed_tp=pert_tp,
        perturbed_fn=pert_fn,
        perturbed_fp=pert_fp,
        perturbed_recall=pert_recall,
        perturbed_precision=pert_precision,
        delta_recall=pert_recall - base_recall,
        total_regressions=total_regressions,
        total_recovered=total_recovered,
        class_metrics=class_metrics,
        size_metrics=size_metrics,
        comparisons=comparisons
    )
