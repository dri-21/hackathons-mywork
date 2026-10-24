"""
MIRAGE - Failure Diagnostics Engine
Extracts empirical failure patterns, separates observed evidence from hypotheses,
and generates side-by-side computer vision overlay visualizations:
- Classifies failures into Regressions, Persistent Misses, False Alarms, and Confidence Drops
- Performs scale and class vulnerability analysis
- Generates publication-ready bounding box overlays for visual comparison
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from core.evaluator import (
    GroundTruthBox,
    PredictionBox,
    ComparisonEvaluation,
    BenchmarkSummary,
    compute_iou
)


@dataclass
class DiagnosticFinding:
    failure_type: str  # "REGRESSION", "PERSISTENT_MISS", "FALSE_ALARM", "CONFIDENCE_DROP"
    image_id: str
    class_name: str
    size_category: str
    baseline_conf: Optional[float]
    perturbed_conf: Optional[float]
    delta_conf: Optional[float]
    iou_with_gt: Optional[float]
    bbox: Tuple[float, float, float, float]
    details: str
    evidence_level: str = "EMPIRICAL_MEASUREMENT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailurePatternAnalysis:
    perturbation_type: str
    severity: int
    baseline_recall: float
    perturbed_recall: float
    delta_recall: float
    total_regressions: int
    findings: List[DiagnosticFinding]
    observations: List[str]
    hypotheses: List[str]
    vulnerable_classes: List[Dict[str, Any]]
    size_regression_rates: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "severity": self.severity,
            "baseline_recall": round(self.baseline_recall, 4),
            "perturbed_recall": round(self.perturbed_recall, 4),
            "delta_recall": round(self.delta_recall, 4),
            "total_regressions": self.total_regressions,
            "findings_count": len(self.findings),
            "observations": self.observations,
            "hypotheses": self.hypotheses,
            "vulnerable_classes": self.vulnerable_classes,
            "size_regression_rates": self.size_regression_rates,
        }


def extract_diagnostic_findings(
    comparisons: List[ComparisonEvaluation],
    conf_drop_threshold: float = 0.20
) -> List[DiagnosticFinding]:
    """
    Scans image comparisons to extract concrete failure events.
    """
    findings: List[DiagnosticFinding] = []
    
    for comp in comparisons:
        img_id = comp.image_id
        
        # 1. Regressions: Detected at baseline, missed under perturbation
        for gt in comp.regressions:
            # Find baseline matched prediction for context
            base_conf = None
            base_iou = None
            for m in comp.baseline.matched_pairs:
                if (m.ground_truth.id or str(m.ground_truth.bbox)) == (gt.id or str(gt.bbox)):
                    base_conf = m.prediction.confidence
                    base_iou = m.iou
                    break
                    
            conf_str = f"{base_conf:.2f}" if base_conf is not None else "N/A"
            findings.append(DiagnosticFinding(
                failure_type="REGRESSION",
                image_id=img_id,
                class_name=gt.class_name,
                size_category=gt.size_category,
                baseline_conf=base_conf,
                perturbed_conf=None,
                delta_conf=None,
                iou_with_gt=base_iou,
                bbox=gt.bbox,
                details=f"Object was detected at baseline (conf: {conf_str}), but completely missed after perturbation."
            ))
            
        # 2. Significant confidence drop for objects still detected
        for delta in comp.confidence_deltas:
            if delta["delta_conf"] <= -conf_drop_threshold:
                findings.append(DiagnosticFinding(
                    failure_type="CONFIDENCE_DROP",
                    image_id=img_id,
                    class_name=delta["class_name"],
                    size_category="medium",
                    baseline_conf=delta["baseline_conf"],
                    perturbed_conf=delta["perturbed_conf"],
                    delta_conf=delta["delta_conf"],
                    iou_with_gt=delta["perturbed_iou"],
                    bbox=delta["bbox"],
                    details=f"Detection confidence dropped significantly by {abs(delta['delta_conf']):.2f} (from {delta['baseline_conf']:.2f} down to {delta['perturbed_conf']:.2f})."
                ))
                
        # 3. New false alarms (unmatched predictions introduced under perturbation)
        for pred in comp.perturbed.unmatched_predictions:
            findings.append(DiagnosticFinding(
                failure_type="FALSE_ALARM",
                image_id=img_id,
                class_name=pred.class_name,
                size_category="unknown",
                baseline_conf=None,
                perturbed_conf=pred.confidence,
                delta_conf=None,
                iou_with_gt=0.0,
                bbox=pred.bbox,
                details=f"Spurious prediction introduced under perturbation (class: {pred.class_name}, conf: {pred.confidence:.2f}) with no matching ground-truth object."
            ))
            
        # 4. Persistent misses (missed at both baseline and perturbation)
        for gt in comp.persistent_misses:
            findings.append(DiagnosticFinding(
                failure_type="PERSISTENT_MISS",
                image_id=img_id,
                class_name=gt.class_name,
                size_category=gt.size_category,
                baseline_conf=None,
                perturbed_conf=None,
                delta_conf=None,
                iou_with_gt=0.0,
                bbox=gt.bbox,
                details=f"Object was missed at baseline and remained undetected after perturbation."
            ))
            
    return findings


def analyze_failure_patterns(
    summary: BenchmarkSummary,
    perturbation_type: str,
    severity: int
) -> FailurePatternAnalysis:
    """
    Synthesizes empirical findings into verified observations and distinct hypotheses.
    Adheres strictly to the requirement:
    'Only present patterns supported by actual measurements.
     Clearly distinguish observations from hypotheses.'
    """
    findings = extract_diagnostic_findings(summary.comparisons)
    observations: List[str] = []
    hypotheses: List[str] = []
    
    delta_rec = summary.delta_recall
    total_reg = summary.total_regressions
    
    # Empirical observations
    observations.append(
        f"Under {perturbation_type} at severity {severity}, overall detection recall changed by {delta_rec*100:+.1f}% "
        f"(Baseline: {summary.baseline_recall*100:.1f}%, Perturbed: {summary.perturbed_recall*100:.1f}%)."
    )
    
    if total_reg > 0:
        observations.append(
            f"Observed {total_reg} regression failure(s) where a ground-truth object detected at baseline was lost under perturbation."
        )
    else:
        observations.append(
            "No regression failures were observed; all baseline detections remained detectable under this condition."
        )
        
    if summary.perturbed_fp > summary.baseline_fp:
        observations.append(
            f"Perturbation induced {summary.perturbed_fp - summary.baseline_fp} additional false-positive prediction(s) "
            f"(Perturbed FP: {summary.perturbed_fp} vs Baseline FP: {summary.baseline_fp})."
        )
        
    # Scale analysis
    size_reg_rates: Dict[str, float] = {}
    for sz, data in summary.size_metrics.items():
        tot = data.get("total_instances", 0)
        reg = data.get("regressions", 0)
        rate = (reg / tot) if tot > 0 else 0.0
        size_reg_rates[sz] = round(rate, 3)
        if tot > 0 and reg > 0:
            observations.append(
                f"Objects in the '{sz}' size band exhibited a {rate*100:.1f}% regression rate ({reg}/{tot} instances lost)."
            )
            
    # Class analysis
    vulnerable_classes = []
    for cname, data in summary.class_metrics.items():
        if data.get("regressions", 0) > 0:
            vulnerable_classes.append({
                "class_name": cname,
                "regressions": data["regressions"],
                "total": data["total_instances"],
                "delta_recall": data["delta_recall"]
            })
    vulnerable_classes.sort(key=lambda x: x["regressions"], reverse=True)
    
    if vulnerable_classes:
        top_cls = vulnerable_classes[0]
        observations.append(
            f"Class '{top_cls['class_name']}' experienced the highest regression volume with {top_cls['regressions']} lost instance(s)."
        )
        
    # Strictly separated hypotheses (clearly labeled as unverified hypotheses)
    if "blur" in perturbation_type:
        hypotheses.append(
            "Hypothesis: High spatial frequency attenuation from blur degrades sharp edge and texture features essential for feature pyramid activation."
        )
    elif "dark" in perturbation_type or "shadow" in perturbation_type:
        hypotheses.append(
            "Hypothesis: Low pixel luminance in shadow/dark regions compresses gradient magnitudes below activation thresholds in early convolutional layers."
        )
    elif "occlusion" in perturbation_type:
        hypotheses.append(
            "Hypothesis: Structural patch obstruction breaks key object topological cues and contextual relationships needed for bounding-box proposals."
        )
    elif "contrast" in perturbation_type:
        hypotheses.append(
            "Hypothesis: Compressed dynamic range diminishes contrast between foreground silhouettes and background textures."
        )
        
    hypotheses.append(
        "Hypothesis: Small objects are disproportionately vulnerable because downsampling operations in the backbone eliminate fewer remaining salient pixels."
    )
    
    return FailurePatternAnalysis(
        perturbation_type=perturbation_type,
        severity=severity,
        baseline_recall=summary.baseline_recall,
        perturbed_recall=summary.perturbed_recall,
        delta_recall=delta_rec,
        total_regressions=total_reg,
        findings=findings,
        observations=observations,
        hypotheses=hypotheses,
        vulnerable_classes=vulnerable_classes,
        size_regression_rates=size_reg_rates
    )


def draw_bounding_boxes(
    image_bgr: np.ndarray,
    ground_truths: Optional[List[GroundTruthBox]] = None,
    predictions: Optional[List[PredictionBox]] = None,
    show_gt: bool = True,
    show_pred: bool = True,
    gt_color: Tuple[int, int, int] = (0, 230, 115),   # Emerald green
    pred_color: Tuple[int, int, int] = (255, 128, 0)   # Vivid cyan/blue (in BGR)
) -> np.ndarray:
    """
    Renders high-contrast, clean bounding boxes and text badges onto an image.
    Preserves original image array by copying.
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    
    # Dynamic scale factors based on image resolution
    line_thickness = max(1, int(round(min(h, w) / 320.0)))
    font_scale = max(0.40, min(h, w) / 900.0)
    
    # 1. Render Ground Truths
    if show_gt and ground_truths:
        for gt in ground_truths:
            x1, y1, x2, y2 = [int(round(v)) for v in gt.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Ground truth rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), gt_color, line_thickness)
            
            # Label badge
            label_text = f"GT: {gt.class_name}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            badge_y1 = max(0, y1 - text_h - 4)
            badge_y2 = y1
            badge_x2 = min(w - 1, x1 + text_w + 6)
            
            # Badge background
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), gt_color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, badge_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            
    # 2. Render Predictions
    if show_pred and predictions:
        for pred in predictions:
            x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Prediction rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), pred_color, line_thickness)
            
            # Prediction badge with confidence
            label_text = f"{pred.class_name} {pred.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            badge_y1 = min(h - 1, y2)
            badge_y2 = min(h - 1, y2 + text_h + 4)
            badge_x2 = min(w - 1, x1 + text_w + 6)
            
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), pred_color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, badge_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
    return canvas
