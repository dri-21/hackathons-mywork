"""
Unit tests for MIRAGE Evaluation Engine
Tests:
- compute_iou accuracy and edge cases
- 1-to-1 matching logic
- GroundTruthBox scale categorization
- ImageEvaluation metrics (precision, recall, f1)
- ComparisonEvaluation (regressions, persistent misses, false alarms)
- Empty predictions and empty ground truth handling
"""

import pytest
from core.evaluator import (
    compute_iou,
    GroundTruthBox,
    PredictionBox,
    evaluate_single_image,
    ComparisonEvaluation,
    aggregate_comparisons
)


def test_compute_iou_exact_match():
    box1 = (10.0, 10.0, 50.0, 50.0)
    box2 = (10.0, 10.0, 50.0, 50.0)
    assert compute_iou(box1, box2) == pytest.approx(1.0, abs=1e-5)


def test_compute_iou_disjoint():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    box1 = (0.0, 0.0, 20.0, 20.0)    # Area = 400
    box2 = (10.0, 0.0, 30.0, 20.0)   # Area = 400, Inter = 10*20 = 200, Union = 600
    assert compute_iou(box1, box2) == pytest.approx(200.0 / 600.0, abs=1e-5)


def test_compute_iou_degenerate_boxes():
    box1 = (10.0, 10.0, 10.0, 20.0)  # zero width
    box2 = (5.0, 5.0, 25.0, 25.0)
    assert compute_iou(box1, box2) == 0.0


def test_ground_truth_size_categories():
    small_gt = GroundTruthBox(0, "person", (0, 0, 20, 20))    # area 400 < 1024
    assert small_gt.size_category == "small"

    med_gt = GroundTruthBox(0, "person", (0, 0, 50, 50))      # area 2500 (1024 <= area < 9216)
    assert med_gt.size_category == "medium"

    large_gt = GroundTruthBox(0, "person", (0, 0, 120, 120))  # area 14400 >= 9216
    assert large_gt.size_category == "large"


def test_single_image_matching_one_to_one():
    # 1 GT of 'car'
    gt = GroundTruthBox(2, "car", (10.0, 10.0, 100.0, 100.0), id="gt_car_1")
    
    # 2 predictions of 'car' overlapping the same GT
    p1 = PredictionBox(2, "car", 0.90, (12.0, 12.0, 98.0, 98.0))
    p2 = PredictionBox(2, "car", 0.70, (15.0, 15.0, 95.0, 95.0))
    
    eval_res = evaluate_single_image(
        image_id="img_01",
        ground_truths=[gt],
        predictions=[p1, p2],
        iou_threshold=0.50,
        confidence_threshold=0.25
    )
    
    # P1 should match GT, P2 must be marked as unmatched prediction (prevent duplicate matching)
    assert len(eval_res.matched_pairs) == 1
    assert eval_res.matched_pairs[0].prediction == p1
    assert len(eval_res.missed_ground_truths) == 0
    assert len(eval_res.unmatched_predictions) == 1
    assert eval_res.unmatched_predictions[0] == p2
    assert eval_res.recall == pytest.approx(1.0)
    assert eval_res.precision == pytest.approx(0.5)


def test_empty_predictions_and_ground_truth():
    # Both empty
    eval_empty = evaluate_single_image("empty", [], [], iou_threshold=0.5)
    assert eval_empty.recall == 0.0
    assert eval_empty.precision == 0.0
    assert eval_empty.f1_score == 0.0

    # Predictions empty, GT present
    gt = GroundTruthBox(0, "person", (10, 10, 50, 50))
    eval_no_preds = evaluate_single_image("no_preds", [gt], [], iou_threshold=0.5)
    assert eval_no_preds.recall == 0.0
    assert len(eval_no_preds.missed_ground_truths) == 1

    # GT empty, predictions present
    pred = PredictionBox(0, "person", 0.85, (10, 10, 50, 50))
    eval_no_gt = evaluate_single_image("no_gt", [], [pred], iou_threshold=0.5)
    assert eval_no_gt.recall == 0.0
    assert eval_no_gt.precision == 0.0
    assert len(eval_no_gt.unmatched_predictions) == 1


def test_comparison_evaluation_regressions():
    gt1 = GroundTruthBox(2, "car", (10, 10, 80, 80), id="car_1")
    gt2 = GroundTruthBox(0, "person", (100, 100, 150, 200), id="person_1")
    
    # Baseline: both detected
    p_base_car = PredictionBox(2, "car", 0.88, (12, 12, 78, 78))
    p_base_person = PredictionBox(0, "person", 0.75, (102, 102, 148, 198))
    
    base_eval = evaluate_single_image("img_test", [gt1, gt2], [p_base_car, p_base_person])
    
    # Perturbed: only person detected, car missed (regression!)
    p_pert_person = PredictionBox(0, "person", 0.65, (102, 102, 148, 198))
    
    pert_eval = evaluate_single_image("img_test", [gt1, gt2], [p_pert_person])
    
    comp = ComparisonEvaluation("img_test", base_eval, pert_eval)
    
    assert len(comp.regressions) == 1
    assert comp.regressions[0].id == "car_1"
    assert len(comp.confidence_deltas) == 1
    assert comp.confidence_deltas[0]["class_name"] == "person"
    assert comp.confidence_deltas[0]["delta_conf"] == pytest.approx(-0.10, abs=1e-4)
    assert comp.delta_recall == pytest.approx(-0.50, abs=1e-4)
