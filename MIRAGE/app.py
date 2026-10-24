"""
MIRAGE — AI Vision Failure Testing & Diagnosis Workbench
Full-featured Streamlit application for empirical reliability testing of computer vision models.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.evaluator import (
    compute_iou,
    GroundTruthBox,
    PredictionBox,
    evaluate_single_image,
    ComparisonEvaluation,
    aggregate_comparisons,
    BenchmarkSummary
)
from core.perturbations import (
    apply_perturbation,
    PERTURBATION_REGISTRY
)
from core.diagnostics import (
    extract_diagnostic_findings,
    analyze_failure_patterns,
    draw_bounding_boxes
)
from core.replay import (
    export_experiment_report,
    load_experiment_manifest,
    replay_experiment,
    get_environment_info
)
from core.detector import ObjectDetector
from core.dataset import setup_dataset, DatasetReport, DatasetSample

# Ensure Ultralytics config is cached locally in workspace
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath("data/.ultralytics"))


# Page Configuration
st.set_page_config(
    page_title="MIRAGE — Vision Failure Workbench",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
st.markdown("""
<style>
    /* Dark Theme Surface Colors */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Sleek Header & Badges */
    .mirage-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    .mirage-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f0f6fc;
        margin: 0;
    }
    .mirage-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-top: 4px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-card-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-card-delta-pos {
        color: #3fb950;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card-delta-neg {
        color: #f85149;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card-delta-neutral {
        color: #8b949e;
        font-size: 0.85rem;
    }
    
    /* Pill Badges */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-ready { background-color: #238636; color: #ffffff; }
    .badge-error { background-color: #da3633; color: #ffffff; }
    .badge-info  { background-color: #1f6feb; color: #ffffff; }
    .badge-warn  { background-color: #9e6a03; color: #ffffff; }
    
    /* Evidence vs Hypothesis Blocks */
    .evidence-box {
        background: #0f1c14;
        border-left: 3px solid #2ea043;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .hypothesis-box {
        background: #1f1a10;
        border-left: 3px solid #d29922;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# Cached Resources
@st.cache_resource(show_spinner="Loading YOLO Object Detector...")
def get_detector(model_name: str = "yolo11n.pt") -> ObjectDetector:
    return ObjectDetector(model_name=model_name, device="cpu")


@st.cache_resource(show_spinner="Preparing Benchmark Dataset...")
def get_dataset(preferred: str = "coco8") -> DatasetReport:
    return setup_dataset(data_dir="data", preferred_dataset=preferred)


def run_benchmark_experiment(
    detector: ObjectDetector,
    samples: List[DatasetSample],
    perturbation_type: str,
    severity: int,
    seed: int,
    iou_thr: float,
    conf_thr: float
) -> Tuple[BenchmarkSummary, Dict[str, Any]]:
    """Runs baseline and perturbed inference deterministically across selected samples."""
    comparisons: List[ComparisonEvaluation] = []
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    t0 = time.perf_counter()
    for idx, sample in enumerate(samples):
        status_text.text(f"Evaluating {sample.image_id} ({idx+1}/{len(samples)})...")
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            continue
            
        # Baseline inference
        base_preds = detector.predict(img_bgr, confidence_threshold=conf_thr, iou_threshold=iou_thr)
        base_eval = evaluate_single_image(
            sample.image_id, sample.ground_truths, base_preds, iou_thr, conf_thr
        )
        
        # Perturbation
        pert_img, pert_meta = apply_perturbation(img_bgr, perturbation_type, severity=severity, seed=seed)
        
        # Perturbed inference
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
        progress_bar.progress((idx + 1) / len(samples))
        
    total_time = time.perf_counter() - t0
    progress_bar.empty()
    status_text.empty()
    
    summary = aggregate_comparisons(comparisons)
    meta = {
        "benchmark_latency_sec": round(total_time, 2),
        "avg_latency_per_img_ms": round((total_time / len(samples)) * 1000.0, 1) if samples else 0.0,
    }
    return summary, meta


def main():
    # Header
    st.markdown("""
    <div class="mirage-header">
        <h1 class="mirage-title">MIRAGE</h1>
        <div class="mirage-subtitle">AI Vision Failure Testing & Diagnosis Workbench</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Detector & Dataset
    with st.sidebar:
        st.markdown("### ⚙️ Engine Configuration")
        model_choice = st.selectbox(
            "Model Selection",
            ["yolo11n.pt", "yolov8n.pt"],
            index=0,
            help="Small supported pretrained YOLO model running on CPU."
        )
        detector = get_detector(model_choice)
        
        dataset_choice = st.selectbox(
            "Benchmark Dataset",
            ["coco8", "coco128"],
            index=0,
            help="Official Ultralytics verified COCO subsets."
        )
        dataset_report = get_dataset(dataset_choice)
        
        st.markdown("---")
        st.markdown("### 🧪 Perturbation Parameters")
        
        perturbation_type = st.selectbox(
            "Perturbation Type",
            list(PERTURBATION_REGISTRY.keys()),
            format_func=lambda k: PERTURBATION_REGISTRY[k],
            index=2  # default: gaussian_blur
        )
        
        severity = st.slider("Severity Level (1-5)", min_value=1, max_value=5, value=3, step=1)
        seed = st.number_input("Deterministic Random Seed", min_value=0, max_value=999999, value=42, step=1)
        
        st.markdown("---")
        st.markdown("### 📐 Evaluation Thresholds")
        iou_threshold = st.slider("IoU Match Threshold", min_value=0.30, max_value=0.90, value=0.50, step=0.05)
        conf_threshold = st.slider("Confidence Cutoff", min_value=0.10, max_value=0.80, value=0.25, step=0.05)
        
        st.markdown("---")
        # Subset selection
        max_available = len(dataset_report.samples)
        default_limit = min(8, max_available)
        image_limit = st.slider(
            "Image Subset Size",
            min_value=1,
            max_value=max_available if max_available > 0 else 8,
            value=default_limit,
            step=1,
            help="Evaluate a quick subset or full set."
        )
        
        run_btn = st.button("▶ Run Full Benchmark", type="primary", use_container_width=True)
        demo_btn = st.button("⚡ Quick 1-Click Demo", use_container_width=True)
        
    # Status Banner
    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 2, 2, 2])
    with col_s1:
        if detector.is_ready:
            st.markdown(f'<span class="status-badge badge-ready">READY</span> **Model:** `{detector.model_name}`', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge badge-error">ERROR</span> **Model:** Failed', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<span class="status-badge badge-info">INFO</span> **Device:** `{detector.device.upper()}`', unsafe_allow_html=True)
    with col_s3:
        if dataset_report.valid_images > 0:
            status_cls = "badge-warn" if dataset_report.is_fallback else "badge-ready"
            status_txt = "FALLBACK" if dataset_report.is_fallback else "VERIFIED"
            st.markdown(f'<span class="status-badge {status_cls}">{status_txt}</span> **Data:** {dataset_report.valid_images} imgs', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge badge-error">EMPTY</span> **Data:** 0 imgs', unsafe_allow_html=True)
    with col_s4:
        st.markdown(f'<span class="status-badge badge-info">LABELS</span> **GT Objects:** {dataset_report.total_ground_truth_boxes}', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Session state initialization
    if "current_summary" not in st.session_state:
        st.session_state.current_summary = None
    if "current_pert_meta" not in st.session_state:
        st.session_state.current_pert_meta = None
        
    # Execute Demo or Run
    if demo_btn or (run_btn and dataset_report.valid_images > 0):
        target_p_type = "gaussian_blur" if demo_btn else perturbation_type
        target_sev = 3 if demo_btn else severity
        target_seed = 42 if demo_btn else seed
        target_limit = min(4, len(dataset_report.samples)) if demo_btn else image_limit
        
        eval_samples = dataset_report.get_subset(target_limit)
        
        with st.spinner("Executing real model inference and benchmark evaluation..."):
            summary, meta = run_benchmark_experiment(
                detector=detector,
                samples=eval_samples,
                perturbation_type=target_p_type,
                severity=target_sev,
                seed=target_seed,
                iou_thr=iou_threshold,
                conf_thr=conf_threshold
            )
            st.session_state.current_summary = summary
            st.session_state.current_pert_meta = {
                "perturbation_type": target_p_type,
                "label": PERTURBATION_REGISTRY.get(target_p_type, target_p_type),
                "severity": target_sev,
                "seed": target_seed,
                "latency": meta,
            }
            st.toast("Benchmark evaluation completed successfully!", icon="✅")
            
    summary: Optional[BenchmarkSummary] = st.session_state.current_summary
    pert_meta = st.session_state.current_pert_meta
    
    if summary is None:
        st.info("👈 Select parameters in the sidebar and click **'Run Full Benchmark'** or **'⚡ Quick 1-Click Demo'** to begin real model evaluation.")
        
        # Display dataset license and info
        st.markdown("### 📚 Dataset Provenance & License")
        st.markdown(f"""
        - **Dataset Source:** {dataset_report.source_name}
        - **License:** {dataset_report.license_info}
        - **Taxonomy:** Standard COCO 80-Class Object Detection Taxonomy
        - **Integrity Status:** {dataset_report.notes}
        """)
        return

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Benchmark & Metrics",
        "🔍 Visual Inspection",
        "🔬 Failure Diagnostics",
        "💾 Reports & Replay"
    ])
    
    # TAB 1: Benchmark & Metrics
    with tab1:
        st.markdown(f"#### Results for **{pert_meta['label']}** (Severity {pert_meta['severity']})")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Baseline Recall</div>
                <div class="metric-card-value">{summary.baseline_recall*100:.1f}%</div>
                <div class="metric-card-delta-neutral">{summary.baseline_tp}/{summary.total_ground_truth} objects detected</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            delta_val = summary.delta_recall * 100
            delta_cls = "metric-card-delta-pos" if delta_val >= 0 else "metric-card-delta-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Perturbed Recall</div>
                <div class="metric-card-value">{summary.perturbed_recall*100:.1f}%</div>
                <div class="{delta_cls}">{delta_val:+.1f}% vs baseline</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            reg_cls = "metric-card-delta-neg" if summary.total_regressions > 0 else "metric-card-delta-pos"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Regressions (Lost Objects)</div>
                <div class="metric-card-value">{summary.total_regressions}</div>
                <div class="{reg_cls}">Detected at baseline, lost after perturbation</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">False Alarms (FP)</div>
                <div class="metric-card-value">{summary.perturbed_fp}</div>
                <div class="metric-card-delta-neutral">Baseline FP: {summary.baseline_fp}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📏 Object Scale Vulnerability")
        col_s1, col_s2, col_s3 = st.columns(3)
        sizes = [("small", "< 32px", col_s1), ("medium", "32px - 96px", col_s2), ("large", "> 96px", col_s3)]
        for sz_name, sz_lbl, sz_col in sizes:
            sz_data = summary.size_metrics.get(sz_name, {})
            tot = sz_data.get("total_instances", 0)
            reg = sz_data.get("regressions", 0)
            b_rec = sz_data.get("baseline_recall", 0.0) * 100
            p_rec = sz_data.get("perturbed_recall", 0.0) * 100
            with sz_col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">{sz_name.upper()} OBJECTS ({sz_lbl})</div>
                    <div style="font-size: 1.1rem; font-weight:600;">Recall: {b_rec:.1f}% → {p_rec:.1f}%</div>
                    <div style="color: {'#f85149' if reg > 0 else '#8b949e'}; font-size:0.85rem; margin-top:4px;">
                        Regressions: <b>{reg}</b> / {tot} total instances
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📋 Per-Image Evaluation Summary")
        table_rows = []
        for c in summary.comparisons:
            table_rows.append({
                "Image ID": c.image_id,
                "GT Count": len(c.baseline.ground_truths),
                "Base TP": len(c.baseline.matched_pairs),
                "Pert TP": len(c.perturbed.matched_pairs),
                "Regressions": len(c.regressions),
                "False Alarms": len(c.perturbed.unmatched_predictions),
                "Base Recall": f"{c.baseline.recall*100:.1f}%",
                "Pert Recall": f"{c.perturbed.recall*100:.1f}%",
                "Delta Recall": f"{c.delta_recall*100:+.1f}%",
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # TAB 2: Visual Inspection
    with tab2:
        st.markdown("#### Side-by-Side Visual Verification")
        
        img_options = [c.image_id for c in summary.comparisons]
        # Mark images with regressions
        formatted_options = {}
        for c in summary.comparisons:
            tag = "❌ REGRESSION" if len(c.regressions) > 0 else "✓ STABLE"
            formatted_options[c.image_id] = f"{tag} | {c.image_id} (GT: {len(c.baseline.ground_truths)}, Lost: {len(c.regressions)})"
            
        selected_img_id = st.selectbox(
            "Select Image to Inspect",
            img_options,
            format_func=lambda x: formatted_options.get(x, x)
        )
        
        selected_comp = next((c for c in summary.comparisons if c.image_id == selected_img_id), summary.comparisons[0])
        
        # Overlay toggles
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            show_gt = st.checkbox("Show Ground Truth Boxes (Green)", value=True)
        with col_t2:
            show_base_pred = st.checkbox("Show Baseline Predictions (Cyan)", value=True)
        with col_t3:
            show_pert_pred = st.checkbox("Show Perturbed Predictions (Cyan)", value=True)
            
        # Find sample path
        target_sample = next((s for s in dataset_report.samples if s.image_id == selected_img_id), None)
        if target_sample and os.path.exists(target_sample.image_path):
            raw_bgr = cv2.imread(target_sample.image_path)
            pert_bgr, _ = apply_perturbation(
                raw_bgr,
                pert_meta["perturbation_type"],
                severity=pert_meta["severity"],
                seed=pert_meta["seed"]
            )
            
            # Render overlays
            canvas_base = draw_bounding_boxes(
                raw_bgr,
                ground_truths=selected_comp.baseline.ground_truths if show_gt else None,
                predictions=selected_comp.baseline.predictions if show_base_pred else None,
            )
            
            canvas_pert = draw_bounding_boxes(
                pert_bgr,
                ground_truths=selected_comp.perturbed.ground_truths if show_gt else None,
                predictions=selected_comp.perturbed.predictions if show_pert_pred else None,
            )
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**Original Image (Baseline)**")
                st.image(cv2.cvtColor(canvas_base, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Recall: {selected_comp.baseline.recall*100:.1f}% | Precision: {selected_comp.baseline.precision*100:.1f}%")
                
            with col_v2:
                st.markdown(f"**Perturbed Image ({pert_meta['label']} Sev {pert_meta['severity']})**")
                st.image(cv2.cvtColor(canvas_pert, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Recall: {selected_comp.perturbed.recall*100:.1f}% | Precision: {selected_comp.perturbed.precision*100:.1f}%")

        # Object level detail table
        st.markdown("##### Object Status Breakdown for Selected Image")
        obj_rows = []
        for gt in selected_comp.baseline.ground_truths:
            gt_key = gt.id or str(gt.bbox)
            
            base_m = next((m for m in selected_comp.baseline.matched_pairs if (m.ground_truth.id or str(m.ground_truth.bbox)) == gt_key), None)
            pert_m = next((m for m in selected_comp.perturbed.matched_pairs if (m.ground_truth.id or str(m.ground_truth.bbox)) == gt_key), None)
            
            if base_m and pert_m:
                status = "✅ PRESERVED"
                delta_c = f"{pert_m.prediction.confidence - base_m.prediction.confidence:+.2f}"
            elif base_m and not pert_m:
                status = "❌ REGRESSED (LOST)"
                delta_c = f"-{base_m.prediction.confidence:.2f}"
            elif not base_m and not pert_m:
                status = "⚠️ PERSISTENT MISS"
                delta_c = "N/A"
            else:
                status = "✨ RECOVERED"
                delta_c = f"+{pert_m.prediction.confidence:.2f}"
                
            obj_rows.append({
                "Class": gt.class_name,
                "Scale": gt.size_category,
                "Baseline Conf": f"{base_m.prediction.confidence:.2f}" if base_m else "Missed",
                "Perturbed Conf": f"{pert_m.prediction.confidence:.2f}" if pert_m else "Missed",
                "Delta Conf": delta_c,
                "Failure Status": status,
            })
            
        st.dataframe(pd.DataFrame(obj_rows), use_container_width=True, hide_index=True)

    # TAB 3: Failure Diagnostics & Patterns
    with tab3:
        st.markdown("#### Evidence-Based Failure Pattern Analysis")
        
        diag_analysis = analyze_failure_patterns(
            summary=summary,
            perturbation_type=pert_meta["perturbation_type"],
            severity=pert_meta["severity"]
        )
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### 📌 Measured Empirical Observations")
            for obs in diag_analysis.observations:
                st.markdown(f'<div class="evidence-box"><b>[MEASUREMENT]</b> {obs}</div>', unsafe_allow_html=True)
                
        with col_d2:
            st.markdown("##### 💡 Separated Candidate Hypotheses")
            for hyp in diag_analysis.hypotheses:
                st.markdown(f'<div class="hypothesis-box"><b>[HYPOTHESIS]</b> {hyp}</div>', unsafe_allow_html=True)

        st.markdown("##### 🎯 Class Vulnerability Ranking")
        if diag_analysis.vulnerable_classes:
            vuln_df = pd.DataFrame(diag_analysis.vulnerable_classes)
            vuln_df.rename(columns={
                "class_name": "Class Name",
                "regressions": "Lost Instances (Regressions)",
                "total": "Total Dataset Instances",
                "delta_recall": "Change in Recall"
            }, inplace=True)
            st.dataframe(vuln_df, use_container_width=True, hide_index=True)
        else:
            st.success("No class regressions were observed under the current test conditions.")

    # TAB 4: Reports & Replay
    with tab4:
        st.markdown("#### Experiment Serialization & Replay")
        
        # Export Report
        manifest, report_path = export_experiment_report(
            summary=summary,
            model_info=detector.get_info(),
            dataset_info={
                "source_name": dataset_report.source_name,
                "license_info": dataset_report.license_info,
                "is_fallback": dataset_report.is_fallback,
            },
            perturbation_info=pert_meta,
            diagnostic_info=analyze_failure_patterns(
                summary, pert_meta["perturbation_type"], pert_meta["severity"]
            ).to_dict()
        )
        
        json_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        st.download_button(
            label="📥 Download Experiment Report (JSON)",
            data=json_bytes,
            file_name=os.path.basename(report_path),
            mime="application/json",
            type="primary"
        )
        st.caption(f"Cached report file: `{report_path}`")
        
        st.markdown("---")
        st.markdown("#### 🔄 Replay Past Experiment")
        uploaded_file = st.file_uploader("Upload Saved Mirage Experiment Manifest (.json)", type=["json"])
        
        if uploaded_file is not None:
            try:
                uploaded_manifest = json.loads(uploaded_file.getvalue().decode("utf-8"))
                st.success(f"Loaded manifest for model: `{uploaded_manifest.get('model', {}).get('name')}` | Perturbation: `{uploaded_manifest.get('perturbation', {}).get('type')}`")
                
                if st.button("Execute Deterministic Replay", use_container_width=True):
                    with st.spinner("Replaying experiment with identical parameters..."):
                        replay_res = replay_experiment(
                            manifest=uploaded_manifest,
                            detector=detector,
                            dataset_samples=dataset_report.samples
                        )
                        
                        if replay_res["status"] == "SUCCESS":
                            if replay_res["is_exact_match"]:
                                st.success(f"🎯 **REPLAY VERDICT: EXACT REPRODUCIBILITY MATCH**")
                            else:
                                st.warning(f"⚠️ **REPLAY VERDICT: DIVERGENCE DETECTED**")
                                
                            st.json(replay_res)
                        else:
                            st.error(f"Replay failed: {replay_res.get('reason')}")
            except Exception as e:
                st.error(f"Error reading manifest: {e}")


if __name__ == "__main__":
    main()
