# MIRAGE — AI Vision Failure Testing & Diagnosis Workbench

**MIRAGE** is a visual workbench designed to systematically test and diagnose how reliably an object-detection model functions when real-world image distortions and environmental perturbations are applied.

The central research question investigated by MIRAGE is:
> **When does an object detector stop detecting an object it could detect before, and what evidence can help us investigate that failure?**

---

## 1. What MIRAGE Does

Modern object detection models trained on clean benchmark images often fail unpredictably when deployed into real-world settings with varying illumination, lens blur, weather conditions, or partial occlusions.

MIRAGE enables you to:
1. Load a real pretrained object detection model (YOLO11n / YOLOv8n) running locally on CPU.
2. Evaluate a verified, small, public labelled dataset (COCO8 / COCO128 with CC BY 4.0 ground-truth annotations).
3. Systematically apply 7 types of physically-motivated, reproducible perturbations:
   - **Dark Exposure (Underexposed)**: Low-light / night conditions.
   - **Bright Exposure (Overexposed)**: Harsh lighting and direct glare.
   - **Gaussian Blur**: Lens defocus and camera motion blur.
   - **Synthetic Shadows**: Feathred polygonal shadow masks with deterministic seeds.
   - **Patch Occlusion**: Physical object obstruction / rectangular masking.
   - **Low Contrast (Haze)**: Fog and atmospheric scattering.
   - **High Contrast**: Dynamic range compression.
4. Run real model inference on both baseline and perturbed images.
5. Compute rigorous one-to-one ground truth matching with configurable IoU and confidence thresholds.
6. Quantify regressions (objects detected at baseline that disappeared under perturbation) and false alarms.
7. Separate **empirical measurements** from **hypotheses** in diagnostic reports.
8. Save compact, non-binary JSON experiment manifests and replay past experiments deterministically.

---

## 2. Model & Dataset Setup

- **Model**: Automatic download of Ultralytics `yolo11n.pt` (or `yolov8n.pt` as fallback) using the official Ultralytics loader. Cached locally in `data/models/`.
- **Dataset**: Automatic download and caching of public COCO subsets (`COCO8` - 8 images, 433 KB; or `COCO128` - 128 images, 6.8 MB), licensed under Creative Commons Attribution 4.0 (CC BY 4.0).
- **Fallback**: If network is offline, MIRAGE seamlessly initializes an isolated benchmark with verified bounding boxes, reporting that it is running in local fallback mode.

---

## 3. Evaluation Methodology & Metrics

MIRAGE uses a strict one-to-one greedy matching algorithm based on Intersection over Union (IoU):
- **IoU Match Threshold (default 0.50)**: A predicted box must have IoU $\ge$ 0.50 with a ground-truth box of the exact same class to be considered a match.
- **One-to-One Assignment**: A single ground-truth object can only be matched by one prediction. A single prediction can only match one ground-truth object. Duplicate predictions over the same object are classified as false alarms (FP).

### Metric Definitions:
- **True Positives (TP)**: Ground-truth objects matched by a valid prediction.
- **False Negatives / Missed Objects (FN)**: Ground-truth objects with no matching prediction above the confidence cutoff.
- **False Positives / Unmatched Predictions (FP)**: Model detections with no matching ground-truth object.
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — fraction of actual objects successfully found.
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — fraction of predicted boxes that correspond to real objects.
- **Recall Delta**: $\text{Recall}_{\text{perturbed}} - \text{Recall}_{\text{baseline}}$.
- **Regressions (Lost Objects)**: Specific ground-truth objects that were successfully detected at baseline, but became undetected under perturbation.
- **Confidence Delta**: Change in confidence score ($\text{Conf}_{\text{perturbed}} - \text{Conf}_{\text{baseline}}$) for objects that remained detected.

---

## 4. What MIRAGE Can and Cannot Conclude

> [!IMPORTANT]
> **Scientific Integrity Notice**
> - **What MIRAGE CAN conclude**: MIRAGE measures exact empirical performance drops (e.g. "Recall dropped from 85% to 50% under Gaussian blur severity 4") and catalogs specific instances of regression.
> - **What MIRAGE CANNOT conclude**: MIRAGE does not claim to establish the true root cause of neural network failures. It explicitly separates observed evidence from unproven hypotheses (such as training data bias, backbone architecture limitations, or illumination sensitivity).

---

## 5. Quickstart & Installation (macOS)

### Prerequisites
- Python 3.9+ or Python 3.10–3.14
- macOS terminal (zsh)

### Installation Steps

1. Navigate to the project directory:
   ```bash
   cd /Users/dhruthi/.gemini/antigravity/scratch/mirage
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```
   Or use the included launch script:
   ```bash
   ./run.sh
   ```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 6. Running Tests

MIRAGE includes unit and integration tests covering IoU calculations, matching rules, perturbation geometry, reproducibility, and JSON manifests.

Run tests using pytest:
```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## 7. Reports & Replay

All experiment reports are saved to `reports/` as compact JSON files containing:
- Exact software versions, model name, and execution device (`cpu`).
- Dataset source, subset image identifiers, and evaluation parameters.
- Perturbation type, severity, and deterministic random seed.
- Aggregate metrics and per-image detection outcomes.

You can download any report directly from the UI or reload it via **Tab 4 (Reports & Replay)** to verify 100% deterministic reproducibility.

---

## 8. Common Troubleshooting

- **Port 8501 already in use**:
  Launch on another port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Slow inference on large images**:
  Use the **Image Subset Size** slider in the sidebar to run benchmarks on a small subset (e.g., 4 to 8 images).
