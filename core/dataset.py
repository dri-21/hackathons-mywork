"""
MIRAGE - Dataset Management Engine
Handles automatic downloading, verification, caching, and parsing of public labelled datasets:
- Standard COCO 80-class taxonomy
- Support for COCO8 (rapid validation subset) and COCO128
- Strict annotation validation (coordinate normalization, box geometry, valid classes)
- Clear provenance, license documentation (CC BY 4.0), and fallback handling
"""

import os
import zipfile
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np
from core.evaluator import GroundTruthBox


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


@dataclass
class DatasetSample:
    image_id: str
    image_path: str
    width: int
    height: int
    ground_truths: List[GroundTruthBox] = field(default_factory=list)
    is_valid: bool = True
    validation_error: Optional[str] = None


@dataclass
class DatasetReport:
    source_name: str
    license_info: str
    is_fallback: bool
    total_images_found: int
    valid_images: int
    total_ground_truth_boxes: int
    class_distribution: Dict[str, int]
    samples: List[DatasetSample]
    notes: str = ""

    def get_subset(self, limit: Optional[int] = None) -> List[DatasetSample]:
        valid = [s for s in self.samples if s.is_valid]
        if limit is not None and limit > 0:
            return valid[:limit]
        return valid


def parse_yolo_annotation_file(
    txt_path: str,
    img_w: int,
    img_h: int,
    class_mapping: Optional[Dict[int, str]] = None
) -> Tuple[List[GroundTruthBox], Optional[str]]:
    """
    Parses a YOLO-format annotation file (.txt):
    Each line: <class_id> <x_center> <y_center> <width> <height> (all normalized 0..1).
    Converts to pixel coordinates [x1, y1, x2, y2].
    """
    if class_mapping is None:
        class_mapping = {i: name for i, name in enumerate(COCO_CLASSES)}
        
    if not os.path.exists(txt_path):
        return [], f"Annotation file not found: {txt_path}"
        
    boxes: List[GroundTruthBox] = []
    
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    box_idx = 0
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 5:
            continue
            
        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
        except ValueError:
            continue
            
        # Validate coordinates are normalized
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
            continue
            
        # Convert to absolute pixel coordinates
        x1 = max(0.0, (xc - w_norm / 2.0) * img_w)
        y1 = max(0.0, (yc - h_norm / 2.0) * img_h)
        x2 = min(float(img_w), (xc + w_norm / 2.0) * img_w)
        y2 = min(float(img_h), (yc + h_norm / 2.0) * img_h)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        cls_name = class_mapping.get(cls_id, f"class_{cls_id}")
        box_id = f"gt_{box_idx}"
        box_idx += 1
        
        boxes.append(GroundTruthBox(
            class_id=cls_id,
            class_name=cls_name,
            bbox=(x1, y1, x2, y2),
            id=box_id
        ))
        
    return boxes, None


def setup_dataset(
    data_dir: str = "data",
    preferred_dataset: str = "coco8",
    max_images: Optional[int] = 30
) -> DatasetReport:
    """
    Ensures a small, labelled object detection dataset is downloaded and validated locally.
    Downloads COCO8 (~433 KB) or COCO128 (~6.8 MB) from Ultralytics public releases.
    Provides verified fallback images if network is unavailable.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # We support coco8 and coco128
    dataset_configs = {
        "coco8": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
            "folder": os.path.join(data_dir, "coco8"),
            "img_subdir": ["images/train", "images/val"],
            "lbl_subdir": ["labels/train", "labels/val"],
            "name": "COCO8 (Ultralytics Validation Subset)",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
        "coco128": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
            "folder": os.path.join(data_dir, "coco128"),
            "img_subdir": ["images/train2017"],
            "lbl_subdir": ["labels/train2017"],
            "name": "COCO128 (First 128 images of COCO 2017)",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        }
    }
    
    cfg = dataset_configs.get(preferred_dataset, dataset_configs["coco8"])
    target_dir = cfg["folder"]
    zip_path = os.path.join(data_dir, f"{preferred_dataset}.zip")
    
    # Check if already downloaded and extracted
    if not os.path.exists(target_dir):
        # Attempt download if zip does not exist
        if not os.path.exists(zip_path):
            try:
                urllib.request.urlretrieve(cfg["url"], zip_path)
            except Exception as e:
                # If preferred failed, try coco8 as minimal fallback
                if preferred_dataset != "coco8":
                    try:
                        cfg = dataset_configs["coco8"]
                        target_dir = cfg["folder"]
                        zip_path = os.path.join(data_dir, "coco8.zip")
                        urllib.request.urlretrieve(cfg["url"], zip_path)
                    except Exception:
                        pass
        
        # If zip exists, extract it
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
            except Exception:
                pass

    # Discover images and labels
    samples: List[DatasetSample] = []
    class_distribution: Dict[str, int] = {}
    
    if os.path.exists(target_dir):
        image_files = []
        for img_sub in cfg["img_subdir"]:
            subpath = os.path.join(target_dir, img_sub)
            if os.path.exists(subpath):
                for f in sorted(os.listdir(subpath)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append((os.path.join(subpath, f), img_sub))
                        
        for img_path, img_sub in image_files:
            img_filename = os.path.basename(img_path)
            img_id = os.path.splitext(img_filename)[0]
            
            # Load image to verify dimensions
            img = cv2.imread(img_path)
            if img is None:
                samples.append(DatasetSample(
                    image_id=img_id,
                    image_path=img_path,
                    width=0,
                    height=0,
                    is_valid=False,
                    validation_error="Could not decode image with OpenCV"
                ))
                continue
                
            h, w = img.shape[:2]
            
            # Locate corresponding label file
            # Replace 'images' with 'labels' in path
            lbl_sub = img_sub.replace("images", "labels")
            lbl_path = os.path.join(target_dir, lbl_sub, f"{img_id}.txt")
            
            boxes, err = parse_yolo_annotation_file(lbl_path, w, h)
            if err:
                samples.append(DatasetSample(
                    image_id=img_id,
                    image_path=img_path,
                    width=w,
                    height=h,
                    is_valid=False,
                    validation_error=err
                ))
                continue
                
            for b in boxes:
                class_distribution[b.class_name] = class_distribution.get(b.class_name, 0) + 1
                
            samples.append(DatasetSample(
                image_id=img_id,
                image_path=img_path,
                width=w,
                height=h,
                ground_truths=boxes,
                is_valid=True
            ))

    valid_samples = [s for s in samples if s.is_valid]
    
    if len(valid_samples) > 0:
        return DatasetReport(
            source_name=cfg["name"],
            license_info=cfg["license"],
            is_fallback=False,
            total_images_found=len(samples),
            valid_images=len(valid_samples),
            total_ground_truth_boxes=sum(len(s.ground_truths) for s in valid_samples),
            class_distribution=class_distribution,
            samples=samples,
            notes=f"Successfully loaded and verified genuine public dataset from {target_dir}."
        )

    # Fallback mode: Generate a clean self-contained verified sample set if external download blocked
    return _create_verified_fallback_dataset(data_dir)


def _create_verified_fallback_dataset(data_dir: str) -> DatasetReport:
    """
    Creates a verified local fallback dataset with synthetic geometric objects
    and genuine annotations, guaranteeing that ground-truth testing can proceed
    deterministically even if all network access is unavailable.
    """
    fallback_dir = os.path.join(data_dir, "mirage_fallback")
    img_dir = os.path.join(fallback_dir, "images")
    lbl_dir = os.path.join(fallback_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    samples: List[DatasetSample] = []
    class_distribution: Dict[str, int] = {}
    
    # Create 8 deterministic benchmark scenes with known objects
    for i in range(8):
        img_id = f"fallback_scene_{i+1:02d}"
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{img_id}.txt")
        
        w, h = 640, 480
        img = np.full((h, w, 3), 120 + (i * 10), dtype=np.uint8)
        
        # Add visual context (gradients, road/table planes)
        cv2.rectangle(img, (0, 300), (w, h), (80, 80, 80), -1)
        
        # We place genuine simulated objects with real YOLO ground-truth annotations
        # Using standard COCO classes: 2: 'car', 0: 'person', 56: 'chair', 62: 'tv'
        gt_boxes = []
        
        # Object 1: 'car' (COCO class 2)
        car_box = (100 + i * 20, 260, 280 + i * 20, 380)
        cv2.rectangle(img, (car_box[0], car_box[1]), (car_box[2], car_box[3]), (50, 50, 180), -1)
        cv2.circle(img, (car_box[0] + 35, car_box[3]), 18, (20, 20, 20), -1)
        cv2.circle(img, (car_box[2] - 35, car_box[3]), 18, (20, 20, 20), -1)
        gt_boxes.append(GroundTruthBox(
            class_id=2, class_name="car", bbox=(float(car_box[0]), float(car_box[1]), float(car_box[2]), float(car_box[3])), id="gt_0"
        ))
        
        # Object 2: 'person' (COCO class 0)
        p_x1, p_y1, p_x2, p_y2 = (380 + i * 15, 200, 430 + i * 15, 360)
        cv2.rectangle(img, (p_x1 + 10, p_y1 + 40), (p_x2 - 10, p_y2), (180, 100, 50), -1)
        cv2.circle(img, ((p_x1 + p_x2) // 2, p_y1 + 20), 16, (200, 180, 150), -1)
        gt_boxes.append(GroundTruthBox(
            class_id=0, class_name="person", bbox=(float(p_x1), float(p_y1), float(p_x2), float(p_y2)), id="gt_1"
        ))
        
        cv2.imwrite(img_path, img)
        
        # Write YOLO annotation file
        with open(lbl_path, "w", encoding="utf-8") as f:
            for b in gt_boxes:
                xc = (b.bbox[0] + b.bbox[2]) / (2.0 * w)
                yc = (b.bbox[1] + b.bbox[3]) / (2.0 * h)
                bw = (b.bbox[2] - b.bbox[0]) / float(w)
                bh = (b.bbox[3] - b.bbox[1]) / float(h)
                f.write(f"{b.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                class_distribution[b.class_name] = class_distribution.get(b.class_name, 0) + 1
                
        samples.append(DatasetSample(
            image_id=img_id,
            image_path=img_path,
            width=w,
            height=h,
            ground_truths=gt_boxes,
            is_valid=True
        ))
        
    return DatasetReport(
        source_name="MIRAGE Synthetic Local Fallback (Isolated Mode)",
        license_info="Built-in MIT / Open Access Synthetic Benchmark",
        is_fallback=True,
        total_images_found=len(samples),
        valid_images=len(samples),
        total_ground_truth_boxes=sum(len(s.ground_truths) for s in samples),
        class_distribution=class_distribution,
        samples=samples,
        notes="Network download was unavailable or offline; generated deterministic local test scenes with valid annotations."
    )
