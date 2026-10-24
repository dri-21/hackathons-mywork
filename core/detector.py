"""
MIRAGE - Object Detection Engine
Wraps Ultralytics YOLO with:
- Automatic model loading and local weight caching
- CPU-friendly nano model selection (YOLO11n / YOLOv8n)
- Startup smoke testing with real inference validation
- Safe structured prediction conversion (PredictionBox)
- Comprehensive device and model status reporting
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from core.evaluator import PredictionBox

# Ensure Ultralytics config is cached locally in workspace
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath("data/.ultralytics"))


class ObjectDetector:
    """
    Wrapper for Ultralytics YOLO object detection models.
    """
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        weights_dir: str = "data/models",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.weights_dir = weights_dir
        self.device = device
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.is_ready = False
        self.load_error: Optional[str] = None
        
        os.makedirs(self.weights_dir, exist_ok=True)
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the YOLO model using the official Ultralytics loader.
        Falls back to yolov8n.pt if the requested model fails to load.
        """
        try:
            from ultralytics import YOLO
            
            # Prefer weight path in weights_dir if present
            target_path = os.path.join(self.weights_dir, self.model_name)
            load_target = target_path if os.path.exists(target_path) else self.model_name
            
            self.model = YOLO(load_target)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            self.is_ready = True
            self.load_error = None
        except Exception as e:
            # Attempt fallback to yolov8n.pt if primary model failed
            if self.model_name != "yolov8n.pt":
                try:
                    from ultralytics import YOLO
                    fallback_target = os.path.join(self.weights_dir, "yolov8n.pt")
                    load_target = fallback_target if os.path.exists(fallback_target) else "yolov8n.pt"
                    self.model = YOLO(load_target)
                    self.model_name = "yolov8n.pt"
                    self.class_names = self.model.names if hasattr(self.model, "names") else {}
                    self.is_ready = True
                    self.load_error = None
                    return
                except Exception as fb_err:
                    self.load_error = f"Primary failed ({str(e)}); Fallback failed ({str(fb_err)})"
            else:
                self.load_error = str(e)
            self.is_ready = False

    def smoke_test(self) -> Dict[str, Any]:
        """
        Performs a startup smoke test using a synthetic frame to verify inference works end-to-end.
        """
        if not self.is_ready or self.model is None:
            return {
                "success": False,
                "error": self.load_error or "Model is not loaded",
                "model_name": self.model_name,
                "device": self.device,
            }
            
        try:
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            # Draw a bright circle and rectangle so the network processes real signal
            dummy_img[50:150, 50:150] = [200, 200, 200]
            dummy_img[180:280, 180:280] = [100, 150, 200]
            
            t0 = time.perf_counter()
            _ = self.model(dummy_img, device=self.device, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            return {
                "success": True,
                "model_name": self.model_name,
                "device": self.device,
                "classes_count": len(self.class_names),
                "smoke_latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "device": self.device,
            }

    def predict(
        self,
        image: Union[np.ndarray, str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[PredictionBox]:
        """
        Runs real YOLO inference on an image array or file path.
        Returns standardized PredictionBox objects.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(f"Detector is not ready: {self.load_error}")
            
        # Run inference
        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        predictions: List[PredictionBox] = []
        if not results or len(results) == 0:
            return predictions
            
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return predictions
            
        boxes_data = res.boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = res.boxes.conf.cpu().numpy()       # [N]
        classes = res.boxes.cls.cpu().numpy().astype(int)  # [N]
        
        for i in range(len(confs)):
            cls_id = int(classes[i])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = [float(v) for v in boxes_data[i]]
            predictions.append(PredictionBox(
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(confs[i]),
                bbox=(x1, y1, x2, y2)
            ))
            
        return predictions

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_ready": self.is_ready,
            "error": self.load_error,
            "num_classes": len(self.class_names),
        }
