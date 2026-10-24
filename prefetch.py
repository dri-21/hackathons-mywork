"""
MIRAGE - Asset Prefetch Script
Pre-downloads the pretrained model and validation dataset for immediate local use.
"""

import os
import urllib.request
import zipfile
from ultralytics import YOLO

os.makedirs("data/models", exist_ok=True)

print("1. Downloading / caching YOLO model (yolo11n.pt)...")
model = YOLO("yolo11n.pt")
print(f"Model ready: {model.names[0]} to {model.names[len(model.names)-1]} ({len(model.names)} classes)")

print("2. Downloading / caching dataset (coco8)...")
coco8_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
zip_path = "data/coco8.zip"
if not os.path.exists("data/coco8"):
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(coco8_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("COCO8 dataset extracted successfully to data/coco8")
else:
    print("COCO8 dataset already present.")

print("All assets successfully downloaded and cached.")
