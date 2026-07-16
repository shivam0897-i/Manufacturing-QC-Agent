"""
Vision Detector Module
======================

YOLOv8-based defect detection for solar module EL images.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Defect class names (must match dataset.yaml)
DEFECT_CLASSES = [
    'black_core',
    'corner',
    'crack',
    'finger',
    'fragment',
    'horizontal_dislocation',
    'printing_error',
    'scratch',
    'short_circuit',
    'star_crack',
    'thick_line',
    'vertical_dislocation'
]

# Severity mapping based on defect type
SEVERITY_MAP = {
    'short_circuit': 'critical',
    'fragment': 'critical',
    'crack': 'high',
    'star_crack': 'high',
    'black_core': 'high',
    'finger': 'medium',
    'thick_line': 'medium',
    'printing_error': 'medium',
    'corner': 'medium',
    'horizontal_dislocation': 'low',
    'vertical_dislocation': 'low',
    'scratch': 'low'
}


class DefectDetector:
    """YOLOv8-based solar panel defect detector."""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            confidence_threshold: Minimum confidence for detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._loaded = False

    def _defects_from_results(self, results) -> List[Dict[str, Any]]:
        defects = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                if confidence < self.confidence_threshold:
                    continue

                bbox = boxes.xyxy[i].tolist()
                class_name = DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else f"class_{cls_id}"

                defects.append({
                    "defect_type": class_name,
                    "class_id": cls_id,
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x1": round(bbox[0], 1),
                        "y1": round(bbox[1], 1),
                        "x2": round(bbox[2], 1),
                        "y2": round(bbox[3], 1),
                        "width": round(bbox[2] - bbox[0], 1),
                        "height": round(bbox[3] - bbox[1], 1)
                    },
                    "severity": SEVERITY_MAP.get(class_name, "medium")
                })

        defects.sort(key=lambda x: x["confidence"], reverse=True)
        return defects

    @staticmethod
    def _severity_counts(defects: List[Dict[str, Any]]) -> Dict[str, int]:
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for defect in defects:
            severity = defect.get("severity", "medium")
            if severity in severity_counts:
                severity_counts[severity] += 1
        return severity_counts

    @staticmethod
    def _save_annotated_image(image_path: str, annotated_path: str, defects: List[Dict[str, Any]]) -> None:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        for defect in defects:
            bbox = defect.get("bounding_box", {})
            x1 = int(round(bbox.get("x1", 0)))
            y1 = int(round(bbox.get("y1", 0)))
            x2 = int(round(bbox.get("x2", 0)))
            y2 = int(round(bbox.get("y2", 0)))

            label = f"{defect.get('defect_type', 'defect')} {defect.get('confidence', 0):.2f}"
            color = (255, 255, 255)
            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

            text_box = draw.textbbox((x1, y1), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_y = max(0, y1 - text_height - 6)
            draw.rectangle([x1, label_y, x1 + text_width + 6, label_y + text_height + 6], fill=color)
            draw.text((x1 + 3, label_y + 3), label, fill=(0, 0, 0), font=font)

        image.save(annotated_path, quality=95)
        
    def load_model(self) -> bool:
        """Load the YOLOv8 model."""
        if self._loaded:
            return True
            
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Model not found: {self.model_path}")
            return False

        if YOLO is None:
            print("YOLO dependency not installed: ultralytics")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            self._loaded = True
            print(f"Model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect defects in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Detection results with defects, confidence, and bounding boxes
        """
        if not os.path.exists(image_path):
            return {
                "status": "failed",
                "error": f"Image not found: {image_path}"
            }
        
        # Ensure model is loaded
        if not self._loaded and not self.load_model():
            return {
                "status": "failed", 
                "error": "Model not loaded. Please provide a valid model path."
            }
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            defects = self._defects_from_results(results)
            severity_counts = self._severity_counts(defects)
            
            return {
                "status": "success",
                "image_path": image_path,
                "defects": defects,
                "total_defects": len(defects),
                "has_defects": len(defects) > 0,
                "severity_counts": severity_counts,
                "model_used": self.model_path
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Detection failed: {str(e)}"
            }
    
    def detect_and_annotate(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Detect defects and save annotated image with bounding boxes.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save annotated image (defaults to temp)
            
        Returns:
            Detection results plus path to annotated image
        """
        if not os.path.exists(image_path):
            return {
                "status": "failed",
                "error": f"Image not found: {image_path}"
            }
        
        # Ensure model is loaded
        if not self._loaded and not self.load_model():
            return {
                "status": "failed", 
                "error": "Model not loaded. Please provide a valid model path."
            }
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            defects = self._defects_from_results(results)
            severity_counts = self._severity_counts(defects)
            
            # Save annotated image
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate annotated filename
            input_filename = Path(image_path).stem
            annotated_filename = f"{input_filename}_annotated.jpg"
            annotated_path = os.path.join(output_dir, annotated_filename)
            
            self._save_annotated_image(image_path, annotated_path, defects)
            
            return {
                "status": "success",
                "image_path": image_path,
                "annotated_image_path": annotated_path,
                "defects": defects,
                "total_defects": len(defects),
                "has_defects": len(defects) > 0,
                "severity_counts": severity_counts,
                "model_used": self.model_path
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Detection and annotation failed: {str(e)}"
            }
    
    def detect_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Detect defects in multiple images."""
        return [self.detect(path) for path in image_paths]


# Singleton instance
_detector_instance: Optional[DefectDetector] = None


def get_detector(model_path: Optional[str] = None) -> DefectDetector:
    """Get or create detector instance."""
    global _detector_instance
    
    if _detector_instance is None:
        # Default model path
        if model_path is None:
            # Look for production model first (Root or Training paths)
            root_model = "best.pt"
            production_model = "training/models/pvel_yolov8_production/weights/best.pt"
            baseline_model = "runs/detect/training/models/pvel_yolov8/weights/best.pt"
            
            if os.path.exists(root_model):
                model_path = root_model
            elif os.path.exists(production_model):
                model_path = production_model
            elif os.path.exists(baseline_model):
                model_path = baseline_model
        
        _detector_instance = DefectDetector(model_path)
    
    return _detector_instance
