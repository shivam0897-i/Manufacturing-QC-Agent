"""
RGB Image Classifier using EfficientNet B0
===========================================

Classifies RGB solar panel images for surface contamination and damage.

Classes:
- Bird-drop: Bird droppings on panels
- Clean: No issues detected
- Dusty: Dust accumulation
- Electrical-damage: Visible electrical damage
- Physical-Damage: Physical damage to panels
- Snow-Covered: Snow covering panels
"""

import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Class names from the trained model
CLASS_NAMES = [
    'Bird-drop',
    'Clean', 
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

# Severity mapping for each class
SEVERITY_MAP = {
    'Bird-drop': 'medium',
    'Clean': 'none',
    'Dusty': 'low',
    'Electrical-damage': 'critical',
    'Physical-Damage': 'high',
    'Snow-Covered': 'medium'
}

# Default model path
DEFAULT_MODEL_PATH = "pv_defect_efficientnet_b0_97.pth.zip"

# Image preprocessing (EfficientNet B0 standard)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class RGBClassifier:
    """
    EfficientNet B0 classifier for RGB solar panel images.
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = CLASS_NAMES
        self._loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self) -> bool:
        """Load the EfficientNet model."""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model not found: {self.model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get class names from checkpoint
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            
            # Create EfficientNet B0 model
            self.model = models.efficientnet_b0(weights=None)
            
            # Modify classifier for our number of classes
            num_classes = len(self.class_names)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 
                num_classes
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print(f"RGB Classifier loaded: {self.model_path} ({num_classes} classes)")
            return True
            
        except Exception as e:
            print(f"Failed to load RGB classifier: {e}")
            return False
    
    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify an RGB image.
        
        Args:
            image_path: Path to the RGB image
            
        Returns:
            Classification results with class, confidence, and all probabilities
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
                "error": "Model not loaded"
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs = probabilities[0].cpu().numpy()
            
            # Get top prediction
            top_idx = int(np.argmax(probs))
            top_class = self.class_names[top_idx]
            top_confidence = float(probs[top_idx])
            
            # Get all class probabilities
            all_predictions = [
                {
                    "class": self.class_names[i],
                    "confidence": round(float(probs[i]), 4),
                    "severity": SEVERITY_MAP.get(self.class_names[i], "unknown")
                }
                for i in range(len(self.class_names))
            ]
            all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Determine if there's an issue
            has_issue = top_class != "Clean" and top_confidence >= self.confidence_threshold
            
            return {
                "status": "success",
                "image_path": image_path,
                "model_type": "efficientnet",
                "predicted_class": top_class,
                "confidence": round(top_confidence, 4),
                "severity": SEVERITY_MAP.get(top_class, "unknown"),
                "has_issue": has_issue,
                "all_predictions": all_predictions,
                "model_used": self.model_path
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Classification failed: {str(e)}"
            }
    
    def classify_and_annotate(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Classify image and create annotated version with label overlay.
        """
        result = self.classify(image_path)
        
        if result["status"] != "success":
            return result
        
        try:
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Create label text
            label = f"{result['predicted_class']}: {result['confidence']*100:.1f}%"
            severity = result['severity']
            
            # Color based on severity
            color_map = {
                'none': (0, 255, 0),      # Green
                'low': (255, 255, 0),     # Yellow
                'medium': (255, 165, 0),  # Orange
                'high': (255, 0, 0),      # Red
                'critical': (139, 0, 0)   # Dark Red
            }
            color = color_map.get(severity, (255, 255, 255))
            
            # Draw label background
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((10, 10), label, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=color)
            draw.text((10, 10), label, fill=(0, 0, 0), font=font)
            
            # Save annotated image
            input_filename = Path(image_path).stem
            annotated_filename = f"{input_filename}_classified.jpg"
            annotated_path = os.path.join(output_dir, annotated_filename)
            image.save(annotated_path)
            
            result["annotated_image_path"] = annotated_path
            return result
            
        except Exception as e:
            # Return classification result even if annotation fails
            result["annotation_error"] = str(e)
            return result


# Singleton instance
_classifier_instance: Optional[RGBClassifier] = None


def get_rgb_classifier(model_path: str = None) -> RGBClassifier:
    """Get or create classifier instance."""
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = RGBClassifier(model_path)
    
    return _classifier_instance


def is_grayscale(image_path: str) -> bool:
    """Check if an image is grayscale."""
    try:
        img = Image.open(image_path)
        
        # Check mode
        if img.mode == 'L':
            return True
        
        if img.mode == 'RGB':
            # Check if R == G == B for all pixels (sample check)
            img_array = np.array(img)
            # Sample 100 random pixels
            h, w = img_array.shape[:2]
            sample_size = min(100, h * w)
            indices = np.random.choice(h * w, sample_size, replace=False)
            
            for idx in indices:
                y, x = divmod(idx, w)
                r, g, b = img_array[y, x]
                if not (r == g == b):
                    return False
            return True
        
        return False
    except Exception:
        return False
