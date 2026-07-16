"""
Analyze Image Tool
==================

Detects defects in solar module images using:
- YOLOv8: For EL grayscale images (manufacturing defects)
- EfficientNet: For RGB photos (surface contamination)
"""

import os
import tempfile
from typing import Dict, Any, List
from point9_platform.tools.decorator import tool
from settings import QCSettings
from tools._utils import find_document

try:
    from point9_platform.storage import get_s3_storage
except ImportError:
    get_s3_storage = None

# Import YOLOv8 detector
try:
    from vision.detector import get_detector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False

# Import EfficientNet classifier
try:
    from vision.rgb_classifier import get_rgb_classifier, is_grayscale, CLASS_NAMES as RGB_CLASSES
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False


# Default model paths (relative to project root)
MODEL_PATHS = [
    "best.pt",                                                     # Production root path
    "models/best.pt",                                              # Deployment path
    "dev/training/models/pvel_yolov8_production/weights/best.pt",  # Dev path 1
    "dev/runs/detect/training/models/pvel_yolov8/weights/best.pt", # Dev path 2
    "training/models/pvel_yolov8_production/weights/best.pt",      # Legacy path
    "runs/detect/training/models/pvel_yolov8/weights/best.pt",     # Legacy path
]

RGB_MODEL_PATHS = [
    "pv_defect_efficientnet_b0_97.pth.zip",                        # Production root path
    "models/pv_defect_efficientnet_b0_97.pth.zip",                 # Deployment path
]


def find_model_path() -> str:
    """Find the best available YOLOv8 model."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def find_rgb_model_path() -> str:
    """Find the best available EfficientNet model."""
    for path in RGB_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def _resolve_confidence_threshold(confidence_threshold: float = None) -> float:
    if confidence_threshold is None:
        return QCSettings().CONFIDENCE_THRESHOLD
    return confidence_threshold


@tool(
    name="analyze_image",
    description="Analyze solar module images for defects. Supports EL grayscale (YOLOv8) and RGB photos (EfficientNet). Can process single image, batch, or ALL images.",
    parameters={
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string", 
                "description": "Single document ID (e.g. 'doc_1')"
            },
            "image_ids": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of document IDs for batch processing"
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence for detection (defaults to configured QC confidence threshold)"
            },
            "model_type": {
                "type": "string",
                "enum": ["auto", "yolo", "efficientnet"],
                "description": "Model to use: 'auto' (detect image type), 'yolo' (EL images), 'efficientnet' (RGB photos). Default: auto"
            }
        },
        "required": []
    }
)
def analyze_image(
    image_id: str = None, 
    image_ids: List[str] = None,
    confidence_threshold: float = None,
    model_type: str = "auto",
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze solar module images for defects.
    
    Supports dual-model analysis:
    - YOLOv8: For EL grayscale images (12 manufacturing defects)
    - EfficientNet: For RGB photos (6 surface conditions)
    
    Args:
        image_id: Single document ID (e.g., 'doc_1')
        image_ids: List of document IDs for batch processing
        confidence_threshold: Minimum confidence for detection. Uses configured default when omitted.
        model_type: 'auto' (detect image type), 'yolo', or 'efficientnet'
        state: Current agent state (injected by executor)
    
    Returns:
        Detection result with summary and details for all processed images.
    """
    confidence_threshold = _resolve_confidence_threshold(confidence_threshold)
    documents = state.get("documents", {})
    
    # Resolve input targets
    targets = []
    if image_ids:
        targets = image_ids
    elif image_id:
        targets = [image_id]
        
    # Auto-discovery logic (Explicit None OR Fallback for invalid IDs)
    use_auto_discovery = False
    
    if not targets:
        use_auto_discovery = True
    else:
        # Check if targets are valid. If ALL are missing, likely hallucination.
        valid_count = 0
        for t in targets:
            if find_document(t, documents):
                valid_count += 1
        
        if valid_count == 0 and len(targets) > 0:
            print(f"Warning: None of the provided IDs {targets[:3]}... found. Falling back to auto-discovery.")
            use_auto_discovery = True
            targets = []  # Reset targets
            
    if use_auto_discovery:
        # Process ALL image documents
        for doc_key, doc_info in documents.items():
            fname = doc_info.get("filename", "").lower()
            if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                targets.append(doc_key)
    
    if not targets:
        return {
            "status": "failed",
            "error": "No valid image_ids provided and no image files found in uploaded documents."
        }

    # Check detector availability
    if not DETECTOR_AVAILABLE:
        return {
            "status": "failed",
            "error": "Vision detector module not available. Check installation."
        }
        
    # Find models
    yolo_model_path = find_model_path()
    rgb_model_path = find_rgb_model_path()
    
    # Validate model availability based on requested type
    if model_type == "yolo" and not yolo_model_path:
        return {
            "status": "failed",
            "error": "YOLOv8 model not found. Please ensure best.pt is available."
        }
    if model_type == "efficientnet" and not rgb_model_path:
        return {
            "status": "failed",
            "error": "EfficientNet model not found. Please ensure pv_defect_efficientnet_b0_97.pth.zip is available."
        }
    if model_type == "auto" and not yolo_model_path and not rgb_model_path:
        return {
            "status": "failed",
            "error": "No trained models found. Please train or add model files."
        }

    # Initialize results container
    batch_results = []
    summary = {
        "total_images": len(targets),
        "processed": 0,
        "failed": 0,
        "images_with_defects": 0,
        "total_defects": 0,
        "severity_counts": {},
        "models_used": []
    }
    
    try:
        # Load models based on mode
        detector = None
        classifier = None
        
        if model_type in ["auto", "yolo"] and yolo_model_path and DETECTOR_AVAILABLE:
            detector = get_detector(yolo_model_path)
            detector.confidence_threshold = confidence_threshold
            if not detector.load_model():
                detector = None
            else:
                summary["models_used"].append("yolov8")
        
        if model_type in ["auto", "efficientnet"] and rgb_model_path and CLASSIFIER_AVAILABLE:
            classifier = get_rgb_classifier(rgb_model_path)
            classifier.confidence_threshold = confidence_threshold
            if not classifier.load_model():
                classifier = None
            else:
                summary["models_used"].append("efficientnet")
        
        if not detector and not classifier:
            return {
                "status": "failed",
                "error": "Failed to load any models"
            }
            
        # Process each target
        for target_id in targets:
            # Find doc
            doc_info = find_document(target_id, documents)
            
            if not doc_info:
                batch_results.append({
                    "image_id": target_id, 
                    "status": "failed", 
                    "error": "Document not found"
                })
                summary["failed"] += 1
                continue
                
            image_path = doc_info.get("path")
            filename = doc_info.get("filename", "unknown")
            
            if not image_path or not os.path.exists(image_path):
                batch_results.append({
                    "image_id": target_id, 
                    "status": "failed", 
                    "error": "File not found on disk"
                })
                summary["failed"] += 1
                continue
            
            # Run detection WITH annotation
            try:
                # Create temp output directory for annotated images
                output_dir = tempfile.mkdtemp(prefix="qc_annotated_")
                
                # Determine which model to use
                use_yolo = False
                use_efficientnet = False
                
                if model_type == "yolo":
                    use_yolo = True
                elif model_type == "efficientnet":
                    use_efficientnet = True
                elif model_type == "auto":
                    # Auto-detect based on image characteristics
                    if CLASSIFIER_AVAILABLE and classifier:
                        try:
                            grayscale = is_grayscale(image_path)
                            if grayscale and detector:
                                use_yolo = True
                            elif classifier:
                                use_efficientnet = True
                            elif detector:
                                use_yolo = True
                        except Exception:
                            # Fallback to YOLO if detection fails
                            use_yolo = detector is not None
                    else:
                        use_yolo = detector is not None
                
                result = None
                model_used = None
                
                if use_yolo and detector:
                    result = detector.detect_and_annotate(image_path, output_dir=output_dir)
                    model_used = "yolov8"
                elif use_efficientnet and classifier:
                    result = classifier.classify_and_annotate(image_path, output_dir=output_dir)
                    model_used = "efficientnet"
                    # Normalize result format for classification
                    if result.get("status") == "success":
                        # Convert classification to defect-like format
                        if result.get("has_issue", False):
                            result["defects"] = [{
                                "defect_type": result["predicted_class"],
                                "class_id": RGB_CLASSES.index(result["predicted_class"]) if result["predicted_class"] in RGB_CLASSES else -1,
                                "confidence": result["confidence"],
                                "severity": result["severity"]
                            }]
                            result["total_defects"] = 1
                            result["has_defects"] = True
                        else:
                            result["defects"] = []
                            result["total_defects"] = 0
                            result["has_defects"] = False
                        result["severity_counts"] = {result["severity"]: 1} if result.get("has_issue") else {}
                        # Keep only top 3 predictions to reduce response size
                        all_preds = result.get("all_predictions", [])[:3]
                        result["classification"] = {
                            "predicted_class": result["predicted_class"],
                            "confidence": result["confidence"],
                            "all_predictions": all_preds
                        }
                else:
                    result = {"status": "failed", "error": "No suitable model available for this image"}
                
                if result["status"] == "success":
                    # Update summary
                    summary["processed"] += 1
                    summary["total_defects"] += result["total_defects"]
                    if result["has_defects"]:
                        summary["images_with_defects"] += 1
                    
                    # Merge severity counts
                    for sev, count in result.get("severity_counts", {}).items():
                        summary["severity_counts"][sev] = summary["severity_counts"].get(sev, 0) + count
                    
                    # Upload annotated image to S3 if available
                    annotated_url = None
                    annotated_local_path = result.get("annotated_image_path")
                    
                    if annotated_local_path and os.path.exists(annotated_local_path):
                        session_id = state.get("session_id", "unknown")
                        try:
                            s3_storage = get_s3_storage() if get_s3_storage else None
                            if s3_storage:
                                s3_key = f"outputs/{session_id}/annotated/{os.path.basename(annotated_local_path)}"
                                upload_result = s3_storage.upload_file(annotated_local_path, s3_key)
                                upload_ok = upload_result.get("success") if isinstance(upload_result, dict) else upload_result
                                if upload_ok:
                                    url_result = s3_storage.get_presigned_url(s3_key, expiration=86400)
                                    if isinstance(url_result, dict) and url_result.get("success"):
                                        annotated_url = url_result.get("url")
                                    elif isinstance(url_result, str):
                                        annotated_url = url_result
                        except Exception:
                            pass
                    
                    # Build result - only include annotated_image_path if no URL
                    img_result = {
                        "image_id": target_id,
                        "filename": filename,
                        "status": "success",
                        "model_used": model_used,
                        "defects": result["defects"],
                        "severity_summary": result.get("severity_counts", {}),
                        "has_defects": result["has_defects"]
                    }
                    if result.get("classification"):
                        img_result["classification"] = result["classification"]
                    if annotated_url:
                        img_result["annotated_image_url"] = annotated_url
                    elif annotated_local_path:
                        img_result["annotated_image_path"] = annotated_local_path
                    batch_results.append(img_result)
                else:
                    batch_results.append({
                        "image_id": target_id,
                        "filename": filename,
                        "status": "failed",
                        "error": result.get("error", "Unknown detection error")
                    })
                    summary["failed"] += 1
                    
            except Exception as e:
                batch_results.append({
                    "image_id": target_id,
                    "filename": filename,
                    "status": "failed",
                    "error": str(e)
                })
                summary["failed"] += 1

    except Exception as e:
        return {
            "status": "failed",
            "error": f"Batch processing fatal error: {str(e)}"
        }

    return {
        "status": "success",
        "summary": summary,
        "results": batch_results,
        "models_available": summary.get("models_used", []),
        "confidence_threshold": confidence_threshold
    }

