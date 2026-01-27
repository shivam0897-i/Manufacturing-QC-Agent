"""
Analyze Image Tool
==================

Detects defects in solar module EL images using YOLOv8.
"""

import os
from typing import Dict, Any, List
from point9_platform.tools.decorator import tool
from tools._utils import find_document, get_available_doc_ids

# Import detector
try:
    from vision.detector import get_detector, DEFECT_CLASSES, SEVERITY_MAP
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False


# Default model paths (relative to project root)
MODEL_PATHS = [
    "best.pt",                                                     # Production root path
    "models/best.pt",                                              # Deployment path
    "dev/training/models/pvel_yolov8_production/weights/best.pt",  # Dev path 1
    "dev/runs/detect/training/models/pvel_yolov8/weights/best.pt", # Dev path 2
    "training/models/pvel_yolov8_production/weights/best.pt",      # Legacy path
    "runs/detect/training/models/pvel_yolov8/weights/best.pt",     # Legacy path
]


def find_model_path() -> str:
    """Find the best available model."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


@tool(
    name="analyze_image",
    description="Analyze solar module EL images for defects. Can process a single image ('image_id'), a batch ('image_ids'), or ALL images if no arguments are provided.",
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
                "description": "Minimum confidence for detection (default 0.25)"
            }
        },
        "required": []
    }
)
def analyze_image(
    image_id: str = None, 
    image_ids: List[str] = None,
    confidence_threshold: float = 0.25,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze solar module images for defects using YOLOv8.
    Supports both single-file and batch processing.
    
    Args:
        image_id: Single document ID (e.g., 'doc_1')
        image_ids: List of document IDs for batch processing
        confidence_threshold: Minimum confidence for detection (default 0.25)
        state: Current agent state (injected by executor)
    
    Returns:
        Detection result with summary and details for all processed images.
    """
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
        
    # Find model
    model_path = find_model_path()
    if not model_path:
        return {
            "status": "failed",
            "error": "No trained model found. Please train the model first."
        }

    # Initialize results container
    batch_results = []
    summary = {
        "total_images": len(targets),
        "processed": 0,
        "failed": 0,
        "images_with_defects": 0,
        "total_defects": 0,
        "severity_counts": {}
    }
    
    try:
        # Load model ONCE for the batch
        detector = get_detector(model_path)
        detector.confidence_threshold = confidence_threshold
        
        if not detector.load_model():
            return {
                "status": "failed",
                "error": f"Failed to load model: {model_path}"
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
            
            # Run detection
            try:
                result = detector.detect(image_path)
                
                if result["status"] == "success":
                    # Update summary
                    summary["processed"] += 1
                    summary["total_defects"] += result["total_defects"]
                    if result["has_defects"]:
                        summary["images_with_defects"] += 1
                    
                    # Merge severity counts
                    for sev, count in result.get("severity_counts", {}).items():
                        summary["severity_counts"][sev] = summary["severity_counts"].get(sev, 0) + count
                    
                    batch_results.append({
                        "image_id": target_id,
                        "filename": filename,
                        "status": "success",
                        "defects": result["defects"],
                        "severity_summary": result.get("severity_counts", {}),
                        "has_defects": result["has_defects"]
                    })
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

    # Final Output Construction
    return {
        "status": "success",
        "summary": summary,
        "results": batch_results,
        "model_used": os.path.basename(model_path),
        "confidence_threshold": confidence_threshold
    }

