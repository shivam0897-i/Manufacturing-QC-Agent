# Vision module for solar panel defect detection
from .detector import DefectDetector, get_detector, DEFECT_CLASSES, SEVERITY_MAP
from .rgb_classifier import RGBClassifier, get_rgb_classifier, CLASS_NAMES as RGB_CLASSES

__all__ = [
    'DefectDetector', 'get_detector', 'DEFECT_CLASSES', 'SEVERITY_MAP',
    'RGBClassifier', 'get_rgb_classifier', 'RGB_CLASSES'
]
