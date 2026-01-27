# Vision module for solar panel defect detection
from .detector import DefectDetector, get_detector, DEFECT_CLASSES, SEVERITY_MAP

__all__ = ['DefectDetector', 'get_detector', 'DEFECT_CLASSES', 'SEVERITY_MAP']
