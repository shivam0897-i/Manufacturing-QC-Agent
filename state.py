"""
QC Agent State
==============

Domain-specific state for Manufacturing QC Agent.
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict
from point9_platform.agent.state import BaseAgentState, DocumentInfo, ProcessingResult, message_reducer


class DefectInfo(TypedDict):
    """Information about a detected defect."""
    defect_type: str          # e.g., "crack", "black_core"
    confidence: float         # 0.0 to 1.0
    image_id: str
    bounding_box: Optional[Dict[str, int]]  # {x, y, width, height}
    severity: Optional[str]   # "low", "medium", "high"


class LogAnomaly(TypedDict):
    """Information about a log anomaly."""
    anomaly_type: str         # e.g., "temp_spike", "speed_variation"
    timestamp: str
    value: Any
    expected_range: Optional[str]
    severity: str


class Recommendation(TypedDict):
    """Optimization recommendation."""
    recommendation: str
    priority: str             # "high", "medium", "low"
    parameter: Optional[str]  # e.g., "lamination_temperature"
    suggested_value: Optional[str]
    rationale: str


class QCAgentState(TypedDict):
    """
    State for Manufacturing QC Agent.
    
    Extends base state with QC-specific fields for defects,
    log anomalies, and optimization recommendations.
    """
    
    # === BASE FIELDS (required) ===
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    session_id: str
    should_continue: bool
    error: Optional[str]
    iteration: int
    max_iterations: int
    model: str
    
    # === DOCUMENT FIELDS ===
    documents: Dict[str, DocumentInfo]
    results: Dict[str, ProcessingResult]
    
    # === PLANNING FIELDS ===
    plan: List[str]
    current_step: int
    current_task: Optional[str]
    
    # === DEBUG/AUDIT FIELDS ===
    thoughts: List[str]
    needs_human_input: bool
    
    # === QC-SPECIFIC FIELDS ===
    defects: List[DefectInfo]           # Detected defects from images
    log_anomalies: List[LogAnomaly]     # Anomalies from production logs
    recommendations: List[Recommendation]  # Optimization suggestions
