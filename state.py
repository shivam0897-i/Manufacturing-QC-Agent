"""
QC Agent State
==============

Domain-specific state for Manufacturing QC Agent.
"""

from typing import Dict, Any, List, Optional, TypedDict
from point9_platform.agent.state import BaseAgentState


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


class QCAgentState(BaseAgentState):
    """
    State for Manufacturing QC Agent.
    
    Extends BaseAgentState with QC-specific fields for defects,
    log anomalies, and optimization recommendations.
    Base fields (messages, session_id, should_continue, error, iteration,
    max_iterations, model, plan, current_step, current_task, thoughts,
    results, documents) are inherited from BaseAgentState.
    """
    
    # === QC-SPECIFIC FIELDS ===
    needs_human_input: bool
    defects: List[DefectInfo]           # Detected defects from images
    log_anomalies: List[LogAnomaly]     # Anomalies from production logs
    recommendations: List[Recommendation]  # Optimization suggestions
