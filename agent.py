"""
Manufacturing QC Agent
======================

AI agent for solar module defect detection and process optimization.
"""

from typing import Dict, List, Any
from point9_platform.agent.base import BaseAgent
from state import QCAgentState
from settings import QCSettings
from prompts.templates import PROMPTS


class QCAgent(BaseAgent[QCAgentState]):
    """
    Manufacturing Quality Control Agent.
    
    Analyzes solar module images for defects and provides
    optimization recommendations based on production logs.
    """
    
    def __init__(self, session_id: str):
        super().__init__(
            session_id=session_id,
            tools_package="tools",
            settings=QCSettings()
        )
    
    def get_agent_name(self) -> str:
        return "manufacturing_qc_agent"
    
    def get_domain_keywords(self) -> List[str]:
        return [
            # Defect detection
            "defect", "crack", "damage", "inspection", "quality",
            # Solar/Manufacturing
            "solar", "module", "panel", "cell", "photovoltaic",
            "manufacturing", "production", "assembly", "line",
            # Analysis
            "analyze", "detect", "check", "scan", "image",
            # Optimization
            "optimize", "improve", "recommend", "suggestion",
            # Logs
            "log", "report", "trend", "anomaly", "parameter"
        ]
    
    def create_initial_state(self, session_id: str) -> QCAgentState:
        from point9_platform.settings.system import SYSTEM_SETTINGS
        
        return QCAgentState(
            messages=[],
            session_id=session_id,
            documents=None,
            results={},
            plan=[],
            current_step=0,
            current_task=None,
            thoughts=[],
            should_continue=True,
            needs_human_input=False,
            error=None,
            iteration=0,
            max_iterations=SYSTEM_SETTINGS.MAX_ITERATIONS,
            model=self.settings.DEFAULT_LLM_MODEL,
            # QC-specific fields
            defects=[],
            log_anomalies=[],
            recommendations=[]
        )
    
    def get_prompts(self) -> Dict[str, str]:
        return PROMPTS
