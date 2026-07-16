"""
QC Agent Settings
=================

Domain-specific settings for Manufacturing QC Agent.
"""

from point9_platform.settings.user import UserSettings
from typing import Optional, List


class QCSettings(UserSettings):
    """
    Manufacturing QC Agent settings.
    
    Extends UserSettings with QC-specific configuration.
    """
    
    # === MODEL ===
    DEFAULT_LLM_MODEL: str = "gemini/gemini-2.5-pro"
    RECOMMENDATION_LLM_MODEL: str = "gemini/gemini-2.5-flash"
    
    # === AGENT IDENTITY ===
    AGENT_NAME: str = "Manufacturing QC Agent"
    AGENT_DESCRIPTION: str = "AI agent for solar module defect detection and process optimization"
    
    # === DEFECT DETECTION ===
    CONFIDENCE_THRESHOLD: float = 0.5
    FALSE_POSITIVE_TARGET: float = 0.05
    
    DEFECT_CATEGORIES: List[str] = [
        "black_core",
        "corner", 
        "crack",
        "finger",
        "fragment",
        "horizontal_dislocation",
        "printing_error",
        "scratch",
        "short_circuit",
        "star_crack",
        "thick_line",
        "vertical_dislocation"
    ]
    
    # === DOMAIN CONFIGURATION ===
    ALLOWED_OPERATIONS: List[str] = [
        "analyze_image",
        "analyze_logs",
        "query_knowledge",
        "recommend_optimization",
        "generate_report",
    ]
    
    DOMAIN_KEYWORDS: List[str] = [
        "defect", "crack", "solar", "module", "quality",
        "analyze", "optimize", "log", "recommendation"
    ]
    
    # === EXTERNAL SERVICES (set in .env or config.yaml) ===
    MODEL_PATH: Optional[str] = None  # Path to TFLite/ONNX model
    KNOWLEDGE_BASE_PATH: Optional[str] = None  # Path to manual embeddings
    
    # === S3 STORAGE ===
    S3_BUCKET_NAME: Optional[str] = None  # S3 bucket for file storage
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # === MONGODB ===
    MONGODB_URI: Optional[str] = None  # MongoDB connection URI
    MONGODB_DB: str = "qc_agent"
    MONGODB_COLLECTION: str = "sessions"
    
    # === STORAGE FLAGS ===
    ENABLE_S3_STORAGE: bool = True  # Set to True when S3 is configured
    ENABLE_MONGODB: bool = True  # Set to True when MongoDB is configured

    # === OBSERVABILITY ===
    ENABLE_MLFLOW: bool = False
    MLFLOW_TRACKING_URI: Optional[str] = None
    MLFLOW_EXPERIMENT_NAME: str = "manufacturing-qc-agent"
    
    class Config(UserSettings.Config):
        env_prefix = "QC_"
