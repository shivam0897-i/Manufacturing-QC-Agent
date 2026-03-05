"""
Manufacturing QC Agent API
==========================

FastAPI endpoints for the Manufacturing QC Agent.
Uses LLM-based planning and tool calling.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime, timezone
import uuid
import os
import shutil
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv()

from agent import QCAgent
from point9_platform.observability.emitter import remove_emitter
from point9_platform.health import create_health_router
from point9_platform.observability import setup_logging, get_logger
from litellm import completion
from settings import QCSettings

# Import storage services from platform
try:
    from point9_platform.storage import get_s3_storage, get_mongo_store
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

# Setup logging BEFORE anything else
setup_logging(level="INFO", agent_name="manufacturing_qc", filter_noise=True)

logger = get_logger("api")

# Load settings
settings = QCSettings()

# Constants
AGENT_NAME = "manufacturing_qc_agent"  # Used to identify sessions in shared MongoDB

# Initialize storage (if enabled)
mongo_store = None
s3_storage = None

if STORAGE_AVAILABLE and settings.ENABLE_MONGODB:
    try:
        mongo_store = get_mongo_store()
        logger.info("MongoDB storage enabled")
    except Exception as e:
        logger.warning("MongoDB not available: %s", e)

if STORAGE_AVAILABLE and settings.ENABLE_S3_STORAGE:
    try:
        s3_storage = get_s3_storage()
        logger.info("S3 storage enabled")
    except Exception as e:
        logger.warning("S3 not available: %s", e)

# In-memory session store (fallback when MongoDB not available)

class SessionData(TypedDict):
    created_at: datetime
    analysis_results: Dict[str, Any]
    defects: List[Dict]
    anomalies: List[Dict]
    recommendations: List[Dict]
    chat_history: List[Dict]  # Track conversation history

_session_store: Dict[str, SessionData] = {}

def save_session(session_id: str, results: Dict, defects: List, anomalies: List, recommendations: List):
    """Save analysis results for chat follow-ups."""
    # Save to MongoDB if available
    if mongo_store:
        mongo_store.store_result(session_id, "final_results", {
            "defects": defects,
            "anomalies": anomalies,
            "recommendations": recommendations,
            "raw_results": results
        })
        mongo_store.update_status(session_id, "completed")
    else:
        # Fallback to in-memory store
        _session_store[session_id] = {
            "created_at": datetime.now(),
            "analysis_results": results,
            "defects": defects,
            "anomalies": anomalies,
            "recommendations": recommendations,
            "chat_history": []  # Initialize empty chat history
        }

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data."""
    # Try MongoDB first
    if mongo_store:
        session = mongo_store.get_session(session_id)
        if session:
            # Extract data from intermediate_results
            intermediate = session.get("intermediate_results", {})
            final = intermediate.get("final_results", {})
            return {
                "created_at": session.get("created_at"),
                "analysis_results": final.get("raw_results", {}),
                "defects": final.get("defects", []),
                "anomalies": final.get("anomalies", []),
                "recommendations": final.get("recommendations", []),
                "chat_history": session.get("chat_history", [])
            }
        return None
    
    # Fallback to in-memory store
    return _session_store.get(session_id)

# OpenAPI tags for organizing endpoints
tags_metadata = [
    {"name": "Session", "description": "Session management - create, list, and clear sessions"},
    {"name": "Agent", "description": "AI agent for defect detection and process optimization"},
    {"name": "Streaming", "description": "Real-time updates via Server-Sent Events"},
]

app = FastAPI(
    title="Manufacturing QC Agent",
    description="AI agent for solar module defect detection and process optimization",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# Include platform health endpoints
app.include_router(create_health_router())


# === EAGER MODEL LOADING ===
@app.on_event("startup")
async def load_models_on_startup():
    """Pre-load defect detection models at server startup."""
    try:
        from vision.detector import get_detector
        from vision.rgb_classifier import get_rgb_classifier
        from tools.analyze_image import find_model_path, find_rgb_model_path
        
        # Load YOLOv8
        yolo_path = find_model_path()
        if yolo_path:
            logger.info("Loading YOLOv8: %s", yolo_path)
            detector = get_detector(yolo_path)
            if detector.load_model():
                logger.info("✅ YOLOv8 ready")
            else:
                logger.warning("⚠️ YOLOv8 failed to load")
        
        # Load EfficientNet
        rgb_path = find_rgb_model_path()
        if rgb_path:
            logger.info("Loading EfficientNet: %s", rgb_path)
            classifier = get_rgb_classifier(rgb_path)
            if classifier.load_model():
                logger.info("✅ EfficientNet ready")
            else:
                logger.warning("⚠️ EfficientNet failed to load")
                
    except ImportError as e:
        logger.warning("⚠️ Vision module not available: %s", e)
    except Exception as e:
        logger.error("❌ Error loading models: %s", e)


@app.post("/session/create", summary="Create a new session for a user", tags=["Session"])
async def create_session():
    """
    Create a new QC session for a user.
    Each user/request gets their own session_id.
    
    Flow:
    1. POST /session/create → Get session_id
    2. GET /stream/{session_id} → Subscribe to SSE
    3. POST /process?session_id={session_id} → Start processing
    """
    from point9_platform.observability.emitter import get_or_create_emitter
    
    session_id = str(uuid.uuid4())
    
    # Register emitter for this session so stream endpoint can find it
    get_or_create_emitter(session_id)
    
    logger.info("[%s] Session created with emitter", session_id)
    
    return {
        "session_id": session_id,
        "stream_url": f"/stream/{session_id}",
        "next_step": "Subscribe to stream_url, then call /process with session_id"
    }


def generate_explanation(defects: list, anomalies: list) -> str:
    """Generate LLM explanation of analysis results for quick review."""
    # Build context
    defect_summary = ""
    if defects:
        defect_types = [d.get("defect_type", "unknown") for d in defects]
        severities = [d.get("severity", "unknown") for d in defects]
        defect_summary = f"Found {len(defects)} defect(s): {', '.join(defect_types)}. Severities: {', '.join(severities)}."
    
    anomaly_summary = ""
    if anomalies:
        critical_count = sum(1 for a in anomalies if a.get("severity") == "critical")
        anomaly_fields = list(set(a.get("field", "unknown") for a in anomalies))
        anomaly_summary = f"Found {len(anomalies)} anomaly(ies) ({critical_count} critical) in: {', '.join(anomaly_fields)}."
    
    if not defects and not anomalies:
        return "No defects or anomalies detected. All systems operating within normal parameters."
    
    prompt = f"""You are a manufacturing QC expert. Provide a brief 2-3 sentence summary of these findings for quick review:

Image Analysis: {defect_summary or "No defects found."}
Log Analysis: {anomaly_summary or "No anomalies found."}

Be concise and actionable."""

    try:
        response = completion(
            model=settings.DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Failed to generate explanation: %s", e)
        # Fallback to simple summary
        return f"Analysis complete: {defect_summary} {anomaly_summary}"


@app.post("/process", summary="Analyze files using AI agent with planning", tags=["Agent"])
async def process_qc(
    message: str = Form("Analyze for defects"),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    Upload and analyze solar module images and/or production logs.
    Uses LLM-based planning and tool calling.
    
    - **message**: Description of what to analyze
    - **session_id**: Optional - use from /start for streaming
    - **files**: Upload images (.jpg, .png) or logs (.csv, .txt)
    """
    # Use provided session_id or generate new one
    if not session_id:
        session_id = str(uuid.uuid4())
    
    documents = {}
    temp_files = []  # Track temp files for cleanup
    input_files_meta = []  # Track file metadata for MongoDB
    
    logger.info("[%s] New request: %s", session_id, message)
    logger.info("[%s] Files received: %s", session_id, len(files))
    
    # Create MongoDB session if available (tagged with agent name)
    if mongo_store:
        # Check if session exists to avoid duplicates (idempotency)
        if not mongo_store.get_session(session_id):
            mongo_store.create_session(session_id, metadata={"agent_name": AGENT_NAME})
        
        mongo_store.add_log(session_id, "upload", f"Processing {len(files)} files")
        mongo_store.update_status(session_id, "processing")
    
    # Save uploaded files to temp storage
    for i, file in enumerate(files):
        if not file.filename:
            continue
            
        file_ext = Path(file.filename).suffix.lower()
        doc_id = f"doc_{i+1}"
        
        # Create temp file (will be cleaned up after processing)
        temp_file = tempfile.NamedTemporaryFile(
            suffix=file_ext, 
            prefix=f"{session_id}_{doc_id}_",
            delete=False  # We'll delete manually after processing
        )
        temp_files.append(temp_file.name)
        
        # Write content
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        file_type = "image" if file_ext in [".jpg", ".jpeg", ".png", ".bmp"] else "log"
        
        documents[doc_id] = {
            "type": file_type,
            "path": temp_file.name,
            "filename": file.filename
        }
        
        # Upload to S3 if available
        s3_key = None
        if s3_storage:
            s3_key = f"inputs/{session_id}/{file.filename}"
            upload_result = s3_storage.upload_file(temp_file.name, s3_key)
            if upload_result.get("success"):
                logger.info("[%s] Uploaded to S3: %s", session_id, s3_key)
                documents[doc_id]["s3_key"] = s3_key
        
        # Track file metadata for MongoDB
        input_files_meta.append({
            "filename": file.filename,
            "type": file_type,
            "s3_key": s3_key,
            "size_bytes": os.path.getsize(temp_file.name)
        })
        
        logger.info("[%s] Stored %s: %s -> %s", session_id, file_type, file.filename, doc_id)
    
    # Update MongoDB with file metadata
    if mongo_store:
        mongo_store.update_session(session_id, {"input_files": input_files_meta})
    
    # Build enhanced message with document info
    enhanced_message = message
    if documents:
        doc_info = "\n\nUploaded documents available for analysis:\n"
        for doc_id, info in documents.items():
            doc_info += f"- {doc_id}: {info['filename']} (type: {info['type']})\n"
        doc_info += "\nPlease:\n1. Analyze images with analyze_image tool\n2. Analyze logs with analyze_logs tool\n3. Generate recommendations with recommend_optimization based on findings"
        enhanced_message = message + doc_info
    
    try:
        # Create agent and process
        logger.info("[%s] Initializing QCAgent...", session_id)
        agent = QCAgent(session_id=session_id)
        
        logger.info("[%s] Starting agent processing with %s documents...", session_id, len(documents))
        logger.info("[%s] Enhanced message: %s...", session_id, enhanced_message[:200])
        
        # Run in thread pool to avoid blocking event loop (enables real-time SSE streaming)
        import asyncio
        result = await asyncio.to_thread(
            agent.process,
            message=enhanced_message,
            documents=documents
        )
        
        logger.info("[%s] Agent processing complete", session_id)

        
        # Extract results from agent response
        agent_results = result.get("results", {})
        
        # Get image analysis results (prefix-matching for v1.0 dynamic keys)
        image_results = {}
        for key, val in agent_results.items():
            if key.startswith("analyze_image"):
                image_results = val
                break
        defects = []
        if image_results.get("status") == "success":
            # Handle Batch Format
            if "results" in image_results and isinstance(image_results["results"], list):
                for res in image_results["results"]:
                    if res.get("status") == "success":
                        defects.extend(res.get("defects", []))
            # Handle Legacy/Single Format
            elif "defects" in image_results:
                defects = image_results.get("defects", [])
        
        # Get log analysis results (prefix-matching for v1.0 dynamic keys)
        log_results = {}
        for key, val in agent_results.items():
            if key.startswith("analyze_logs"):
                log_results = val
                break
        anomalies = log_results.get("anomalies", []) if log_results.get("status") == "success" else []
        
        # Get recommendations - if agent didn't call it, call directly (prefix-matching)
        rec_results = {}
        for key, val in agent_results.items():
            if key.startswith("recommend_optimization"):
                rec_results = val
                break
        if not rec_results or rec_results.get("status") != "success":
            # Agent skipped recommend_optimization - call it directly
            try:
                from tools.recommend_optimization import recommend_optimization
                state = {"results": {"analyze_image": image_results, "analyze_logs": log_results}}
                rec_results = recommend_optimization(state=state)
                logger.info("[%s] Generated %s recommendations (direct call)", session_id, len(rec_results.get('recommendations', [])))
            except Exception as e:
                logger.warning("[%s] Direct recommendation failed: %s", session_id, e)
                rec_results = {}
        recommendations = rec_results.get("recommendations", []) if rec_results.get("status") == "success" else []
        
        # Store intermediate results in MongoDB
        if mongo_store:
            if image_results:
                mongo_store.store_result(session_id, "analyze_image", image_results)
                mongo_store.add_log(session_id, "analyze_image", f"Found {len(defects)} defects")
            if log_results:
                mongo_store.store_result(session_id, "analyze_logs", log_results)
                mongo_store.add_log(session_id, "analyze_logs", f"Found {len(anomalies)} anomalies")
            if rec_results:
                mongo_store.store_result(session_id, "recommend_optimization", rec_results)
                mongo_store.add_log(session_id, "recommend_optimization", f"Generated {len(recommendations)} recommendations")
        
        # Generate LLM explanation for quick review
        explanation = generate_explanation(defects, anomalies)
        
        # Build final results
        final_results = {
            "session_id": session_id,
            "completed_at": datetime.now().isoformat(),
            "summary": {
                "total_defects": len(defects),
                "total_anomalies": len(anomalies),
                "total_recommendations": len(recommendations)
            },
            "defects": defects,
            "anomalies": anomalies,
            "recommendations": recommendations
        }
        
        # Upload results.json to S3
        if s3_storage:
            s3_key = f"outputs/{session_id}/results.json"
            upload_result = s3_storage.upload_json(final_results, s3_key)
            if upload_result.get("success"):
                logger.info("[%s] Uploaded results to S3: %s", session_id, s3_key)
                if mongo_store:
                    mongo_store.set_output(session_id, s3_key)
        
        # Save session for chat follow-ups
        save_session(session_id, agent_results, defects, anomalies, recommendations)
        
        if mongo_store:
            mongo_store.add_log(session_id, "complete", "Analysis finished")
        
        # Build clean response - no duplicate data
        # Determine highest severity across all findings
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        all_severities = []
        for d in defects:
            all_severities.append(d.get("severity", "low"))
        for a in anomalies:
            all_severities.append(a.get("severity", "low"))
        highest_severity = max(all_severities, key=lambda s: severity_order.get(s, 0)) if all_severities else None
        
        # Build response, omitting null/empty fields
        response = {
            "success": True,
            "session_id": session_id,
            "processed_at": datetime.now().isoformat() + "Z",
            "explanation": explanation,
            "summary": {
                "total_defects": len(defects),
                "total_anomalies": len(anomalies),
                "total_recommendations": len(recommendations),
                "documents_processed": len(documents),
                "action_required": len(defects) > 0 or any(a.get("severity") in ["critical", "high"] for a in anomalies),
                "highest_severity": highest_severity
            },
            "recommendations": recommendations
        }
        
        # Only include non-null analysis results
        if image_results:
            response["image_analysis"] = image_results
        if log_results:
            response["log_analysis"] = log_results
        if result.get("error"):
            response["error"] = result["error"]
        
        return response
        
    except Exception as e:
        logger.error("[%s] Error: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    
    finally:
        # Cleanup temp files (always runs)
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        remove_emitter(session_id)


@app.post("/chat", summary="Continue conversation with the agent", tags=["Agent"])
async def chat(message: str = Form(...), session_id: str = Form(...)):
    """Continue conversation about analysis results using session_id."""
    logger.info("[%s] Chat: %s", session_id, message)
    
    try:
        # Get stored session context
        session = get_session(session_id)
        
        if not session:
            return {
                "success": False,
                "message": "Session not found. Please run /process first to analyze files.",
                "session_id": session_id,
                "results": None
            }
        
        # Build FULL context from stored results (not just counts)
        defects = session.get("defects", [])
        anomalies = session.get("anomalies", [])
        recommendations = session.get("recommendations", [])
        analysis_results = session.get("analysis_results", {})
        
        # Build detailed context with FULL data for meaningful answers
        context_parts = ["=== PREVIOUS ANALYSIS RESULTS ===\n"]
        
        # Defect details
        context_parts.append(f"DEFECTS FOUND: {len(defects)}")
        if defects:
            for i, d in enumerate(defects, 1):
                defect_type = d.get('defect_type', d.get('class_name', 'unknown'))
                confidence = d.get('confidence', 0)
                severity = d.get('severity', 'unknown')
                bbox = d.get('bounding_box', {})
                context_parts.append(f"  {i}. Type: {defect_type}, Severity: {severity}, Confidence: {confidence:.1%}")
                if bbox:
                    context_parts.append(f"     Location: x1={bbox.get('x1')}, y1={bbox.get('y1')}, x2={bbox.get('x2')}, y2={bbox.get('y2')}")
        
        # Anomaly details
        context_parts.append(f"\nANOMALIES FOUND: {len(anomalies)}")
        if anomalies:
            for i, a in enumerate(anomalies, 1):
                field = a.get('field', 'unknown')
                value = a.get('value', 'N/A')
                unit = a.get('unit', '')
                severity = a.get('severity', 'unknown')
                expected_mean = a.get('expected_mean', 'N/A')
                expected_range = a.get('expected_range', 'N/A')
                z_score = a.get('z_score', 'N/A')
                timestamp = a.get('timestamp', '')
                context_parts.append(f"  {i}. {field}: value={value}{' ' + unit if unit else ''}, severity={severity}, z_score={z_score}")
                context_parts.append(f"     Expected: mean={expected_mean}, range={expected_range}, at={timestamp}")
        
        # Recommendation details - FULL info
        context_parts.append(f"\nRECOMMENDATIONS: {len(recommendations)}")
        if recommendations:
            for i, r in enumerate(recommendations, 1):
                rec_text = r.get('recommendation', r.get('title', r.get('action', 'Recommendation')))
                priority = r.get('priority', 'medium')
                rationale = r.get('rationale', '')
                parameter = r.get('parameter', '')
                source = r.get('source', '')
                anomaly_value = r.get('anomaly_value', '')
                expected_value = r.get('expected_value', '')
                unit = r.get('unit', '')
                context_parts.append(f"  {i}. [{priority.upper()}] {rec_text}")
                if rationale:
                    context_parts.append(f"     Rationale: {rationale}")
                if parameter:
                    context_parts.append(f"     Parameter: {parameter}, Current: {anomaly_value}{' ' + unit if unit else ''}, Expected: {expected_value}{' ' + unit if unit else ''}")
                if source:
                    context_parts.append(f"     Source: {source}")
        
        full_context = "\n".join(context_parts)
        
        # Get previous chat history (limit to last 10 exchanges to manage context size)
        chat_history = session.get("chat_history", [])[-10:]
        
        # Build messages with history
        from prompts.templates import CHAT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(context=full_context)}
        ]
        
        # Add chat history
        for exchange in chat_history:
            messages.append({"role": "user", "content": exchange.get("user", "")})
            messages.append({"role": "assistant", "content": exchange.get("assistant", "")})
        
        # Add current question
        messages.append({"role": "user", "content": message})
        
        # Use LLM to answer question with FULL context + history
        response = completion(
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            max_tokens=2000  # Increased for detailed responses
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Save this exchange to chat history
        new_exchange = {"user": message, "assistant": answer}
        if session_id in _session_store:
            if "chat_history" not in _session_store[session_id]:
                _session_store[session_id]["chat_history"] = []
            _session_store[session_id]["chat_history"].append(new_exchange)
        elif mongo_store:
            # Append to MongoDB chat history
            # NOTE: We access .sessions directly to use $push, because update_session wraps input in $set
            # which causes "The dollar ($) prefixed field '$push' ... is not allowed" error
            if hasattr(mongo_store, "sessions"):
                mongo_store.sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"chat_history": new_exchange},
                        "$set": {"updated_at": datetime.now(timezone.utc)}
                    }
                )
            else:
                # Fallback: Read-Modify-Write if direct access fails
                # Warning: Potential race condition, but safe for single-user chat
                try:
                    current = mongo_store.get_session(session_id)
                    history = current.get("chat_history", []) if current else []
                    history.append(new_exchange)
                    mongo_store.update_session(session_id, {"chat_history": history})
                except Exception as e:
                    logger.error("[%s] Failed to update chat history: %s", session_id, e)
        
        return {
            "success": True,
            "message": answer,
            "session_id": session_id,
            "context": {
                "defects": len(defects),
                "anomalies": len(anomalies),
                "recommendations": len(recommendations),
                "chat_turns": len(chat_history) + 1
            }
        }
        
    except Exception as e:
        logger.error("[%s] Chat error: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        remove_emitter(session_id)


@app.delete("/session/{session_id}", summary="Clear/delete a session", tags=["Session"])
async def clear_session(session_id: str):
    """
    Clear all data for a session.
    
    Removes:
    - Analysis results
    - Chat history
    - All stored data
    
    Use this to start fresh or clean up after processing.
    
    **Safety**: Only deletes sessions belonging to this agent (manufacturing_qc_agent).
    """
    logger.info("[%s] Clearing session...", session_id)
    
    deleted_from = []
    
    # === AGENT VALIDATION ===
    # Ensure we only delete sessions that belong to THIS agent
    if mongo_store:
        try:
            session_doc = mongo_store.get_session(session_id)
            if session_doc:
                session_agent = session_doc.get("metadata", {}).get("agent_name", "")
                if session_agent and session_agent != AGENT_NAME:
                    logger.warning("[%s] Blocked: Session belongs to '%s', not '%s'", session_id, session_agent, AGENT_NAME)
                    return {
                        "success": False,
                        "message": f"Cannot delete: Session belongs to a different agent ({session_agent})",
                        "session_id": session_id
                    }
        except Exception as e:
            logger.warning("[%s] Could not verify session owner: %s", session_id, e)
    
    # Clear from in-memory store
    if session_id in _session_store:
        del _session_store[session_id]
        deleted_from.append("memory")
    
    # Clear from MongoDB (already validated above)
    if mongo_store:
        try:
            result = mongo_store.delete_session(session_id)
            if result:
                deleted_from.append("mongodb")
        except Exception as e:
            logger.warning("[%s] MongoDB delete warning: %s", session_id, e)
    
    # Clear from S3 (optional - delete uploaded files)
    if s3_storage:
        try:
            # Helper to delete by prefix
            def delete_by_prefix(prefix):
                files = s3_storage.list_files(prefix)
                count = 0
                for f in files:
                    if "key" in f:
                        try:
                            if s3_storage.delete_file(f["key"]):
                                count += 1
                        except Exception:
                            pass
                return count

            # Delete input files
            c1 = delete_by_prefix(f"inputs/{session_id}/")
            # Delete output files
            c2 = delete_by_prefix(f"outputs/{session_id}/")
            
            if c1 + c2 > 0:
                deleted_from.append("s3")
        except Exception as e:
            logger.warning("[%s] S3 delete warning: %s", session_id, e)
    
    if not deleted_from:
        return {
            "success": False,
            "message": "Session not found",
            "session_id": session_id
        }
    
    logger.info("[%s] Session cleared from: %s", session_id, ', '.join(deleted_from))
    
    return {
        "success": True,
        "message": f"Session cleared from: {', '.join(deleted_from)}",
        "session_id": session_id,
        "deleted_from": deleted_from
    }


@app.post("/session/{session_id}/clear-chat", summary="Clear chat history only", tags=["Session"])
async def clear_chat_history(session_id: str):
    """
    Clear only the chat history for a session, keeping analysis results.
    
    Useful when you want to start a new conversation about the same analysis.
    
    **Safety**: Only clears chat for sessions belonging to this agent.
    """
    logger.info("[%s] Clearing chat history...", session_id)
    
    # === AGENT VALIDATION ===
    if mongo_store:
        try:
            session_doc = mongo_store.get_session(session_id)
            if session_doc:
                session_agent = session_doc.get("metadata", {}).get("agent_name", "")
                if session_agent and session_agent != AGENT_NAME:
                    logger.warning("[%s] Blocked: Session belongs to '%s'", session_id, session_agent)
                    return {
                        "success": False,
                        "message": f"Cannot modify: Session belongs to a different agent ({session_agent})",
                        "session_id": session_id
                    }
        except Exception as e:
            logger.warning("[%s] Could not verify session owner: %s", session_id, e)
    
    # Clear from in-memory store
    if session_id in _session_store:
        _session_store[session_id]["chat_history"] = []
        return {
            "success": True,
            "message": "Chat history cleared",
            "session_id": session_id
        }
    
    # Clear from MongoDB
    if mongo_store:
        try:
            mongo_store.update_session(session_id, {"chat_history": []})
            return {
                "success": True,
                "message": "Chat history cleared",
                "session_id": session_id
            }
        except Exception as e:
            logger.error("[%s] Failed to clear chat history: %s", session_id, e)
    
    return {
        "success": False,
        "message": "Session not found",
        "session_id": session_id
    }

@app.get("/stream/{session_id}", summary="Stream processing updates (SSE)", tags=["Streaming"])
async def stream_updates(session_id: str):
    """
    Stream processing updates for a session using Server-Sent Events.
    Connect via EventSource in frontend for real-time updates.
    """
    import asyncio
    import json
    
    async def event_generator():
        # Yield connection event immediately
        yield f"data: {json.dumps({'event': 'connected', 'session_id': session_id})}\n\n"
        
        # Get or create emitter for this session
        try:
            from point9_platform.observability.emitter import get_or_create_emitter
            emitter = get_or_create_emitter(session_id)
            
            if emitter is None:
                yield f"data: {json.dumps({'event': 'error', 'message': 'Emitter not found'})}\n\n"
                return
            
            # Subscribe to the queue
            queue = emitter.subscribe()
            ping_interval = 0
            
            try:
                while True:
                    try:
                        # Wait for event with timeout for keep-alive
                        step = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        if step is None:
                            # End signal
                            break
                        
                        # Yield the SSE event
                        yield step.to_sse()
                        
                    except asyncio.TimeoutError:
                        # Send keep-alive comment to prevent timeout
                        ping_interval += 1
                        if ping_interval >= 15:  # Every 15 seconds
                            yield ": keep-alive\n\n"
                            ping_interval = 0
                        continue
                        
            finally:
                emitter.unsubscribe(queue)
                
        except ImportError as e:
            yield f"data: {json.dumps({'event': 'error', 'message': f'Streaming not available: {e}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        
        yield f"data: {json.dumps({'event': 'complete'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )


@app.get("/sessions", summary="List recent sessions", tags=["Session"])
async def list_sessions(limit: int = 10, status: Optional[str] = None):
    """
    List recent QC analysis sessions.
    
    - **limit**: Maximum number of sessions to return (default 10)
    - **status**: Optional filter by status (created, processing, completed, failed)
    """
    if not mongo_store:
        return {
            "success": False,
            "error": "Session persistence not enabled. Set ENABLE_MONGODB=True",
            "sessions": []
        }
    
    sessions = mongo_store.list_sessions(limit=limit, status=status)
    
    # Return summary for each session (filter out docs without session_id)
    session_list = []
    for s in sessions:
        if not s.get("session_id"):
            continue  # Skip documents without session_id
        try:
            session_list.append({
                "session_id": s["session_id"],
                "status": s.get("status", "unknown"),
                "created_at": s["created_at"].isoformat() if s.get("created_at") else None,
                "input_files": len(s.get("input_files", [])),
                "has_results": bool(s.get("intermediate_results"))
            })
        except Exception:
            continue  # Skip malformed documents
    
    return {
        "success": True,
        "total": len(session_list),
        "sessions": session_list
    }


@app.get("/sessions/{session_id}", summary="Get session details", tags=["Session"])
async def get_session_details(session_id: str):
    """
    Get detailed information about a specific session.
    
    Returns session data, intermediate results, and presigned URLs for files.
    """
    if not mongo_store:
        # Try in-memory store
        session = _session_store.get(session_id)
        if session:
            return {
                "success": True,
                "session_id": session_id,
                "status": "completed",
                "created_at": session["created_at"].isoformat() if session.get("created_at") else None,
                "results": {
                    "defects": session.get("defects", []),
                    "anomalies": session.get("anomalies", []),
                    "recommendations": session.get("recommendations", [])
                }
            }
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = mongo_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate presigned URLs for input files if S3 is available
    file_urls = []
    if s3_storage and session.get("input_files"):
        for f in session["input_files"]:
            if f.get("s3_key"):
                url = s3_storage.get_presigned_url(f["s3_key"])
                if url:
                    file_urls.append({
                        "filename": f.get("filename"),
                        "download_url": url
                    })
    
    # Extract results
    intermediate = session.get("intermediate_results", {})
    
    return {
        "success": True,
        "session_id": session_id,
        "status": session.get("status"),
        "created_at": session["created_at"].isoformat() if session.get("created_at") else None,
        "updated_at": session["updated_at"].isoformat() if session.get("updated_at") else None,
        "input_files": file_urls or session.get("input_files", []),
        "results": {
            "image_analysis": intermediate.get("analyze_image"),
            "log_analysis": intermediate.get("analyze_logs"),
            "recommendations": intermediate.get("recommend_optimization"),
            "final": intermediate.get("final_results")
        },
        "logs": session.get("logs", [])[:20],  # Last 20 logs
        "chat_history": session.get("chat_history", [])
    }