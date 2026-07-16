---
title: Manufacturing QC Agent
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.11"
pinned: false
---

# Manufacturing Quality Control & Optimization Agent

![Platform](https://img.shields.io/badge/Platform-Point9%20v1.0-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/LLM-Gemini%202.5%20Pro-purple)
![Deployment](https://img.shields.io/badge/Deployment-Hugging%20Face%20Spaces-yellow)

> AI agent for automated solar module defect detection, production log anomaly analysis, and process optimization — built on the [Point9 Agent Platform](https://github.com/shivam0897-i/agent-platform).

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────────────┐
                         │                   FastAPI Server (api/main.py)               │
                         │                                                              │
                         │  POST /process    POST /chat    GET /stream/{id}  (+ 6 more) │
                         └──────────┬────────────┬───────────────────────────────────────┘
                                    │            │
                    ┌───────────────┘            │
                    ▼                            ▼
    ┌───────────────────────────────┐   ┌──────────────────────────┐
    │   QCAgent (agent.py)          │   │  Direct LLM (litellm)    │
    │   extends BaseAgent           │   │  /chat, /explain         │
    │                               │   │  model from settings     │
    │   LangGraph Orchestration:    │   └──────────────────────────┘
    │   Planner → Executor →        │
    │   Reflector → Responder       │
    │                               │
    │   State: QCAgentState         │
    │   (extends BaseAgentState)    │
    └───────────┬───────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                   Tool Registry (tools/)                      │
    │   Auto-discovered by platform at startup                      │
    │                                                               │
    │   analyze_image          YOLOv8 (EL) + EfficientNet (RGB)     │
    │   analyze_logs           Pandas + 3-sigma statistical engine  │
    │   recommend_optimization Rule engine + LLM correlation        │
    │   generate_report        Structured JSON/PDF report builder   │
    │   query_knowledge        RAG over domain knowledge (Phase 4)  │
    └───────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                   Storage Layer (optional)                     │
    │                                                               │
    │   MongoDB                          S3                         │
    │   Sessions, results, chat history  Images, logs, output JSON  │
    │   Falls back to in-memory store    Falls back to local temp   │
    └───────────────────────────────────────────────────────────────┘
```

---

## How It Works

1. **User uploads** solar module images (`.jpg`, `.png`) and/or production logs (`.csv`, `.txt`) via `POST /process`
2. **Agent plans** — Gemini 2.5 Pro analyzes the request and creates an execution plan (e.g., "analyze image → analyze logs → recommend optimization")
3. **Tools execute** — The agent calls tools in sequence via LangGraph's Planner → Executor → Reflector loop
4. **Results aggregate** — Defects, anomalies, and recommendations are extracted, correlated, and stored
5. **User chats** — Follow-up questions via `POST /chat` use the full analysis context + chat history

The agent runs inside `asyncio.to_thread()` to avoid blocking the FastAPI event loop, enabling real-time SSE streaming via `/stream/{session_id}`.

---

## Tools

### `analyze_image`
- **Input:** Image file path(s) from uploaded documents
- **Models:** YOLOv8 (`best.pt`, grayscale EL images) + EfficientNet (`pv_defect_efficientnet_b0_97.pth.zip`, RGB surface images)
- **Output:** List of detected defects with `defect_type`, `confidence`, `bounding_box`, `severity`
- **12 defect categories:** `black_core`, `corner`, `crack`, `finger`, `fragment`, `horizontal_dislocation`, `printing_error`, `scratch`, `short_circuit`, `star_crack`, `thick_line`, `vertical_dislocation`
- **6 surface conditions** (RGB): `Bird-drop`, `Clean`, `Dusty`, `Electrical-damage`, `Physical-Damage`, `Snow-Covered`
- **Batch processing:** Handles multiple images in a single request

### `analyze_logs`
- **Input:** CSV/TXT/LOG file path from uploaded documents
- **Engine:** Pandas-based statistical analysis with 3-sigma anomaly detection
- **Monitored parameters:** Temperature (°C), Pressure (bar), Humidity (%), Speed (m/min), Voltage (V), Current (A), Power (W), and more (see `config.yaml` FIELD_UNITS)
- **Output:** List of anomalies with `anomaly_type`, `value`, `expected_range`, `severity`, plus statistical summary (min, max, mean, std per field)

### `recommend_optimization`
- **Input:** Results from `analyze_image` and `analyze_logs` (accessed via prefix-matching on state keys)
- **Engine:** Rule-based mapping (defect→parameter→action) + LLM-powered contextual recommendations
- **Output:** Prioritized recommendations with `recommendation`, `priority` (critical/high/medium/low), `parameter`, `rationale`, `anomaly_source`

### `generate_report`
- **Input:** Combined results from all prior tools
- **Output:** Structured report JSON combining defects, anomalies, and recommendations

### `query_knowledge`
- **Status:** Phase 4 (RAG dependencies commented out in `requirements.txt`)
- **Purpose:** Query domain-specific knowledge base for manufacturing best practices

---

## API Endpoints

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process` | Upload images/logs and run full analysis pipeline |
| `POST` | `/chat` | Follow-up conversation about analysis results |
| `GET` | `/health` | Health check (from platform) |

### Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/session/create` | Create session + SSE emitter |
| `GET` | `/stream/{session_id}` | Real-time processing updates (SSE) |
| `DELETE` | `/session/{session_id}` | Delete session data (agent-scoped) |
| `POST` | `/session/{session_id}/clear-chat` | Clear chat history only |
| `GET` | `/sessions` | List recent sessions (requires MongoDB) |
| `GET` | `/sessions/{session_id}` | Get session details + S3 presigned URLs |
| `GET` | `/observability/mlflow` | Non-sensitive MLflow tracing status |

---

## MLflow Observability

MLflow tracing is optional and disabled by default. When enabled, the app:

- turns on `mlflow.litellm.autolog()` so LiteLLM/Gemini calls record prompts, completions, latency, token usage, cost metadata when available, cache hits, and exceptions
- creates application spans for `/process`, `/process` summaries, and `/chat`
- tags traces with `session_id`, endpoint, and agent name
- records metadata summaries only for app spans; uploaded file bytes and annotated images are not logged to MLflow

Local setup:

```bash
pip install "mlflow[genai]>=3.10.0"
mlflow server --host 0.0.0.0 --port 5000
```

App configuration:

```bash
QC_ENABLE_MLFLOW=true
QC_MLFLOW_TRACKING_URI=http://localhost:5000
QC_MLFLOW_EXPERIMENT_NAME=manufacturing-qc-agent
```

Runtime check:

```bash
curl http://localhost:8000/observability/mlflow
```

On Hugging Face Spaces, keep MLflow disabled unless `QC_MLFLOW_TRACKING_URI` points to a separate MLflow server. For future EC2 deployment, run MLflow as a separate service with persistent backend storage instead of writing local tracking data inside the API container.

---

### `POST /process`

Upload images and/or logs for analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/process \
  -F "message=Analyze for defects" \
  -F "session_id=optional-session-id" \
  -F "files=@panel_001.jpg" \
  -F "files=@production_log.csv"
```

**Response (200):**
```json
{
  "success": true,
  "session_id": "79d93e27-5277-4f3d-845e-344112bbc550",
  "processed_at": "2026-03-05T16:51:14.424367Z",
  "explanation": "Image analysis shows no visual defects",
  "summary": {
    "total_defects": 0,
    "total_anomalies": 5,
    "total_recommendations": 4,
    "documents_processed": 2,
    "action_required": true,
    "highest_severity": "critical"
  },
  "recommendations": [
    {
      "recommendation": "Reduce heating zone setpoint",
      "priority": "high",
      "parameter": "heating_setpoint",
      "rationale": "Temperature exceeded safe limits.",
      "anomaly_source": "temperature_anomaly",
      "anomaly_value": 92.3,
      "expected_value": 82.03,
      "unit": "°C",
      "source": "rules"
    }
  ],
  "image_analysis": { "..." : "included if images were analyzed" },
  "log_analysis": { "..." : "included if logs were analyzed" }
}
```

**Notes:**
- `image_analysis` and `log_analysis` are only included when non-empty
- `action_required` is `true` when defects > 0 or any anomaly has `critical`/`high` severity
- `explanation` is an LLM-generated summary of findings
- If the agent skips `recommend_optimization`, it is called directly as a fallback

---

### `POST /chat`

Continue conversation about analysis results.

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -F "session_id=79d93e27-5277-4f3d-845e-344112bbc550" \
  -F "message=What is the most critical issue?"
```

**Response (200):**
```json
{
  "success": true,
  "message": "The most critical issue is the temperature spike at 92.3°C...",
  "session_id": "79d93e27-5277-4f3d-845e-344112bbc550",
  "context": {
    "defects": 0,
    "anomalies": 5,
    "recommendations": 4,
    "chat_turns": 1
  }
}
```

**Notes:**
- Keeps last 10 conversation turns in context
- Full analysis context (defects, anomalies, recommendations) is injected into the LLM prompt
- Chat history is persisted in MongoDB (if enabled) or in-memory

---

## Project Structure

```
Manufacturing-QC-Agent/
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI endpoints (10 routes)
├── tools/
│   ├── __init__.py              # Auto-discovered by platform
│   ├── _utils.py                # Shared tool utilities
│   ├── analyze_image.py         # YOLOv8 + EfficientNet defect detection
│   ├── analyze_logs.py          # Pandas statistical anomaly detection
│   ├── recommend_optimization.py # Rule engine + LLM recommendations
│   ├── generate_report.py       # Structured report builder
│   └── query_knowledge.py       # RAG knowledge base (Phase 4)
├── vision/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8 model wrapper (grayscale EL)
│   └── rgb_classifier.py        # EfficientNet classifier (RGB surface)
├── prompts/
│   ├── __init__.py
│   └── templates.py             # LLM prompt templates (PROMPTS dict)
├── docs/
│   ├── ARCHITECTURE.md          # Detailed architecture documentation
│   └── OUTPUT_GUIDE.md          # API output format guide
├── dev/
│   └── tests/                   # Test images and CSV logs
│
├── agent.py                     # QCAgent class (extends BaseAgent)
├── state.py                     # QCAgentState (extends BaseAgentState)
├── settings.py                  # QCSettings (extends UserSettings)
├── config.yaml                  # Non-sensitive configuration
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build (python:3.11-slim)
├── .env.example                 # Environment variable template
├── best.pt                      # YOLOv8 weights (22MB)
└── pv_defect_efficientnet_b0_97.pth.zip  # EfficientNet weights (16MB)
```

---

## Configuration

### `settings.py` — QCSettings

All configuration is centralized in `QCSettings` (extends `UserSettings` from the platform). Values can be overridden via `.env` with the `QC_` prefix or via `config.yaml`.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `DEFAULT_LLM_MODEL` | `str` | `gemini/gemini-2.5-pro` | LLM for agent planning, chat, and explanations |
| `CONFIDENCE_THRESHOLD` | `float` | `0.85` | Minimum confidence for defect reporting |
| `FALSE_POSITIVE_TARGET` | `float` | `0.05` | Target false positive rate |
| `ENABLE_MONGODB` | `bool` | `True` | Enable MongoDB session persistence |
| `ENABLE_S3_STORAGE` | `bool` | `True` | Enable S3 file storage |
| `MONGODB_URI` | `str?` | `None` | MongoDB connection string |
| `S3_BUCKET_NAME` | `str?` | `None` | S3 bucket name |

### `.env` — Secrets

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional — MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DB=qc_agent
QC_ENABLE_MONGODB=True

# Optional — S3
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=qc-agent-storage
QC_ENABLE_S3_STORAGE=True
```

### `config.yaml` — Non-Sensitive Config

Contains defect categories, surface categories, field-to-unit mappings for anomaly detection (temperature→°C, pressure→bar, etc.), and model paths. See the file for full details.

---

## State Management

### `QCAgentState` (extends `BaseAgentState`)

The agent state inherits base fields from the platform and adds QC-specific fields:

**Inherited fields:** `messages`, `session_id`, `should_continue`, `error`, `iteration`, `max_iterations`, `model`, `plan`, `current_step`, `current_task`, `thoughts`, `results`, `documents`

**QC-specific fields:**

| Field | Type | Description |
|-------|------|-------------|
| `defects` | `List[DefectInfo]` | Detected defects (type, confidence, bbox, severity) |
| `log_anomalies` | `List[LogAnomaly]` | Process anomalies (type, value, expected range, severity) |
| `recommendations` | `List[Recommendation]` | Optimization suggestions (action, priority, parameter, rationale) |
| `needs_human_input` | `bool` | Whether agent requires human intervention |

---

## Deployment

### Hugging Face Spaces (Production)

The agent is deployed as a Docker Space at [`point9/ManufacturingQC_agent`](https://huggingface.co/spaces/point9/ManufacturingQC_agent).

- **Runtime:** Docker (`python:3.11-slim`)
- **Port:** 7860 (HF Spaces standard)
- **User:** Non-root (UID 1000, HF requirement)
- **Secrets:** Configured via HF Space secrets (GEMINI_API_KEY, MONGODB_URI, AWS keys)

Push to deploy:
```bash
git push hf main
```

### Local Development

```bash
# Clone
git clone https://huggingface.co/spaces/point9/ManufacturingQC_agent
cd ManufacturingQC_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\Activate.ps1 # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Run
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker (Local)

```bash
docker build -t manufacturing-qc-agent .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your-key" \
  manufacturing-qc-agent
```

> **Note:** Docker exposes port 7860 by default (for HF Spaces). Override with `-p 8000:7860` if accessing locally.

---

## Security

- **Session isolation:** Each session is tagged with `agent_name = "manufacturing_qc_agent"`
- **Cross-agent protection:** `DELETE /session/{id}` and `POST /session/{id}/clear-chat` validate that the session belongs to this agent before modifying
- **Non-root container:** Docker runs as UID 1000
- **Emitter cleanup:** `remove_emitter(session_id)` is called in `finally` blocks on `/process` and `/chat` to prevent memory leaks
- **Exception handling:** All endpoints use `except Exception:` (no bare `except:`) and chain exceptions with `from e`

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Platform** | `agent-platform` (git), `langgraph>=0.2.0` |
| **Server** | `fastapi>=0.100.0`, `uvicorn>=0.23.0`, `python-multipart` |
| **LLM** | `litellm>=1.0.0` |
| **Vision** | `ultralytics>=8.0.0` (YOLOv8), `torch>=2.0.0`, `torchvision>=0.15.0` (EfficientNet) |
| **Data** | `pandas>=2.0.0`, `numpy>=1.24.0`, `pillow>=10.0.0`, `opencv-python-headless>=4.8.0` |
| **Storage** | `boto3>=1.26.0` (S3), `pymongo>=4.3.0` (MongoDB) |

---

## Platform Compliance (Point9 v1.0)

This agent follows all Point9 Platform v1.0 conventions:

- `QCAgent` extends `BaseAgent[QCAgentState]`
- `QCAgentState` extends `BaseAgentState` (no duplicated base fields)
- `QCSettings` extends `UserSettings`
- Tools are auto-discovered from the `tools/` package
- Tool results are accessed via prefix-matching (e.g., `key.startswith("analyze_image")`)
- `asyncio.to_thread()` wraps `agent.process()` to avoid blocking the event loop
- `remove_emitter()` is called in `finally` blocks for cleanup
- `datetime.now(timezone.utc)` used instead of deprecated `datetime.utcnow()`
- Lazy `%s` logging used instead of f-string interpolation in all logger calls
- `langgraph>=0.2.0` for `START` constant support

---

*Built by the Point9 AI Engineering Team*
