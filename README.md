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

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Point9-orange)
![License](https://img.shields.io/badge/License-Proprietary-red)

> **Enterprise AI Agent** for automated solar module inspection, production log analysis, and process optimization.

---

## 🎯 Overview

The Manufacturing QC Agent is an **autonomous AI system** designed for high-volume solar panel manufacturing environments. It acts as a "Virtual Quality Engineer" that can:

- **Inspect** solar module images for defects using computer vision
- **Analyze** production logs for process anomalies
- **Recommend** optimizations based on correlated findings
- **Converse** about analysis results with contextual memory

Built on the [Point9 Agent Platform](https://github.com/shivam0897-i/point9-agent-platform), it leverages LangGraph for orchestration and Gemini for reasoning.

---

## ✨ Features

### 🔍 Visual Defect Detection
| Capability | Description |
|------------|-------------|
| **Batch Processing** | Analyze entire manufacturing lots in a single request |
| **12 Defect Types** | Cracks, Black Cores, Finger Interruptions, Thick Lines, etc. |
| **Auto-Discovery** | Automatically finds and processes all uploaded images |
| **YOLOv8 Model** | Custom-trained on PVEL-AD dataset (640px inference) |

### 📊 Log Anomaly Analysis
| Capability | Description |
|------------|-------------|
| **Multi-Format Support** | CSV, TXT, LOG files |
| **Statistical Detection** | Flags deviations beyond 3-sigma thresholds |
| **Parameter Monitoring** | Temperature, Pressure, Humidity, Belt Speed |
| **Smart Fallback** | Auto-discovers logs when file IDs are missing |

### 💡 Process Optimization
| Capability | Description |
|------------|-------------|
| **Root Cause Correlation** | Links defects to process anomalies |
| **Prioritized Actions** | High/Medium/Low priority recommendations |
| **Domain Knowledge** | Rule-based mappings for solar manufacturing |

### 💬 Conversational Interface
| Capability | Description |
|------------|-------------|
| **Context Memory** | Remembers last 10 conversation exchanges |
| **Full Analysis Context** | Accesses defect details, anomaly values, recommendations |
| **Session Persistence** | Chat history saved across requests |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for deployment)
- Gemini API Key

### Local Development
```bash
# Clone repository
git clone https://github.com/shivam0897-i/Manufacturing-QC-Agent.git
cd Manufacturing-QC-Agent

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-api-key"

# Run server
uvicorn api.main:app --port 8000 --reload
```

### Docker Deployment
```bash
docker build -t manufacturing-qc-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" manufacturing-qc-agent
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process` | Upload files and analyze |
| `POST` | `/chat` | Continue conversation about results |
| `POST` | `/session/create` | Create new session with SSE support |
| `GET` | `/stream/{session_id}` | Real-time processing updates (SSE) |
| `DELETE` | `/session/{session_id}` | Clear session data |
| `POST` | `/session/{session_id}/clear-chat` | Clear chat history only |
| `GET` | `/sessions` | List recent sessions |
| `GET` | `/sessions/{session_id}` | Get session details |

---

### Process Files

**Endpoint:** `POST /process`

Upload images and/or logs for analysis.

```bash
curl -X POST http://localhost:8000/process \
  -F "message=Analyze for defects" \
  -F "files=@panel_001.jpg" \
  -F "files=@panel_002.jpg" \
  -F "files=@production_log.csv"
```

**Response:**
```json
{
  "success": true,
  "message": "Analysis complete. Found 3 defects across 2 images...",
  "session_id": "abc123-def456",
  "results": {
    "analyze_image": { ... },
    "analyze_logs": { ... },
    "recommend_optimization": { ... }
  }
}
```

---

### Chat with Context

**Endpoint:** `POST /chat`

Ask follow-up questions about analysis results.

```bash
curl -X POST http://localhost:8000/chat \
  -F "session_id=abc123-def456" \
  -F "message=What caused the cracks?"
```

**Response:**
```json
{
  "success": true,
  "message": "Based on the analysis, the crack defect (85.3% confidence) correlates with temperature anomalies...",
  "session_id": "abc123-def456",
  "context": {
    "defects": 3,
    "anomalies": 4,
    "recommendations": 6,
    "chat_turns": 2
  }
}
```

---

### Clear Session

**Endpoint:** `DELETE /session/{session_id}`

Delete all session data (analysis, chat history, files).

```bash
curl -X DELETE http://localhost:8000/session/abc123-def456
```

> **Note:** Only deletes sessions belonging to this agent (manufacturing_qc_agent).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  /process  │  /chat  │  /stream  │  /session/*              │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                     QC Agent (LangGraph)                     │
├──────────────┬──────────────┬───────────────┬───────────────┤
│   Planner    │   Executor   │   Reflector   │   Responder   │
│  (Gemini)    │  (Tools)     │  (Control)    │  (Gemini)     │
└──────────────┴──────────────┴───────────────┴───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Tool Registry                         │
├─────────────┬─────────────┬─────────────────┬───────────────┤
│ analyze_    │ analyze_    │ recommend_      │ query_        │
│ image       │ logs        │ optimization    │ knowledge     │
│ (YOLOv8)    │ (Pandas)    │ (Rule Engine)   │ (RAG)         │
└─────────────┴─────────────┴─────────────────┴───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
├──────────────────────────┬──────────────────────────────────┤
│        MongoDB           │              S3                   │
│   (Sessions, Results)    │     (Images, Logs, Outputs)       │
└──────────────────────────┴──────────────────────────────────┘
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `MONGODB_URI` | ❌ | MongoDB connection string |
| `AWS_ACCESS_KEY_ID` | ❌ | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | ❌ | S3 secret key |
| `S3_BUCKET_NAME` | ❌ | S3 bucket for file storage |

### config.yaml

```yaml
# Model Settings
CONFIDENCE_THRESHOLD: 0.25
IMAGE_SIZE: 640
MODEL_PATH: best.pt

# LLM Settings
llm:
  primary_model: gemini/gemini-2.5-pro
  fallback_model: gemini/gemini-2.0-flash

# Defect Categories
defect_categories:
  - crack
  - black_core
  - finger_interruption
  - thick_line
  - scratch
  - star_crack
  - corner_crack
  - edge_crack
  - short_circuit
  - horizontal_crack
  - vertical_crack
  - cross_crack
```

---

## 📁 Project Structure

```
Manufacturing-QC-Agent/
├── api/
│   └── main.py              # FastAPI endpoints
├── agent/
│   └── qc_agent.py          # LangGraph agent definition
├── tools/
│   ├── analyze_image.py     # Vision tool (YOLOv8)
│   ├── analyze_logs.py      # Log analysis tool
│   └── recommend_optimization.py
├── prompts/
│   └── templates.py         # LLM prompts
├── vision/
│   └── detector.py          # YOLO model wrapper
├── config.yaml              # Agent configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
└── README.md
```

---

## 🔒 Security

- **Session Isolation**: Each agent tags sessions with `agent_name`
- **Cross-Agent Protection**: Cannot delete sessions from other agents
- **Non-Root Docker**: Runs as UID 1000 in container
- **Secret Management**: API keys via environment variables

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Image Processing | ~500ms per image |
| Log Analysis | ~200ms per file |
| End-to-End (5 images + log) | ~8-12 seconds |
| Concurrent Sessions | Unlimited (stateless) |

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
ruff check .
black --check .
```

---

## 📄 License

Proprietary - Point9 AI Engineering Team

---

## 🔗 Related Projects

- [Point9 Agent Platform](https://github.com/shivam0897-i/point9-agent-platform) - Core agent framework
- [PVEL-AD Dataset](https://github.com/) - Training data for defect detection

---

*Built with ❤️ by the Point9 AI Engineering Team*
