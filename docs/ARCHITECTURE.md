# Manufacturing QC Agent Architecture

> All diagrams verified against source code. See references below each diagram.

---

## Low-Level Agent Internals

```mermaid
flowchart TB
    subgraph "1. Request Entry"
        A[POST /process] --> B[Save temp files]
        B --> C[Build documents dict]
    end
    
    subgraph "2. Agent Initialization"
        C --> D["QCAgent(session_id)"]
        D --> E["tools_package = 'tools'"]
        D --> F["settings = QCSettings()"]
    end
    
    subgraph "3. Process Method"
        F --> G["process(message, documents)"]
        G --> H{validate_domain}
        H -->|fail| I[Return error]
        H -->|pass| J[create_initial_state]
        
        J --> K["State:<br/>messages=[]<br/>documents={}<br/>plan=[]<br/>current_step=0<br/>defects=[]<br/>anomalies=[]"]
    end
    
    subgraph "4. Graph Execution"
        K --> L[graph.invoke]
        
        L --> M[PLANNER]
        M --> N["Analyze request<br/>Create plan[]"]
        
        N --> O[EXECUTOR]
        O --> P["Get plan[current_step]<br/>Call tool<br/>Store result<br/>Increment step"]
        
        P --> Q[REFLECTOR]
        Q --> R{current_step < len plan?}
        R -->|yes| O
        R -->|no| S[RESPONDER]
        
        S --> T[Format response]
    end
    
    subgraph "5. Response"
        T --> U[_extract_result]
        U --> V[on_complete]
        V --> W[Return JSON]
    end
```

**Source**: `agent.py`, `base.py:128-189`, `builder.py:53-64`

---

## System Overview

```mermaid
flowchart LR
    subgraph Client
        API_CLIENT[API Client]
    end
    
    subgraph Server
        FASTAPI[FastAPI]
        AGENT[QCAgent]
        TOOLS[Tools]
        YOLO[YOLOv8]
    end
    
    subgraph Storage
        S3[(S3)]
        MONGO[(MongoDB)]
    end
    
    subgraph External
        LLM[Gemini API]
    end
    
    API_CLIENT --> FASTAPI
    FASTAPI --> AGENT
    AGENT --> TOOLS
    AGENT --> LLM
    TOOLS --> YOLO
    FASTAPI --> S3
    FASTAPI --> MONGO
```

---

## Agent Execution Flow

```mermaid
flowchart TB
    START([POST /process]) --> INIT[Initialize QCAgent]
    INIT --> VALIDATE{Domain Valid?}
    VALIDATE -->|No| ERROR[Return Error]
    VALIDATE -->|Yes| STATE[Create Initial State]
    STATE --> GRAPH[Invoke LangGraph]
    
    subgraph LangGraph
        GRAPH --> P[Planner]
        P --> E[Executor]
        E --> R[Reflector]
        R -->|More Steps| E
        R -->|Complete| RESP[Responder]
    end
    
    RESP --> EXTRACT[Extract Results]
    EXTRACT --> RETURN([Return JSON])
```

**Source**: `agent/base.py:128-189`, `graph/builder.py:53-64`

---

## LangGraph Nodes

```mermaid
flowchart LR
    P[Planner] -->|"plan[]"| E[Executor]
    E -->|"results{}"| R[Reflector]
    R -->|"current_step < len(plan)"| E
    R -->|"else"| RESP[Responder]
    RESP --> END([END])
```

**Routing Logic** (`nodes.py:292-296`):
```python
if should_continue and current_step < len(plan):
    return "executor"
return "responder"
```

---

## State Structure

```mermaid
classDiagram
    class QCAgentState {
        +messages: List
        +session_id: str
        +documents: Dict
        +results: Dict
        +plan: List
        +current_step: int
        +should_continue: bool
        +error: Optional[str]
        +defects: List
        +log_anomalies: List
        +recommendations: List
    }
```

**Source**: `state.py:39-72`

---

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| `analyze_image` | YOLO defect detection (Batch Supported) | summary + defects[] |
| `analyze_logs` | Statistical anomaly detection | anomalies[] |
| `recommend_optimization` | Rule-based suggestions | recommendations[] |
| `query_knowledge` | RAG document search | context |
| `generate_report` | PDF/HTML generation | report_url |

**Source**: `tools/` directory, `agent.py:26`

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/session/create` | Create session with SSE emitter |
| GET | `/stream/{id}` | SSE event stream |
| POST | `/process` | Analyze files |
| POST | `/chat` | Follow-up conversation |
| GET | `/sessions` | List sessions |
| GET | `/sessions/{id}` | Session details |

**Source**: `api/main.py`

---

## Storage

```mermaid
flowchart LR
    subgraph Input
        FILES[Uploaded Files]
    end
    
    subgraph Process
        AGENT[QCAgent]
    end
    
    subgraph Storage
        S3[(S3 Bucket)]
        MONGO[(MongoDB)]
    end
    
    FILES -->|inputs/session/| S3
    AGENT -->|session state| MONGO
    AGENT -->|results.json| S3
```

**Source**: `api/main.py:54-66`

---

## File Structure

```
agent/
├── agent.py          # QCAgent class
├── state.py          # QCAgentState TypedDict
├── settings.py       # Configuration
├── api/main.py       # FastAPI endpoints
├── tools/            # Tool implementations
├── vision/           # YOLOv8 wrapper
├── models/best.pt    # Trained model
└── prompts/          # LLM prompt templates
```
