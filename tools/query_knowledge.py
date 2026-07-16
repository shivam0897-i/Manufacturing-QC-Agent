"""
Query Knowledge Tool
====================

Searches equipment manuals and best practices for manufacturing QC.
Uses keyword matching with relevance scoring (lightweight RAG alternative).
"""

import re
from typing import Dict, Any
from point9_platform.tools.decorator import tool


# Knowledge base - Solar panel manufacturing best practices
KNOWLEDGE_BASE = [
    {
        "source": "Equipment Manual v2.3",
        "section": "4.2 - Crack Prevention",
        "keywords": ["crack", "cracks", "mechanical", "stress", "handling", "lamination"],
        "content": """To minimize cell cracks during lamination:
1. Maintain lamination temperature between 145-150°C
2. Ensure uniform pressure distribution (0.8-1.0 bar)
3. Check conveyor alignment weekly
4. Handle cells carefully during loading - avoid mechanical stress
5. Inspect incoming wafers for pre-existing micro-cracks"""
    },
    {
        "source": "Equipment Manual v2.3",
        "section": "4.3 - Temperature Control",
        "keywords": ["temperature", "heat", "thermal", "lamination", "spike", "overheat"],
        "content": """Temperature control guidelines:
- Pre-heating zone: 80-90°C (gradual warm-up)
- Lamination zone: 145-150°C (±2°C tolerance)
- Cooling zone: Gradual cooling to 40°C before exit
- Maximum temperature spike: 155°C (triggers alarm)
- Temperature spikes above 155°C cause EVA degradation and cell cracks"""
    },
    {
        "source": "Equipment Manual v2.3",
        "section": "4.4 - Finger Electrode Defects",
        "keywords": ["finger", "electrode", "grid", "interruption", "broken", "line"],
        "content": """Finger interruption prevention:
1. Check screen printing paste viscosity (120-150 Pa·s)
2. Maintain screen tension at 18-22 N/cm
3. Clean screens every 500 prints
4. Verify paste temperature: 23-25°C
5. Inspect finger width uniformity (target: 40-50 μm)"""
    },
    {
        "source": "Equipment Manual v2.3",
        "section": "4.5 - Black Core Defects",
        "keywords": ["black_core", "black", "core", "dark", "center", "degradation"],
        "content": """Black core defect causes and prevention:
- Caused by oxygen contamination during diffusion
- Check POCl3 bubbler temperature (28-30°C)
- Maintain nitrogen flow rate at 2-3 slm
- Monitor furnace tube for leaks
- Ensure wafer loading density ≤ 200 wafers/run"""
    },
    {
        "source": "Best Practices Guide",
        "section": "5.1 - Defect Correlation with Process Parameters",
        "keywords": ["defect", "correlation", "parameter", "process", "optimization", "trend"],
        "content": """Key correlations observed:
- Temperature +5°C above setpoint → 15% increase in cracks
- Pressure variation >0.1 bar → 20% increase in finger defects
- Humidity >60% → increased printing errors
- Speed >105 modules/min → quality degradation
Recommendation: Monitor parameters in real-time, trigger alerts at ±10% deviation"""
    },
    {
        "source": "Best Practices Guide",
        "section": "5.2 - Short Circuit Prevention",
        "keywords": ["short", "circuit", "short_circuit", "shunt", "electrical"],
        "content": """Short circuit prevention:
1. Ensure proper isolation between cell edges
2. Check solder ribbon alignment (±0.5mm tolerance)
3. Inspect for metal debris before lamination
4. Verify string voltage before encapsulation
5. Use edge isolation paste on all cells"""
    },
    {
        "source": "Best Practices Guide",
        "section": "5.3 - Star Crack Analysis",
        "keywords": ["star", "crack", "star_crack", "impact", "point", "radial"],
        "content": """Star crack characteristics and prevention:
- Star cracks originate from point impact damage
- Common causes: Dropping, sharp objects, improper stacking
- Prevention: Use pneumatic handling systems, avoid manual cell handling
- Detection: EL imaging shows radial dark lines from center point
- Action: Investigate handling equipment if star cracks increase"""
    },
    {
        "source": "Maintenance Guide",
        "section": "6.1 - Daily Checklist",
        "keywords": ["daily", "check", "maintenance", "inspection", "routine"],
        "content": """Daily maintenance checklist:
□ Check temperature controller readings
□ Verify pressure gauge calibration
□ Inspect conveyor belt condition
□ Clean EL camera lens
□ Review previous shift defect log
□ Check consumable levels (paste, ribbon, EVA)"""
    },
    {
        "source": "Maintenance Guide",
        "section": "6.2 - Thick Line Defects",
        "keywords": ["thick", "line", "thick_line", "grid", "printing", "screen"],
        "content": """Thick line defect troubleshooting:
- Cause 1: Worn screen mesh → Replace screen
- Cause 2: Paste viscosity too low → Adjust paste temperature
- Cause 3: Incorrect squeegee pressure → Re-calibrate
- Cause 4: Contaminated paste → Use fresh batch
Target finger width: 40-50 μm, lines >60 μm are defective"""
    },
    {
        "source": "Troubleshooting Guide",
        "section": "7.1 - Fragment and Corner Defects",
        "keywords": ["fragment", "corner", "chip", "broken", "edge", "mechanical"],
        "content": """Fragment and corner defect handling:
- Usually caused by mechanical damage during transport
- Check gripper pressure settings
- Inspect edge support on conveyors
- Review loading/unloading procedures
- Consider adding edge protection bumpers"""
    }
]


def calculate_relevance(query: str, doc: Dict) -> float:
    """Calculate relevance score based on keyword matching."""
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))
    
    # Check keyword matches
    keyword_matches = sum(1 for kw in doc["keywords"] if kw.lower() in query_lower)
    keyword_score = keyword_matches / len(doc["keywords"]) if doc["keywords"] else 0
    
    # Check content word overlap
    content_words = set(re.findall(r'\w+', doc["content"].lower()))
    word_overlap = len(query_words & content_words) / len(query_words) if query_words else 0
    
    # Check section/title match
    section_lower = doc["section"].lower()
    section_match = any(word in section_lower for word in query_words)
    
    # Combined score
    score = (keyword_score * 0.5) + (word_overlap * 0.3) + (0.2 if section_match else 0)
    return round(min(score * 1.5, 1.0), 3)  # Scale up but cap at 1.0


@tool(
    name="query_knowledge",
    description="Search equipment manuals and best practices for relevant procedures, troubleshooting guides, and recommendations for solar panel manufacturing"
)
def query_knowledge(
    query: str, 
    top_k: int = 3,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Search the knowledge base for relevant manufacturing information.
    
    Args:
        query: Search query (e.g., "how to reduce crack defects", "temperature settings")
        top_k: Number of results to return (default 3)
        state: Current agent state (injected by executor)
    
    Returns:
        Relevant sections from equipment manuals and best practices
    """
    if not query or not query.strip():
        return {
            "status": "failed",
            "error": "Query cannot be empty"
        }
    
    # Calculate relevance scores for all documents
    scored_results = []
    for doc in KNOWLEDGE_BASE:
        score = calculate_relevance(query, doc)
        if score > 0.1:  # Minimum threshold
            scored_results.append({
                "source": doc["source"],
                "section": doc["section"],
                "content": doc["content"],
                "relevance_score": score
            })
    
    # Sort by relevance and take top_k
    scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = scored_results[:top_k]
    
    if not top_results:
        return {
            "status": "success",
            "query": query,
            "results": [],
            "total_results": 0,
            "message": "No relevant documents found. Try different keywords."
        }
    
    return {
        "status": "success",
        "query": query,
        "results": top_results,
        "total_results": len(top_results),
        "all_matches": len(scored_results)
    }
