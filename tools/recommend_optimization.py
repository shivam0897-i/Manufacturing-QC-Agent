"""
Recommend Optimization Tool
===========================

Generates intelligent process optimization recommendations using:
- LLM analysis with manufacturing rules as context (primary)
- Rule-based fallback if LLM fails
"""

import json
from typing import Dict, Any, List
from point9_platform.tools.decorator import tool

# Import prompts
from prompts.templates import (
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_PROMPT,
    format_defects_summary,
    format_rules_context
)

# Defect-to-recommendation mapping
DEFECT_RECOMMENDATIONS = {
    "crack": {
        "primary": {
            "recommendation": "Reduce lamination temperature by 3-5°C",
            "priority": "high",
            "parameter": "lamination_temperature",
            "rationale": "Cracks are often caused by thermal stress. Equipment Manual Section 4.2 recommends maintaining 145-150°C."
        },
        "secondary": {
            "recommendation": "Check conveyor alignment and handling procedures",
            "priority": "medium",
            "parameter": "mechanical_handling",
            "rationale": "Mechanical stress during transport can cause micro-cracks that propagate during lamination."
        }
    },
    "black_core": {
        "primary": {
            "recommendation": "Check POCl3 diffusion parameters",
            "priority": "high",
            "parameter": "diffusion_settings",
            "rationale": "Black core defects indicate oxygen contamination. Verify bubbler temperature (28-30°C) and N2 flow rate."
        }
    },
    "finger": {
        "primary": {
            "recommendation": "Optimize screen printing parameters",
            "priority": "high",
            "parameter": "screen_printing",
            "rationale": "Finger interruptions caused by paste viscosity or screen wear. Check paste viscosity (120-150 Pa·s)."
        },
        "secondary": {
            "recommendation": "Clean or replace printing screens",
            "priority": "medium",
            "parameter": "screen_maintenance",
            "rationale": "Screens should be cleaned every 500 prints and replaced when worn."
        }
    },
    "thick_line": {
        "primary": {
            "recommendation": "Adjust squeegee pressure",
            "priority": "medium",
            "parameter": "squeegee_pressure",
            "rationale": "Thick lines indicate excessive paste deposition. Reduce pressure or check paste temperature."
        }
    },
    "star_crack": {
        "primary": {
            "recommendation": "Review cell handling procedures",
            "priority": "high",
            "parameter": "handling_procedure",
            "rationale": "Star cracks originate from point impact. Check for dropped cells or sharp objects in handling equipment."
        }
    },
    "short_circuit": {
        "primary": {
            "recommendation": "Inspect solder ribbon alignment",
            "priority": "critical",
            "parameter": "soldering",
            "rationale": "Short circuits cause module failure. Verify ribbon alignment (±0.5mm) and check for metal debris."
        }
    },
    "horizontal_dislocation": {
        "primary": {
            "recommendation": "Recalibrate stringer alignment",
            "priority": "medium",
            "parameter": "stringer_alignment",
            "rationale": "Cell dislocation affects module efficiency. Check stringer positioning accuracy."
        }
    },
    "vertical_dislocation": {
        "primary": {
            "recommendation": "Recalibrate cell placement mechanism",
            "priority": "medium",
            "parameter": "placement_mechanism",
            "rationale": "Vertical misalignment indicates cell placement issues. Check suction cup condition."
        }
    },
    "printing_error": {
        "primary": {
            "recommendation": "Review screen printing setup",
            "priority": "high",
            "parameter": "printing_setup",
            "rationale": "Printing errors indicate alignment or paste issues. Recalibrate print head position."
        }
    },
    "corner": {
        "primary": {
            "recommendation": "Check gripper pressure settings",
            "priority": "medium",
            "parameter": "gripper_pressure",
            "rationale": "Corner damage from excessive gripper force. Reduce pressure to minimum required."
        }
    },
    "fragment": {
        "primary": {
            "recommendation": "Inspect cell transport mechanism",
            "priority": "high",
            "parameter": "transport_mechanism",
            "rationale": "Fragments indicate severe mechanical damage. Check conveyor edges and handling equipment."
        }
    },
    "scratch": {
        "primary": {
            "recommendation": "Clean conveyor belt surfaces",
            "priority": "medium",
            "parameter": "conveyor_maintenance",
            "rationale": "Scratches from debris on transport surfaces. Clean all contact surfaces."
        }
    }
}

# RGB Surface Condition Recommendations (EfficientNet classifications)
RGB_RECOMMENDATIONS = {
    "Snow-Covered": {
        "primary": {
            "recommendation": "Clear snow from panel surface",
            "priority": "high",
            "parameter": "panel_cleaning",
            "rationale": "Snow coverage blocks sunlight and reduces power output. Clear snow to restore generation capacity."
        }
    },
    "Bird-drop": {
        "primary": {
            "recommendation": "Clean panel surface with appropriate cleaning solution",
            "priority": "medium",
            "parameter": "panel_cleaning",
            "rationale": "Bird droppings cause hot spots and reduce efficiency. Schedule cleaning within 24-48 hours."
        }
    },
    "Dusty": {
        "primary": {
            "recommendation": "Schedule routine panel cleaning",
            "priority": "low",
            "parameter": "maintenance_schedule",
            "rationale": "Dust accumulation reduces efficiency by 5-25%. Include in regular maintenance schedule."
        }
    },
    "Electrical-damage": {
        "primary": {
            "recommendation": "Disconnect panel and arrange professional inspection",
            "priority": "critical",
            "parameter": "safety_inspection",
            "rationale": "Electrical damage poses safety risk. Isolate panel immediately and contact qualified technician."
        }
    },
    "Physical-Damage": {
        "primary": {
            "recommendation": "Inspect panel for replacement",
            "priority": "high",
            "parameter": "panel_replacement",
            "rationale": "Physical damage may compromise panel integrity and efficiency. Assess for repair or replacement."
        }
    },
    "Clean": {
        "primary": {
            "recommendation": "No action required",
            "priority": "low",
            "parameter": None,
            "rationale": "Panel is in good condition. Continue regular monitoring schedule."
        }
    }
}

# Anomaly-to-recommendation mapping
ANOMALY_RECOMMENDATIONS = {
    "temperature": {
        "high": {
            "recommendation": "Reduce heating zone setpoint",
            "priority": "high",
            "parameter": "heating_setpoint",
            "rationale": "Temperature exceeded safe limits. Risk of EVA degradation and thermal stress cracks."
        },
        "low": {
            "recommendation": "Check heating element functionality",
            "priority": "medium",
            "parameter": "heating_element",
            "rationale": "Low temperature may cause incomplete lamination."
        }
    },
    "pressure": {
        "high": {
            "recommendation": "Verify pressure regulator settings",
            "priority": "medium",
            "parameter": "pressure_regulator",
            "rationale": "Excessive pressure can cause cell breakage."
        }
    },
    "speed": {
        "low": {
            "recommendation": "Check conveyor motor and belts",
            "priority": "medium",
            "parameter": "conveyor_motor",
            "rationale": "Speed reduction may indicate mechanical issues."
        }
    },
    "humidity": {
        "high": {
            "recommendation": "Increase facility ventilation",
            "priority": "medium",
            "parameter": "hvac_settings",
            "rationale": "High humidity affects paste viscosity and printing quality."
        }
    }
}


def get_defect_recommendations(defects: List[Dict]) -> List[Dict]:
    """Generate recommendations based on detected defects (both YOLOv8 and EfficientNet)."""
    recommendations = []
    seen_types = set()
    
    for defect in defects:
        defect_type = defect.get("defect_type") or defect.get("type") or defect.get("class", "").lower()
        
        if defect_type in seen_types:
            continue
        seen_types.add(defect_type)
        
        # Check manufacturing defects first (YOLOv8)
        if defect_type in DEFECT_RECOMMENDATIONS:
            rec_data = DEFECT_RECOMMENDATIONS[defect_type]
            primary = rec_data["primary"].copy()
            primary["defect_source"] = defect_type
            recommendations.append(primary)
            
            if "secondary" in rec_data:
                secondary = rec_data["secondary"].copy()
                secondary["defect_source"] = defect_type
                recommendations.append(secondary)
        
        # Check RGB surface conditions (EfficientNet)
        elif defect_type in RGB_RECOMMENDATIONS:
            rec_data = RGB_RECOMMENDATIONS[defect_type]
            primary = rec_data["primary"].copy()
            primary["defect_source"] = defect_type
            primary["model_type"] = "rgb"
            recommendations.append(primary)
    
    return recommendations


def get_anomaly_recommendations(anomalies: List[Dict]) -> List[Dict]:
    """Generate recommendations based on log anomalies."""
    recommendations = []
    seen = set()  # Track (recommendation, parameter) to avoid duplicates
    
    for anomaly in anomalies:
        field = anomaly.get("field", "")
        severity = anomaly.get("severity", "medium")
        value = anomaly.get("value", 0)
        expected_mean = anomaly.get("expected_mean", 0)
        unit = anomaly.get("unit")  # Get unit from anomaly
        
        # Determine if high or low anomaly
        direction = "high" if value > expected_mean else "low"
        
        if field in ANOMALY_RECOMMENDATIONS:
            if direction in ANOMALY_RECOMMENDATIONS[field]:
                rec_data = ANOMALY_RECOMMENDATIONS[field][direction].copy()
                
                # Create unique key for deduplication
                key = (rec_data["recommendation"], rec_data["parameter"])
                if key in seen:
                    continue  # Skip duplicate recommendation
                seen.add(key)
                
                # Use clean anomaly_source naming (just the type, not field_type)
                rec_data["anomaly_source"] = anomaly.get("anomaly_type", f"{field}_anomaly")
                rec_data["anomaly_value"] = value
                rec_data["expected_value"] = expected_mean
                rec_data["unit"] = unit  # Include unit for frontend display
                recommendations.append(rec_data)
    
    return recommendations


def generate_llm_recommendations(defects: List[Dict], rules_context: str) -> Dict[str, Any]:
    """Generate recommendations using LLM with manufacturing rules as context."""
    try:
        from litellm import completion
        
        defects_summary = format_defects_summary(defects)
        user_prompt = RECOMMENDATION_USER_PROMPT.format(
            defects_summary=defects_summary,
            rules_context=rules_context
        )
        
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            temperature=0.2,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        result["source"] = "llm"
        return result
        
    except Exception as e:
        print(f"LLM recommendation failed: {e}")
        return None


@tool(
    name="recommend_optimization",
    description="Generate intelligent optimization recommendations using LLM analysis with manufacturing rules as context. Falls back to rule-based if LLM fails."
)
def recommend_optimization(
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate optimization recommendations using LLM with rules as context.
    Falls back to rule-based logic if LLM fails.
    
    Args:
        state: Current agent state (injected by executor)
    
    Returns:
        Optimization recommendations with priority and rationale
    """
    results = state.get("results", {}) if state else {}
    
    # Extract defects from analyze_image results
    defects = []
    image_result = results.get("analyze_image", {})
    if isinstance(image_result, dict) and image_result.get("status") == "success":
        if "results" in image_result and isinstance(image_result["results"], list):
            for res in image_result["results"]:
                if isinstance(res, dict) and res.get("status") == "success":
                    defects.extend(res.get("defects", []))
        elif "defects" in image_result:
            defects = image_result.get("defects", [])
    
    # Extract anomalies from analyze_logs results  
    anomalies = []
    log_result = results.get("analyze_logs", {})
    if isinstance(log_result, dict) and log_result.get("status") == "success":
        anomalies = log_result.get("anomalies", [])
    
    # Try LLM-powered recommendations first
    llm_result = None
    if defects:
        # Combine manufacturing and RGB rules for context
        all_rules = {**DEFECT_RECOMMENDATIONS, **RGB_RECOMMENDATIONS}
        rules_context = format_rules_context(all_rules)
        llm_result = generate_llm_recommendations(defects, rules_context)
    
    if llm_result:
        llm_recs = llm_result.get("recommendations", [])
        
        # Helper to check if value is meaningful (not null/empty/N/A)
        def is_valid(val):
            if not val:
                return False
            if isinstance(val, str) and val.strip().lower() in ("n/a", "na", "none", "null", ""):
                return False
            return True
        
        # Normalize format with proper null handling
        all_recommendations = []
        for rec in llm_recs:
            normalized = {
                "recommendation": rec.get("action") or rec.get("recommendation", ""),
                "priority": rec.get("priority", "medium"),
                "rationale": rec.get("rationale") if is_valid(rec.get("rationale")) else None,
                "source": "llm"
            }
            # Only include optional fields if they have meaningful values
            if is_valid(rec.get("parameter")):
                normalized["parameter"] = rec["parameter"]
            if is_valid(rec.get("target_value")):
                normalized["target_value"] = rec["target_value"]
            if is_valid(rec.get("expected_impact")):
                normalized["expected_impact"] = rec["expected_impact"]
            all_recommendations.append(normalized)
        
        # Add anomaly recommendations from rules (LLM focused on defects)
        if anomalies:
            for rec in get_anomaly_recommendations(anomalies):
                rec["source"] = "rules"
                all_recommendations.append(rec)
        
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
        
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for rec in all_recommendations:
            priority_counts[rec.get("priority", "low")] += 1
        
        return {
            "status": "success",
            "source": "llm",
            "analysis": llm_result.get("analysis", {}),
            "recommendations": all_recommendations,
            "total_recommendations": len(all_recommendations),
            "priority_summary": priority_counts,
            "defects_analyzed": len(defects),
            "anomalies_analyzed": len(anomalies)
        }
    
    # Fallback to rule-based recommendations
    all_recommendations = []
    
    if defects:
        for rec in get_defect_recommendations(defects):
            rec["source"] = "rules"
            all_recommendations.append(rec)
    
    if anomalies:
        for rec in get_anomaly_recommendations(anomalies):
            rec["source"] = "rules"
            all_recommendations.append(rec)
    
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
    
    if not all_recommendations:
        all_recommendations.append({
            "recommendation": "No immediate action required",
            "priority": "low",
            "parameter": None,
            "rationale": "No critical anomalies or defects detected.",
            "source": "rules"
        })
    
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for rec in all_recommendations:
        priority_counts[rec.get("priority", "low")] += 1
    
    return {
        "status": "success",
        "source": "rules",
        "recommendations": all_recommendations,
        "total_recommendations": len(all_recommendations),
        "priority_summary": priority_counts,
        "defects_analyzed": len(defects),
        "anomalies_analyzed": len(anomalies)
    }
