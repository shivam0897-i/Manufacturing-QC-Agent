"""
Recommend Optimization Tool
===========================

Generates intelligent process optimization recommendations based on 
defect analysis and log anomalies using rule-based logic.
"""

from typing import Dict, Any, List, Optional
from point9_platform.tools.decorator import tool


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
    """Generate recommendations based on detected defects."""
    recommendations = []
    seen_types = set()
    
    for defect in defects:
        defect_type = defect.get("defect_type") or defect.get("type") or defect.get("class", "").lower()
        
        if defect_type in seen_types:
            continue
        seen_types.add(defect_type)
        
        if defect_type in DEFECT_RECOMMENDATIONS:
            rec_data = DEFECT_RECOMMENDATIONS[defect_type]
            primary = rec_data["primary"].copy()
            primary["defect_source"] = defect_type
            recommendations.append(primary)
            
            if "secondary" in rec_data:
                secondary = rec_data["secondary"].copy()
                secondary["defect_source"] = defect_type
                recommendations.append(secondary)
    
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
                recommendations.append(rec_data)
    
    return recommendations


@tool(
    name="recommend_optimization",
    description="Generate process optimization recommendations based on defect analysis and log anomalies. Call this after analyze_image and analyze_logs."
)
def recommend_optimization(
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate optimization recommendations using rule-based analysis.
    
    Args:
        state: Current agent state (injected by executor)
    
    Returns:
        Optimization recommendations with priority and rationale
    """
    all_recommendations = []
    
    # Get results from state (populated by previous tool calls)
    results = state.get("results", {}) if state else {}
    
    # Get defects from analyze_image results
    defects = []
    image_result = results.get("analyze_image", {})
    if isinstance(image_result, dict) and image_result.get("status") == "success":
        # Handle Batch Format
        if "results" in image_result and isinstance(image_result["results"], list):
            for res in image_result["results"]:
                if isinstance(res, dict) and res.get("status") == "success":
                    defects.extend(res.get("defects", []))
        # Handle Legacy/Single Formatting
        elif "defects" in image_result:
            defects = image_result.get("defects", [])
    
    # Get anomalies from analyze_logs results  
    anomalies = []
    log_result = results.get("analyze_logs", {})
    if isinstance(log_result, dict) and log_result.get("status") == "success":
        anomalies = log_result.get("anomalies", [])
    
    # Generate recommendations
    if defects:
        all_recommendations.extend(get_defect_recommendations(defects))
    
    if anomalies:
        all_recommendations.extend(get_anomaly_recommendations(anomalies))
    
    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
    
    # If no recommendations, provide general advice
    if not all_recommendations:
        all_recommendations.append({
            "recommendation": "No immediate action required",
            "priority": "low",
            "parameter": None,
            "rationale": "No critical anomalies or defects detected. Continue monitoring and follow daily maintenance checklist."
        })
    
    # Count by priority
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for rec in all_recommendations:
        priority_counts[rec.get("priority", "low")] += 1
    
    return {
        "status": "success",
        "recommendations": all_recommendations,
        "total_recommendations": len(all_recommendations),
        "priority_summary": priority_counts,
        "defects_analyzed": len(defects),
        "anomalies_analyzed": len(anomalies)
    }
