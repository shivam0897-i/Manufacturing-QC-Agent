"""
Generate Report Tool
====================

Compiles QC analysis into structured reports for solar panel manufacturing.
Supports summary, detailed, and shift reports.
"""

from datetime import datetime
from typing import Dict, Any, List
from point9_platform.tools.decorator import tool


def _count_by_type(items: list, type_key: str) -> Dict[str, int]:
    """Count items by type."""
    counts = {}
    for item in items:
        item_type = item.get(type_key, "unknown")
        counts[item_type] = counts.get(item_type, 0) + 1
    return counts


def _count_by_severity(items: list) -> Dict[str, int]:
    """Count items by severity."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for item in items:
        severity = item.get("severity", "medium")
        if severity in counts:
            counts[severity] += 1
    return counts


def _extract_image_defects(image_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract defects from legacy single-image and current batch image results."""
    if not isinstance(image_result, dict):
        return []

    defects = []
    direct_defects = image_result.get("defects", [])
    if isinstance(direct_defects, list):
        defects.extend(item for item in direct_defects if isinstance(item, dict))

    batch_results = image_result.get("results", [])
    if isinstance(batch_results, list):
        for result in batch_results:
            if not isinstance(result, dict) or result.get("status") != "success":
                continue
            result_defects = result.get("defects", [])
            if isinstance(result_defects, list):
                defects.extend(item for item in result_defects if isinstance(item, dict))

    return defects


def _generate_summary_report(defects: list, anomalies: list, recommendations: list) -> Dict:
    """Generate a summary report."""
    defect_counts = _count_by_type(defects, "defect_type")
    severity_counts = _count_by_severity(defects)
    
    return {
        "title": "QC Analysis Summary Report",
        "generated_at": datetime.now().isoformat(),
        "sections": [
            {
                "heading": "Executive Summary",
                "content": f"Analyzed production data. Found {len(defects)} defects across {len(defect_counts)} categories. "
                          f"{severity_counts['critical'] + severity_counts['high']} require immediate attention."
            },
            {
                "heading": "Defect Analysis",
                "content": f"Total defects detected: {len(defects)}",
                "data": {
                    "total_defects": len(defects),
                    "by_type": defect_counts,
                    "by_severity": severity_counts,
                    "critical_count": severity_counts["critical"],
                    "high_severity_count": severity_counts["high"]
                }
            },
            {
                "heading": "Process Anomalies",
                "content": f"Total anomalies found: {len(anomalies)}",
                "data": {
                    "total_anomalies": len(anomalies),
                    "by_type": _count_by_type(anomalies, "anomaly_type"),
                    "by_severity": _count_by_severity(anomalies)
                }
            },
            {
                "heading": "Recommendations",
                "content": f"Total recommendations: {len(recommendations)}",
                "data": {
                    "total": len(recommendations),
                    "by_priority": _count_by_type(recommendations, "priority"),
                    "top_recommendations": recommendations[:3] if recommendations else []
                }
            }
        ]
    }


def _generate_detailed_report(defects: list, anomalies: list, recommendations: list, results: dict) -> Dict:
    """Generate a detailed report with all findings."""
    return {
        "title": "Detailed QC Analysis Report",
        "generated_at": datetime.now().isoformat(),
        "sections": [
            {
                "heading": "Defect Details",
                "content": f"Complete list of {len(defects)} detected defects",
                "data": {
                    "defects": defects[:50],  # Limit to 50
                    "summary": _count_by_type(defects, "defect_type")
                }
            },
            {
                "heading": "Anomaly Details",  
                "content": f"Complete list of {len(anomalies)} process anomalies",
                "data": {
                    "anomalies": anomalies[:50],
                    "summary": _count_by_type(anomalies, "field") if anomalies else {}
                }
            },
            {
                "heading": "All Recommendations",
                "content": f"Complete list of {len(recommendations)} recommendations",
                "data": {
                    "recommendations": recommendations
                }
            },
            {
                "heading": "Analysis Results",
                "content": "Raw analysis results from all tools",
                "data": results
            }
        ]
    }


def _generate_shift_report(defects: list, anomalies: list, recommendations: list) -> Dict:
    """Generate a shift handover report."""
    critical_issues = [d for d in defects if d.get("severity") in ["critical", "high"]]
    critical_anomalies = [a for a in anomalies if a.get("severity") in ["critical", "high"]]
    high_priority_recs = [r for r in recommendations if r.get("priority") in ["critical", "high"]]
    
    return {
        "title": "Shift Handover Report",
        "generated_at": datetime.now().isoformat(),
        "shift_summary": {
            "total_defects": len(defects),
            "critical_issues": len(critical_issues),
            "process_anomalies": len(anomalies),
            "pending_actions": len(high_priority_recs)
        },
        "sections": [
            {
                "heading": "Critical Issues - Immediate Attention Required",
                "content": f"{len(critical_issues)} critical/high severity defects detected",
                "data": {
                    "issues": critical_issues[:10],
                    "anomalies": critical_anomalies[:10]
                }
            },
            {
                "heading": "Priority Actions for Next Shift",
                "content": f"{len(high_priority_recs)} high-priority recommendations",
                "data": {
                    "actions": high_priority_recs
                }
            },
            {
                "heading": "Production Statistics",
                "content": "Overview of defect rates and trends",
                "data": {
                    "defect_breakdown": _count_by_type(defects, "defect_type"),
                    "severity_breakdown": _count_by_severity(defects)
                }
            }
        ],
        "handover_notes": "Review critical issues before resuming production. Check recommended actions."
    }


@tool(
    name="generate_report",
    description="Generate a QC analysis report summarizing defects, anomalies, and recommendations. Supports 'summary', 'detailed', and 'shift' report types."
)
def generate_report(
    report_type: str = "summary",
    include_raw_data: bool = False,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a QC analysis report.
    
    Args:
        report_type: Type of report - "summary" (default), "detailed", or "shift"
        include_raw_data: Include raw analysis data in report
        state: Current agent state (injected by executor)
    
    Returns:
        Structured report content
    """
    # Get data from state
    defects = state.get("defects", [])
    anomalies = state.get("log_anomalies", [])
    recommendations = state.get("recommendations", [])
    results = state.get("results", {})
    
    # Also check for data in results (prefix-matching for v1.0 dynamic keys)
    if not defects:
        img_result = {}
        for key, val in results.items():
            if key.startswith("analyze_image"):
                img_result = val
                break
        defects = _extract_image_defects(img_result)
    
    if not anomalies:
        log_result = {}
        for key, val in results.items():
            if key.startswith("analyze_logs"):
                log_result = val
                break
        if isinstance(log_result, dict):
            anomalies = log_result.get("anomalies", [])
    
    if not recommendations:
        rec_result = {}
        for key, val in results.items():
            if key.startswith("recommend_optimization"):
                rec_result = val
                break
        if isinstance(rec_result, dict):
            recommendations = rec_result.get("recommendations", [])
    
    # Generate report based on type
    report_type = report_type.lower()
    
    if report_type == "summary":
        report = _generate_summary_report(defects, anomalies, recommendations)
    elif report_type == "detailed":
        report = _generate_detailed_report(defects, anomalies, recommendations, results if include_raw_data else {})
    elif report_type == "shift":
        report = _generate_shift_report(defects, anomalies, recommendations)
    else:
        return {
            "status": "failed",
            "error": f"Unknown report type: {report_type}. Use 'summary', 'detailed', or 'shift'."
        }
    
    return {
        "status": "success",
        "report_type": report_type,
        "report": report,
        "data_summary": {
            "defects_analyzed": len(defects),
            "anomalies_analyzed": len(anomalies),
            "recommendations_included": len(recommendations)
        }
    }
