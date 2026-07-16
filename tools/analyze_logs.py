"""
Analyze Logs Tool
=================

Parses production logs (CSV/TXT) and detects anomalies in manufacturing data.
Supports: temperature, speed, pressure, humidity, and other numeric parameters.
"""

import csv
import io
import os
from typing import Dict, Any, List, Optional
from statistics import mean, stdev
import yaml

from point9_platform.tools.decorator import tool
from tools._utils import find_document, get_available_doc_ids


def _load_field_units() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            field_units = config.get("FIELD_UNITS", {})
            return {k.lower(): v for k, v in field_units.items()}
    except Exception:
        return {}


def parse_csv_log(content: str) -> List[Dict[str, Any]]:
    """Parse CSV log file content into list of records."""
    records = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        # Convert numeric fields
        parsed_row = {}
        for key, value in row.items():
            key = key.strip().lower().replace(' ', '_')
            try:
                # Try to convert to float
                parsed_row[key] = float(value)
            except (ValueError, TypeError):
                # Keep as string (e.g., timestamps)
                parsed_row[key] = value.strip() if value else None
        records.append(parsed_row)
    return records


def parse_txt_log(content: str) -> List[Dict[str, Any]]:
    """Parse TXT log file (key=value or tab-separated format)."""
    records = []
    lines = content.strip().split('\n')
    
    # Try tab-separated format first
    if '\t' in lines[0]:
        headers = [h.strip().lower().replace(' ', '_') for h in lines[0].split('\t')]
        for line in lines[1:]:
            values = line.split('\t')
            if len(values) == len(headers):
                record = {}
                for i, header in enumerate(headers):
                    try:
                        record[header] = float(values[i])
                    except (ValueError, TypeError):
                        record[header] = values[i].strip()
                records.append(record)
    else:
        # Key=value format
        current_record = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    current_record[key] = float(value.strip())
                except ValueError:
                    current_record[key] = value.strip()
            elif line.strip() == '' and current_record:
                records.append(current_record)
                current_record = {}
        if current_record:
            records.append(current_record)
    
    return records


def detect_anomalies(
    records: List[Dict[str, Any]], 
    numeric_fields: List[str],
    threshold_std: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using statistical analysis (z-score method).
    Anomalies are values that deviate more than threshold_std standard deviations.
    """
    # Load field units from config
    field_units = _load_field_units()
    
    anomalies = []
    
    for field in numeric_fields:
        # Get all numeric values for this field
        values = []
        for i, record in enumerate(records):
            val = record.get(field)
            if isinstance(val, (int, float)):
                values.append((i, val))
        
        if len(values) < 3:  # Need at least 3 values for meaningful stats
            continue
        
        # Calculate statistics
        vals = [v[1] for v in values]
        avg = mean(vals)
        std = stdev(vals) if len(vals) > 1 else 0
        
        if std == 0:  # No variation
            continue
        
        # Find anomalies
        for idx, value in values:
            z_score = abs(value - avg) / std
            if z_score > threshold_std:
                record = records[idx]
                timestamp = record.get('timestamp') or record.get('time') or record.get('datetime') or f"record_{idx}"
                
                # Determine severity based on z-score
                if z_score > 3:
                    severity = "critical"
                elif z_score > 2.5:
                    severity = "high"
                else:
                    severity = "medium"
                
                # Get unit for this field (case-insensitive lookup)
                unit = field_units.get(field.lower())
                unit_suffix = f" {unit}" if unit else ""
                
                anomalies.append({
                    "anomaly_type": f"{field}_anomaly",
                    "field": field,
                    "timestamp": str(timestamp),
                    "value": round(value, 2),
                    "unit": unit,  # Will be None if not found in config
                    "expected_mean": round(avg, 2),
                    "expected_range": f"{round(avg - threshold_std*std, 2)}{unit_suffix} - {round(avg + threshold_std*std, 2)}{unit_suffix}",
                    "z_score": round(z_score, 2),
                    "severity": severity
                })
    
    return sorted(anomalies, key=lambda x: x.get('z_score', 0), reverse=True)


def detect_trends(records: List[Dict[str, Any]], numeric_fields: List[str]) -> List[Dict[str, Any]]:
    """Detect trends in numeric parameters over time."""
    # Load field units from config
    field_units = _load_field_units()
    
    trends = []
    
    for field in numeric_fields:
        values = [r[field] for r in records if field in r and isinstance(r[field], (int, float))]
        
        if len(values) < 5:
            continue
        
        # Simple trend detection: compare first half vs second half
        mid = len(values) // 2
        first_half_avg = mean(values[:mid])
        second_half_avg = mean(values[mid:])
        
        change = second_half_avg - first_half_avg
        change_pct = (change / first_half_avg * 100) if first_half_avg != 0 else 0
        
        if abs(change_pct) > 5:  # More than 5% change
            trend = "increasing" if change > 0 else "decreasing"
            unit = field_units.get(field.lower())
            unit_suffix = f" {unit}" if unit else ""
            
            trends.append({
                "parameter": field,
                "trend": trend,
                "change": f"{'+' if change > 0 else ''}{round(change, 2)}{unit_suffix}",
                "change_percent": f"{round(change_pct, 1)}%",
                "first_half_avg": round(first_half_avg, 2),
                "second_half_avg": round(second_half_avg, 2),
                "unit": unit
            })
    
    return trends


@tool(
    name="analyze_logs",
    description="Parse production logs (CSV/TXT) to find anomalies like temperature spikes, speed variations, and errors. Supports statistical anomaly detection."
)
def analyze_logs(
    log_file_id: str,
    time_range: Optional[str] = None,
    anomaly_threshold: float = 2.0,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze production logs for anomalies using statistical methods.
    
    Args:
        log_file_id: ID of the uploaded log file
        time_range: Optional time range filter (e.g., "2024-01-21 08:00 to 2024-01-21 16:00")
        anomaly_threshold: Standard deviation threshold for anomaly detection (default 2.0)
        state: Current agent state (injected by executor)
    
    Returns:
        Analysis result with anomalies, trends, and statistics
    """
    documents = state.get("documents", {})
    
    # Use shared utility to find document
    doc_info = find_document(log_file_id, documents)
    
    # Smart Fallback: If specific ID failed or wasn't provided, look for any compatible log file
    if not doc_info:
        # scans for any available log file (prioritizing CSV)
        for doc_id, doc in documents.items():
            fname = doc.get("filename", "").lower()
            if fname.endswith('.csv') or fname.endswith('.log') or fname.endswith('.txt'):
                doc_info = doc
                log_file_id = doc_id  # Update ID for reporting
                break

    if not doc_info:
        return {
            "status": "failed",
            "log_file_id": log_file_id,
            "error": f"Log file {log_file_id} not found and no other fallback log files (csv/log/txt) detected. Available: {get_available_doc_ids(documents)}"
        }
    
    file_path = doc_info.get("path")
    filename = doc_info.get("filename", "unknown")
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse based on file extension
        if filename.lower().endswith('.csv'):
            records = parse_csv_log(content)
        else:
            records = parse_txt_log(content)
        
        if not records:
            return {
                "status": "failed",
                "log_file_id": log_file_id,
                "error": "Could not parse log file - no records found"
            }
        
        # Identify numeric fields
        numeric_fields = []
        sample_record = records[0]
        for key, value in sample_record.items():
            if isinstance(value, (int, float)):
                numeric_fields.append(key)
        
        # Detect anomalies
        anomalies = detect_anomalies(records, numeric_fields, anomaly_threshold)
        
        # Detect trends
        trends = detect_trends(records, numeric_fields)
        
        # Calculate summary statistics
        # Load field units for stats
        field_units = _load_field_units()
        
        summary_stats = {}
        for field in numeric_fields:
            values = [r[field] for r in records if field in r and isinstance(r[field], (int, float))]
            if values:
                unit = field_units.get(field.lower())
                summary_stats[field] = {
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "mean": round(mean(values), 2),
                    "std": round(stdev(values), 2) if len(values) > 1 else 0,
                    "unit": unit
                }
        
        # Count anomalies by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0}
        for anomaly in anomalies:
            severity_counts[anomaly.get("severity", "medium")] += 1
        
        return {
            "status": "success",
            "log_file_id": log_file_id,
            "filename": filename,
            "time_range": time_range or "full file",
            "total_records": len(records),
            "numeric_fields_analyzed": numeric_fields,
            "anomalies": anomalies[:20],  # Limit to top 20
            "total_anomalies": len(anomalies),
            "anomalies_by_severity": severity_counts,
            "trends": trends,
            "summary_statistics": summary_stats,
            "anomaly_rate": f"{round(len(anomalies) / len(records) * 100, 2)}%" if records else "0%"
        }
        
    except FileNotFoundError:
        return {
            "status": "failed",
            "log_file_id": log_file_id,
            "error": f"File not found: {file_path}"
        }
    except Exception as e:
        return {
            "status": "failed",
            "log_file_id": log_file_id,
            "error": f"Error analyzing log: {str(e)}"
        }
