"""MLflow tracing integration for the Manufacturing QC agent."""

from contextlib import contextmanager
import sys
from typing import Any, Dict, Optional

try:
    import mlflow as _imported_mlflow
    _mlflow_import_error: Optional[Exception] = None
except ImportError as exc:
    _imported_mlflow = None
    _mlflow_import_error = exc


_mlflow_module = None
_enabled = False
_status: Dict[str, Any] = {"enabled": False, "reason": "disabled"}

_MAX_TRACE_STRING_LENGTH = 200
_MAX_TRACE_LIST_LENGTH = 20
_MAX_TRACE_DEPTH = 4
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_credential",
    "_password",
    "_secret",
    "_token",
)
_OMITTED_TRACE_KEYS = (
    "provider_specific_fields",
    "raw_llm_response",
    "raw_results",
    "thinking_blocks",
    "thought_signature",
    "thought_signatures",
)


def reset_mlflow_observability() -> None:
    """Reset module state. Intended for tests and controlled reconfiguration."""
    global _mlflow_module, _enabled, _status
    _mlflow_module = None
    _enabled = False
    _status = {"enabled": False, "reason": "disabled"}


def setup_mlflow_observability(settings: Any, logger: Any = None) -> Dict[str, Any]:
    """Configure MLflow tracing when explicitly enabled.

    The integration is intentionally opt-in. If enabled without a tracking URI,
    it no-ops instead of creating local tracking artifacts inside the API
    container.
    """
    global _mlflow_module, _enabled, _status

    if not getattr(settings, "ENABLE_MLFLOW", False):
        reset_mlflow_observability()
        _status = {"enabled": False, "reason": "disabled"}
        return dict(_status)

    tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", None)
    if not tracking_uri:
        reset_mlflow_observability()
        _status = {"enabled": False, "reason": "missing_tracking_uri"}
        if logger:
            logger.warning("MLflow is enabled but MLFLOW_TRACKING_URI is not configured")
        return dict(_status)

    experiment_name = getattr(settings, "MLFLOW_EXPERIMENT_NAME", "manufacturing-qc-agent")

    if _imported_mlflow is None:
        reset_mlflow_observability()
        _status = {
            "enabled": False,
            "reason": "missing_dependency",
            "error": str(_mlflow_import_error or "mlflow package is not installed"),
        }
        if logger:
            logger.warning("MLflow is enabled but the mlflow package is not installed")
        return dict(_status)

    try:
        mlflow = _imported_mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.litellm.autolog(log_traces=True, silent=True)
    except Exception as exc:
        reset_mlflow_observability()
        _status = {"enabled": False, "reason": "setup_failed", "error": str(exc)}
        if logger:
            logger.warning("MLflow setup failed: %s", exc)
        return dict(_status)

    _mlflow_module = mlflow
    _enabled = True
    _status = {"enabled": True, "reason": "enabled"}
    if logger:
        logger.info("MLflow tracing enabled: %s", tracking_uri)
    return dict(_status)


def get_mlflow_status() -> Dict[str, Any]:
    return dict(_status)


def is_mlflow_enabled() -> bool:
    return _enabled


def sanitize_trace_payload(value: Any, *, _depth: int = 0) -> Any:
    """Return a bounded, secret-safe copy for app-controlled trace IO."""
    if _depth >= _MAX_TRACE_DEPTH:
        return _summarize_value(value)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_trace_string(value)
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                sanitized[key_text] = "[redacted]"
            elif _is_omitted_key(key_text):
                sanitized[key_text] = "[omitted]"
            elif key_text in {"documents", "files", "input_files"} and isinstance(item, list):
                sanitized[key_text] = [_summarize_document(doc) for doc in item[:_MAX_TRACE_LIST_LENGTH]]
                if len(item) > _MAX_TRACE_LIST_LENGTH:
                    sanitized[key_text].append({"omitted_count": len(item) - _MAX_TRACE_LIST_LENGTH})
            else:
                sanitized[key_text] = sanitize_trace_payload(item, _depth=_depth + 1)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized_items = [
            sanitize_trace_payload(item, _depth=_depth + 1)
            for item in items[:_MAX_TRACE_LIST_LENGTH]
        ]
        if len(items) > _MAX_TRACE_LIST_LENGTH:
            sanitized_items.append({"omitted_count": len(items) - _MAX_TRACE_LIST_LENGTH})
        return sanitized_items

    return _summarize_value(value)


@contextmanager
def trace_request(
    name: str,
    endpoint: str,
    session_id: str,
    agent: str,
    attributes: Optional[Dict[str, Any]] = None,
    inputs: Any = None,
):
    """Create a request trace span with consistent tags and sanitized IO."""
    request_context = {
        "agent": agent,
        "endpoint": endpoint,
        "session_id": session_id,
    }
    span_attributes = dict(request_context)
    if attributes:
        span_attributes.update(attributes)

    with trace_span(
        name,
        span_type="CHAIN",
        attributes=span_attributes,
        inputs=inputs,
    ) as span:
        update_current_trace(
            tags=request_context,
            metadata=request_context,
            client_request_id=session_id,
        )
        yield span


@contextmanager
def trace_span(
    name: str,
    span_type: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    inputs: Any = None,
):
    """Create an MLflow span when enabled, otherwise act as a no-op context."""
    if not _enabled or _mlflow_module is None:
        yield None
        return

    span_kwargs = {"name": name}
    if span_type:
        span_kwargs["span_type"] = span_type
    if attributes:
        span_kwargs["attributes"] = attributes

    try:
        span_context = _mlflow_module.start_span(**span_kwargs)
        span = span_context.__enter__()
    except Exception:
        yield None
        return

    try:
        if inputs is not None:
            _safe_call(span, "set_inputs", sanitize_trace_payload(inputs))
        yield span
    except Exception:
        exc_info = sys.exc_info()
        try:
            span_context.__exit__(*exc_info)
        except Exception:
            pass
        raise
    else:
        try:
            span_context.__exit__(None, None, None)
        except Exception:
            return


def set_span_outputs(span: Any, outputs: Any) -> None:
    if span is not None:
        _safe_call(span, "set_outputs", sanitize_trace_payload(outputs))


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    if span is not None:
        _safe_call(span, "set_attribute", key, sanitize_trace_payload(value))


def update_current_trace(
    tags: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    client_request_id: Optional[str] = None,
) -> None:
    if not _enabled or _mlflow_module is None:
        return

    kwargs = {}
    if tags:
        kwargs["tags"] = tags
    if metadata:
        kwargs["metadata"] = metadata
    if client_request_id:
        kwargs["client_request_id"] = client_request_id
    if not kwargs:
        return

    try:
        _mlflow_module.update_current_trace(**kwargs)
    except Exception:
        return


def _safe_call(target: Any, method_name: str, *args: Any) -> None:
    try:
        method = getattr(target, method_name, None)
        if method:
            method(*args)
    except Exception:
        return


def _truncate_trace_string(value: str) -> str:
    if len(value) <= _MAX_TRACE_STRING_LENGTH:
        return value
    return f"{value[:_MAX_TRACE_STRING_LENGTH]}...[truncated]"


def _summarize_document(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"value_type": type(value).__name__}

    summary = {}
    for key in ("document_id", "filename", "type", "size_bytes"):
        if key in value:
            summary[key] = sanitize_trace_payload(value[key], _depth=1)
    return summary or {"keys": sorted(str(key) for key in value.keys())}


def _summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        return f"<dict keys={len(value)}>"
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__} items={len(value)}>"
    return f"<{type(value).__name__}>"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_KEYS or lowered.endswith(_SENSITIVE_KEY_SUFFIXES)


def _is_omitted_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _OMITTED_TRACE_KEYS)
