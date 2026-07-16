import sys
from unittest.mock import patch
from types import SimpleNamespace
from unittest import TestCase

from settings import QCSettings
from observability import mlflow as mlflow_observability


class _FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message, *args):
        self.info_messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.warning_messages.append(message % args if args else message)


class _FakeLiteLLM:
    def __init__(self):
        self.autolog_calls = []

    def autolog(self, **kwargs):
        self.autolog_calls.append(kwargs)


class _FakeSpan:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.attributes = {}

    def set_inputs(self, value):
        self.inputs = value

    def set_outputs(self, value):
        self.outputs = value

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeSpanContext:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMLflow:
    def __init__(self):
        self.litellm = _FakeLiteLLM()
        self.tracking_uri = None
        self.experiment_name = None
        self.start_span_calls = []
        self.trace_updates = []
        self.next_span = _FakeSpan()

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_span(self, **kwargs):
        self.start_span_calls.append(kwargs)
        return _FakeSpanContext(self.next_span)

    def update_current_trace(self, **kwargs):
        self.trace_updates.append(kwargs)


class _FailingSpanMLflow(_FakeMLflow):
    def start_span(self, **kwargs):
        raise RuntimeError("mlflow span failed")


def _install_fake_mlflow(fake_mlflow):
    mlflow_observability._imported_mlflow = fake_mlflow
    mlflow_observability._mlflow_import_error = None


class MlflowObservabilityTests(TestCase):
    def setUp(self):
        self._original_imported_mlflow = mlflow_observability._imported_mlflow
        self._original_mlflow_import_error = mlflow_observability._mlflow_import_error

    def tearDown(self):
        mlflow_observability.reset_mlflow_observability()
        mlflow_observability._imported_mlflow = self._original_imported_mlflow
        mlflow_observability._mlflow_import_error = self._original_mlflow_import_error
        sys.modules.pop("mlflow", None)

    def test_setup_noops_when_disabled(self):
        fake_mlflow = _FakeMLflow()
        _install_fake_mlflow(fake_mlflow)

        status = mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(ENABLE_MLFLOW=False),
            logger=_FakeLogger(),
        )

        self.assertEqual({"enabled": False, "reason": "disabled"}, status)
        self.assertIsNone(fake_mlflow.tracking_uri)

    def test_qc_settings_default_mlflow_off(self):
        settings = QCSettings()

        self.assertFalse(settings.ENABLE_MLFLOW)
        self.assertIsNone(settings.MLFLOW_TRACKING_URI)
        self.assertEqual("manufacturing-qc-agent", settings.MLFLOW_EXPERIMENT_NAME)

    def test_qc_prefixed_env_overrides_yaml_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "QC_ENABLE_MLFLOW": "true",
                "QC_MLFLOW_TRACKING_URI": "file:///tmp/mlruns",
                "QC_MLFLOW_EXPERIMENT_NAME": "hf-space",
            },
            clear=False,
        ):
            settings = QCSettings()

        self.assertTrue(settings.ENABLE_MLFLOW)
        self.assertEqual("file:///tmp/mlruns", settings.MLFLOW_TRACKING_URI)
        self.assertEqual("hf-space", settings.MLFLOW_EXPERIMENT_NAME)

    def test_setup_requires_tracking_uri_when_enabled(self):
        fake_mlflow = _FakeMLflow()
        _install_fake_mlflow(fake_mlflow)

        status = mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(
                ENABLE_MLFLOW=True,
                MLFLOW_TRACKING_URI=None,
                MLFLOW_EXPERIMENT_NAME="manufacturing-qc-agent",
            ),
            logger=_FakeLogger(),
        )

        self.assertEqual({"enabled": False, "reason": "missing_tracking_uri"}, status)
        self.assertIsNone(fake_mlflow.tracking_uri)

    def test_setup_enables_litellm_autologging(self):
        fake_mlflow = _FakeMLflow()
        _install_fake_mlflow(fake_mlflow)

        status = mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(
                ENABLE_MLFLOW=True,
                MLFLOW_TRACKING_URI="http://mlflow.local:5000",
                MLFLOW_EXPERIMENT_NAME="manufacturing-qc-agent",
            ),
            logger=_FakeLogger(),
        )

        self.assertEqual({"enabled": True, "reason": "enabled"}, status)
        self.assertEqual("http://mlflow.local:5000", fake_mlflow.tracking_uri)
        self.assertEqual("manufacturing-qc-agent", fake_mlflow.experiment_name)
        self.assertEqual(
            [{"log_traces": True, "silent": True}],
            fake_mlflow.litellm.autolog_calls,
        )

    def test_trace_span_is_noop_when_disabled(self):
        mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(ENABLE_MLFLOW=False),
            logger=_FakeLogger(),
        )

        with mlflow_observability.trace_span("qc.process") as span:
            self.assertIsNone(span)

    def test_trace_span_records_inputs_outputs_and_attributes(self):
        fake_mlflow = _FakeMLflow()
        _install_fake_mlflow(fake_mlflow)
        mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(
                ENABLE_MLFLOW=True,
                MLFLOW_TRACKING_URI="http://mlflow.local:5000",
                MLFLOW_EXPERIMENT_NAME="manufacturing-qc-agent",
            ),
            logger=_FakeLogger(),
        )

        with mlflow_observability.trace_span(
            "qc.process",
            span_type="CHAIN",
            attributes={"session_id": "session-123"},
            inputs={"message": "Analyze"},
        ) as span:
            mlflow_observability.set_span_outputs(span, {"success": True})
            mlflow_observability.update_current_trace(
                tags={"endpoint": "/process"},
                metadata={"session_id": "session-123"},
                client_request_id="session-123",
            )

        self.assertEqual(
            {
                "name": "qc.process",
                "span_type": "CHAIN",
                "attributes": {"session_id": "session-123"},
            },
            fake_mlflow.start_span_calls[0],
        )
        self.assertEqual({"message": "Analyze"}, fake_mlflow.next_span.inputs)
        self.assertEqual({"success": True}, fake_mlflow.next_span.outputs)
        self.assertEqual(
            [
                {
                    "tags": {"endpoint": "/process"},
                    "metadata": {"session_id": "session-123"},
                    "client_request_id": "session-123",
                }
            ],
            fake_mlflow.trace_updates,
        )

    def test_sanitize_trace_payload_redacts_secrets_and_summarizes_nested_content(self):
        payload = {
            "message": "Analyze " + ("x" * 300),
            "api_key": "secret-value",
            "documents": [
                {"filename": "img000005.jpg", "path": "C:/tmp/raw-image.jpg", "type": "image"},
                {"filename": "run.csv", "s3_key": "inputs/session/run.csv", "type": "log"},
            ],
            "raw_results": {"tool": "large internal result"},
        }

        sanitized = mlflow_observability.sanitize_trace_payload(payload)

        self.assertEqual("[redacted]", sanitized["api_key"])
        self.assertEqual("[omitted]", sanitized["raw_results"])
        self.assertLessEqual(len(sanitized["message"]), 220)
        self.assertEqual(
            [
                {"filename": "img000005.jpg", "type": "image"},
                {"filename": "run.csv", "type": "log"},
            ],
            sanitized["documents"],
        )

    def test_sanitize_trace_payload_preserves_token_usage_metrics(self):
        sanitized = mlflow_observability.sanitize_trace_payload(
            {
                "prompt_tokens": 56,
                "completion_tokens": 147,
                "total_tokens": 203,
                "access_token": "secret-value",
            }
        )

        self.assertEqual(56, sanitized["prompt_tokens"])
        self.assertEqual(147, sanitized["completion_tokens"])
        self.assertEqual(203, sanitized["total_tokens"])
        self.assertEqual("[redacted]", sanitized["access_token"])

    def test_trace_request_applies_standard_tags_and_sanitized_io(self):
        fake_mlflow = _FakeMLflow()
        _install_fake_mlflow(fake_mlflow)
        mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(
                ENABLE_MLFLOW=True,
                MLFLOW_TRACKING_URI="http://mlflow.local:5000",
                MLFLOW_EXPERIMENT_NAME="manufacturing-qc-agent",
            ),
            logger=_FakeLogger(),
        )

        with mlflow_observability.trace_request(
            name="qc.process",
            endpoint="/process",
            session_id="session-123",
            agent="manufacturing_qc_agent",
            attributes={"documents_count": 1},
            inputs={"message": "Analyze", "token": "secret"},
        ) as span:
            mlflow_observability.set_span_outputs(
                span,
                {"success": True, "raw_results": {"thought": "internal"}},
            )

        self.assertEqual(
            {
                "name": "qc.process",
                "span_type": "CHAIN",
                "attributes": {
                    "agent": "manufacturing_qc_agent",
                    "endpoint": "/process",
                    "session_id": "session-123",
                    "documents_count": 1,
                },
            },
            fake_mlflow.start_span_calls[0],
        )
        self.assertEqual({"message": "Analyze", "token": "[redacted]"}, fake_mlflow.next_span.inputs)
        self.assertEqual(
            {"success": True, "raw_results": "[omitted]"},
            fake_mlflow.next_span.outputs,
        )
        self.assertEqual(
            [
                {
                    "tags": {
                        "agent": "manufacturing_qc_agent",
                        "endpoint": "/process",
                        "session_id": "session-123",
                    },
                    "metadata": {
                        "session_id": "session-123",
                        "endpoint": "/process",
                        "agent": "manufacturing_qc_agent",
                    },
                    "client_request_id": "session-123",
                }
            ],
            fake_mlflow.trace_updates,
        )

    def test_trace_span_noops_when_mlflow_span_start_fails(self):
        fake_mlflow = _FailingSpanMLflow()
        _install_fake_mlflow(fake_mlflow)
        mlflow_observability.setup_mlflow_observability(
            SimpleNamespace(
                ENABLE_MLFLOW=True,
                MLFLOW_TRACKING_URI="http://mlflow.local:5000",
                MLFLOW_EXPERIMENT_NAME="manufacturing-qc-agent",
            ),
            logger=_FakeLogger(),
        )

        with mlflow_observability.trace_span("qc.process") as span:
            self.assertIsNone(span)
