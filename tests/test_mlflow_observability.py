import sys
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
