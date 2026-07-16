from unittest import TestCase

from settings import QCSettings
from tools.analyze_image import _resolve_confidence_threshold


class AnalyzeImageTests(TestCase):
    def test_missing_confidence_threshold_uses_configured_default(self):
        self.assertEqual(
            QCSettings().CONFIDENCE_THRESHOLD,
            _resolve_confidence_threshold(None),
        )

    def test_explicit_confidence_threshold_is_preserved(self):
        self.assertEqual(0.42, _resolve_confidence_threshold(0.42))
