import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from settings import QCSettings
from tools.recommend_optimization import generate_llm_recommendations


class RecommendOptimizationTests(TestCase):
    def test_llm_recommendations_use_configured_flash_model(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "analysis": {},
                                "recommendations": [],
                            }
                        )
                    )
                )
            ]
        )

        with patch("litellm.completion", return_value=response) as completion:
            result = generate_llm_recommendations(
                [{"defect_type": "crack", "severity": "high"}],
                "rules",
            )

        self.assertEqual("llm", result["source"])
        self.assertEqual(
            QCSettings().RECOMMENDATION_LLM_MODEL,
            completion.call_args.kwargs["model"],
        )
