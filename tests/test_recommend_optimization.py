import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from settings import QCSettings
from tools.recommend_optimization import generate_llm_recommendations, recommend_optimization


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

    def test_llm_recommendations_are_merged_with_rule_recommendations(self):
        state = {
            "results": {
                "analyze_image": {
                    "status": "success",
                    "results": [
                        {
                            "status": "success",
                            "defects": [
                                {"defect_type": "crack", "severity": "high"},
                                {"defect_type": "crack", "severity": "high"},
                            ],
                        }
                    ],
                }
            }
        }
        llm_result = {
            "analysis": {"root_cause": "thermal stress"},
            "recommendations": [
                {
                    "action": "Review lamination temperature profile",
                    "priority": "high",
                    "parameter": "lamination_temperature",
                    "rationale": "Crack pattern indicates heat stress.",
                }
            ],
        }

        with patch("tools.recommend_optimization.generate_llm_recommendations", return_value=llm_result):
            result = recommend_optimization(state=state)

        recommendation_text = [r["recommendation"] for r in result["recommendations"]]
        self.assertIn("Review lamination temperature profile", recommendation_text)
        self.assertIn("Reduce lamination temperature by 3-5°C", recommendation_text)
        self.assertIn("Check conveyor alignment and handling procedures", recommendation_text)
