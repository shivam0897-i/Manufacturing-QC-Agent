from unittest import TestCase

from tools.generate_report import generate_report


class GenerateReportTests(TestCase):
    def test_batch_image_results_contribute_defects_to_report(self):
        state = {
            "results": {
                "analyze_image_call_1": {
                    "status": "success",
                    "results": [
                        {
                            "status": "success",
                            "defects": [
                                {
                                    "defect_type": "crack",
                                    "severity": "high",
                                }
                            ],
                        }
                    ],
                }
            }
        }

        result = generate_report(state=state)

        self.assertEqual("success", result["status"])
        self.assertEqual(1, result["data_summary"]["defects_analyzed"])
        defect_section = result["report"]["sections"][1]
        self.assertEqual(1, defect_section["data"]["total_defects"])
