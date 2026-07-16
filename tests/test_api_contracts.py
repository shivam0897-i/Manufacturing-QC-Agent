import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
from uuid import UUID

os.environ.setdefault("QC_ENABLE_MONGODB", "False")
os.environ.setdefault("QC_ENABLE_S3_STORAGE", "False")

from fastapi.testclient import TestClient

import api.main as api_main


class ApiContractTests(TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)

    def test_process_files_openapi_schema_uses_binary_upload_items(self):
        schema = self.client.get("/openapi.json").json()
        process_schema_ref = schema["paths"]["/process"]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]["$ref"]
        process_schema_name = process_schema_ref.rsplit("/", 1)[-1]
        files_schema = schema["components"]["schemas"][process_schema_name]["properties"]["files"]

        self.assertEqual("array", files_schema["type"])
        self.assertEqual("string", files_schema["items"]["type"])
        self.assertEqual("binary", files_schema["items"]["format"])

    def test_local_annotated_path_is_published_as_served_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_outputs_dir = api_main.LOCAL_OUTPUTS_DIR
            api_main.LOCAL_OUTPUTS_DIR = Path(temp_dir) / "outputs"
            try:
                source_path = Path(temp_dir) / "annotated.jpg"
                source_path.write_bytes(b"annotated-image")

                image_results = {
                    "status": "success",
                    "results": [
                        {
                            "status": "success",
                            "annotated_image_path": str(source_path),
                        }
                    ],
                }

                api_main._publish_local_annotated_artifacts(image_results, "session-123")

                result = image_results["results"][0]
                self.assertNotIn("annotated_image_path", result)
                self.assertEqual(
                    "/outputs/session-123/annotated/annotated.jpg",
                    result["annotated_image_url"],
                )

                response = self.client.get(result["annotated_image_url"])
                self.assertEqual(200, response.status_code)
                self.assertEqual(b"annotated-image", response.content)
            finally:
                api_main.LOCAL_OUTPUTS_DIR = previous_outputs_dir

    def test_swagger_placeholder_session_id_is_replaced_with_uuid(self):
        session_id = api_main._normalize_session_id("string")

        self.assertNotEqual("string", session_id)
        UUID(session_id)

    def test_empty_explanation_llm_content_uses_fallback_without_warning(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

        with patch("api.main.completion", return_value=response), patch.object(
            api_main.logger, "warning"
        ) as warning:
            explanation = api_main.generate_explanation(
                [{"defect_type": "crack", "severity": "high"}],
                [],
            )

        self.assertIn("Found 1 defect(s): crack", explanation)
        warning.assert_not_called()
