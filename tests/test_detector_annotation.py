import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image

from vision.detector import DefectDetector


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Box:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Boxes:
    cls = [_Scalar(2), _Scalar(2)]
    conf = [_Scalar(0.70), _Scalar(0.40)]
    xyxy = [_Box([10, 10, 60, 60]), _Box([20, 20, 40, 40])]

    def __len__(self):
        return 2


class _Result:
    boxes = _Boxes()

    def plot(self):
        raise AssertionError("annotation must be rendered from filtered defects")


class _Model:
    def __init__(self):
        self.last_confidence = None

    def __call__(self, image_path, conf, verbose=False):
        self.last_confidence = conf
        return [_Result()]


class DetectorAnnotationTests(TestCase):
    def test_annotation_matches_filtered_detection_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panel.jpg"
            Image.new("RGB", (80, 80), (0, 0, 0)).save(image_path)

            model = _Model()
            detector = DefectDetector(confidence_threshold=0.5)
            detector.model = model
            detector._loaded = True

            result = detector.detect_and_annotate(str(image_path), output_dir=temp_dir)

            self.assertEqual("success", result["status"])
            self.assertEqual(1, result["total_defects"])
            self.assertEqual(0.7, result["defects"][0]["confidence"])
            self.assertEqual(0.5, model.last_confidence)

            annotated = Image.open(result["annotated_image_path"]).convert("RGB")
            self.assertEqual((0, 0, 0), annotated.getpixel((0, 0)))
            self.assertNotEqual((0, 0, 0), annotated.getpixel((10, 10)))
