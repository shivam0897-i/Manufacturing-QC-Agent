import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image, ImageDraw

from vision.rgb_classifier import is_grayscale


class RGBClassifierImageTypeTests(TestCase):
    def test_grayscale_detection_tolerates_small_colored_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "annotated_el.jpg"
            image = Image.new("RGB", (200, 200), (36, 36, 36))
            draw = ImageDraw.Draw(image)
            draw.rectangle([70, 120, 170, 160], outline=(255, 255, 255), width=3)
            draw.text((70, 100), "crack 0.70", fill=(16, 28, 102))
            image.save(image_path, quality=95)

            self.assertTrue(is_grayscale(str(image_path)))

    def test_grayscale_detection_rejects_color_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rgb_panel.jpg"
            image = Image.new("RGB", (200, 200), (34, 120, 54))
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 180, 180], fill=(80, 160, 210))
            image.save(image_path, quality=95)

            self.assertFalse(is_grayscale(str(image_path)))
