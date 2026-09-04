from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ifdr_yolo.data.interventions.schema import InterventionKind
from ifdr_yolo.eval.robustness_view import build_robustness_view


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("failed to encode test image")
    path.write_bytes(encoded.tobytes())


class RobustnessViewTest(unittest.TestCase):
    def test_builds_reproducible_global_condition_without_resizing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            image = np.indices((24, 32)).sum(axis=0) % 2
            image = np.repeat((image * 255).astype(np.uint8)[..., None], 3, 2)
            write_png(source / "000001.png", image)

            first = build_robustness_view(
                source_image_dir=source,
                image_ids=("000001",),
                output_dir=root / "first",
                kind=InterventionKind.SAMPLING,
                strength=0.8,
                seed=17,
            )
            second = build_robustness_view(
                source_image_dir=source,
                image_ids=("000001",),
                output_dir=root / "second",
                kind=InterventionKind.SAMPLING,
                strength=0.8,
                seed=17,
            )

            degraded = cv2.imdecode(
                np.frombuffer(
                    (first.image_dir / "000001.png").read_bytes(),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(degraded.shape, image.shape)
            self.assertFalse(np.array_equal(degraded, image))
            self.assertEqual(first.view_sha256, second.view_sha256)
            self.assertEqual(first.image_count, 1)

    def test_strength_zero_is_pixel_exact_clean_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            image = np.full((8, 12, 3), 73, dtype=np.uint8)
            write_png(source / "000001.png", image)

            view = build_robustness_view(
                source_image_dir=source,
                image_ids=("000001",),
                output_dir=root / "clean",
                kind=InterventionKind.VISIBILITY,
                strength=0.0,
                seed=17,
            )

            restored = cv2.imdecode(
                np.frombuffer(
                    (view.image_dir / "000001.png").read_bytes(),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            np.testing.assert_array_equal(restored, image)

    def test_refuses_nonempty_output_to_preserve_previous_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_png(
                source / "000001.png",
                np.zeros((8, 8, 3), dtype=np.uint8),
            )
            output = root / "occupied"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                build_robustness_view(
                    source_image_dir=source,
                    image_ids=("000001",),
                    output_dir=output,
                    kind=InterventionKind.SAMPLING,
                    strength=0.5,
                    seed=17,
                )


if __name__ == "__main__":
    unittest.main()
