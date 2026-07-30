import unittest

import numpy as np

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)
from ifdr_yolo.data.interventions.targets import FactorTarget
from ifdr_yolo.data.interventions.transforms import apply_intervention


def checkerboard(size: int = 32) -> np.ndarray:
    grid = np.indices((size, size)).sum(axis=0) % 2
    image = np.repeat((grid * 255).astype(np.uint8)[..., None], 3, axis=2)
    return image


def spec(
    kind: InterventionKind,
    *,
    strength: float,
    seed: int = 17,
) -> InterventionSpec:
    return InterventionSpec(
        image_id="000123",
        kind=kind,
        role=InterventionRole.OBJECT,
        strength=strength,
        seed=seed,
        object_id=4,
        region_xyxy=(0.25, 0.25, 0.75, 0.75),
    )


class InterventionTransformTest(unittest.TestCase):
    def test_identity_is_exact_and_does_not_mutate_input(self) -> None:
        image = checkerboard()
        original = image.copy()
        identity = InterventionSpec(
            image_id="000123",
            kind=InterventionKind.IDENTITY,
            role=InterventionRole.GLOBAL,
            strength=0.0,
            seed=17,
        )
        target = FactorTarget(
            sampling=0.0,
            visibility=0.0,
            sampling_valid=True,
            visibility_valid=False,
        )

        result = apply_intervention(image, identity, target)

        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(result.image, original)
        self.assertIsNot(result.image, image)
        self.assertEqual(result.parameters["operator"], "identity")
        self.assertTrue(np.all(result.sampling_weight == 1.0))
        self.assertTrue(np.all(result.visibility_weight == 0.0))

    def test_sampling_is_deterministic_and_local(self) -> None:
        image = checkerboard()
        intervention = spec(
            InterventionKind.SAMPLING,
            strength=0.8,
        )
        target = FactorTarget(
            sampling=0.8,
            visibility=0.0,
            sampling_valid=True,
            visibility_valid=False,
        )

        first = apply_intervention(image, intervention, target)
        second = apply_intervention(image, intervention, target)

        np.testing.assert_array_equal(first.image, second.image)
        np.testing.assert_array_equal(
            first.sampling_weight,
            second.sampling_weight,
        )
        outside = np.ones(image.shape[:2], dtype=bool)
        outside[8:24, 8:24] = False
        np.testing.assert_array_equal(first.image[outside], image[outside])
        self.assertLess(
            float(first.image[10:22, 10:22].var()),
            float(image[10:22, 10:22].var()),
        )
        self.assertEqual(first.parameters["operator"], "sampling")
        self.assertLess(first.parameters["downsample_scale"], 1.0)

    def test_visibility_uses_soft_mask_and_local_seed(self) -> None:
        image = checkerboard()
        target = FactorTarget(
            sampling=0.0,
            visibility=0.6,
            sampling_valid=False,
            visibility_valid=True,
        )

        first = apply_intervention(
            image,
            spec(InterventionKind.VISIBILITY, strength=0.6, seed=17),
            target,
        )
        repeated = apply_intervention(
            image,
            spec(InterventionKind.VISIBILITY, strength=0.6, seed=17),
            target,
        )
        other_seed = apply_intervention(
            image,
            spec(InterventionKind.VISIBILITY, strength=0.6, seed=18),
            target,
        )

        np.testing.assert_array_equal(first.image, repeated.image)
        self.assertFalse(np.array_equal(first.image, other_seed.image))
        weights = first.visibility_weight
        self.assertEqual(float(weights[:8].max()), 0.0)
        self.assertEqual(float(weights[:, :8].max()), 0.0)
        self.assertEqual(float(weights[24:].max()), 0.0)
        self.assertEqual(float(weights[:, 24:].max()), 0.0)
        self.assertGreater(float(weights.max()), 0.99)
        self.assertTrue(np.any((weights > 0.0) & (weights < 1.0)))
        self.assertEqual(first.parameters["operator"], "visibility")

    def test_factor_maps_respect_validity_masks(self) -> None:
        image = checkerboard()
        target = FactorTarget(
            sampling=0.7,
            visibility=0.0,
            sampling_valid=True,
            visibility_valid=False,
        )

        result = apply_intervention(
            image,
            spec(InterventionKind.SAMPLING, strength=0.7),
            target,
        )

        support = result.sampling_weight > 0
        self.assertTrue(np.all(result.sampling_target[support] == 0.7))
        self.assertTrue(np.all(result.sampling_target[~support] == 0.0))
        self.assertTrue(np.all(result.visibility_target == 0.0))
        self.assertTrue(np.all(result.visibility_weight == 0.0))

    def test_rejects_non_uint8_three_channel_image(self) -> None:
        invalid_images = (
            np.zeros((32, 32), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.float32),
        )
        target = FactorTarget(
            sampling=0.0,
            visibility=0.0,
            sampling_valid=True,
            visibility_valid=True,
        )
        for image in invalid_images:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaisesRegex(ValueError, "uint8 HWC"):
                    apply_intervention(
                        image,
                        spec(InterventionKind.IDENTITY, strength=0.0),
                        target,
                    )


if __name__ == "__main__":
    unittest.main()
