import unittest

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)
from ifdr_yolo.data.interventions.targets import (
    FactorTarget,
    factor_target_for_spec,
)


def object_spec(
    kind: InterventionKind,
    strength: float,
) -> InterventionSpec:
    return InterventionSpec(
        image_id="000123",
        kind=kind,
        role=InterventionRole.OBJECT,
        strength=strength,
        seed=17,
        object_id=4,
        region_xyxy=(0.1, 0.2, 0.3, 0.6),
    )


class InterventionTargetTest(unittest.TestCase):
    def test_visibility_combines_natural_and_synthetic_occlusion(self) -> None:
        target = factor_target_for_spec(
            object_spec(InterventionKind.VISIBILITY, 0.5),
            natural_sampling=0.2,
            natural_occlusion=0.25,
        )

        self.assertEqual(
            target,
            FactorTarget(
                sampling=0.2,
                visibility=0.625,
                sampling_valid=True,
                visibility_valid=True,
            ),
        )

    def test_sampling_combines_natural_proxy_and_masks_unknown_visibility(
        self,
    ) -> None:
        target = factor_target_for_spec(
            object_spec(InterventionKind.SAMPLING, 0.6),
            natural_sampling=0.2,
        )

        self.assertAlmostEqual(target.sampling, 0.68)
        self.assertTrue(target.sampling_valid)
        self.assertEqual(target.visibility, 0.0)
        self.assertFalse(target.visibility_valid)

    def test_identity_object_masks_unknown_natural_sampling(self) -> None:
        target = factor_target_for_spec(
            object_spec(InterventionKind.IDENTITY, 0.0),
            natural_occlusion=0.75,
        )

        self.assertEqual(target.sampling, 0.0)
        self.assertEqual(target.visibility, 0.75)
        self.assertFalse(target.sampling_valid)
        self.assertTrue(target.visibility_valid)

    def test_background_sampling_is_known_not_to_add_occlusion(self) -> None:
        spec = InterventionSpec(
            image_id="000123",
            kind=InterventionKind.SAMPLING,
            role=InterventionRole.BACKGROUND,
            strength=0.4,
            seed=17,
            region_xyxy=(0.6, 0.2, 0.8, 0.6),
        )

        target = factor_target_for_spec(spec)

        self.assertEqual(target.sampling, 0.4)
        self.assertEqual(target.visibility, 0.0)
        self.assertTrue(target.sampling_valid)
        self.assertTrue(target.visibility_valid)

    def test_rejects_invalid_natural_occlusion(self) -> None:
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "natural_occlusion",
                ):
                    factor_target_for_spec(
                        object_spec(InterventionKind.IDENTITY, 0.0),
                        natural_occlusion=value,
                    )

    def test_rejects_invalid_natural_sampling(self) -> None:
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "natural_sampling",
                ):
                    factor_target_for_spec(
                        object_spec(InterventionKind.IDENTITY, 0.0),
                        natural_sampling=value,
                    )


if __name__ == "__main__":
    unittest.main()
