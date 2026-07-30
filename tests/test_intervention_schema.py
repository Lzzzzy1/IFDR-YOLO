import math
import unittest

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)


class InterventionSchemaTest(unittest.TestCase):
    def test_payload_round_trip_preserves_reproducibility_fields(self) -> None:
        spec = InterventionSpec(
            image_id="000123",
            kind=InterventionKind.SAMPLING,
            role=InterventionRole.OBJECT,
            strength=0.625,
            seed=1700123,
            object_id=4,
            region_xyxy=(0.1, 0.2, 0.3, 0.6),
        )

        restored = InterventionSpec.from_payload(spec.to_payload())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.schema_version, 1)
        self.assertEqual(
            restored.to_payload()["kind"],
            "sampling",
        )

    def test_rejects_strength_outside_closed_unit_interval(self) -> None:
        for strength in (-0.01, 1.01, math.nan, math.inf):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "strength"):
                    InterventionSpec(
                        image_id="000123",
                        kind=InterventionKind.SAMPLING,
                        role=InterventionRole.GLOBAL,
                        strength=strength,
                        seed=17,
                    )

    def test_identity_requires_zero_strength(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity strength"):
            InterventionSpec(
                image_id="000123",
                kind=InterventionKind.IDENTITY,
                role=InterventionRole.GLOBAL,
                strength=0.1,
                seed=17,
            )

    def test_object_role_requires_object_and_normalized_region(self) -> None:
        invalid_cases = (
            {
                "object_id": None,
                "region_xyxy": (0.1, 0.2, 0.3, 0.6),
            },
            {
                "object_id": 4,
                "region_xyxy": None,
            },
            {
                "object_id": 4,
                "region_xyxy": (0.3, 0.2, 0.1, 0.6),
            },
            {
                "object_id": 4,
                "region_xyxy": (-0.1, 0.2, 0.3, 0.6),
            },
        )
        for fields in invalid_cases:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    InterventionSpec(
                        image_id="000123",
                        kind=InterventionKind.VISIBILITY,
                        role=InterventionRole.OBJECT,
                        strength=0.5,
                        seed=17,
                        **fields,
                    )

    def test_background_role_forbids_object_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "background"):
            InterventionSpec(
                image_id="000123",
                kind=InterventionKind.VISIBILITY,
                role=InterventionRole.BACKGROUND,
                strength=0.5,
                seed=17,
                object_id=4,
                region_xyxy=(0.1, 0.2, 0.3, 0.6),
            )

    def test_from_payload_rejects_unknown_fields(self) -> None:
        payload = {
            "schema_version": 1,
            "image_id": "000123",
            "kind": "identity",
            "role": "global",
            "strength": 0.0,
            "seed": 17,
            "object_id": None,
            "region_xyxy": None,
            "hidden_default": True,
        }

        with self.assertRaisesRegex(ValueError, "unknown intervention fields"):
            InterventionSpec.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
