import random
import unittest

from ifdr_yolo.data.interventions.sampler import (
    DeterministicInterventionSampler,
    SamplingPolicy,
)
from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
)


class InterventionSamplerTest(unittest.TestCase):
    def test_matched_pair_is_reproducible_across_sampler_instances(self) -> None:
        first = DeterministicInterventionSampler(base_seed=17)
        second = DeterministicInterventionSampler(base_seed=17)
        arguments = {
            "image_id": "000123",
            "object_id": 4,
            "epoch": 9,
            "slot": 2,
            "object_region": (0.1, 0.2, 0.3, 0.6),
            "background_region": (0.6, 0.2, 0.8, 0.6),
        }

        first_pair = first.sample_matched_pair(**arguments)
        second_pair = second.sample_matched_pair(**arguments)

        self.assertEqual(first_pair, second_pair)

    def test_matched_pair_shares_factor_but_not_local_seed(self) -> None:
        sampler = DeterministicInterventionSampler(base_seed=17)

        object_spec, background_spec = sampler.sample_matched_pair(
            image_id="000123",
            object_id=4,
            epoch=9,
            slot=2,
            object_region=(0.1, 0.2, 0.3, 0.6),
            background_region=(0.6, 0.2, 0.8, 0.6),
        )

        self.assertEqual(object_spec.kind, background_spec.kind)
        self.assertEqual(object_spec.strength, background_spec.strength)
        self.assertNotEqual(object_spec.seed, background_spec.seed)
        self.assertEqual(object_spec.role, InterventionRole.OBJECT)
        self.assertEqual(background_spec.role, InterventionRole.BACKGROUND)
        self.assertEqual(object_spec.object_id, 4)
        self.assertIsNone(background_spec.object_id)

    def test_slot_changes_stable_sample_identity(self) -> None:
        sampler = DeterministicInterventionSampler(base_seed=17)
        common = {
            "image_id": "000123",
            "object_id": 4,
            "epoch": 9,
            "object_region": (0.1, 0.2, 0.3, 0.6),
            "background_region": (0.6, 0.2, 0.8, 0.6),
        }

        first = sampler.sample_matched_pair(slot=1, **common)
        second = sampler.sample_matched_pair(slot=2, **common)

        self.assertNotEqual(first[0].seed, second[0].seed)

    def test_sampler_does_not_consume_global_random_state(self) -> None:
        random.seed(12345)
        expected = random.random()
        random.seed(12345)

        DeterministicInterventionSampler(base_seed=17).sample_matched_pair(
            image_id="000123",
            object_id=4,
            epoch=9,
            slot=2,
            object_region=(0.1, 0.2, 0.3, 0.6),
            background_region=(0.6, 0.2, 0.8, 0.6),
        )

        self.assertEqual(random.random(), expected)

    def test_non_identity_strength_respects_policy_bounds(self) -> None:
        policy = SamplingPolicy(
            identity_probability=0.0,
            sampling_probability=1.0,
            visibility_probability=0.0,
            minimum_strength=0.25,
            maximum_strength=0.5,
        )
        sampler = DeterministicInterventionSampler(
            base_seed=17,
            policy=policy,
        )

        object_spec, _ = sampler.sample_matched_pair(
            image_id="000123",
            object_id=4,
            epoch=9,
            slot=2,
            object_region=(0.1, 0.2, 0.3, 0.6),
            background_region=(0.6, 0.2, 0.8, 0.6),
        )

        self.assertEqual(object_spec.kind, InterventionKind.SAMPLING)
        self.assertGreaterEqual(object_spec.strength, 0.25)
        self.assertLessEqual(object_spec.strength, 0.5)

    def test_policy_rejects_probability_sum_other_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "probabilities"):
            SamplingPolicy(
                identity_probability=0.2,
                sampling_probability=0.2,
                visibility_probability=0.2,
            )


if __name__ == "__main__":
    unittest.main()
