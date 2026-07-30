import unittest

from ifdr_yolo.eval.robustness import (
    RobustnessCondition,
    summarize_robustness,
)


def metrics(ap40: float) -> dict[str, object]:
    return {
        "classes": {
            "Pedestrian": {
                "moderate": {"ap40": ap40},
            },
        },
    }


class RobustnessMetricsTest(unittest.TestCase):
    def test_computes_normalized_auc_and_maximum_strength_drop(self) -> None:
        conditions = (
            RobustnessCondition(
                kind="sampling",
                strength=0.0,
                seed=17,
                metrics=metrics(60.0),
            ),
            RobustnessCondition(
                kind="sampling",
                strength=0.5,
                seed=17,
                metrics=metrics(45.0),
            ),
            RobustnessCondition(
                kind="sampling",
                strength=1.0,
                seed=17,
                metrics=metrics(30.0),
            ),
        )

        result = summarize_robustness(conditions)

        curve = result["curves"]["sampling"]["Pedestrian"]["moderate"]
        self.assertEqual(curve["clean_ap40"], 60.0)
        self.assertEqual(curve["ap40_at_max_strength"], 30.0)
        self.assertEqual(curve["absolute_drop"], 30.0)
        self.assertAlmostEqual(curve["relative_retention"], 0.5)
        self.assertAlmostEqual(curve["normalized_auc"], 0.75)

    def test_averages_repeated_seeds_before_integrating_curve(self) -> None:
        conditions = (
            RobustnessCondition("visibility", 0.0, 17, metrics(40.0)),
            RobustnessCondition("visibility", 0.0, 23, metrics(60.0)),
            RobustnessCondition("visibility", 1.0, 17, metrics(20.0)),
            RobustnessCondition("visibility", 1.0, 23, metrics(30.0)),
        )

        result = summarize_robustness(conditions)

        curve = result["curves"]["visibility"]["Pedestrian"]["moderate"]
        self.assertEqual(curve["mean_ap40"], [50.0, 25.0])
        self.assertAlmostEqual(curve["normalized_auc"], 0.75)
        self.assertEqual(curve["seed_count"], 2)

    def test_rejects_curve_without_clean_reference(self) -> None:
        conditions = (
            RobustnessCondition("sampling", 0.5, 17, metrics(20.0)),
            RobustnessCondition("sampling", 1.0, 17, metrics(10.0)),
        )

        with self.assertRaisesRegex(ValueError, "strength 0"):
            summarize_robustness(conditions)

    def test_rejects_inconsistent_seed_grid(self) -> None:
        conditions = (
            RobustnessCondition("sampling", 0.0, 17, metrics(40.0)),
            RobustnessCondition("sampling", 1.0, 17, metrics(20.0)),
            RobustnessCondition("sampling", 0.0, 23, metrics(40.0)),
        )

        with self.assertRaisesRegex(ValueError, "same strength grid"):
            summarize_robustness(conditions)


if __name__ == "__main__":
    unittest.main()
