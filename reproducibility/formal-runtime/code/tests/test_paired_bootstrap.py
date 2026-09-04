import unittest

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.paired_bootstrap import paired_bootstrap_ap40


def ground_truth(box: BoundingBox) -> KittiObject:
    return KittiObject(
        kind="Pedestrian",
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 0.0),
        rotation_y=0.0,
    )


class PairedBootstrapAP40Test(unittest.TestCase):
    def test_is_deterministic_and_preserves_image_pairing(self) -> None:
        first = BoundingBox(0, 0, 20, 80)
        second = BoundingBox(100, 0, 120, 80)
        ground_truth_by_image = {
            "000001": (ground_truth(first),),
            "000002": (ground_truth(second),),
        }
        reference = {
            "000001": (Detection("000001", "Pedestrian", 0.9, first),),
            "000002": (),
        }
        candidate = {
            "000001": (Detection("000001", "Pedestrian", 0.9, first),),
            "000002": (Detection("000002", "Pedestrian", 0.8, second),),
        }

        first_result = paired_bootstrap_ap40(
            gt_by_image=ground_truth_by_image,
            reference_by_image=reference,
            candidate_by_image=candidate,
            class_name="Pedestrian",
            difficulty=Difficulty.HARD,
            iterations=200,
            seed=17,
        )
        second_result = paired_bootstrap_ap40(
            gt_by_image=ground_truth_by_image,
            reference_by_image=reference,
            candidate_by_image=candidate,
            class_name="Pedestrian",
            difficulty=Difficulty.HARD,
            iterations=200,
            seed=17,
        )

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.reference_ap40, 50.0)
        self.assertEqual(first_result.candidate_ap40, 100.0)
        self.assertEqual(first_result.difference_ap40, 50.0)
        self.assertLessEqual(first_result.ci_lower, 50.0)
        self.assertGreaterEqual(first_result.ci_upper, 50.0)
        self.assertGreater(first_result.probability_improvement, 0.0)

    def test_rejects_unpaired_image_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "same image IDs"):
            paired_bootstrap_ap40(
                gt_by_image={"000001": ()},
                reference_by_image={"000001": ()},
                candidate_by_image={"000002": ()},
                class_name="Pedestrian",
                difficulty=Difficulty.HARD,
                iterations=10,
                seed=17,
            )


if __name__ == "__main__":
    unittest.main()
