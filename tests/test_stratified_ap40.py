import unittest

from ifdr_yolo.data.kitti_types import BoundingBox, Detection, KittiObject
from ifdr_yolo.eval.stratified_ap40 import (
    KITTI_RESEARCH_SLICES,
    evaluate_target_slices,
)


def ground_truth(
    *,
    height: float,
    occluded: int = 0,
    truncated: float = 0.0,
    depth: float = 10.0,
) -> KittiObject:
    return KittiObject(
        kind="Pedestrian",
        truncated=truncated,
        occluded=occluded,
        alpha=0.0,
        bbox=BoundingBox(0.0, 0.0, 20.0, height),
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, depth),
        rotation_y=0.0,
    )


class TargetSliceTest(unittest.TestCase):
    def test_height_slices_follow_kitti_hard_and_easy_boundaries(self) -> None:
        height_slices = [
            target_slice
            for target_slice in KITTI_RESEARCH_SLICES
            if target_slice.axis == "height"
        ]

        expected = {
            25.0: (),
            25.01: ("small_25_40",),
            40.0: ("small_25_40",),
            40.01: ("medium_40_80",),
            80.0: ("medium_40_80",),
            80.01: ("large_gt_80",),
        }
        for height, names in expected.items():
            matched = tuple(
                target_slice.name
                for target_slice in height_slices
                if target_slice.matches(ground_truth(height=height))
            )
            self.assertEqual(matched, names)

    def test_occlusion_and_truncation_slices_do_not_overlap(self) -> None:
        obj = ground_truth(
            height=60.0,
            occluded=1,
            truncated=0.3,
            depth=40.0,
        )

        matched = {
            target_slice.axis: target_slice.name
            for target_slice in KITTI_RESEARCH_SLICES
            if target_slice.matches(obj)
        }

        self.assertEqual(matched["height"], "medium_40_80")
        self.assertEqual(matched["depth"], "mid_20_40m")
        self.assertEqual(matched["occlusion"], "occlusion_1")
        self.assertEqual(matched["truncation"], "truncation_015_030")

    def test_depth_slices_use_camera_z_distance(self) -> None:
        depth_slices = [
            target_slice
            for target_slice in KITTI_RESEARCH_SLICES
            if target_slice.axis == "depth"
        ]

        for depth, expected in (
            (0.0, ()),
            (0.01, ("near_0_20m",)),
            (20.0, ("near_0_20m",)),
            (20.01, ("mid_20_40m",)),
            (40.0, ("mid_20_40m",)),
            (40.01, ("far_gt_40m",)),
        ):
            matched = tuple(
                target_slice.name
                for target_slice in depth_slices
                if target_slice.matches(
                    ground_truth(height=60.0, depth=depth)
                )
            )
            self.assertEqual(matched, expected)

    def test_evaluation_reports_conditional_ap40_and_target_count(self) -> None:
        obj = ground_truth(height=30.0)
        box = obj.bbox
        payload = evaluate_target_slices(
            gt_by_image={"000001": (obj,)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Pedestrian", 0.9, box),
                )
            },
        )

        small = payload["slices"]["height"]["small_25_40"]["classes"]
        medium = payload["slices"]["height"]["medium_40_80"]["classes"]
        self.assertEqual(small["Pedestrian"]["num_valid_gt"], 1)
        self.assertEqual(small["Pedestrian"]["ap40"], 100.0)
        self.assertNotIn("precision", small["Pedestrian"])
        self.assertNotIn("recall", small["Pedestrian"])
        self.assertNotIn("scores", small["Pedestrian"])
        self.assertEqual(medium["Pedestrian"]["num_valid_gt"], 0)


if __name__ == "__main__":
    unittest.main()
