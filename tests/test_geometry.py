import math
import unittest

from waytous_curation.domain import Box3D, Sample
from waytous_curation.geometry import axis_aligned_iou_3d, extract_features, point_inside_box


class GeometryTests(unittest.TestCase):
    def test_rotated_point_membership(self):
        box = Box3D(center=(0, 0, 0), size=(4, 2, 2), yaw=math.pi / 2)
        self.assertTrue(point_inside_box((0, 1.5, 0), box))
        self.assertFalse(point_inside_box((1.5, 0, 0), box))

    def test_iou(self):
        first = Box3D(center=(0, 0, 0), size=(2, 2, 2))
        second = Box3D(center=(1, 0, 0), size=(2, 2, 2))
        self.assertAlmostEqual(axis_aligned_iou_3d(first, second), 1 / 3)

    def test_features_count_only_points_inside(self):
        sample = Sample(
            sample_id="one",
            class_name="car",
            bbox=Box3D(center=(0, 0, 1), size=(2, 2, 2)),
            points=[(0, 0, 1), (5, 5, 5)],
            ground_z=0,
        )
        features = extract_features(sample)
        self.assertEqual(features.point_count, 1)
        self.assertEqual(features.point_density, 0.125)
        self.assertEqual(features.ground_offset, 0)


if __name__ == "__main__":
    unittest.main()
