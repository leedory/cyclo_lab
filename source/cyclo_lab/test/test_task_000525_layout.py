"""Pure-Python geometry contracts for the approved Task000525 B layout."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


LAYOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
    / "layout.py"
)


def _load_layout_module():
    module_name = "task000525_layout_contract"
    spec = spec_from_file_location(module_name, LAYOUT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Task000525LayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = _load_layout_module()

    def test_requested_can_names_and_four_regions(self):
        self.assertEqual(
            self.layout.TASK000525_CAN_NAMES,
            (
                "coffee_can_black",
                "coffee_can_brown",
                "coffee_can_green",
                "coffee_can_orange",
            ),
        )
        for key in ("A", "B", "C"):
            self.assertEqual(len(self.layout.candidate_sampling_regions(key)), 4)

    def test_all_candidate_regions_stay_inside_measured_shelf(self):
        shelf = self.layout.TASK000525_SHELF_BOUNDS
        for key in ("A", "B", "C"):
            regions = self.layout.candidate_sampling_regions(key)
            for region in regions:
                self.assertGreaterEqual(region.x_min_back_m, shelf.x_min_back_m)
                self.assertLessEqual(region.x_max_front_m, shelf.x_max_front_m)
                self.assertGreaterEqual(region.y_min_m, shelf.y_min_m)
                self.assertLessEqual(region.y_max_m, shelf.y_max_m)
            for left, right in zip(regions, regions[1:]):
                self.assertGreater(right.y_min_m, left.y_max_m)

    def test_user_approved_forward_candidate_numbers(self):
        selected = self.layout.TASK000525_LAYOUT_CANDIDATES["B"]
        regions = self.layout.build_sampling_regions(selected)
        self.assertAlmostEqual(
            regions[0].depth_x_m * 1000.0, 150.0, delta=0.001
        )
        self.assertAlmostEqual(regions[0].width_y_m * 1000.0, 62.8423, places=3)
        clearances = self.layout.guaranteed_clearances_mm(selected)
        self.assertAlmostEqual(clearances["between_can_surfaces"], 33.0)
        self.assertAlmostEqual(clearances["can_to_low_y_edge"], 86.5)
        self.assertAlmostEqual(clearances["can_to_high_y_edge"], 86.5)
        self.assertAlmostEqual(clearances["can_to_back_edge"], 86.5)
        self.assertAlmostEqual(clearances["can_to_front_edge"], -3.5)

    def test_b_is_the_approved_selection(self):
        self.assertEqual(self.layout.TASK000525_SELECTED_LAYOUT_KEY, "B")
        self.assertEqual(
            self.layout.selected_sampling_regions(),
            self.layout.candidate_sampling_regions("B"),
        )


if __name__ == "__main__":
    unittest.main()
