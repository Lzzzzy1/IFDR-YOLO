from pathlib import Path
import unittest


class RepositorySafetyTest(unittest.TestCase):
    def test_generated_data_is_git_ignored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ignore = (root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/processed/", ignore)

    def test_raw_dataset_is_not_under_generated_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = (root / "kitti_raw").resolve()
        generated = (root / "data" / "processed").resolve()
        self.assertNotEqual(raw, generated)
        self.assertNotIn(raw, generated.parents)


if __name__ == "__main__":
    unittest.main()
