from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
V163 = HERE.parent / "stage11-v163-gitless-provenance-successor" / "stage11-v163-gitless-provenance-envelope.tar"
V163_SHA256 = "d24d7841f33d6992b225336e904382ee1e0d14352f41f59631ae6c8f60daf278"

from build_stage11_v164_runtime_provenance_adapter import build_package, verify_package


class RuntimeProvenanceAdapterTest(unittest.TestCase):
    def test_runtime_records_verified_gitless_identity_instead_of_unknown_git(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE.parent / ".tmp") as directory:
            output = Path(directory) / "v164.tar"
            manifest = build_package(HERE, V163, output)
            self.assertEqual(verify_package(output), manifest)
            extraction = Path(directory) / "runtime"
            extraction.mkdir()
            with tarfile.open(output, "r:") as archive:
                archive.extractall(extraction, filter="data")
            provenance = json.loads((extraction / "code" / ".stage11-provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["source_provenance_package_sha256"], V163_SHA256)
            module_path = extraction / "code" / "scripts" / "run_p2_interaction_s0.py"
            spec = importlib.util.spec_from_file_location("v164_stage11_dcli", module_path)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            previous_path = list(sys.path)
            sys.path.insert(0, str(extraction / "code"))
            try:
                self.assertIsNotNone(spec.loader)
                spec.loader.exec_module(module)
                self.assertEqual(module._git_commit(extraction / "code"), provenance["runtime_identity"])
            finally:
                sys.path[:] = previous_path


if __name__ == "__main__":
    unittest.main()
