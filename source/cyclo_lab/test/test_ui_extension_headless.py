"""CPU-only regression for loading the Cyclo Lab extension without omni.ui."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cyclo_lab/ui_extension_example.py"


class TestUiExtensionHeadless(unittest.TestCase):
    def test_startup_without_omni_ui_skips_example_window(self):
        saved = {
            name: sys.modules.get(name)
            for name in ("omni", "omni.ext", "omni.ui")
        }
        omni = types.ModuleType("omni")
        omni_ext = types.ModuleType("omni.ext")
        omni_ext.IExt = object
        omni.ext = omni_ext
        sys.modules["omni"] = omni
        sys.modules["omni.ext"] = omni_ext
        sys.modules.pop("omni.ui", None)
        try:
            spec = importlib.util.spec_from_file_location(
                "cyclo_lab_ui_extension_headless_test", MODULE_PATH
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            extension = module.ExampleExtension()
            extension.on_startup("cyclo_lab-test")
            self.assertEqual(extension._count, 0)
            self.assertIsNone(extension._window)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
