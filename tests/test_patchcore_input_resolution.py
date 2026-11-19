"""Verifies that experimental resolution matches training and v1 remains untouched."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import patchcore_inference as module


class PatchCoreInputResolutionTests(unittest.TestCase):
    def make_inspector(self, size):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.ckpt"
            checkpoint.write_bytes(b"only-for-constructor-test")
            mock_patchcore = Mock()
            mock_patchcore.configure_pre_processor.return_value = "preprocessor-explicit"
            with (
                patch.object(module, "Patchcore", mock_patchcore),
                patch.object(module, "Engine", Mock()),
            ):
                inspector = module.PatchCoreInspector(
                    str(checkpoint), image_size=size
                )
            return inspector, mock_patchcore

    def test_v1_256_preserves_historical_preprocessor(self):
        inspector, patchcore = self.make_inspector(256)
        self.assertEqual(inspector.image_size, (256, 256))
        patchcore.configure_pre_processor.assert_not_called()
        self.assertNotIn("pre_processor", patchcore.call_args.kwargs)

    def test_384_uses_explicit_training_resolution(self):
        inspector, patchcore = self.make_inspector(384)
        self.assertEqual(inspector.image_size, (384, 384))
        patchcore.configure_pre_processor.assert_called_once_with(
            image_size=(384, 384)
        )
        self.assertEqual(
            patchcore.call_args.kwargs["pre_processor"],
            "preprocessor-explicit",
        )

    def test_512_uses_explicit_training_resolution(self):
        inspector, patchcore = self.make_inspector(512)
        self.assertEqual(inspector.image_size, (512, 512))
        patchcore.configure_pre_processor.assert_called_once_with(
            image_size=(512, 512)
        )
        self.assertEqual(
            patchcore.call_args.kwargs["pre_processor"],
            "preprocessor-explicit",
        )

    def test_nonpositive_resolution_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ckpt=Path(temp)/"checkpoint.ckpt"
            ckpt.write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "mayor que cero"):
                module.PatchCoreInspector(str(ckpt),image_size=0)


if __name__ == "__main__":
    unittest.main()
