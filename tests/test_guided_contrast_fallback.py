"""Guided camera may verify a light garment when adaptive edges fail.

No changes to PatchCore inspection, persisted datasets, or min-coverage config.
"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class GuidedContrastFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module=importlib.import_module("app")

    def sample(self):
        image=np.full((520,820,3),234,dtype=np.uint8)
        image[:,:38]=31
        image[:,-38:]=31
        outline=np.array([[290,105],[340,144],[410,158],[480,142],[532,101],
                          [592,145],[622,205],[579,231],[562,206],[560,410],
                          [320,410],[317,212],[285,240],[253,201]],np.int32)
        cv2.fillPoly(image,[outline],(105,115,135))
        return image

    def test_verified_secondary_contour_is_used(self):
        m=self.module
        f=self.sample()
        roi=(110,50,710,480)
        mask=np.zeros(f.shape[:2],np.uint8)
        cv2.rectangle(mask,(240,110),(600,400),255,-1)
        with (patch.object(m,"create_guided_garment_mask",
                           side_effect=lambda frame,diag:diag.update({"method":"ninguno"}) or np.zeros(f.shape[:2],np.uint8)),
              patch.object(m,"garment_visual_preview",return_value=mask) as secondary):
            result,bbox,cov,diagnostics=m._ai_guided_segment(f,roi)
        secondary.assert_called_once()
        self.assertEqual(diagnostics["method"],"contraste_verificado")
        self.assertEqual(bbox,(240,110,601,401))
        self.assertGreater(cov,0.2)
        self.assertLess(cov,0.75)
        self.assertTrue(diagnostics["contrast_candidate"]["inside_roi"])

    def test_does_not_promote_mask_cut_at_roi_border(self):
        m=self.module
        f=self.sample()
        roi=(110,50,710,480)
        mask=np.zeros(f.shape[:2],np.uint8)
        cv2.rectangle(mask,(105,100),(600,420),255,-1)
        with (patch.object(m,"create_guided_garment_mask",
                           return_value=np.zeros(f.shape[:2],np.uint8)),
              patch.object(m,"garment_visual_preview",return_value=mask)):
            _,bbox,cov,diag=m._ai_guided_segment(f,roi)
        self.assertIsNone(bbox)
        self.assertEqual(cov,0.)
        self.assertFalse(diag["contrast_candidate"]["inside_roi"])

    def test_blank_scene_remains_unverified(self):
        m=self.module
        frame=np.full((520,820,3),234,dtype=np.uint8)
        mask=m.garment_visual_preview(frame,mask_only=True)
        self.assertIsNone(mask)

    def test_coverage_threshold_is_not_lowered(self):
        m=self.module
        frame=self.sample()
        roi=(110,50,710,480)
        tiny=np.zeros(frame.shape[:2],np.uint8)
        cv2.rectangle(tiny,(300,170),(400,250),255,-1)
        with (patch.object(m,"create_guided_garment_mask",
                           return_value=np.zeros(frame.shape[:2],np.uint8)),
              patch.object(m,"garment_visual_preview",return_value=tiny)):
            _,bbox,cov,diag=m._ai_guided_segment(frame,roi)
        self.assertIsNone(bbox)
        self.assertEqual(cov,0.)
        self.assertFalse("method" in diag and diag["method"] == "contraste_verificado")

    def test_markup_does_not_cramp_action_buttons(self):
        doc=(ROOT/"templates/guided_capture.html").read_text(encoding="utf-8")
        self.assertIn("@container guided-panel (max-width: 360px)",doc)
        self.assertIn('[data-review="discard"]',doc)
        self.assertIn('data-review="accept"',doc)
        self.assertIn('data-review="repeat"',doc)

if __name__=="__main__":
    unittest.main()
