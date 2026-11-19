"""Display-only mask polarity: retain the actual garment, never its backdrop."""
import unittest
import cv2
import numpy as np
from garment_background_preview import (
    _catalog_canvas, _presentation_foreground_mask, _render_presentation,
    garment_visual_preview,
)
from pathlib import Path
import hashlib


def garment():
    frame=np.full((480,420,3),255,np.uint8)
    mask=np.zeros(frame.shape[:2],np.uint8)
    polygon=np.array([(130,70),(170,50),(185,70),(235,70),(250,50),(290,70),
                      (365,125),(325,195),(285,170),(285,420),(135,420),(135,170),
                      (95,195),(55,125)],np.int32)
    cv2.fillPoly(mask,[polygon],1)
    frame[mask>0]=(150,55,120)
    frame[210:230,185:235]=(20,25,25)  # Real plate/label: not background.
    return frame,mask


class MaskPolarityTests(unittest.TestCase):
    def test_inverted_mask_retains_the_whole_garment(self):
        frame,expected=garment()
        actual=_presentation_foreground_mask(frame,1-expected)
        self.assertIsNotNone(actual)
        self.assertGreater(np.count_nonzero((actual>0)&(expected>0))/np.count_nonzero(expected),.99)
        self.assertLess(np.count_nonzero((actual>0)&(expected==0))/np.count_nonzero(expected),.01)

    def test_zero_one_and_zero_255_render_identically_even_when_resized(self):
        frame,mask=garment()
        for resized in (False,True):
            binary=cv2.resize(mask,(210,240),interpolation=cv2.INTER_NEAREST) if resized else mask
            normal=_render_presentation(frame,binary,None)
            encoded=_render_presentation(frame,binary*255,None)
            self.assertIsNotNone(normal)
            self.assertEqual(normal,encoded)

    def test_inverted_render_has_white_background_and_visible_cloth(self):
        frame,mask=garment();before=frame.copy()
        rendered=_render_presentation(frame,(1-mask)*255,np.full_like(frame,60))
        self.assertIsNotNone(rendered)
        image=cv2.imdecode(np.frombuffer(rendered,np.uint8),cv2.IMREAD_COLOR)
        self.assertTrue(np.all(image[0]==255));self.assertTrue(np.all(image[-1]==255))
        purple=np.all(np.abs(image.astype(int)-[150,55,120])<3,axis=2)
        self.assertGreater(np.count_nonzero(purple)/purple.size,.45)
        self.assertTrue(np.array_equal(frame,before))

    def test_small_interior_label_hole_preserves_original_plate(self):
        frame,mask=garment();mask[210:230,185:235]=0
        foreground=_presentation_foreground_mask(frame,mask)
        self.assertIsNotNone(foreground)
        self.assertTrue(np.all(foreground[210:230,185:235]==1))
        rendered=_render_presentation(frame,mask,None)
        image=cv2.imdecode(np.frombuffer(rendered,np.uint8),cv2.IMREAD_COLOR)
        self.assertGreater(np.count_nonzero(np.all(image==[20,25,25],axis=2)),500)

    def test_light_table_panel_and_large_missing_body_are_rejected(self):
        frame,garment_mask=garment()
        panel=np.zeros(frame.shape[:2],np.uint8);panel[25:455,25:395]=1
        self.assertIsNone(_render_presentation(frame,panel,None))
        incomplete=garment_mask.copy();incomplete[240:390,150:270]=0
        self.assertIsNone(_render_presentation(frame,incomplete,None))

    def test_degenerate_and_grabcut_class_masks_are_not_used_as_alpha(self):
        frame,mask=garment()
        for invalid in (np.zeros_like(mask),np.ones_like(mask),mask*3,np.full(mask.shape,np.nan)):
            self.assertIsNone(_render_presentation(frame,invalid,None))

    def test_neutral_backdrop_with_interior_garment_void_is_rejected(self):
        frame,_=garment()
        frame=cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        panel=np.zeros(frame.shape[:2],np.uint8);panel[25:455,25:395]=1
        panel[240:320,180:240]=0
        self.assertIsNone(_render_presentation(frame,panel,None))

    def test_catalog_canvas_normalizes_binary_alpha(self):
        frame,mask=garment();bbox=cv2.boundingRect(mask)
        self.assertEqual(_catalog_canvas(frame,mask,bbox),_catalog_canvas(frame,mask*255,bbox))

    def test_authorized_purple_original_remains_intact(self):
        source=Path(__file__).resolve().parents[1]/'diagnostics/catalog-presentation-20261020-172735/purple-2074-original.jpg'
        before=source.read_bytes();digest=hashlib.sha256(before).hexdigest()
        result=garment_visual_preview(before)
        self.assertIsNotNone(result)
        image=cv2.imdecode(np.frombuffer(result,np.uint8),cv2.IMREAD_COLOR)
        self.assertTrue(np.all(image[0]==255))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
