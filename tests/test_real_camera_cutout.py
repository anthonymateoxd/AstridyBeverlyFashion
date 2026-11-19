"""Real camera cutouts retain photographic pixels, resolution and evidence."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import unittest
import cv2
import numpy as np
import garment_background_preview as g

FIXTURES=Path(__file__).resolve().parents[1]/'diagnostics/catalog-presentation-20261020-172735'

class RealCameraCutoutTests(unittest.TestCase):
    def test_real_camera_silhouette_preserves_cloth_and_excludes_station(self):
        raw=(FIXTURES/'purple-2076-original.jpg').read_bytes()
        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        with patch.object(g,'_catalog_canvas',wraps=g._catalog_canvas) as render:
            encoded=g.garment_visual_preview(raw)
        self.assertIsNotNone(encoded)
        source,mask,bbox=render.call_args.args[:3]
        # Actual photo coordinates: both sleeves, collar, plate and lower hem.
        for x,y in ((620,110),(580,890),(465,510),(625,500),(1300,480)):
            self.assertEqual(mask[y,x],1,(x,y))
        for x,y in ((300,500),(1400,500),(900,120),(900,900),(620,45)):
            self.assertEqual(mask[y,x],0,(x,y))
        x,y,w,h=bbox; margin=max(8,round(max(w,h)*.04))
        image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2],(h+2*margin,w+2*margin))
        content=image[margin:margin+h,margin:margin+w]
        foreground=mask[y:y+h,x:x+w]>0
        self.assertTrue(np.array_equal(content[foreground],frame[y:y+h,x:x+w][foreground]))
        self.assertTrue(np.all(content[~foreground]==255))
        self.assertTrue(np.all(image[:margin]==255))
        self.assertGreater(w/image.shape[1],.9)
        self.assertGreater(h/image.shape[0],.9)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),hashlib.sha256((FIXTURES/'purple-2076-original.jpg').read_bytes()).hexdigest())

    def test_renderer_does_not_recolour_edges_or_shrink_native_pixels(self):
        rng=np.random.default_rng(99)
        frame=rng.integers(0,230,(1200,1500,3),dtype=np.uint8)
        mask=np.zeros(frame.shape[:2],np.uint8);mask[100:1100,100:1400]=1
        encoded=g._catalog_canvas(frame,mask,(100,100,1300,1000),np.full_like(frame,180))
        image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        margin=52
        self.assertTrue(np.array_equal(image[margin:margin+1000,margin:margin+1300],frame[100:1100,100:1400]))

    def test_ambiguous_blank_photo_is_not_fabricated(self):
        ok,encoded=cv2.imencode('.jpg',np.full((480,640,3),245,np.uint8))
        self.assertTrue(ok)
        self.assertIsNone(g.garment_visual_preview(encoded.tobytes()))
