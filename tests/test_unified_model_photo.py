"""La imagen principal coincide entre catalogo, ficha y editor."""
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

class MainPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=importlib.import_module("app")
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.original=self.a.app.static_folder
        self.a.app.static_folder=self.tmp.name
        self.ref=Path(self.tmp.name)/"results/model_references/2069/photo.jpg"
        self.cover=Path(self.tmp.name)/"results/model_covers/2069.jpg"
        self.ref.parent.mkdir(parents=True,exist_ok=True)
        self.cover.parent.mkdir(parents=True,exist_ok=True)
        self.relref="results/model_references/2069/photo.jpg"
    def tearDown(self):
        self.a.app.static_folder=self.original
        self.tmp.cleanup()
    def visual(self,path):
        with self.a.app.test_request_context():
            return self.a.garment_cover_url(2069,path)
    def test_creation_reference_photo_is_catalog_cover(self):
        self.ref.write_bytes(b"image-reference")
        url=self.visual(self.relref)
        self.assertIn(self.relref,url)
        self.assertIn("?v=",url)
    def test_editing_cover_supersedes_older_reference(self):
        self.ref.write_bytes(b"reference")
        self.cover.write_bytes(b"new-edit")
        os.utime(self.ref,ns=(100000000000,100000000000))
        os.utime(self.cover,ns=(200000000000,200000000000))
        self.assertIn("results/model_covers/2069.jpg",self.visual(self.relref))
    def test_replacing_reference_supersedes_older_cover(self):
        self.cover.write_bytes(b"prior-cover")
        self.ref.write_bytes(b"new-reference")
        os.utime(self.cover,ns=(100000000000,100000000000))
        os.utime(self.ref,ns=(300000000000,300000000000))
        self.assertIn(self.relref,self.visual(self.relref))
    def test_fallback_shirt_if_no_actual_photo(self):
        self.assertIsNone(self.visual(self.relref))
    def test_bad_db_path_cannot_escape_static(self):
        self.assertIsNone(self.visual("../../secret.png"))
    def test_legacy_cover_used_when_reference_missing(self):
        self.cover.write_bytes(b"legacy-cover")
        self.assertIn("results/model_covers/2069.jpg",self.visual(self.relref))
    def test_optional_reference_query_for_edit_page(self):
        self.ref.write_bytes(b"photo")
        with (
            self.a.app.test_request_context(),
            patch.object(self.a,"fetch_one",return_value={"image_path":self.relref}) as query
        ):
            url=self.a.garment_cover_url(2069)
        self.assertIn(self.relref,url)
        query.assert_called_once()
    def test_template_uses_same_main_photo(self):
        html=(ROOT/"templates/garment_model_detail.html").read_text(encoding="utf-8")
        self.assertIn("primary_photo_url if loop.first",html)
        catalog=(ROOT/"templates/garment_models.html").read_text(encoding="utf-8")
        self.assertIn("model.cover_url",catalog)

if __name__=="__main__":unittest.main()
