"""Catalog references and inspection evidence must remain distinct."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from jinja2 import ChoiceLoader, DictLoader
from werkzeug.datastructures import FileStorage
from tests.test_support import configure_test_environment
configure_test_environment()
import app as A

ROOT = Path(__file__).resolve().parents[1]


class ImageSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.static = Path(self.tmp.name)
        self.static_patch = patch.object(A.app, '_static_folder', str(self.static))
        self.static_patch.start(); self.addCleanup(self.static_patch.stop)
        self.request = A.app.test_request_context('/')
        self.request.push(); self.addCleanup(self.request.pop)

    def store(self, relative, data=b'original evidence'):
        path = self.static / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_catalog_1219_and_1216_show_original_without_derivative(self):
        environment = A.app.jinja_env.overlay(loader=ChoiceLoader([
            DictLoader({'base.html':'{% block content %}{% endblock %}'}), A.app.jinja_loader]))
        for code, identifier in [('BLUSA-1219',2072),('BLUSA-1216',2069)]:
            relative=f'results/model_references/{identifier}/source.jpg'
            self.store(relative,(ROOT/'diagnostics/catalog-presentation-20261020-172735/purple-2074-original.jpg').read_bytes())
            url, cutout=A.garment_cover_display_status(identifier,relative)
            self.assertIn(relative,url); self.assertFalse(cutout)
            model=SimpleNamespace(id=identifier,code=code,name=code,color='Morado',size='S',
                inspection_side='Frente',creator_name='Prueba',status='APROBADO',
                cover_url=url,cover_has_cutout=cutout,cover_display_label='Referencia original',ai_ready=False)
            html=environment.get_template('garment_models.html').render(models=[model])
            self.assertIn('src="'+url.replace('&','&amp;')+'"',html)
            self.assertNotIn('Presentaci&oacute;n pendiente',html)

    def test_catalog_and_detail_agree_for_legacy_original(self):
        relative='garment_models/2069/source.jpg';self.store(relative)
        catalog,cutout=A.garment_cover_display_status(2069,relative)
        self.assertEqual(catalog,A.garment_cover_url(2069,relative))
        self.assertEqual(catalog,A.garment_reference_display_url(relative))
        self.assertFalse(cutout)

    def test_catalog_prefers_derived_then_original_then_no_image(self):
        relative='results/model_references/2074/source.jpg';self.store(relative)
        self.store('results/model_previews/2074/source.png')
        url,cutout=A.garment_cover_display_status(2074,relative)
        self.assertTrue(cutout);self.assertIn('/model_previews/',url)
        self.assertEqual((None,False),A.garment_cover_display_status(999,'results/model_references/999/missing.jpg'))

    def test_inspection_evidence_uses_original_and_never_substitutes_result(self):
        original='captures/inspection-99.jpg';result='results/inspection-99.png'
        self.store(original);self.store(result)
        self.assertIn(original,A.inspection_evidence_url(original,result))
        (self.static/original).unlink()
        self.assertIsNone(A.inspection_evidence_url(original,result))
        self.assertIn(result,A.inspection_evidence_url(result))

    def test_inspection_never_substitutes_model_picture_or_traverses_root(self):
        for path in ('results/model_previews/2074/source.png','results/model_covers/2074.jpg','garment_models/2074/photo.jpg'):
            self.store(path)
            self.assertIsNone(A.inspection_evidence_url(path,path))
        self.assertIsNone(A.inspection_evidence_url('../outside.jpg'))

    def test_real_purple_capture_generates_white_preview_and_preserves_original(self):
        source=ROOT/'diagnostics/catalog-presentation-20261020-172735/purple-2074-original.jpg'
        original=source.read_bytes()
        upload=FileStorage(stream=io.BytesIO(original),filename='camera.jpg')
        images=A.prepare_garment_reference_images([upload])
        queries=[]
        class Cursor:
            def execute(self,sql,params):queries.append((sql,params))
        A.save_garment_reference_images(2074,images,cur=Cursor())
        stored=list((self.static/'results/model_references/2074').glob('*'))
        self.assertEqual(stored[0].read_bytes(),original)
        preview=list((self.static/'results/model_previews/2074').glob('*.png'))[0]
        image=cv2.imread(str(preview))
        self.assertTrue(np.all(image[0]==255));self.assertTrue(np.all(image[-1]==255))
        x,y,w,h=cv2.boundingRect(np.any(image<245,axis=2).astype(np.uint8))
        self.assertGreater(w/image.shape[1],.85);self.assertGreater(h/image.shape[0],.85)
        self.assertEqual(source.read_bytes(),original)
        self.assertIn('/model_references/',queries[0][1][1])

    def test_failed_segmentation_saves_exact_original_without_manual_requirement(self):
        ok,encoded=cv2.imencode('.jpg',np.full((200,200,3),255,np.uint8));self.assertTrue(ok)
        original=encoded.tobytes()
        with patch.object(A,'garment_visual_preview',return_value=None):
            images=A.prepare_garment_reference_images([FileStorage(stream=io.BytesIO(original),filename='photo.jpg')])
            class Cursor:
                def execute(self,*args):pass
            A.save_garment_reference_images(2074,images,cur=Cursor())
        self.assertEqual(list((self.static/'results/model_references/2074').glob('*'))[0].read_bytes(),original)
        self.assertFalse((self.static/'results/model_previews/2074').exists())

    def test_processing_exception_falls_back_without_losing_source(self):
        with patch.object(A,'garment_visual_preview',side_effect=RuntimeError('processing failed')):
            self.assertIsNone(A.garment_reference_presentation(b'original'))

    def test_no_manual_editor_is_loaded_by_capture_edit_or_detail(self):
        for name in ('garment_model_form.html','garment_model_detail.html','garment_model_edit.html'):
            text=(ROOT/'templates'/name).read_text(encoding='utf-8')
            self.assertNotIn('garment-presentation-editor.js',text)
            self.assertNotIn('Ajustar contorno',text)
            self.assertNotIn('openGarmentPresentationEditor',text)

    def test_history_and_batch_cards_render_only_inspection_evidence(self):
        for name in ('records.html','batch_review.html'):
            text=(ROOT/'templates'/name).read_text(encoding='utf-8')
            self.assertIn('inspection_evidence_url',text)
            self.assertNotIn('row.model_presentation_url',text)
        js=(ROOT/'static/js/garment-reference-preview.js').read_text(encoding='utf-8')
        self.assertNotIn('openGarmentPresentationEditor',js)
        self.assertIn('Ver original',js)
