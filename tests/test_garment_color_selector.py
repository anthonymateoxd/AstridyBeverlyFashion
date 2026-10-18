"""Color options and server validation for garment create/edit forms."""
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class GarmentColorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=importlib.import_module("app")

    def setUp(self):
        self.client=self.m.app.test_client()
        self.user={"id":31,"username":"qa-model-color","full_name":"Prueba","role":"ADMIN","active":1}
        self.model={"id":871,"code":"BLUSA-TEST","name":"Blusa antigua",
                    "color":"Rosa","description":"Antiguo modelo",
                    "status":"BORRADOR","active":1,"created_by":31}
        with self.client.session_transaction() as sess:
            sess.update(user_id=31,username="qa-model-color",role="ADMIN")

    def test_catalog_and_normalization(self):
        self.assertEqual(len(self.m.GARMENT_COLORS),14)
        self.assertIn("Multicolor",self.m.GARMENT_COLORS)
        self.assertIn("Café",self.m.GARMENT_COLORS)
        for before,after in [
            (" Blanca ","Blanco"),("blanco","Blanco"),
            ("ROSADO","Rosado"),("Rosa","Rosado"),
            ("Cafe","Café"),("café","Café"),
            (" azul  ","Azul"),("Multicolor","Multicolor"),
        ]:
            with self.subTest(before=before):
                self.assertEqual(self.m.normalize_garment_color(before),after)

    def test_form_validates_predefined_and_other(self):
        from werkzeug.datastructures import MultiDict
        for raw,expected in [
            ({"color":"Blanco"},"Blanco"),
            ({"color":"Blanca"},"Blanco"),
            ({"color":"Rosa"},"Rosado"),
            ({"color":"__otro__","custom_color":"Verde olivo"},"Verde olivo"),
            ({"color":"__otro__","custom_color":"rosa"},"Rosado"),
            ({"color":"__otro__","custom_color":"Azul petróleo"},"Azul petróleo"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(self.m.garment_color_from_form(MultiDict(raw)),expected)
        for raw in [
            {"color":""},
            {"color":"Ultravioleta"},
            {"color":"__otro__","custom_color":""},
            {"color":"__otro__","custom_color":"a"},
            {"color":"__otro__","custom_color":"Rojo2"},
            {"color":"__otro__","custom_color":"<script>"},
            {"color":"__otro__","custom_color":"Q"*81},
        ]:
            with self.subTest(invalid=raw):
                with self.assertRaises(ValueError):
                    self.m.garment_color_from_form(MultiDict(raw))

    def test_legacy_edit_color_shows_equivalent_or_custom(self):
        self.assertEqual(self.m.garment_color_selection("Blanca"),("Blanco",""))
        self.assertEqual(self.m.garment_color_selection("Rosa"),("Rosado",""))
        self.assertEqual(self.m.garment_color_selection("Verde olivo"),("__otro__","Verde olivo"))
        self.assertEqual(self.m.garment_color_selection(""),("",""))

    def test_new_form_has_one_shared_selector_and_legacy_photo_form(self):
        with patch.object(self.m,"fetch_one",return_value=self.user):
            res=self.client.get("/modelos-prenda/nuevo")
        self.assertEqual(res.status_code,200)
        body=res.get_data(as_text=True)
        self.assertIn('id="gmfColorSelect"',body)
        self.assertIn('<option value="Multicolor"',body)
        self.assertIn('<option value="__otro__"',body)
        self.assertIn('name="custom_color"',body)
        self.assertIn('name="reference_images"',body)
        self.assertIn('id="gmfTakePhoto"',body)
        import re
        self.assertEqual(len(re.findall(r'<select[^>]+name="color"',body)),1)

    def test_edit_draft_preselects_legacy_color(self):
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model",return_value=self.model),
              patch.object(self.m,"garment_cover_url",return_value=None)):
            res=self.client.get("/modelos-prenda/871/editar")
        self.assertEqual(res.status_code,200)
        html=res.get_data(as_text=True)
        self.assertIn('value="Rosado" selected',html)
        self.assertIn('name="custom_color"',html)

    def test_edit_custom_historical_color_survives(self):
        self.model["color"]="Verde olivo"
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model",return_value=self.model),
              patch.object(self.m,"garment_cover_url",return_value=None)):
            html=self.client.get("/modelos-prenda/871/editar").get_data(as_text=True)
        self.assertIn('value="__otro__" selected',html)
        self.assertIn('value="Verde olivo"',html)

    def test_approved_model_keeps_readonly_original_color(self):
        self.model["status"]="APROBADO"
        self.model["color"]="Rosa"
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model",return_value=self.model),
              patch.object(self.m,"garment_cover_url",return_value=None)):
            html=self.client.get("/modelos-prenda/871/editar").get_data(as_text=True)
        self.assertIn('value="Rosa" readonly',html)
        self.assertNotIn('name="custom_color"',html)

    def test_bad_color_create_rejected_without_db_changes(self):
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"db") as db):
            res=self.client.post("/modelos-prenda/nuevo",
                data={"name":"Blusa nueva","color":"Inventado"})
        self.assertEqual(res.status_code,200)
        db.assert_not_called()
        self.assertIn("Seleccione un color",res.get_data(as_text=True))

    def test_invalid_other_color_create_rejected_without_db_changes(self):
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"db") as db):
            res=self.client.post("/modelos-prenda/nuevo",
                data={"name":"Blusa nueva","color":"__otro__","custom_color":" "})
        self.assertEqual(res.status_code,200)
        db.assert_not_called()
        self.assertIn("Ingrese otro color",res.get_data(as_text=True))

    def test_edit_custom_color_persists_on_draft_only(self):
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model",return_value=self.model),
              patch.object(self.m,"execute") as execute):
            res=self.client.post("/modelos-prenda/871/editar",
                data={"name":"Blusa antigua","color":"__otro__",
                      "custom_color":"Verde olivo","description":"Antiguo modelo"})
        self.assertEqual(res.status_code,302)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[1][1],"Verde olivo")

    def test_edit_rejects_unknown_direct_color_without_write(self):
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model",return_value=self.model),
              patch.object(self.m,"execute") as execute):
            res=self.client.post("/modelos-prenda/871/editar",
                data={"name":"Blusa antigua","color":"Marfil"})
        self.assertEqual(res.status_code,302)
        execute.assert_not_called()

    def test_edit_option_is_not_accessible_to_quality_manager(self):
        self.user["role"]="QUALITY_MANAGER"
        with (patch.object(self.m,"fetch_one",return_value=self.user),
              patch.object(self.m,"get_garment_model") as model_fetch):
            res=self.client.get("/modelos-prenda/871/editar")
        self.assertNotEqual(res.status_code,200)
        model_fetch.assert_not_called()

if __name__=="__main__":
    unittest.main()
