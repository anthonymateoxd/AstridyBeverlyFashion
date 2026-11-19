"""Synthetic QA for display-only garment cutouts and original preservation."""
import importlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from garment_background_preview import garment_visual_preview


ROOT = Path(__file__).resolve().parents[1]
REAL_REFERENCE = ROOT / "diagnostics" / "authorized_blusa_1218_original.jpg"


def legacy_sample(color=(105, 115, 135)):
    """Reference fixture of the original display-only cutout contract."""
    frame = np.full((520, 820, 3), 234, dtype=np.uint8)
    frame[:, :38] = 31
    frame[:, -38:] = 31
    outline = np.array([
        [290, 105], [340, 144], [410, 158], [480, 142],
        [532, 101], [592, 145], [622, 205], [579, 231],
        [562, 206], [560, 410], [320, 410], [317, 212],
        [285, 240], [253, 201],
    ], np.int32)
    cv2.fillPoly(frame, [outline], color)
    ok, enc = cv2.imencode(".jpg", frame)
    assert ok
    return enc.tobytes()


def _lab_distance(image, color):
    source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = cv2.cvtColor(
        np.uint8([[np.clip(color, 0, 255)]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    return np.linalg.norm(source - reference, axis=2)


def _fill_ratio(mask):
    """Foreground area over its bounding box, a shape-only measure."""
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return 0.0
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x, y, bw, bh, area = stats[largest]
    return area / max(1, bw * bh)


def _expected_silhouette(resolution=(720, 540)):
    """The photographed shape of the synthetic garment, as a mask."""
    h, w = resolution
    sx, sy = w / 540, h / 720
    points = np.array([
        [190, 125], [235, 155], [270, 170], [305, 155], [350, 125], [410, 185],
        [375, 230], [352, 207], [350, 555], [190, 555], [188, 207], [165, 230],
        [130, 185],
    ], np.float32) * np.array([sx, sy], np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    cv2.ellipse(
        mask, (round(270 * sx), round(167 * sy)),
        (max(2, round(31 * sx)), max(2, round(19 * sy))),
        0, 0, 180, 0, -1,
    )
    return mask


def sample(color=(242, 242, 242), background=(248, 248, 248), *,
           style="shirt", resolution=(720, 540), seed=31):
    h, w = resolution
    rng = np.random.default_rng(seed)
    frame = np.empty((h, w, 3), np.uint8)
    # Slight light-table gradient is representative of nonuniform studio light.
    gradient = np.linspace(-4, 4, h, dtype=np.float32)[:, None, None]
    frame[:] = np.clip(np.asarray(background, np.float32) + gradient, 0, 255)
    if style == "shirt":
        sx, sy = w / 540, h / 720
        points = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.float32)
        points *= np.array([sx, sy], np.float32)
        cv2.fillPoly(frame, [points.astype(np.int32)], color)
        cv2.ellipse(frame, (round(270*sx),round(167*sy)),
                    (max(2,round(31*sx)),max(2,round(19*sy))),
                    0, 0, 180, background, -1)
    elif style == "problem-background":
        cv2.rectangle(frame, (round(w*.07),round(h*.1)),
                      (round(w*.93),round(h*.9)), (70,70,70), -1)
        cv2.rectangle(frame, (round(w*.35),round(h*.2)),
                      (round(w*.65),round(h*.8)), color, -1)
    elif style == "blank":
        pass
    if seed is not None:
        noise = rng.normal(0, 0.35, frame.shape)
        frame = np.clip(frame.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 97])
    assert ok
    return encoded.tobytes(), frame


class GarmentPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = importlib.import_module("app")

    def test_colorful_garment_is_cut_out_on_a_white_canvas(self):
        """The catalog cutout is opaque on a uniform white background.

        The original contract returned a transparent PNG. The catalog now
        shows a uniform white background, so the derivative is an opaque
        image whose canvas is white and whose cloth keeps its colours.
        """
        raw = legacy_sample()
        png = garment_visual_preview(raw)
        self.assertIsNotNone(png)
        cut = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(cut.shape[2], 3)
        self.assertLess(cut.shape[1], 820)
        self.assertTrue(np.all(cut[0, 0] > 245))
        self.assertTrue(np.all(cut[-1, -1] > 245))
        self.assertTrue(raw.startswith(b"\xff\xd8"))

    def test_near_white_background_without_object_falls_back(self):
        blank = np.full((480, 720, 3), 240, np.uint8)
        _, jpg = cv2.imencode(".jpg", blank)
        self.assertIsNone(garment_visual_preview(jpg.tobytes()))

    def test_dark_scene_falls_back(self):
        dark = np.full((480, 720, 3), 30, np.uint8)
        cv2.rectangle(dark, (230, 120), (520, 400), (60, 70, 85), -1)
        _, jpg = cv2.imencode(".jpg", dark)
        self.assertIsNone(garment_visual_preview(jpg.tobytes()))

    def test_preserve_original_and_write_derivative_with_no_db_change(self):
        m = self.app_module
        original = m.app.static_folder
        with tempfile.TemporaryDirectory() as temporary:
            m.app.static_folder = temporary
            try:
                with m.app.test_request_context():
                    with patch.dict(m.session, {"user_id": 15}):
                        cursor = Mock()
                        raw = legacy_sample()
                        import inspect
                        if "cur" in inspect.signature(
                            m.save_garment_reference_images
                        ).parameters:
                            originals = m.save_garment_reference_images(
                                417, [(".jpg", raw)], cur=cursor
                            )
                            cursor.execute.assert_called_once()
                            sql, values = cursor.execute.call_args.args
                            self.assertEqual(len(originals), 2)
                        else:
                            with patch.object(m, "execute") as execute:
                                m.save_garment_reference_images(
                                    417, [(".jpg", raw)]
                                )
                                execute.assert_called_once()
                                sql, values = execute.call_args.args
                        self.assertIn("INSERT INTO garment_model_images", sql)
                        stored = Path(temporary) / values[1]
                        self.assertEqual(stored.read_bytes(), raw)
                        visual = m.garment_reference_preview_path(values[1])
                        self.assertTrue(visual.is_file())
                        self.assertTrue(visual.read_bytes().startswith(b"\x89PNG"))
                        url = m.garment_cover_url(417, values[1])
                        self.assertIn("model_previews/417", url)
            finally:
                m.app.static_folder = original

    def test_preview_endpoint_requires_authorized_session(self):
        m = self.app_module
        client = m.app.test_client()
        response = client.post("/modelos-prenda/camara-referencia/recorte")
        self.assertNotEqual(response.status_code, 200)

    def test_preview_endpoint_returns_png_for_authorized_user(self):
        m = self.app_module
        client = m.app.test_client()
        with client.session_transaction() as session:
            session.update(user_id=35, username="qa", role="ADMIN")
        with patch.object(m, "fetch_one", return_value={
            "id": 35, "username": "qa", "role": "ADMIN",
            "full_name": "Prueba", "active": 1,
        }):
            response = client.post(
                "/modelos-prenda/camara-referencia/recorte",
                data={"photo": (BytesIO(legacy_sample()), "foto.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertTrue(response.data.startswith(b"\x89PNG"))

    def test_existing_cover_wins_if_newer_than_reference_preview(self):
        m = self.app_module
        original = m.app.static_folder
        with tempfile.TemporaryDirectory() as temporary:
            m.app.static_folder = temporary
            try:
                with m.app.test_request_context():
                    raw = legacy_sample()
                    ref = Path(temporary) / "results/model_references/44/a.jpg"
                    ref.parent.mkdir(parents=True, exist_ok=True)
                    ref.write_bytes(raw)
                    visual = m.garment_reference_preview_path(
                        "results/model_references/44/a.jpg"
                    )
                    visual.parent.mkdir(parents=True, exist_ok=True)
                    visual.write_bytes(garment_visual_preview(raw))
                    cover = Path(temporary) / "results/model_covers/44.jpg"
                    cover.parent.mkdir(parents=True, exist_ok=True)
                    cover.write_bytes(raw)
                    import os
                    os.utime(ref, ns=(10 ** 12, 10 ** 12))
                    os.utime(cover, ns=(2 * 10 ** 12, 2 * 10 ** 12))
                    url = m.garment_cover_url(
                        44, "results/model_references/44/a.jpg"
                    )
                    self.assertIn("model_covers/44.jpg", url)
            finally:
                m.app.static_folder = original

    def test_white_garment_keeps_whole_silhouette_and_high_resolution(self):
        """A pale garment on a lit table keeps its shape on the canvas."""
        raw, source = sample(color=(228, 228, 228))
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(cut.shape[2], 3)
        self.assertGreater(np.count_nonzero(np.all(cut[0] > 245, axis=1)),
                           cut.shape[1] * .95)
        self.assertGreater(cut.shape[0], 400)
        self.assertGreater(cut.shape[1], 300)
        self.assertTrue(raw.startswith(b"\xff\xd8"))
        nonwhite = np.any(cut[:, :, :3] < 248, axis=2)
        ys, xs = np.where(nonwhite)
        self.assertLess(abs((xs.mean()+0.5)/cut.shape[1] - 0.5), 0.06)
        self.assertLess(abs((ys.mean()+0.5)/cut.shape[0] - 0.5), 0.12)
        self.assertGreater(source.shape[0], 0)

    def test_white_garment_without_contrast_is_not_published(self):
        """A white garment on a white table cannot be cut reliably.

        The separation against the table is a couple of levels, so any cutout
        would eat into the shoulders. The original photograph is kept and
        the interface labels it as requiring review.
        """
        raw, _ = sample(color=(242, 242, 242))
        self.assertIsNone(garment_visual_preview(raw))

    def test_colored_garment_is_segmented(self):
        raw, _ = sample(color=(35, 65, 220))
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.all(cut[0, 0] > 245))
        self.assertGreater(np.count_nonzero(np.any(cut < 240, axis=2)), 60_000)

    def test_cutout_has_no_halo_of_light_table_pixels(self):
        """The table captured around the cloth must not survive the cutout.

        A halo is the band of light-table pixels that the segmentation keeps
        around the garment. On the white canvas it is visible as a grey rim.
        The previous implementation kept half of the band around the cloth;
        the boundary itself is antialiased, so the measurement is taken a
        few pixels away from the silhouette.
        """
        raw, _ = sample(color=(35, 65, 220))
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        source = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        table = np.median(source[:40].reshape(-1, 3), axis=0)
        to_table = _lab_distance(cut, table)
        to_white = _lab_distance(cut, np.array([255.0, 255.0, 255.0]))
        # Table-coloured pixels are neither the cloth nor the white canvas.
        table_like = (to_table < 4.0) & (to_white > 4.0)
        cloth = (to_table > 25).astype(np.uint8)
        near = cv2.dilate(
            cloth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ) > 0
        band = cv2.dilate(
            cloth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ) > 0
        band &= ~near
        self.assertGreater(np.count_nonzero(band), 500)
        ratio = np.count_nonzero(table_like & band) / np.count_nonzero(band)
        self.assertLess(ratio, 0.15)

    def test_white_garment_keeps_the_photographed_silhouette(self):
        """A pale garment must keep its shape, without table fringe."""
        raw, _ = sample(color=(228, 228, 228))
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        foreground = np.any(cut < 250, axis=2)
        expected = _expected_silhouette() > 0
        self.assertGreater(np.count_nonzero(foreground), 20_000)
        self.assertLess(
            abs(_fill_ratio(foreground) - _fill_ratio(expected)), 0.12
        )

    def test_empty_light_scene_is_rejected(self):
        raw, _ = sample(style="blank")
        self.assertIsNone(garment_visual_preview(raw))

    def test_dark_problematic_background_is_rejected(self):
        raw, _ = sample(style="problem-background")
        self.assertIsNone(garment_visual_preview(raw))

    def test_visible_print_and_brand_detail_are_preserved(self):
        h, w = 720, 540
        frame = np.full((h, w, 3), 248, np.uint8)
        outline = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.int32)
        cv2.fillPoly(frame, [outline], (242, 242, 242))
        cv2.ellipse(frame, (270, 167), (31, 19), 0, 0, 180, (248,248,248), -1)
        # High-contrast print and brand plate remain fully inside the garment.
        cv2.rectangle(frame, (245, 315), (285, 345), (30, 40, 210), -1)
        cv2.rectangle(frame, (295, 430), (307, 438), (20, 20, 20), -1)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        output = garment_visual_preview(encoded.tobytes())
        self.assertIsNotNone(output)
        cut = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(np.count_nonzero(
            (cut[:, :, 2] > cut[:, :, 1] * 1.7)
            & (cut[:, :, 2] > cut[:, :, 0] * 1.7)
        ), 900)

    def test_variable_light_background_keeps_pale_garment_candidate(self):
        """A pale garment on a lit table is cut out when it still separates."""
        h, w = 720, 540
        y = np.linspace(238, 255, h, dtype=np.float32)[:, None, None]
        frame = np.broadcast_to(y, (h, w, 3)).copy().astype(np.uint8)
        outline = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.int32)
        cv2.fillPoly(frame, [outline], (228, 228, 228))
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        result = garment_visual_preview(encoded.tobytes())
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        foreground = np.any(cut < 250, axis=2)
        self.assertGreater(np.count_nonzero(foreground), 20_000)
        self.assertLess(
            abs(_fill_ratio(foreground) - _fill_ratio(_expected_silhouette())),
            0.12,
        )

    def test_garment_indistinguishable_from_the_table_is_not_published(self):
        """When cloth and table cannot be told apart, no cutout is produced.

        Publishing a cutout in that situation would present a defective
        segmentation as a successful one, so the original photograph is kept
        and the interface shows it as "requires review".
        """
        h, w = 720, 540
        y = np.linspace(238, 255, h, dtype=np.float32)[:, None, None]
        frame = np.broadcast_to(y, (h, w, 3)).copy().astype(np.uint8)
        outline = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.int32)
        cv2.fillPoly(frame, [outline], (244, 244, 244))
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        self.assertIsNone(garment_visual_preview(encoded.tobytes()))

    def test_incomplete_segmentation_with_large_hole_is_rejected(self):
        h, w = 720, 540
        frame = np.full((h, w, 3), 248, np.uint8)
        points = np.array([[190,125],[235,155],[270,170],[305,155],
                           [350,125],[410,185],[375,230],[352,207],
                           [350,555],[190,555],[188,207],[165,230],
                           [130,185]], np.int32)
        cv2.fillPoly(frame, [points], (240, 240, 240))
        cv2.rectangle(frame, (160, 225), (380, 520), (248, 248, 248), -1)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        self.assertIsNone(garment_visual_preview(encoded.tobytes()))

    def test_asymmetric_dark_background_with_no_garment_is_rejected(self):
        blank = np.full((520, 820, 3), 230, np.uint8)
        blank[:, :40] = (30, 30, 30)
        blank[:, -40:] = (30, 30, 30)
        ok, encoded = cv2.imencode(".jpg", blank)
        self.assertTrue(ok)
        self.assertIsNone(garment_visual_preview(encoded.tobytes()))

    def test_invalid_input_fails_closed(self):
        self.assertIsNone(garment_visual_preview(b"invalid-image"))
        self.assertIsNone(garment_visual_preview(b""))

    def test_mask_only_returns_full_frame_mask_for_legacy_caller(self):
        raw, source = sample()
        mask = garment_visual_preview(raw, mask_only=True)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, source.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)).issubset({0, 255}), True)

    def test_garment_print_and_brand_plate_pixels_survive_compositing(self):
        frame = np.full((720, 540, 3), 248, np.uint8)
        outline = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.int32)
        cv2.fillPoly(frame, [outline], (242,242,242))
        cv2.ellipse(frame, (270,167), (31,19), 0, 0, 180, (248,248,248), -1)
        cv2.rectangle(frame, (245,315), (285,345), (30,40,210), -1)
        cv2.rectangle(frame, (295,430), (307,438), (20,20,20), -1)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        result = garment_visual_preview(encoded.tobytes())
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        red_pixels = ((cut[:,:,2] > cut[:,:,1]*1.7)
                      & (cut[:,:,2] > cut[:,:,0]*1.7))
        dark_pixels = np.all(cut < 70, axis=2)
        self.assertGreater(np.count_nonzero(red_pixels), 900)
        self.assertGreater(np.count_nonzero(dark_pixels), 40)

    def test_small_collar_decoration_pixel_survives_compositing(self):
        frame = np.full((720, 540, 3), 248, np.uint8)
        outline = np.array([
            [190,125],[235,155],[270,170],[305,155],[350,125],[410,185],
            [375,230],[352,207],[350,555],[190,555],[188,207],[165,230],
            [130,185],
        ], np.int32)
        cv2.fillPoly(frame, [outline], (242,242,242))
        cv2.ellipse(frame, (270,167), (31,19), 0, 0, 180, (248,248,248), -1)
        cv2.rectangle(frame, (263,182), (277,196), (35,35,180), -1)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 97])
        self.assertTrue(ok)
        result = garment_visual_preview(encoded.tobytes())
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(np.count_nonzero(cut[:,:,2] > cut[:,:,0]*1.7), 50)

    @unittest.skipUnless(
        REAL_REFERENCE.is_file(), "authorized reference photo is not present"
    )
    def test_authorized_blusa_1218_reference_is_cut_out_cleanly(self):
        """Real reference photo: garment isolated on a white canvas."""
        raw = REAL_REFERENCE.read_bytes()
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        cut = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.all(cut[0, :] > 250))
        self.assertTrue(np.all(cut[-1, :] > 250))
        foreground = np.any(cut < 250, axis=2)
        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), 8
        )
        self.assertEqual(count - 1, 1)
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.assertGreater(stats[largest + 1, cv2.CC_STAT_AREA], 50_000)
        # The printed brand survives the cutout.
        self.assertGreater(np.count_nonzero(np.all(cut < 90, axis=2)), 200)
        # The neck opening keeps the light table seen through it instead of
        # being painted with cloth, so the collar keeps its outline.
        collar = np.all(cut < 235, axis=2) & ~np.all(cut < 120, axis=2)
        self.assertGreater(np.count_nonzero(collar), 2000)


if __name__ == "__main__":
    unittest.main()
