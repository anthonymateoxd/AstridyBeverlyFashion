"""Conservative, visual-only garment cutout. Originals always remain untouched.

Return None when the photo cannot be segmented with sufficient confidence.
The utility never edits training datasets or existing inspection evidence.
"""
import cv2
import numpy as np

MAX_EDGE = 1040


def garment_visual_preview(image_bytes: bytes, *, mask_only=False):
    """Return visual PNG, or its verified full-frame mask for guided capture.

    mask_only uses the same conservative detection but skips PNG encoding.
    It is solely an alternative to the guided segmentation, not PatchCore.
    """
    if isinstance(image_bytes, np.ndarray):
        frame = image_bytes
    else:
        if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
            return None
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    if frame is None or frame.ndim != 3:
        return None
    original_h, original_w = frame.shape[:2]
    height, width = frame.shape[:2]
    if min(height, width) < 130 or height * width > 18_000_000:
        return None
    scale = min(1.0, MAX_EDGE / max(height, width))
    if scale < 1:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x0, x1 = round(w * .085), round(w * .915)
    y0, y1 = round(h * .07), round(h * .93)
    inside = gray[y0:y1, x0:x1]
    brightest = float(np.percentile(inside, 85))
    # Intended for a well-lit background/light table. Reject dark/unclear scenes.
    if brightest < 165:
        return None
    bright = frame[y0:y1, x0:x1][inside >= np.percentile(inside, 78)]
    if len(bright) < 100:
        return None
    background = np.median(bright, axis=0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(
        np.uint8([[background]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    distance = np.sqrt(np.sum((lab - lab_bg) ** 2, axis=2))
    initial = np.zeros((h, w), dtype=np.uint8)
    core = (
        (distance[y0:y1, x0:x1] > 34)
        & (inside < brightest - 16)
    )
    initial[y0:y1, x0:x1] = (core * 255).astype(np.uint8)
    k = max(3, round(min(w, h) / 130)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    initial = cv2.morphologyEx(
        initial, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        initial, 8
    )
    plausible = []
    for idx in range(1, n):
        x, y, box_w, box_h, area = stats[idx]
        if area < h * w * .025 or area > h * w * .62:
            continue
        cx, cy = centroids[idx]
        if (
            abs(cx / w - .5) < .25
            and abs(cy / h - .5) < .27
            and box_w > w * .18 and box_h > h * .18
        ):
            plausible.append((area, idx))
    if not plausible:
        return None
    component = max(plausible)[1]
    mask = np.uint8(labels == component) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    mask[:] = 0
    cv2.drawContours(
        mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
    )
    x, y, box_w, box_h = cv2.boundingRect(mask)
    if (
        x <= x0 + 2 or y <= y0 + 2
        or x + box_w >= x1 - 2 or y + box_h >= y1 - 2
        or min(box_w, box_h) < min(w, h) * .19
    ):
        return None
    if mask_only:
        if (h, w) != (original_h, original_w):
            mask = cv2.resize(
                mask, (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask
    padding = max(12, round(max(box_w, box_h) * .075))
    left, top = max(0, x-padding), max(0, y-padding)
    right, bottom = min(w, x+box_w+padding), min(h, y+box_h+padding)
    alpha = cv2.GaussianBlur(mask, (5, 5), 1.0)
    alpha[mask == 0] = np.minimum(alpha[mask == 0], 100)
    bgra = cv2.cvtColor(
        frame[top:bottom, left:right], cv2.COLOR_BGR2BGRA
    )
    bgra[:, :, 3] = alpha[top:bottom, left:right]
    success, encoded = cv2.imencode(
        ".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 5]
    )
    return encoded.tobytes() if success and encoded.size else None
