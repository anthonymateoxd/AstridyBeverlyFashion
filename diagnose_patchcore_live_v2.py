from pathlib import Path
from urllib.parse import quote
import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")

OUT = Path(
    "/data/BLUSA_002_S_ROSA_FRONT_V2/diagnostics"
)
OUT.mkdir(parents=True, exist_ok=True)

ip = os.environ["CAMERA_IP"]
user = quote(os.environ["CAMERA_USER"], safe="")
password = quote(os.environ["CAMERA_PASSWORD"], safe="")
port = os.environ.get("CAMERA_RTSP_PORT", "554")
channel = os.environ.get("CAMERA_CHANNEL", "1")
subtype = os.environ.get("CAMERA_SUBTYPE", "0")

url = (
    f"rtsp://{user}:{password}@{ip}:{port}"
    f"/cam/realmonitor?channel={channel}&subtype={subtype}"
)

RX1 = float(os.environ.get("ROI_X1", "0.16"))
RY1 = float(os.environ.get("ROI_Y1", "0.05"))
RX2 = float(os.environ.get("ROI_X2", "0.84"))
RY2 = float(os.environ.get("ROI_Y2", "0.88"))

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise SystemExit("ERROR: no se pudo abrir RTSP.")

frame = None

for _ in range(12):
    ok, candidate = cap.read()
    if ok and candidate is not None:
        frame = candidate

cap.release()

if frame is None:
    raise SystemExit("ERROR: no se obtuvo frame.")

h, w = frame.shape[:2]

x1 = int(RX1 * w)
y1 = int(RY1 * h)
x2 = int(RX2 * w)
y2 = int(RY2 * h)

roi = frame[y1:y2, x1:x2]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

sat = hsv[:, :, 1]
lab_a = lab[:, :, 1]

seed = np.where(
    (sat >= 30) & (lab_a >= 136),
    255,
    0,
).astype(np.uint8)

seed_raw = seed.copy()

seed = cv2.morphologyEx(
    seed,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    ),
    iterations=1,
)

seed = cv2.morphologyEx(
    seed,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    ),
    iterations=1,
)

seed = cv2.morphologyEx(
    seed,
    cv2.MORPH_OPEN,
    np.ones((5, 5), dtype=np.uint8),
    iterations=1,
)

count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        seed,
        connectivity=8,
    )
)

roi_area = float(roi.shape[0] * roi.shape[1])

components = []

for label in range(1, count):
    area = int(stats[label, cv2.CC_STAT_AREA])
    cx, cy = centroids[label]

    components.append(
        (area, area / roi_area * 100.0, cx, cy)
    )

components.sort(reverse=True)

print("===== FRAME ACTUAL =====")
print(f"Resolucion: {w}x{h}")
print(f"ROI: {roi.shape[1]}x{roi.shape[0]}")
print()

print("===== COLOR ROI =====")
print(f"Saturacion media: {sat.mean():.2f}")
print(f"Saturacion P50:   {np.percentile(sat, 50):.2f}")
print(f"Saturacion P75:   {np.percentile(sat, 75):.2f}")
print(f"Saturacion P90:   {np.percentile(sat, 90):.2f}")
print(f"LAB-A media:      {lab_a.mean():.2f}")
print(f"LAB-A P50:        {np.percentile(lab_a, 50):.2f}")
print(f"LAB-A P75:        {np.percentile(lab_a, 75):.2f}")
print(f"LAB-A P90:        {np.percentile(lab_a, 90):.2f}")
print()

raw_pixels = cv2.countNonZero(seed_raw)
final_pixels = cv2.countNonZero(seed)

print("===== SEMILLA =====")
print(
    f"Pixeles raw:       {raw_pixels} "
    f"({raw_pixels / roi_area * 100:.2f}%)"
)
print(
    f"Pixeles post-morf: {final_pixels} "
    f"({final_pixels / roi_area * 100:.2f}%)"
)
print(f"Componentes:       {count - 1}")
print()

print("===== COMPONENTES MAYORES =====")

for i, (area, pct, cx, cy) in enumerate(
    components[:10],
    start=1,
):
    mark = "VALIDO >=4%" if pct >= 4.0 else "pequeno"
    print(
        f"{i:02d}. area={area:8d} "
        f"coverage={pct:6.2f}% "
        f"centro=({cx:.1f},{cy:.1f}) "
        f"{mark}"
    )

cv2.imwrite(
    str(OUT / "live_raw.jpg"),
    frame,
    [cv2.IMWRITE_JPEG_QUALITY, 97],
)

cv2.imwrite(
    str(OUT / "live_roi.jpg"),
    roi,
    [cv2.IMWRITE_JPEG_QUALITY, 97],
)

cv2.imwrite(
    str(OUT / "seed_raw.png"),
    seed_raw,
)

cv2.imwrite(
    str(OUT / "seed_processed.png"),
    seed,
)

overlay = roi.copy()
overlay[seed > 0] = (
    overlay[seed > 0] * 0.45
    + np.array([0, 0, 255]) * 0.55
).astype(np.uint8)

cv2.imwrite(
    str(OUT / "seed_overlay.jpg"),
    overlay,
    [cv2.IMWRITE_JPEG_QUALITY, 97],
)

print()
print("===== ARCHIVOS DIAGNOSTICO =====")
print(OUT)
print("OK: diagnostico terminado.")
