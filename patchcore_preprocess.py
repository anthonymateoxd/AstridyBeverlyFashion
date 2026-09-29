"""Preprocesamiento compartido entre entrenamiento e inferencia.

Produccion (app.detect_defect) y el entrenador (ai_training) deben
consumir EXACTAMENTE la misma representacion antes de PatchCore:

    frame completo -> mascara de prenda -> recorte ROI -> fondo blanco
    (si la mascara no describe la prenda: ROI completo, nunca blanco
    uniforme; ver MIN_MASK_COVERAGE).

Mantener esta logica en un unico modulo evita el training-serving skew.
Este modulo no importa Flask ni app.py y puede usarse desde el worker.
"""

from __future__ import annotations


def _require_cv2():
    import cv2

    return cv2


def _require_numpy():
    import numpy

    return numpy


# Cobertura minima de la mascara de prenda dentro del ROI. Coincide con
# el limite inferior que ya aplicaba create_garment_mask: por debajo de
# este valor la "silueta" no describe la prenda y se cae al ROI.
MIN_MASK_COVERAGE = 0.20


def roi_bounds_from_fractions(frame, roi):
    """Convierte ROI en fracciones (x1, y1, x2, y2) a pixeles validos."""
    h, w = frame.shape[:2]
    x1 = int(max(0.0, min(float(roi[0]), 0.99)) * w)
    y1 = int(max(0.0, min(float(roi[1]), 0.99)) * h)
    x2 = int(max(0.01, min(float(roi[2]), 1.0)) * w)
    y2 = int(max(0.01, min(float(roi[3]), 1.0)) * h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("La region ROI configurada no es valida.")

    return x1, y1, x2, y2


def mask_coverage(roi_garment_mask, *, cv2_mod=None) -> float:
    """Cobertura (0..1) de la mascara dentro del ROI."""
    cv2 = cv2_mod if cv2_mod is not None else _require_cv2()

    if roi_garment_mask is None or roi_garment_mask.size == 0:
        return 0.0

    return float(
        cv2.countNonZero(roi_garment_mask)
    ) / float(max(1, roi_garment_mask.size))


def build_patchcore_regions(image, garment_mask, roi_bounds, *, cv2_mod=None):
    """Devuelve (roi_image, roi_mask, roi con fondo blanco).

    Reproduce exactamente el preprocesamiento previo a PatchCore en
    produccion (app.detect_defect).

    FALLBACK ROI: si la mascara de la prenda esta vacia o no cubre al
    menos MIN_MASK_COVERAGE del ROI, la silueta no describe la prenda.
    En ese caso se usa el ROI rectangular completo (mismo recorte que
    ve produccion) y NUNCA se pinta toda la imagen de blanco: una
    imagen colapsada a un unico color deja a PatchCore sin informacion.
    """
    cv2 = cv2_mod if cv2_mod is not None else _require_cv2()

    if cv2 is None:
        raise RuntimeError("OpenCV no esta disponible.")

    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_garment_mask = garment_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    coverage = mask_coverage(roi_garment_mask, cv2_mod=cv2)

    if coverage < MIN_MASK_COVERAGE:
        print(
            "[SEGMENTACION] "
            f"Cobertura mascara: {coverage * 100:.2f}% "
            f"(minimo {MIN_MASK_COVERAGE * 100:.2f}%): "
            "se usa el ROI completo."
        )
        roi_garment_mask = _require_numpy().full(
            roi_garment_mask.shape,
            255,
            dtype=_require_numpy().uint8,
        )
        return roi_image, roi_garment_mask, roi_image.copy()

    patchcore_input = _require_numpy().full_like(roi_image, 255)
    patchcore_input[roi_garment_mask > 0] = roi_image[roi_garment_mask > 0]

    return roi_image, roi_garment_mask, patchcore_input


def build_patchcore_input(image, garment_mask, roi_bounds, *, cv2_mod=None):
    """Recorta el ROI y pinta de blanco el fondo. Devuelve el ROI."""
    return build_patchcore_regions(
        image,
        garment_mask,
        roi_bounds,
        cv2_mod=cv2_mod,
    )[2]


def create_garment_mask(image, roi_bounds, *, cv2_mod=None, np_mod=None):
    """
    Obtiene la silueta completa de la blusa rosa talla S.

    El color rosa se utiliza solamente como semilla para encontrar
    la prenda. Después se rellena su contorno exterior para conservar
    cualquier alteración visual situada sobre la tela, aunque tenga
    un color diferente.
    """
    if image is None:
        raise ValueError(
            "No se recibió una imagen válida."
        )

    height, width = image.shape[:2]

    cv2 = cv2_mod if cv2_mod is not None else _require_cv2()
    np = np_mod if np_mod is not None else _require_numpy()

    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds

    roi = image[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]

    if roi is None or roi.size == 0:
        raise ValueError(
            "El ROI de inspección está vacío."
        )

    roi_height, roi_width = roi.shape[:2]
    roi_area = float(
        roi_height * roi_width
    )

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    )

    saturation = hsv[:, :, 1]
    lab_a = lab[:, :, 1]

    # Semilla: tela rosa.
    seed = np.where(
        (saturation >= 30)
        & (lab_a >= 136),
        255,
        0,
    ).astype(np.uint8)

    # Cerrar discontinuidades producidas por manchas,
    # reflejos, costuras y pequeños huecos en la tela.
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
        np.ones(
            (5, 5),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            seed,
            connectivity=8,
        )
    )

    if count <= 1:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    candidates = []

    for label in range(1, count):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < roi_area * 0.04:
            continue

        cx, cy = centroids[label]

        dx = abs(
            cx - roi_width / 2.0
        ) / max(
            1.0,
            roi_width,
        )

        dy = abs(
            cy - roi_height / 2.0
        ) / max(
            1.0,
            roi_height,
        )

        score = (
            area / roi_area
            - dx * 0.20
            - dy * 0.10
        )

        candidates.append(
            (
                score,
                label,
            )
        )

    if not candidates:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_label = (
        candidates[0][1]
    )

    component = np.where(
        labels == selected_label,
        255,
        0,
    ).astype(np.uint8)

    # Recuperar el CONTORNO EXTERIOR de la prenda.
    #
    # Es la diferencia fundamental respecto del algoritmo anterior:
    # los huecos internos de otro color ya no desaparecen.
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    garment_contour = max(
        contours,
        key=cv2.contourArea,
    )

    garment_roi = np.zeros(
        (roi_height, roi_width),
        dtype=np.uint8,
    )

    cv2.drawContours(
        garment_roi,
        [garment_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # Ligero cierre del borde exterior sin convertirlo
    # en un rectángulo ni en un convex hull.
    garment_roi = cv2.morphologyEx(
        garment_roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        ),
        iterations=1,
    )

    garment_roi = cv2.dilate(
        garment_roi,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    coverage = (
        cv2.countNonZero(
            garment_roi
        )
        / roi_area
    )

    print(
        "[SEGMENTACION] "
        f"Cobertura silueta: "
        f"{coverage * 100:.2f}%"
    )

    if coverage < 0.20:
        raise RuntimeError(
            "La silueta detectada es demasiado pequeña."
        )

    if coverage > 0.75:
        raise RuntimeError(
            "La silueta detectada es demasiado grande."
        )

    garment_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    garment_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ] = garment_roi

    return garment_mask
