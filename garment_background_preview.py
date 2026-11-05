"""Conservative, display-only garment cutouts for product reference photos.

Original uploads, capture masks, PatchCore inputs and inspection artifacts are
not changed by this module. Ambiguous/incomplete masks are rejected so callers
can display the unmodified original instead.
"""
import cv2
import numpy as np


MIN_COMPONENT_FRACTION = 0.018
MAX_COMPONENT_FRACTION = 0.82
# Minimum separation (LAB distance) between cloth and backdrop along the
# silhouette. Below this the cutout is not trustworthy and the original
# photograph must be shown instead of a defective preview.
MIN_BOUNDARY_CONTRAST = 3.0
# Fixed seed for the colour models of the segmentation, so the same photo
# always produces the same catalog cutout.
SEED_SEGMENTATION = 20240517


def _decode(image_bytes):
    if isinstance(image_bytes, np.ndarray):
        frame = image_bytes
    else:
        if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
            return None
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    return frame


def _border_background(frame):
    """Estimate a possibly uneven light-table background from border bands."""
    h, w = frame.shape[:2]
    band = max(3, round(min(h, w) * 0.035))
    rows = np.concatenate((frame[:band], frame[-band:]), axis=0)
    cols = np.concatenate((frame[:, :band], frame[:, -band:]), axis=1)
    samples = np.concatenate((rows.reshape(-1, 3), cols.reshape(-1, 3)), axis=0)
    samples = samples.astype(np.float32)
    center = np.percentile(samples, 90, axis=0)
    spread = np.median(np.abs(samples - np.median(samples, axis=0)), axis=0)
    # Per-channel MAD validates a light neutral backdrop; segmentation also
    # considers local and spatial edges instead of relying on color thresholds.
    neutral = (abs(float(center[0] - center[1])) < 65
               and abs(float(center[1] - center[2])) < 65)
    light = float(np.mean(center)) >= 145
    consistent = float(np.mean(spread)) < 100
    if not (neutral and light and consistent):
        return None, spread
    return center, spread


def _grabcut_mask(frame):
    h, w = frame.shape[:2]
    background, spread = _border_background(frame)
    if background is None:
        return None

    # Trim only the outermost border; the garment is expected to be completely
    # inside the photo. GrabCut uses color models plus neighborhood contrast.
    margin_x, margin_y = max(2, round(w * .012)), max(2, round(h * .012))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    labels = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    labels[:margin_y] = cv2.GC_BGD
    labels[-margin_y:] = cv2.GC_BGD
    labels[:, :margin_x] = cv2.GC_BGD
    labels[:, -margin_x:] = cv2.GC_BGD

    # Seed probable foreground by a locally adaptive color distance from the
    # estimated border color. A soft seed, not a final threshold mask, lets
    # GrabCut retain white cloth and patterned details using spatial edges.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(
        np.uint8([[np.clip(background, 0, 255)]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    local = cv2.GaussianBlur(dist, (0, 0), max(2, min(h, w) * 0.055))
    valid = labels == cv2.GC_PR_BGD
    inner = np.zeros((h, w), dtype=bool)
    inner[round(h * .08):round(h * .92), round(w * .08):round(w * .92)] = True
    # Keep a conservative lower bound for textured/shaded garment pixels while
    # adapting to border noise and broad background gradients.
    threshold = max(5.0, float(np.percentile(dist[valid], 65)) * 0.22)
    # Brightness/chroma evidence is supplemented by spatial edge strength,
    # preserving white garments where the background and cloth have similar
    # color but a real contour separates them.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge_context = cv2.GaussianBlur(edge, (0, 0), max(2, min(h, w) * 0.012))
    probable = ((dist > threshold) | (local > threshold * 0.72) |
                (edge_context > 5)) & inner
    # Multi-cue region growing: candidate cloth must have either substantial
    # chroma distance from the border model, local contrast, or edge structure.
    labels[probable & valid] = cv2.GC_PR_FGD

    # Strong foreground seeds are selected away from the frame edge, where a
    # centered product normally occupies the image. If none exist, fail closed.
    strong = ((dist > max(9.0, threshold * 1.4)) |
              ((edge_context > 5) & (local > 0.2))) & inner
    if np.count_nonzero(strong) < h * w * 0.0005:
        return None
    labels[strong] = cv2.GC_FGD
    # Prefer probable-background on the outer surround while retaining the
    # light-cloth candidates within the image's central product area.
    labels[~inner & (labels == cv2.GC_PR_BGD)] = cv2.GC_BGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    # GrabCut seeds its colour models with k-means, which draws from the
    # global OpenCV RNG. A fixed seed keeps the same photo producing the same
    # cutout on every run instead of depending on the process state. No other
    # module of the project uses that generator.
    try:
        cv2.setRNGSeed(SEED_SEGMENTATION)
        if float(np.mean(spread)) >= 48:
            # Variable illumination: keep the structured prior from LAB/local
            # contrast and let GrabCut's GMM adapt rather than rejecting the
            # backdrop or selecting one global color cutoff.
            iterations = 7
        else:
            iterations = 5
        cv2.grabCut(frame, labels, rect, bg_model, fg_model, iterations,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return mask


def _validate_and_refine(mask, frame_shape):
    h, w = frame_shape[:2]
    total = h * w
    count, components, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        fraction = area / total
        if not MIN_COMPONENT_FRACTION <= fraction <= MAX_COMPONENT_FRACTION:
            continue
        cx, cy = centers[label]
        if not (0.12 < cx / w < 0.88 and 0.10 < cy / h < 0.90):
            continue
        candidates.append((area, label))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    # Keep smaller connected garment features (e.g. a separated sleeve/collar)
    # if substantial; reject rather than silently excluding them.
    largest_area = candidates[0][0]
    if len(candidates) > 1 and candidates[1][0] >= largest_area * 0.12:
        return None
    chosen = candidates[0][1]
    clean = (components == chosen).astype(np.uint8)
    x, y, bw, bh, area = stats[chosen]
    if bw < w * 0.18 or bh < h * 0.20:
        return None
    # Reject likely clipped garments and overly fragmented/interior-only masks.
    if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
        return None
    solidity = area / max(1, bw * bh)
    if solidity < 0.12 or solidity > 0.99:
        return None

    # Close only tiny boundary gaps without smoothing away narrow garment edges.
    kernel_size = max(3, round(min(h, w) / 360) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean, cv2.boundingRect(clean)


def _mask_is_complete(mask, frame_shape):
    """Reject masks with large interior voids or clipped outer contours."""
    h, w = frame_shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    outer = np.zeros((h, w), np.uint8)
    cv2.drawContours(outer, [max(contours, key=cv2.contourArea)], -1,
                     1, cv2.FILLED)
    mask_area = int(np.count_nonzero(mask))
    outer_area = int(np.count_nonzero(outer))
    # Large holes are unsafe omissions. Also inspect row/column occupancy of
    # the silhouette to catch a missing panel spanning much of the torso.
    if outer_area <= 0 or mask_area / outer_area < 0.90:
        return False
    holes = (outer > 0) & (mask == 0)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), 8
    )
    # A neck opening is a real garment detail, so only holes large enough to
    # hide a panel of cloth are treated as a segmentation failure.
    if any(
        int(hole_stats[index, cv2.CC_STAT_AREA]) > max(48, outer_area * 0.12)
        for index in range(1, hole_count)
    ):
        return False
    x, y, w, h = cv2.boundingRect(outer)
    interior = mask[y:y + h, x:x + w]
    active_rows = np.mean(interior > 0, axis=1)
    active_cols = np.mean(interior > 0, axis=0)
    # A long continuous low-occupancy run in the torso indicates a large
    # missing panel; natural waist/hem narrowing is allowed.
    row_bad = ((active_rows < 0.16) & (np.arange(h) > h * 0.20)
               & (np.arange(h) < h * 0.82))
    if np.any(np.convolve(row_bad.astype(np.int8), np.ones(12, np.int8),
                          mode="valid") >= 12):
        return False
    if np.any((active_cols < 0.10) & (np.arange(w) > w * 0.18)
              & (np.arange(w) < w * 0.82)):
        return False
    return True


def _contour_mask(frame):
    """Foreground candidate from closed contours around the garment.

    Reference photos are taken over an illuminated table, so the frame often
    shows the room instead of a uniform backdrop. The garment is still the
    only region enclosed by its own contour: everything that can be reached
    from the image border without crossing a strong edge is background.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 60, 60)
    median = float(np.median(blurred))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    h, w = gray.shape
    count, labels = cv2.connectedComponents((edges == 0).astype(np.uint8), 8)
    if count <= 1:
        return None
    border = set(labels[0, :].tolist())
    border |= set(labels[-1, :].tolist())
    border |= set(labels[:, 0].tolist())
    border |= set(labels[:, -1].tolist())
    border.discard(0)
    if not border:
        return None
    background = np.isin(labels, list(border))
    return (~background).astype(np.uint8)


def _display_candidates(frame):
    """Yield plausible foreground masks for the display cutout."""
    grabcut = _grabcut_mask(frame)
    if grabcut is not None:
        yield grabcut
    contour = _contour_mask(frame)
    if contour is not None:
        yield contour


def _surface_model(frame, mask):
    """Local quadratic model of the surface that surrounds the garment.

    A light table is a smooth surface with a gentle gradient. Fitting that
    gradient on the ring around the garment reproduces the backdrop at every
    position, so the table can be told apart from the cloth even when both
    are pale. The fit is robust, so dark surroundings and the frame of the
    table do not distort it, and it is rejected when the surface is not
    smooth enough to be described by a quadratic.
    """
    h, w = frame.shape[:2]
    outside_distance = cv2.distanceTransform(
        (mask == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    ring = (outside_distance >= 3.0) & (outside_distance <= 45.0)
    if int(np.count_nonzero(ring)) < 500:
        return None
    ys, xs = np.nonzero(ring)
    if len(ys) > 24000:
        step = len(ys) // 24000 + 1
        ys, xs = ys[::step], xs[::step]
    x = (xs / max(1, w - 1)).astype(np.float32) * 2.0 - 1.0
    y = (ys / max(1, h - 1)).astype(np.float32) * 2.0 - 1.0
    basis = np.stack(
        [np.ones_like(x), x, y, x * x, x * y, y * y], axis=1
    )
    pixels = frame[ys, xs].astype(np.float32)
    weights = np.ones(len(ys), dtype=np.float32)
    solution = None
    for _ in range(3):
        try:
            solution = np.linalg.lstsq(
                basis * weights[:, None], pixels * weights[:, None], rcond=None
            )[0]
        except np.linalg.LinAlgError:
            return None
        predicted = basis @ solution
        residual = np.abs(predicted - pixels).max(axis=1)
        scale = float(np.median(residual)) * 1.4826 + 0.5
        weights = (residual < 3.0 * scale).astype(np.float32)
        if not weights.any():
            return None
    residual = np.abs(basis @ solution - pixels).max(axis=1)
    if float(np.median(residual)) > 4.0:
        return None
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    gx = (grid_x / max(1, w - 1)).astype(np.float32) * 2.0 - 1.0
    gy = (grid_y / max(1, h - 1)).astype(np.float32) * 2.0 - 1.0
    estimate = np.empty((h, w, 3), np.float32)
    for channel in range(3):
        estimate[:, :, channel] = (
            solution[0][channel]
            + solution[1][channel] * gx
            + solution[2][channel] * gy
            + solution[3][channel] * gx * gx
            + solution[4][channel] * gx * gy
            + solution[5][channel] * gy * gy
        )
    return np.clip(estimate, 0, 255).astype(np.uint8)


def _background_estimate(frame, mask):
    """Estimate the light table per pixel around the garment.

    The studio backdrop is a smooth, slowly varying surface. Modelling it
    per position (instead of with one global colour) is what allows the mask
    to be tightened without eroding real cloth.
    """
    modelled = _surface_model(frame, mask)
    if modelled is not None:
        return modelled
    h, w = frame.shape[:2]
    scale = 0.25
    small = cv2.resize(
        frame,
        (max(16, int(w * scale)), max(16, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    hole = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    hole = cv2.resize(
        hole, (small.shape[1], small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    if np.count_nonzero(hole) < hole.size * 0.02:
        return None
    try:
        estimated = cv2.inpaint(small, hole, 3, cv2.INPAINT_TELEA)
    except cv2.error:
        return None
    return cv2.resize(estimated, (w, h), interpolation=cv2.INTER_LINEAR)


def _foreground_score(frame, background):
    """Per-pixel colour distance between the photo and the local backdrop."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    backdrop = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(lab - backdrop, axis=2)


def _cloth_contrast(score, core):
    """Typical separation between cloth and backdrop inside the candidate.

    The first segmentation can be loose and include part of the table, and
    the garment can carry high contrast details (prints, brand plates). The
    dominant peak of the separation histogram identifies the cloth itself,
    where a median would mix the table with the details of the garment.
    """
    values = score[core]
    if values.size < 100:
        return None
    values = values[values > 1.0]
    if values.size < 100:
        return None
    scaled = np.clip(values, 0, 255).astype(np.uint8)
    histogram = np.bincount(scaled, minlength=256).astype(np.float64)
    histogram = np.convolve(histogram, np.ones(5) / 5.0, mode="same")
    peak = int(np.argmax(histogram[2:])) + 2
    return float(peak)


def _score_threshold(score, mask, cloth_score=None):
    """Robust noise level of the backdrop close to the garment.

    The samples are taken in a ring just outside the silhouette, where the
    surface is the one the garment rests on. Measuring the noise there (and
    not on the whole frame) keeps the threshold valid when the photo also
    shows the room, the table frame or other dark surroundings.
    """
    outside = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) == 0
    near = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ) > 0
    values = score[outside & near]
    if values.size < 200:
        values = score[cv2.dilate(
            mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ) == 0]
    if values.size < 200:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.8, median + 3.0 * mad)
    if cloth_score:
        # Pale cloth barely differs from the light table. The threshold then
        # has to follow that weak contrast, otherwise the transition band of
        # the real edge is cut away and the silhouette turns ragged.
        threshold = max(threshold, min(3.0, 0.45 * cloth_score))
    return threshold


def _boundary_contrast(mask, score):
    """Median separation between cloth and backdrop along the silhouette.

    The band is measured a little inside the contour, where the cloth has to
    be present. A mask that still ends in the backdrop shows a low contrast
    and is rejected instead of being published as a cutout.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    band = (mask > 0) & (distance > 2.0) & (distance <= 6.0)
    if not band.any():
        return 0.0
    return float(np.median(score[band]))


def _drop_table_specks(mask, score, threshold):
    """Remove isolated table fragments left by the segmentation."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 2:
        return mask
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= largest * 0.02:
            keep[label] = True
            continue
        component = labels == label
        if float(np.median(score[component])) >= threshold * 1.5:
            keep[label] = True
    return keep[labels].astype(np.uint8)


def _refine_against_score(score, mask, low, high, growth=5):
    """Keep the cloth and drop the table captured around it.

    The rule is spatial, not only chromatic: the table is the region that
    reaches the outside of the candidate, while the garment is what is
    enclosed by it. Colour decides how the two are told apart:
    * a region that does not look like cloth and reaches the outside is
      table, and that is what removes the halo around the silhouette;
    * a region that does not look like cloth but is enclosed by cloth is a
      highlight on the fabric, and it is kept;
    * a region that looks like cloth and touches the clearly cloth core is
      the garment with its antialiased transition band.
    """
    outside = (mask == 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    weak = (score >= low) & (mask > 0)
    strong = (score >= high) & (mask > 0)
    if not strong.any():
        return None
    reach = cv2.dilate(
        strong.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (growth * 2 + 1, growth * 2 + 1)
        ),
    ) > 0
    adjacent = cv2.dilate(outside.astype(np.uint8), kernel) > 0
    work = np.zeros(mask.shape, dtype=bool)
    for region, is_weak in ((weak, True), (~weak & (mask > 0), False)):
        count, labels = cv2.connectedComponents(region.astype(np.uint8), 8)
        touches_outside = np.zeros(count, dtype=bool)
        touches_outside[np.unique(labels[adjacent])] = True
        touches_core = np.zeros(count, dtype=bool)
        if is_weak:
            touches_core[np.unique(labels[reach])] = True
        keep = ~touches_outside | touches_core
        keep[0] = False
        work |= keep[labels]
    return work.astype(np.uint8)


def _boundary_is_clean(mask):
    """Reject a silhouette that the refinement left ragged.

    When cloth and backdrop are almost the same colour the refinement can
    eat into the garment and leave a ragged contour. Such a cutout is a
    defective segmentation, so it is not published: the caller keeps the
    original photograph instead. A clean cutout loses less than one percent
    of its area when the small notches of the contour are closed.
    """
    closed = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    area = float(np.count_nonzero(mask))
    closed_area = float(np.count_nonzero(closed))
    if area <= 0 or closed_area <= 0:
        return False
    return area / closed_area >= 0.99


def _tighten_against_background(frame, mask):
    """Drop table pixels captured by the initial segmentation.

    GrabCut marks a band of the light table as probable foreground. On a white
    catalog canvas that band renders as a grey halo. Removing only the pixels
    that do not belong to the cloth keeps collars, sleeves, seams and thin
    details, because cloth separates from the table by more than the backdrop
    noise.
    """
    background = _background_estimate(frame, mask)
    if background is None:
        return None
    score = _foreground_score(frame, background)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    core = (mask > 0) & (distance > 6.0)
    if not core.any():
        return None
    cloth_score = _cloth_contrast(score, core)
    if cloth_score is None:
        return None
    low = _score_threshold(score, mask, cloth_score)
    if low is None:
        return None
    # The strong threshold must stay clearly below the cloth contrast, or a
    # pale garment would have no core at all and nothing could grow from it.
    high = min(max(0.35 * cloth_score, low * 1.2), 0.85 * cloth_score)
    if high <= low:
        high = min(low * 1.5, max(low + 0.1, 0.85 * cloth_score))
    if high <= low:
        return None
    work = _refine_against_score(score, mask, low, high)
    if work is None:
        return None
    area = int(np.count_nonzero(work))
    if area == 0 or area < int(np.count_nonzero(mask)) * 0.30:
        return None
    work = _drop_table_specks(work, score, low)
    if not np.count_nonzero(work):
        return None
    contrast = _boundary_contrast(work, score)
    if contrast < MIN_BOUNDARY_CONTRAST:
        # Cloth and backdrop are not separable: refuse to publish a cutout.
        return None
    if contrast < 0.45 * cloth_score:
        # The silhouette still ends inside the backdrop.
        return None
    if not _boundary_is_clean(work):
        # The refinement left a ragged contour: the cutout is defective.
        return None
    return work, background, score


def _unmix_edge(frame, background, mask):
    """Remove backdrop spill from the antialiased garment boundary.

    Edge pixels are a mix of cloth and backdrop. Unmixing them against the
    local backdrop and recomposing on pure white removes the coloured rim
    that JPEG and studio lighting leave around the silhouette.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    band = (mask > 0) & (distance <= 2.0)
    inner = (mask > 0) & (distance > 2.0) & (distance <= 6.0)
    if not band.any() or not inner.any():
        return frame
    cloth = np.median(frame[inner].astype(np.float32), axis=0)
    table = np.median(background[inner].astype(np.float32), axis=0)
    delta = cloth - table
    norm = float(np.dot(delta, delta))
    if norm < 400.0:
        # Cloth and backdrop are too close in colour for a stable unmix.
        return frame
    pixels = frame.astype(np.float32)
    alpha = np.zeros(pixels.shape[:2], np.float32)
    for channel in range(3):
        alpha += (pixels[:, :, channel] - table[channel]) * delta[channel]
    alpha = np.clip(alpha / norm, 0.0, 1.0)
    # Only genuine mixtures of cloth and table are recomposed. A darker pixel
    # (a fold or a shadow on the fabric) is cloth, not spill, and must keep
    # its original value.
    partial = band & (alpha > 0.02) & (alpha < 0.97)
    if not partial.any():
        return frame
    out = pixels.copy()
    for channel in range(3):
        mixed = alpha * cloth[channel] + (1.0 - alpha) * 255.0
        out[:, :, channel] = np.where(
            partial, mixed, pixels[:, :, channel]
        )
    return out.clip(0, 255).astype(np.uint8)


def _catalog_canvas(frame, mask, bbox, background=None):
    """Tight white photograph of the complete garment, preserving its aspect.

    Margins are measured in output pixels. Canvas dimensions follow the
    garment instead of nesting a small cutout in a fixed portrait frame.
    This rendering function is never used by the legacy AI mask path.
    """
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    garment = frame[y:y + height, x:x + width]
    # Binary encodings 0/1 and 0/255 mean the same thing. Alpha must be 0/1,
    # never 255: multiplying colours by 255 can erase or bleach the garment.
    source_mask = (mask[y:y + height, x:x + width] > 0).astype(np.uint8)
    # Presentation is the captured photograph, not a recolouring of its edge.
    # Keep native pixels and resolution; CSS scales the separate PNG for display.
    scale = 1.0
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    alpha = source_mask.astype(np.float32)
    pixels = garment.astype(np.float32) * alpha[:, :, None]
    if target != (width, height):
        pixels = cv2.resize(pixels, target, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, target, interpolation=cv2.INTER_AREA)
    # Resize premultiplied colour and alpha together to avoid dark edge halos.
    composed = np.clip(np.rint(pixels + 255 * (1 - alpha[:, :, None])), 0, 255).astype(np.uint8)
    margin = max(8, round(max(target) * .04))
    canvas = np.full((target[1] + margin * 2, target[0] + margin * 2, 3), 255, np.uint8)
    canvas[margin:margin + target[1], margin:margin + target[0]] = composed
    ok, encoded = cv2.imencode('.png', canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return encoded.tobytes() if ok and encoded.size else None



def _presentation_foreground_mask(frame, mask):
    """Validate display polarity against visible cloth, without changing IA masks.

    A complement is considered only when an independently detected dominant
    cloth colour supports it. Geometry alone cannot distinguish a light-table
    panel from a garment. Ambiguous/incomplete selections return None.
    """
    mask = np.asarray(mask)
    if mask.shape != frame.shape[:2] or not np.isfinite(mask).all():
        return None
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        return None
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center = hsv[round(h * .2):round(h * .8), round(w * .2):round(w * .8)]
    colored = (center[:, :, 1] > 55) & (center[:, :, 2] > 35)
    cloth = None
    cloth_support = None
    if np.count_nonzero(colored) >= colored.size * .20:
        hues = center[:, :, 0][colored]
        histogram = np.bincount(hues, minlength=180)
        dominant = int(histogram.argmax())
        distance = np.abs(hsv[:, :, 0].astype(np.int16) - dominant)
        distance = np.minimum(distance, 180 - distance)
        central_distance = np.abs(hues.astype(np.int16) - dominant)
        central_distance = np.minimum(central_distance, 180 - central_distance)
        if np.mean(central_distance <= 12) >= .80:
            seeds = ((distance <= 12) & (hsv[:, :, 1] > 55)
                     & (hsv[:, :, 2] > 35)).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
            if count > 1:
                label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                cloth = labels == label
                if np.count_nonzero(cloth) < h * w * MIN_COMPONENT_FRACTION:
                    cloth = None
                else:
                    cloth_support = ((distance <= 18) & (hsv[:, :, 1] > 25)
                                     & (hsv[:, :, 2] > 20))

    # Do not guess a complement for neutral/ambiguous photographs.
    candidates = (binary, 1 - binary) if cloth is not None else (binary,)
    for candidate in candidates:
        validated = _validate_and_refine(candidate, frame.shape)
        if validated is None:
            continue
        selected, _ = validated
        if cloth is not None:
            retained = np.count_nonzero((selected > 0) & cloth) / np.count_nonzero(cloth)
            if retained < .97:
                continue
            # Keeping the garment somewhere inside a light-table rectangle
            # is insufficient: most selected pixels must belong to cloth.
            precision = np.count_nonzero((selected > 0) & cloth_support) / np.count_nonzero(selected)
            if precision < .80:
                continue
        # Small enclosed details (labels/plates) retain their original pixels;
        # never punch them out as white. Large missing panels remain a failure.
        contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer = np.zeros_like(selected)
        cv2.drawContours(outer, contours, -1, 1, cv2.FILLED)
        holes = ((outer > 0) & (selected == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
        maximum = np.count_nonzero(outer) * .025
        if any(stats[label, cv2.CC_STAT_AREA] > maximum for label in range(1, count)):
            continue
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= maximum:
                selected[labels == label] = 1
        if _mask_is_complete(selected, frame.shape):
            return selected
    return None


def _render_presentation(frame, mask, background):
    """Render source pixels with the display mask, never a training mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        return None
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        return None
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != frame.shape[:2]:
        size = (frame.shape[1], frame.shape[0])
        mask = (cv2.resize(mask.astype(np.float32), size,
                           interpolation=cv2.INTER_LINEAR) >= .5).astype(np.uint8)
        if background is not None:
            background = cv2.resize(background, size, interpolation=cv2.INTER_LINEAR)
    foreground = _presentation_foreground_mask(frame, mask)
    if foreground is None:
        return None
    if np.count_nonzero((foreground > 0) & (mask > 0)) < np.count_nonzero(foreground) * .50:
        # An estimate fitted using reversed labels belongs to the wrong side;
        # use unchanged source pixels instead of bleaching corrected edges.
        background = None
    return _catalog_canvas(frame, foreground, cv2.boundingRect(foreground), background=background)


def _chromatic_display_candidate(frame):
    """Separate distinctly coloured cloth even where it overlaps a dark frame.

    A dominant, saturated colour in the central area supplies foreground
    seeds; local GrabCut then recovers hems and shaded sleeves. This is a
    presentation-only candidate, never the mask used for guided AI capture.
    Ambiguous colours, clipped garments and disconnected panels are rejected.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center = hsv[round(h * .2):round(h * .8), round(w * .2):round(w * .8)]
    colored = center[:, :, 1] > 40
    if np.count_nonzero(colored) < center.shape[0] * center.shape[1] * .20:
        return None
    hues = center[:, :, 0][colored]
    histogram = np.bincount(hues, minlength=180)
    # Circular window also supports red cloth crossing hue zero.
    strengths = np.array([sum(histogram[(i + j) % 180]
                              for j in range(-10, 11)) for i in range(180)])
    dominant = int(histogram.argmax())
    if strengths[dominant] / hues.size < .75:
        return None
    distance = np.abs(hsv[:, :, 0].astype(np.int16) - dominant)
    distance = np.minimum(distance, 180 - distance)
    candidate = ((distance <= 16) & (hsv[:, :, 1] > 30)
                 & (hsv[:, :, 2] > 25)).astype(np.uint8)
    # Remove thin coloured reflections of the frame before fitting the model.
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8))
    validated = _validate_and_refine(candidate, frame.shape)
    if validated is None:
        return None
    seed, _ = validated
    core = cv2.erode(seed, np.ones((9, 9), np.uint8))
    if np.count_nonzero(core) < seed.size * .02:
        return None
    labels = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    # A wide unknown band lets the light-box rails join shaded sleeve hems.
    # Colour-supported cloth already supplies the silhouette; refine locally.
    labels[cv2.dilate(seed, np.ones((5, 5), np.uint8)) > 0] = cv2.GC_PR_BGD
    labels[seed > 0] = cv2.GC_PR_FGD
    labels[core > 0] = cv2.GC_FGD
    try:
        cv2.setRNGSeed(SEED_SEGMENTATION)
        cv2.grabCut(frame, labels, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Colour seeds must not erase labels, metallic badges or printed details.
    # Restore their original pixels inside small enclosed silhouette holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = np.zeros_like(mask)
    cv2.drawContours(outer, contours, -1, 1, cv2.FILLED)
    holes = ((outer > 0) & (mask == 0)).astype(np.uint8)
    count, hole_labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    limit = np.count_nonzero(mask) * .025
    fill = np.zeros(count, dtype=bool)
    for label in range(1, count):
        fill[label] = stats[label, cv2.CC_STAT_AREA] <= limit
    mask[fill[hole_labels]] = 1
    final = _validate_and_refine(mask, frame.shape)
    if final is None or not _mask_is_complete(final[0], frame.shape):
        return None
    mask, _ = final
    # Dark rails can contribute thin connected projections at sleeve corners.
    # Remove only narrow projections, rejecting a substantial silhouette change.
    ring = (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0) & (mask == 0)
    surround = frame[ring].astype(np.float32)
    if not surround.size:
        return None
    dark_neutral = (np.mean(surround, axis=1) < 150) & (np.std(surround, axis=1) < 16)
    if np.mean(dark_neutral) > .08:
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        if np.count_nonzero(mask & (1 - cleaned)) > np.count_nonzero(mask) * .01:
            return None
        final = _validate_and_refine(cleaned, frame.shape)
        if final is None or not _mask_is_complete(final[0], frame.shape):
            return None
        mask, _ = final
    # Do not accept a colour match that lost substantial foreground seeds.
    if np.count_nonzero((mask > 0) & (core > 0)) < np.count_nonzero(core) * .995:
        return None
    return mask


def _neutral_backdrop_candidate(frame):
    """Full-frame candidate for a garment on a light neutral surface.

    Unlike the capture mask, presentation must include collars and sleeves
    inside the outer eight percent. Only this display candidate uses those
    pixels; guided capture keeps its previous segmentation unchanged.
    """
    h, w = frame.shape[:2]
    band = max(2, round(min(h, w) * .02))
    samples = np.concatenate((frame[:band].reshape(-1, 3),
                              frame[-band:].reshape(-1, 3),
                              frame[:, :band].reshape(-1, 3),
                              frame[:, -band:].reshape(-1, 3)))
    median = np.median(samples, axis=0)
    deviation = float(np.median(np.abs(samples.astype(np.float32) - median)))
    if float(np.mean(median)) < 175 or deviation > 12:
        return None
    background = np.full_like(frame, np.clip(median, 0, 255).astype(np.uint8))
    score = _foreground_score(frame, background)
    threshold = max(3.0, deviation * 3.0)
    candidate = (score > threshold).astype(np.uint8)
    candidate[:2] = candidate[-2:] = 0
    candidate[:, :2] = candidate[:, -2:] = 0
    return candidate


def _recover_display_candidate(frame, mask):
    """Separate soft table shadows from the complete garment silhouette.

    Try thresholds relative to the cloth, rather than accepting a halo or
    refusing a whole reference because a low noise threshold retained it.
    Enclosed bright fabric and decoration remain part of the garment.
    """
    background = _background_estimate(frame, mask)
    if background is None:
        return None
    score = _foreground_score(frame, background)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    cloth = _cloth_contrast(score, (mask > 0) & (distance > 6))
    if cloth is None or cloth < MIN_BOUNDARY_CONTRAST:
        return None
    best = None
    for fraction in (.12, .22, .32, .45):
        low = max(.8, fraction * cloth)
        high = max(low * 1.2, .55 * cloth)
        work = _refine_against_score(score, mask, low, high)
        if work is None or np.count_nonzero(work) < np.count_nonzero(mask) * .60:
            continue
        refined = _validate_and_refine(work, frame.shape)
        if refined is None:
            continue
        work, bbox = refined
        if not _mask_is_complete(work, frame.shape) or not _boundary_is_clean(work):
            continue
        contrast = _boundary_contrast(work, score)
        if contrast < max(MIN_BOUNDARY_CONTRAST, .40 * cloth):
            continue
        if best is None or contrast > best[0]:
            best = (contrast, work, bbox, background)
    return best


def garment_manual_preview(image_bytes, points, quarter_turns=0):
    """Explicit human outline for catalog presentation; source is read-only."""
    frame = _decode(image_bytes)
    if frame is None or not isinstance(points, list) or not 3 <= len(points) <= 128:
        raise ValueError("Marque entre 3 y 128 puntos alrededor de toda la prenda.")
    if not isinstance(quarter_turns, int) or quarter_turns not in (0, 1, 2, 3):
        raise ValueError("Giro de imagen no valido.")
    try:
        polygon = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError("Contorno no valido.") from None
    if polygon.shape != (len(points), 2) or not np.isfinite(polygon).all() or np.any(polygon <= 0) or np.any(polygon >= 1):
        raise ValueError("El contorno debe quedar dentro de la fotografia original.")
    h, w = frame.shape[:2]
    polygon = np.rint(polygon * [w - 1, h - 1]).astype(np.int32)
    area = abs(cv2.contourArea(polygon)) / (w * h)
    if not .018 <= area <= .90:
        raise ValueError("El contorno no encierra una prenda completa.")
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    if quarter_turns:
        frame = np.ascontiguousarray(np.rot90(frame, -quarter_turns))
        mask = np.ascontiguousarray(np.rot90(mask, -quarter_turns))
    return _catalog_canvas(frame, mask, cv2.boundingRect(mask))


def garment_visual_preview(image_bytes: bytes, *, mask_only=False):
    """Return conservative display cutout PNG, mask for legacy caller, or None.

    The mask_only path is retained for compatibility with guided capture and
    does not use any of the display crop/white-background rendering behavior.
    It is otherwise unchanged in its output contract (full-frame uint8 mask).
    """
    frame = _decode(image_bytes)
    if frame is None:
        return None
    original_h, original_w = frame.shape[:2]
    h, w = original_h, original_w
    if min(h, w) < 130 or h * w > 18_000_000:
        return None

    analysis = frame
    if mask_only:
        # Legacy capture path: unchanged behaviour, full-frame binary mask.
        candidate = _grabcut_mask(analysis)
        if candidate is None:
            return None
        validated = _validate_and_refine(candidate, analysis.shape)
        if validated is None:
            return None
        mask, _bbox = validated
        if not _mask_is_complete(mask, analysis.shape):
            return None
        if mask.shape != (original_h, original_w):
            mask = cv2.resize(mask, (original_w, original_h),
                              interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8) * 255

    # Display analysis is bounded; original pixels are used for the final PNG.
    # This branch is after the legacy mask return, so IA inputs are unchanged.
    display_scale = min(1.0, 1280 / max(original_h, original_w))
    if display_scale < 1.0:
        analysis = cv2.resize(frame, (round(original_w * display_scale),
                                      round(original_h * display_scale)),
                              interpolation=cv2.INTER_AREA)

    chromatic = _chromatic_display_candidate(analysis)
    if chromatic is not None:
        return _render_presentation(frame, chromatic, None)

    # A neutral studio/light-table surround can be segmented directly. This
    # avoids running a full colour-model fit before recovering a clean edge.
    neutral = _neutral_backdrop_candidate(analysis)
    if neutral is not None:
        validated = _validate_and_refine(neutral, analysis.shape)
        if validated is not None and _mask_is_complete(validated[0], analysis.shape):
            recovered = _recover_display_candidate(analysis, validated[0])
            if recovered is not None:
                _quality, mask, bbox, background = recovered
                return _render_presentation(frame, mask, background)

    best = None
    candidates = list(_display_candidates(analysis))
    if neutral is not None:
        candidates.append(neutral)
    for candidate in candidates:
        validated = _validate_and_refine(candidate, analysis.shape)
        if validated is None:
            continue
        mask, bbox = validated
        if not _mask_is_complete(mask, analysis.shape):
            continue
        # Remove the light-table band that the initial segmentation keeps
        # around the cloth; that band is what produced the visible halo.
        tightened = _tighten_against_background(analysis, mask)
        if tightened is None:
            recovered = _recover_display_candidate(analysis, mask)
            if recovered is not None and (best is None or recovered[0] > best[0]):
                best = recovered
            continue
        work, background, score = tightened
        final = _validate_and_refine(work, analysis.shape)
        if final is None:
            continue
        work, bbox = final
        if not _mask_is_complete(work, analysis.shape):
            continue
        quality = _boundary_contrast(work, score)
        if best is None or quality > best[0]:
            best = (quality, work, bbox, background)
    if best is None:
        return None
    _quality, mask, bbox, background = best
    # Do not infer orientation from silhouette proportions: cropped garments
    # may legitimately be wider. Preserve source orientation absent reliable
    # EXIF or semantic evidence.
    return _render_presentation(frame, mask, background)
