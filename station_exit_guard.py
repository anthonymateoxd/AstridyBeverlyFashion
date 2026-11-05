"""Confirm physical empty-panel evidence; independent of defect inference."""
import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np


class StationExitGuard:
    MIN_FRAMES = 5
    MIN_SECONDS = 2.0
    MAX_AGE = 2.0
    MAX_GAP = 1.0

    def __init__(self, reference_path, bounds, shape, log=None):
        reference = cv2.imread(str(reference_path))
        if reference is None or tuple(reference.shape) != tuple(shape):
            raise RuntimeError("Referencia de panel vacio ausente o geometria diferente.")
        metadata = json.loads(Path(reference_path).with_suffix('.json').read_text())
        if (tuple(metadata['bounds']) != tuple(bounds)
                or metadata['sha256'] != hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()):
            raise RuntimeError("Referencia de panel vacio no corresponde a la region configurada.")
        self.bounds = tuple(bounds)
        self.shape = tuple(shape)
        self.reference = self._sample(reference)
        self.state = "WAITING_EMPTY"
        self.last_sequence = None
        self.last_stamp = None
        self.first_stamp = None
        self.frames = 0
        self.log = log or (lambda row: print("[AUTO_TRANSITION] " + json.dumps(dict(row, at_ms=int(time.time()*1000))), flush=True))
        self.last_evidence = {}
        self.last_log = -float("inf")

    def _sample(self, frame):
        x1, y1, x2, y2 = self.bounds
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

    def block(self, now, token):
        previous = self.state
        self.state = "INSPECTED_WAITING_EXIT"
        self.frames = 0
        self.first_stamp = self.last_stamp = None
        self.log(dict(from_state=previous, to_state=self.state, monotonic=now,
                      garment_token=token, reason="capture_reserved_before_registration"))

    def present(self, now, token):
        previous = self.state
        self.state = "PRESENT"
        self.log(dict(from_state=previous, to_state=self.state, monotonic=now,
                      garment_token=token, reason="garment_detected"))

    def observe(self, frame, coverage, sequence, stamp, now, bounds):
        if sequence == self.last_sequence:
            return False
        self.last_sequence = sequence
        valid = (stamp is not None and 0 <= now - stamp <= self.MAX_AGE
                 and tuple(frame.shape) == self.shape and tuple(bounds) == self.bounds)
        evidence = dict(sequence=int(sequence), monotonic=now, frame_monotonic=stamp,
                        coverage=float(coverage), fresh=bool(valid))
        consistent = False
        if valid and coverage <= .12:
            current = self._sample(frame)
            # Exposure gain/offset may change; geometry must still match the empty panel.
            ref = self.reference.astype(np.float64)
            cur = current.astype(np.float64)
            centered = ref - ref.mean(axis=1, keepdims=True)
            current_centered = cur - cur.mean(axis=1, keepdims=True)
            # The physical LED panel has moving horizontal exposure bands.
            # Correct brightness per scanline, never move/warp the panel geometry.
            gain = np.clip(np.sum(centered * current_centered, axis=1, keepdims=True)
                           / np.maximum(np.sum(centered * centered, axis=1, keepdims=True), 1.0), .4, 1.8)
            delta = cur - gain * ref
            residual = np.abs(delta - np.median(delta, axis=1, keepdims=True))
            changed = float(np.mean(residual > 14.0))
            mean = float(residual.mean())
            contrast = float(np.std(current_centered) / max(float(np.std(centered)), 1.0))
            consistent = .4 <= contrast <= 1.8 and changed <= .02 and mean <= 3.5
            evidence.update(gain_min=float(gain.min()), gain_max=float(gain.max()),
                            contrast_ratio=contrast, changed_fraction=changed, mean_residual=mean)
        if consistent:
            if self.last_stamp is None or not 0 < stamp - self.last_stamp <= self.MAX_GAP:
                self.frames = 0
                self.first_stamp = stamp
            self.frames += 1
            self.last_stamp = stamp
        else:
            self.frames = 0
            self.first_stamp = self.last_stamp = None
        span = stamp - self.first_stamp if self.first_stamp is not None else 0.0
        evidence.update(empty_consistent=bool(consistent), stable_frames=self.frames, stable_seconds=span)
        self.last_evidence = evidence
        ready = self.frames >= self.MIN_FRAMES and span >= self.MIN_SECONDS
        if ready:
            previous = self.state
            self.state = "READY"
            self.log(dict(from_state=previous, to_state=self.state, reason="stable_verified_empty", **evidence))
            return True
        if now - self.last_log >= 1.0:
            self.log(dict(from_state=self.state, to_state=self.state, reason="exit_not_confirmed", **evidence))
            self.last_log = now
        return False
