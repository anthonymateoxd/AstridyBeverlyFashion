"""Serialize manual capture requests without changing inference or inspection data."""
import threading
import time
from contextlib import contextmanager


class InspectionBusy(RuntimeError):
    pass


class ManualInspectionGate:
    def __init__(self, cooldown=2.0, clock=time.monotonic):
        self.cooldown = cooldown
        self.clock = clock
        self.lock = threading.Lock()
        self.last_finished = None
        self.last_sequence = None

    @contextmanager
    def acquire(self, connect):
        if not self.lock.acquire(blocking=False):
            raise InspectionBusy("Ya hay una inspección manual en curso. Espere a que finalice.")
        conn = cursor = None
        acquired = False
        try:
            if self.last_finished is not None and self.clock() - self.last_finished < self.cooldown:
                raise InspectionBusy("Espere dos segundos antes de inspeccionar la siguiente prenda.")
            # Session advisory lock also protects installations with several WSGI workers.
            # No table, schema, or historical record is written by this gate.
            conn = connect()
            cursor = conn.cursor()
            cursor.execute("SELECT GET_LOCK('astrid_manual_inspection', 0)")
            acquired = cursor.fetchone()[0] == 1
            if not acquired:
                raise InspectionBusy("Ya hay una inspección manual en curso. Espere a que finalice.")
            cursor.execute(
                "SELECT COUNT(*) FROM inspections "
                "WHERE created_at >= UTC_TIMESTAMP() - INTERVAL 2 SECOND"
            )
            if cursor.fetchone()[0]:
                raise InspectionBusy("Espere dos segundos antes de inspeccionar la siguiente prenda.")
            yield self
        finally:
            try:
                if acquired:
                    cursor.execute("SELECT RELEASE_LOCK('astrid_manual_inspection')")
                    cursor.fetchone()
            finally:
                try:
                    if cursor is not None:
                        cursor.close()
                finally:
                    try:
                        if conn is not None:
                            conn.close()
                    finally:
                        self.lock.release()

    def validate_frame(self, frame, timestamp, sequence):
        if frame is None or timestamp is None or self.clock() - timestamp > 2.0:
            raise InspectionBusy("La imagen de la cámara no está actualizada. Espere una imagen nueva.")
        if sequence == self.last_sequence:
            raise InspectionBusy("Este fotograma ya fue inspeccionado. Espere una imagen nueva.")

    def completed(self, sequence):
        self.last_sequence = sequence
        self.last_finished = self.clock()
