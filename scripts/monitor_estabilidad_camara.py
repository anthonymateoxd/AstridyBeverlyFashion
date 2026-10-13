"""Monitoreo no invasivo de cámara Dahua: TCP, salud HTTP y actividad del lector."""
import argparse
import datetime as dt
import json
import re
import socket
import subprocess
import time
import urllib.request


def prueba_tcp(ip, puerto):
    try:
        with socket.create_connection((ip, puerto), timeout=2):
            return True
    except OSError:
        return False


def prueba_salud():
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=3) as r:
            return r.status == 200 and json.load(r).get("status") == "ok"
    except (OSError, ValueError):
        return False


def actividad(inicio):
    try:
        p = subprocess.run(
            ["docker", "logs", "--timestamps", "--since", inicio, "astrid_quality_app"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"registros": None, "intervalo_maximo_s": None}
    marcas = []
    for linea in (p.stdout + p.stderr).splitlines():
        if "source=camera_capture_worker" not in linea:
            continue
        marca = re.match(r"^(\S+)", linea)
        if marca:
            try:
                marcas.append(dt.datetime.fromisoformat(marca.group(1).replace("Z", "+00:00")))
            except ValueError:
                pass
    marcas.sort()
    intervalos = [(b - a).total_seconds() for a, b in zip(marcas, marcas[1:])]
    return {"registros": len(marcas), "intervalo_maximo_s": round(max(intervalos), 2) if intervalos else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.0.5")
    ap.add_argument("--duracion", type=int, default=120)
    ap.add_argument("--intervalo", type=int, default=5)
    args = ap.parse_args()
    if not 5 <= args.duracion <= 3600 or not 1 <= args.intervalo <= 30:
        ap.error("Duración: 5–3600 segundos; intervalo: 1–30 segundos")
    inicio = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fin = time.monotonic() + args.duracion
    pruebas = fallos_tcp = fallos_web = 0
    while time.monotonic() < fin:
        fallos_tcp += not prueba_tcp(args.ip, 554)
        fallos_web += not prueba_salud()
        pruebas += 1
        if fin > time.monotonic():
            time.sleep(min(args.intervalo, fin - time.monotonic()))
    lector = actividad(inicio)
    print(json.dumps({
        "inicio_utc": inicio, "duracion_s": args.duracion, "muestras": pruebas,
        "fallos_rtsp_tcp": fallos_tcp, "fallos_salud_web": fallos_web,
        "lector": lector,
        "nota": "Las comprobaciones TCP no sustituyen la inspección visual del video.",
    }, indent=2, ensure_ascii=False))
    return 0 if not (fallos_tcp or fallos_web) and lector["registros"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
