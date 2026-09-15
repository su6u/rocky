"""Local development server for the simulator.

  python3 tools/serve.py [--port 8792]      then open http://127.0.0.1:8792/  (?scenario=walk, ?scenario=drive)

Serves the simulator directory over HTTP (module workers and binary assets do not load from file://) with
caching disabled, so edited modules are always reloaded.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # simulator/


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".ktx2": "image/ktx2",
                      ".wasm": "application/wasm", ".webp": "image/webp"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Rocky Motion Lab: http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(ROOT))).serve_forever()


if __name__ == "__main__":
    main()
