#!/usr/bin/env python3
"""Local test UI for Decision-LLM.

Run:   python3 server.py            (then open http://localhost:8787)
       python3 server.py --mock     (no API calls, fake Jev)

The API key stays on this server. The page streams each word as Jev picks it.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import decision_llm as m

HERE = Path(__file__).parent
INDEX = HERE / "static" / "index.html"
ARGS: argparse.Namespace


@lru_cache(maxsize=4)
def dictionary(vocab: int) -> m.Dictionary:
    return m.Dictionary(HERE / "words.txt", vocab)


def clamp(value: str | None, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def rounded(r: m.Ranked, n: int) -> list[list]:
    return [[k, round(v or 0, 4)] for k, v in r[:n]]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/generate":
            self.generate(urllib.parse.parse_qs(url.query))
        else:
            self.send_error(404)

    def event(self, name: str, data: dict):
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def generate(self, qs: dict):
        question = (qs.get("q") or [""])[0].strip()[:500]
        max_words = clamp((qs.get("max_words") or [None])[0], 1, 60, 20)
        vocab = clamp((qs.get("vocab") or [None])[0], 1000, 50000, 20000)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            if not question:
                self.event("failed", {"message": "Type a question first."})
                return
            d = dictionary(vocab)
            jev = m.MockJev() if ARGS.mock else m.Jev(m.load_key())
            ctx = m.Context(question)
            t0 = time.time()
            with ThreadPoolExecutor(16) as pool:
                for i in range(max_words):
                    r = m.generate_word(jev, d, ctx, pool)
                    self.event("word", {
                        "index": i,
                        "token": r.token,
                        "answer": ctx.answer_text(),
                        "calls": r.calls,
                        "seconds": round(r.seconds, 2),
                        "next": rounded(r.next, 5),
                        "letters": rounded(r.letters, 6),
                        "tried": r.tried,
                        "candidates": r.candidates,
                        "final": rounded(r.final, 6),
                    })
                    if r.token == "<end>":
                        break
            s = jev.stats
            self.event("done", {
                "calls": s.calls,
                "seconds": round(time.time() - t0, 1),
                "input_tokens": s.input_tokens,
                "mock": ARGS.mock,
            })
        except (BrokenPipeError, ConnectionResetError):
            return  # the page pressed Stop or closed
        except SystemExit as e:  # Jev API errors surface as SystemExit in decision_llm
            try:
                self.event("failed", {"message": str(e)})
            except OSError:
                pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--mock", action="store_true", help="fake Jev, no API calls")
    ARGS = ap.parse_args()
    if not ARGS.mock:
        m.load_key()  # fail fast if the key is missing
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"Decision-LLM test UI: http://localhost:{ARGS.port}" + ("  (mock Jev)" if ARGS.mock else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
