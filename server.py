#!/usr/bin/env python3
"""Local test page for Decision-LLM.

Run:  python3 server.py      then open http://localhost:8787

The API keys stay on this server. The page asks the same question to Jev and to Clef-omni
and streams each model's steps side by side.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import decision_llm as m

INDEX = Path(__file__).parent / "static" / "index.html"
SPELLING: m.Spelling | None = None

# Display only (Jev never sees this): capitalize names, since Jev spells in lowercase.
EXTRA_NAMES = """shakespeare france london tokyo einstein china africa europe asia earth paris
    jupiter saturn mars venus mercury neptune uranus pluto newton darwin rome italy spain germany
    india japan canada australia mexico brazil russia egypt england britain ireland scotland
    washington lincoln obama berlin madrid moscow beijing sydney chicago texas california""".split()


def load_names() -> set[str]:
    names = set(EXTRA_NAMES)
    try:
        words = Path("/usr/share/dict/words").read_text().split()
    except OSError:
        return names
    lower = {w for w in words if w.islower()}
    return names | {w.lower() for w in words if w[:1].isupper() and w.lower() not in lower}


NAMES: set[str] = set()


def pretty(answer: str, question: str) -> str:
    """Capitalize the first letter, names, and words the question capitalized."""
    first = question.split()[:1]  # the question's first word is capitalized anyway
    asked = {w.lower() for w in re.findall(r"\b[A-Z][a-z]+", question) if [w] != first}
    out =re.sub(r"[a-z]+", lambda mo: mo[0].capitalize() if mo[0] in NAMES or mo[0] in asked
                 else mo[0], answer)
    return out[:1].upper() + out[1:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
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
        model = (qs.get("model") or ["jev"])[0]
        try:
            max_chars = max(1, min(1000, int((qs.get("max_chars") or ["300"])[0])))
        except ValueError:
            max_chars = 300
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            if not question:
                self.event("failed", {"message": "Type a question first."})
                return
            jev = m.make_decider(model)
            t0 = time.time()
            last = None
            for step in m.generate(jev, question, max_chars, spelling=SPELLING):
                last = step
                self.event("step", {
                    "key": step.key,
                    "answer": pretty(step.answer, question),
                    "seconds": round(step.seconds, 2),
                    "letters": [[k, round(p or 0, 4)] for k, p in step.letters],
                    "word_complete": step.word_complete,
                    "on_track": step.on_track,
                    "locked": step.locked,
                    "answer_complete": step.answer_complete,
                    "target": step.target,
                    "pool": step.pool,
                    "typed": step.typed,
                })
            self.event("done", {
                "complete": bool(last and last.key == m.DONE),
                "calls": jev.calls,
                "seconds": round(time.time() - t0, 1),
                "input_tokens": jev.input_tokens,
            })
        except (BrokenPipeError, ConnectionResetError):
            return  # Stop pressed or page closed
        except RuntimeError as e:
            try:
                self.event("failed", {"message": str(e)})
            except OSError:
                pass


def main():
    global SPELLING, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    SPELLING = m.Spelling()
    NAMES = load_names()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Decision-LLM test page: http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
